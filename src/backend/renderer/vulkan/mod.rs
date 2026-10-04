//! Implementation of the rendering traits using Vulkan.
//!
//! The [`VulkanRenderer`] renders using a single graphics queue on a
//! [`PhysicalDevice`]. Rendering happens through [dynamic
//! rendering](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_dynamic_rendering.html)
//! directly into bound [`Dmabuf`]s or offscreen [`VulkanTexture`]s; there is no swapchain
//! involved, presentation is left to e.g. the DRM backend.
//!
//! Internally all work is ordered by a single timeline semaphore: every queue submission
//! signals the next timeline point and resources are destroyed once their last-use point was
//! reached. Returned [`SyncPoint`]s carry a [`VulkanFence`] tracking the submission's timeline
//! point plus a binary semaphore that can be exported as a sync_file for explicit sync.
//!
//! Like the GLES renderer, sampling and blending happen directly on the stored (typically
//! sRGB-encoded) values via `UNORM` formats — no linearization is performed.
//!
//! # Requirements
//!
//! The physical device must support Vulkan 1.3 (for core dynamic rendering and
//! synchronization2) as well as the device extensions returned by
//! [`VulkanRenderer::required_extensions`].

use std::{
    collections::HashMap,
    fmt,
    io::Cursor,
    os::unix::io::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64},
    },
};

use ash::{ext, khr, vk};
use tracing::{info_span, instrument, trace, warn};

use crate::{
    backend::{
        allocator::{
            Buffer as BufferTrait, Format as DrmFormat, Fourcc,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::FormatSet,
        },
        vulkan::{PhysicalDevice, version::Version},
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};

use super::{
    Bind, Blit, Color32F, ContextId, DebugFlags, ExportMem, ImportDma, ImportMem, Offscreen, Renderer,
    RendererSuper, Texture, TextureFilter, sync::SyncPoint,
};
use crate::utils::user_data::UserDataMap;

#[cfg(feature = "wayland_frontend")]
use super::{ImportDmaWl, ImportMemWl};

mod custom;
mod error;
mod fence;
mod format;
mod frame;
mod shaders;
mod texture;

#[cfg(test)]
mod descriptor_pool_tests;

#[cfg(test)]
mod upload_tests;

pub use custom::{
    CustomUniform, CustomUniformDecl, CustomUniformKind, CustomUniformValue, MAX_CUSTOM_PARAMS_SIZE,
    MAX_CUSTOM_TEXTURES, OwnedCustomUniform, VulkanPixelProgram, texture_bindings_glsl, uniform_block_glsl,
};
pub use error::VulkanError;
pub use fence::VulkanFence;
pub use frame::{CustomPass, VulkanFrame, VulkanFrameGuard};
pub use texture::{VulkanTarget, VulkanTexture, VulkanTextureMapping};

use fence::BinarySemaphore;
use format::{FormatInfo, FormatMapping};
use texture::{InnerTexture, RenderBuffer, TargetInner};

/// Resources whose destruction is deferred until a timeline point completes.
#[derive(Debug)]
pub(super) enum CleanupItem {
    Image(vk::Image),
    ImageView(vk::ImageView),
    Memory(vk::DeviceMemory),
    Buffer(vk::Buffer),
    Semaphore(vk::Semaphore),
    DescriptorSet(vk::DescriptorPool, vk::DescriptorSet),
    ShaderModule(vk::ShaderModule),
}

/// Logical device state shared between the renderer, textures and fences.
pub(super) struct Device {
    pub(super) raw: ash::Device,
    pub(super) phd: PhysicalDevice,
    pub(super) queue_family: u32,
    /// Timeline semaphore ordering all submissions of this renderer.
    pub(super) timeline: vk::Semaphore,
    pub(super) external_memory_fd: khr::external_memory_fd::Device,
    pub(super) external_semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    pub(super) memory_props: vk::PhysicalDeviceMemoryProperties,
    cleanup: Mutex<Vec<(u64, CleanupItem)>>,
    /// Pools outlive the renderer when textures still own descriptor sets. The mutex also
    /// provides Vulkan's external synchronization for allocating and freeing pool contents.
    descriptor_pools: Mutex<Vec<(vk::DescriptorPool, u32)>>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("physical_device", &self.phd.name())
            .field("queue_family", &self.queue_family)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Queues resources for destruction once `point` is reached on the timeline.
    pub(super) fn defer_destroy(&self, point: u64, items: Vec<CleanupItem>) {
        let mut guard = self.cleanup.lock().unwrap();
        guard.extend(items.into_iter().map(|item| (point, item)));
    }

    fn completed_point(&self) -> Result<u64, vk::Result> {
        unsafe { self.raw.get_semaphore_counter_value(self.timeline) }
    }

    unsafe fn destroy_item(&self, item: CleanupItem) {
        unsafe {
            match item {
                CleanupItem::Image(image) => self.raw.destroy_image(image, None),
                CleanupItem::ImageView(view) => self.raw.destroy_image_view(view, None),
                CleanupItem::Memory(memory) => self.raw.free_memory(memory, None),
                CleanupItem::Buffer(buffer) => self.raw.destroy_buffer(buffer, None),
                CleanupItem::Semaphore(semaphore) => self.raw.destroy_semaphore(semaphore, None),
                CleanupItem::DescriptorSet(pool, set) => {
                    let mut pools = self.descriptor_pools.lock().unwrap();
                    match self.raw.free_descriptor_sets(pool, &[set]) {
                        Ok(()) => {
                            let (_, free) = pools
                                .iter_mut()
                                .find(|(handle, _)| *handle == pool)
                                .expect("descriptor pool must outlive its sets");
                            debug_assert!(*free < DESCRIPTOR_POOL_SIZE);
                            *free += 1;
                        }
                        Err(err) => warn!(?err, "Failed to free Vulkan descriptor set"),
                    }
                }
                CleanupItem::ShaderModule(module) => self.raw.destroy_shader_module(module, None),
            }
        }
    }

    /// Destroys all queued resources whose timeline point completed.
    fn process_cleanup(&self, up_to: u64) {
        let items = {
            let mut guard = self.cleanup.lock().unwrap();
            let mut items = Vec::new();
            guard.retain_mut(|(point, item)| {
                if *point <= up_to {
                    // Replace with a dummy to move the item out.
                    items.push(std::mem::replace(item, CleanupItem::Image(vk::Image::null())));
                    false
                } else {
                    true
                }
            });
            items
        };
        for item in items {
            unsafe { self.destroy_item(item) };
        }
    }

    /// Finds a memory type index in `type_bits` with the given property flags.
    pub(super) fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory_props.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && self.memory_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.process_cleanup(u64::MAX);
            for (pool, _) in self.descriptor_pools.get_mut().unwrap().drain(..) {
                self.raw.destroy_descriptor_pool(pool, None);
            }
            self.raw.destroy_semaphore(self.timeline, None);
            self.raw.destroy_device(None);
        }
    }
}

/// Per-draw color blend parameters of the built-in texture shader.
///
/// The parameter block is ported from niri's `niri_blend` GLSL stage: it can encode
/// electrical sRGB content into PQ/BT.2020 for HDR outputs, handle extended-linear
/// (scRGB-style) content, convert PQ content back to SDR with ICtCp tone mapping, and
/// re-encode PQ content through a gamut matrix. All fields zero (the default) is a
/// passthrough.
///
/// Boolean parameters use 0.0 / 1.0. See `shaders/texture.frag` for the exact semantics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[allow(missing_docs)]
pub struct ColorBlendParams {
    pub hdr_pq: f32,
    pub ref_lum_scale: f32,
    /// Scale from HDR content reference white to the output reference white.
    /// 1.0 means no adjustment.
    pub hdr_ref_scale: f32,
    pub linear: f32,
    pub linear_scale: f32,
    pub linear_to_ref: f32,
    pub hdr_to_sdr: f32,
    pub pq_gamut: f32,
    pub use_gamut: f32,
    pub tonemap: f32,
    pub tm_v: f32,
    pub tm_ref_scale: f32,
    pub tm_out_scale: f32,
    /// Column-major 3x3 gamut conversion matrix, used when `use_gamut` is set.
    pub gamut: [f32; 9],
}

impl ColorBlendParams {
    /// std140 layout of the shader's parameter block: 13 tightly packed scalars, rounded up
    /// to the next 16-byte boundary, followed by a mat3 (three vec4-aligned columns).
    pub(super) fn to_std140(self) -> [f32; PARAMS_FLOATS] {
        let mut out = [0.0f32; PARAMS_FLOATS];
        out[0] = self.hdr_pq;
        out[1] = self.ref_lum_scale;
        out[2] = self.hdr_ref_scale;
        out[3] = self.linear;
        out[4] = self.linear_scale;
        out[5] = self.linear_to_ref;
        out[6] = self.hdr_to_sdr;
        out[7] = self.pq_gamut;
        out[8] = self.use_gamut;
        out[9] = self.tonemap;
        out[10] = self.tm_v;
        out[11] = self.tm_ref_scale;
        out[12] = self.tm_out_scale;
        for col in 0..3 {
            for row in 0..3 {
                out[16 + col * 4 + row] = self.gamut[col * 3 + row];
            }
        }
        out
    }
}

/// Number of floats in the std140 color blend parameter block.
pub(super) const PARAMS_FLOATS: usize = 28;
/// Fixed range of the ring buffer descriptor; parameter blocks must fit inside.
pub(super) const PARAMS_RANGE: u32 = 512;
const PARAMS_RING_SIZE: u32 = 64 * 1024;
const MAX_CACHED_PARAMS_RINGS: usize = 8;
const MAX_CACHED_UPLOAD_BUFFERS: usize = 4;
const MAX_CACHED_UPLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Mapped transfer storage. A cached allocation is reusable only after its submission completes.
struct UploadBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    capacity: u64,
}

impl UploadBuffer {
    fn defer_destroy(self, device: &Device, point: u64) {
        device.defer_destroy(
            point,
            vec![CleanupItem::Buffer(self.buffer), CleanupItem::Memory(self.memory)],
        );
    }
}

/// Persistently mapped parameter storage, exclusively owned by a frame or the renderer pool.
struct ParamsRing {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    ds_pool: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    capacity: u32,
    used: u32,
}

impl ParamsRing {
    fn defer_destroy(self, device: &Device, point: u64) {
        device.defer_destroy(
            point,
            vec![
                CleanupItem::DescriptorSet(self.ds_pool, self.ds),
                CleanupItem::Buffer(self.buffer),
                CleanupItem::Memory(self.memory),
            ],
        );
    }
}

/// Key identifying a cached graphics pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    format: vk::Format,
    solid: bool,
    blend: bool,
}

/// Push constant block shared by all shaders. Must match `shaders/quad.vert`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(super) struct PushConstants {
    /// Position matrix linear part: m00, m01, m10, m11.
    pub mat_pos: [f32; 4],
    /// Position translation x, y; rect offset x, y.
    pub pos_off_rect: [f32; 4],
    /// Rect size w, h; alpha; tint.
    pub rect_size_misc: [f32; 4],
    /// UV matrix linear part.
    pub mat_uv: [f32; 4],
    /// UV translation x, y.
    pub uv_off: [f32; 4],
    /// Solid color (premultiplied).
    pub color: [f32; 4],
}

impl PushConstants {
    pub(super) fn as_bytes(&self) -> &[u8] {
        // SAFETY: `PushConstants` is `repr(C)` and contains only `f32`s.
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>()) }
    }
}

/// Tracking for a submission that may still be executing.
#[derive(Debug)]
struct InFlight {
    point: u64,
    command_buffers: Vec<vk::CommandBuffer>,
}

const DESCRIPTOR_POOL_SIZE: u32 = 256;

/// A renderer utilizing Vulkan.
pub struct VulkanRenderer {
    device: Arc<Device>,
    queue: vk::Queue,
    context_id: ContextId<VulkanTexture>,

    formats: HashMap<Fourcc, FormatInfo>,
    dmabuf_texture_formats: FormatSet,
    dmabuf_render_formats: FormatSet,

    command_pool: vk::CommandPool,
    free_command_buffers: Vec<vk::CommandBuffer>,
    in_flight: Vec<InFlight>,
    /// Value of the most recently submitted timeline point.
    timeline_point: u64,

    vert_module: vk::ShaderModule,
    tex_frag_module: vk::ShaderModule,
    solid_frag_module: vk::ShaderModule,
    ds_layout: vk::DescriptorSetLayout,
    /// Texture descriptor set layouts by texture count (0..=MAX_CUSTOM_TEXTURES);
    /// index 1 equals `ds_layout`.
    texture_ds_layouts: [vk::DescriptorSetLayout; custom::MAX_CUSTOM_TEXTURES + 1],
    params_ds_layout: vk::DescriptorSetLayout,
    /// Minimum alignment for parameter ring offsets.
    pub(super) params_align: u32,
    /// Parameter blocks retained with their last-use timeline point. Only completed blocks
    /// may be acquired; the pool never waits for an in-flight block to become available.
    params_rings: Vec<(u64, ParamsRing)>,
    upload_buffers: Vec<(u64, UploadBuffer)>,
    /// Pipeline layouts by texture count; index 1 equals `pipeline_layout`.
    pub(super) pipeline_layouts: [vk::PipelineLayout; custom::MAX_CUSTOM_TEXTURES + 1],
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    custom_pipelines: HashMap<(usize, vk::Format, bool), vk::Pipeline>,
    /// Sampler for custom program textures: linear, clamp to transparent border.
    custom_sampler: vk::Sampler,
    shaderc: Option<shaderc::Compiler>,
    /// Samplers per (downscale, upscale) filter combination.
    samplers: HashMap<(TextureFilter, TextureFilter), vk::Sampler>,

    /// Timeline points to wait for on the next submission.
    pending_timeline_waits: Vec<u64>,
    /// Imported binary semaphores to wait for on the next submission.
    pending_binary_waits: Vec<vk::Semaphore>,

    dmabuf_textures: Vec<(WeakDmabuf, VulkanTexture)>,
    render_buffers: Vec<(WeakDmabuf, Arc<RenderBuffer>)>,

    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    debug_flags: DebugFlags,

    /// Default color blend parameters applied to texture draws without a per-draw override.
    pub(super) default_color_params: Option<ColorBlendParams>,
    /// CPU-side transform applied to solid colors (including clears).
    pub(super) solid_color_transform: Option<Box<dyn Fn(Color32F) -> Color32F>>,
    user_data: UserDataMap,

    /// Whether the kernel supports the dmabuf sync_file import/export ioctls.
    ///
    /// `None` until the first attempt.
    implicit_interop: Option<bool>,

    span: tracing::Span,
}

impl fmt::Debug for VulkanRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanRenderer")
            .field("device", &self.device)
            .field("timeline_point", &self.timeline_point)
            .field("debug_flags", &self.debug_flags)
            .finish_non_exhaustive()
    }
}

impl VulkanRenderer {
    /// Returns the list of device extensions required to create a [`VulkanRenderer`].
    ///
    /// The `VK_KHR_external_semaphore_fd` extension is used additionally, if available, to
    /// support exporting and importing sync_files.
    pub fn required_extensions(phd: &PhysicalDevice) -> Vec<&'static std::ffi::CStr> {
        let _ = phd;
        vec![
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
            ext::queue_family_foreign::NAME,
            khr::external_memory_fd::NAME,
        ]
    }

    /// Creates a new [`VulkanRenderer`] from a [`PhysicalDevice`].
    #[instrument(err, skip(phd), fields(physical_device = phd.name()))]
    pub fn new(phd: &PhysicalDevice) -> Result<Self, VulkanError> {
        if phd.api_version() < Version::VERSION_1_3 {
            return Err(VulkanError::UnsupportedVersion(1, 3));
        }

        for extension in Self::required_extensions(phd) {
            if !phd.has_device_extension(extension) {
                return Err(VulkanError::MissingExtension(
                    extension.to_str().unwrap_or("<invalid>"),
                ));
            }
        }
        let has_external_semaphore = phd.has_device_extension(khr::external_semaphore_fd::NAME);

        let instance = phd.instance().handle();

        // Check whether binary semaphores can be exported as sync_files.
        let external_semaphore_features = unsafe {
            let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let mut props = vk::ExternalSemaphoreProperties::default();
            instance.get_physical_device_external_semaphore_properties(phd.handle(), &info, &mut props);
            props.external_semaphore_features
        };
        let sync_fd_export = has_external_semaphore
            && external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE);
        let sync_fd_import = has_external_semaphore
            && external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phd.handle()) };
        let queue_family = queue_families
            .iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(VulkanError::NoGraphicsQueue)? as u32;

        let mut extensions = Self::required_extensions(phd);
        if has_external_semaphore {
            extensions.push(khr::external_semaphore_fd::NAME);
        }
        let extension_pointers: Vec<_> = extensions.iter().map(|ext| ext.as_ptr()).collect();

        let queue_create_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_info)
            .enabled_extension_names(&extension_pointers)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device = unsafe { instance.create_device(phd.handle(), &create_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let timeline = {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            unsafe { device.create_semaphore(&create_info, None) }.inspect_err(|_| {
                unsafe { device.destroy_device(None) };
            })?
        };

        let external_memory_fd = khr::external_memory_fd::Device::new(instance, &device);
        let external_semaphore_fd = (sync_fd_export || sync_fd_import)
            .then(|| khr::external_semaphore_fd::Device::new(instance, &device));
        let memory_props = unsafe { instance.get_physical_device_memory_properties(phd.handle()) };

        let device = Arc::new(Device {
            raw: device,
            phd: phd.clone(),
            queue_family,
            timeline,
            external_memory_fd,
            external_semaphore_fd,
            memory_props,
            cleanup: Mutex::new(Vec::new()),
            descriptor_pools: Mutex::new(Vec::new()),
        });

        // Query supported formats.
        let mut formats = HashMap::new();
        let mut texture_formats = Vec::new();
        let mut render_formats = Vec::new();
        for mapping in format::KNOWN_FORMATS {
            let info = format::query_format_info(phd, *mapping);
            texture_formats.extend(info.texture_modifiers.iter().map(|m| DrmFormat {
                code: mapping.fourcc,
                modifier: m.modifier,
            }));
            render_formats.extend(info.render_modifiers.iter().map(|m| DrmFormat {
                code: mapping.fourcc,
                modifier: m.modifier,
            }));
            formats.insert(mapping.fourcc, info);
        }

        let raw = &device.raw;

        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family);
            unsafe { raw.create_command_pool(&create_info, None) }?
        };

        let vert_module = create_shader_module(raw, shaders::QUAD_VERT)?;
        let tex_frag_module = create_shader_module(raw, shaders::TEXTURE_FRAG)?;
        let solid_frag_module = create_shader_module(raw, shaders::SOLID_FRAG)?;

        let ds_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe { raw.create_descriptor_set_layout(&create_info, None) }?
        };

        let params_ds_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe { raw.create_descriptor_set_layout(&create_info, None) }?
        };

        let params_align = phd.limits().min_uniform_buffer_offset_alignment.max(4) as u32;

        // Texture descriptor set layouts for 0..=MAX_CUSTOM_TEXTURES combined samplers.
        let mut texture_ds_layouts = [vk::DescriptorSetLayout::null(); custom::MAX_CUSTOM_TEXTURES + 1];
        for (count, layout) in texture_ds_layouts.iter_mut().enumerate() {
            if count == 1 {
                *layout = ds_layout;
                continue;
            }
            let bindings: Vec<_> = (0..count as u32)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                })
                .collect();
            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            *layout = unsafe { raw.create_descriptor_set_layout(&create_info, None) }?;
        }

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)];
        let mut pipeline_layouts = [vk::PipelineLayout::null(); custom::MAX_CUSTOM_TEXTURES + 1];
        for (count, layout) in pipeline_layouts.iter_mut().enumerate() {
            let layouts = [texture_ds_layouts[count], params_ds_layout];
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&ranges);
            *layout = unsafe { raw.create_pipeline_layout(&create_info, None) }?;
        }
        let pipeline_layout = pipeline_layouts[1];

        let custom_sampler = {
            let create_info = vk::SamplerCreateInfo::default()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
                .max_lod(0.25);
            unsafe { raw.create_sampler(&create_info, None) }?
        };

        let mut samplers = HashMap::new();
        for downscale in [TextureFilter::Linear, TextureFilter::Nearest] {
            for upscale in [TextureFilter::Linear, TextureFilter::Nearest] {
                let to_vk = |filter| match filter {
                    TextureFilter::Linear => vk::Filter::LINEAR,
                    TextureFilter::Nearest => vk::Filter::NEAREST,
                };
                let create_info = vk::SamplerCreateInfo::default()
                    .min_filter(to_vk(downscale))
                    .mag_filter(to_vk(upscale))
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.25);
                let sampler = unsafe { raw.create_sampler(&create_info, None) }?;
                samplers.insert((downscale, upscale), sampler);
            }
        }

        Ok(Self {
            device,
            queue,
            context_id: ContextId::new(),
            formats,
            dmabuf_texture_formats: texture_formats.into_iter().collect(),
            dmabuf_render_formats: render_formats.into_iter().collect(),
            command_pool,
            free_command_buffers: Vec::new(),
            in_flight: Vec::new(),
            timeline_point: 0,
            vert_module,
            tex_frag_module,
            solid_frag_module,
            ds_layout,
            texture_ds_layouts,
            params_ds_layout,
            params_align,
            params_rings: Vec::new(),
            upload_buffers: Vec::new(),
            pipeline_layouts,
            pipeline_layout,
            pipelines: HashMap::new(),
            custom_pipelines: HashMap::new(),
            custom_sampler,
            shaderc: None,
            samplers,
            pending_timeline_waits: Vec::new(),
            pending_binary_waits: Vec::new(),
            dmabuf_textures: Vec::new(),
            render_buffers: Vec::new(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            debug_flags: DebugFlags::empty(),
            default_color_params: None,
            solid_color_transform: None,
            user_data: UserDataMap::default(),
            implicit_interop: None,
            span: info_span!("renderer_vulkan"),
        })
    }

    /// The underlying [`PhysicalDevice`].
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.device.phd
    }

    /// Returns a [`UserDataMap`] for renderer-associated state.
    pub fn user_data(&self) -> &UserDataMap {
        &self.user_data
    }

    /// Sets the default [`ColorBlendParams`] applied to texture draws.
    ///
    /// A per-draw override set on the frame takes precedence. `None` (the default) is a
    /// passthrough.
    pub fn set_default_color_params(&mut self, params: Option<ColorBlendParams>) {
        self.default_color_params = params;
    }

    /// The current default [`ColorBlendParams`].
    pub fn default_color_params(&self) -> Option<ColorBlendParams> {
        self.default_color_params
    }

    /// Sets a transform applied on the CPU to all solid colors (including clears).
    pub fn set_solid_color_transform(&mut self, transform: Option<Box<dyn Fn(Color32F) -> Color32F>>) {
        self.solid_color_transform = transform;
    }

    /// Compiles a custom fragment shader program from GLSL source.
    ///
    /// The source must target Vulkan GLSL (`#version 450`) and use the uniform block and
    /// sampler declarations generated by [`uniform_block_glsl`] and [`texture_bindings_glsl`]
    /// for the same `uniforms` and `texture_names`.
    pub fn compile_custom_pixel_shader(
        &mut self,
        src: &str,
        uniforms: &[CustomUniformDecl],
        texture_names: &[&str],
    ) -> Result<VulkanPixelProgram, VulkanError> {
        if self.shaderc.is_none() {
            self.shaderc =
                Some(shaderc::Compiler::new().map_err(|err| VulkanError::ShaderCompile(err.to_string()))?);
        }
        custom::compile_program(
            &self.device,
            self.shaderc.as_ref().unwrap(),
            src,
            uniforms,
            texture_names,
        )
    }

    /// Returns (creating if necessary) the pipeline for a custom program and target format.
    pub(super) fn get_custom_pipeline(
        &mut self,
        program: &VulkanPixelProgram,
        format: vk::Format,
        blend: bool,
    ) -> Result<vk::Pipeline, VulkanError> {
        let key = (program.0.id, format, blend);
        if let Some(pipeline) = self.custom_pipelines.get(&key) {
            return Ok(*pipeline);
        }

        let layout = self.pipeline_layouts[program.0.texture_names.len()];
        let raw = &self.device.raw;
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert_module)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(program.0.module)
                .name(entry),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(blend)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline =
            unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) }
                .map_err(|_| VulkanError::PipelineCreation)?
                .into_iter()
                .next()
                .ok_or(VulkanError::PipelineCreation)?;

        self.custom_pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Allocates a transient descriptor set binding the given textures with the custom
    /// sampler, for use within the current frame.
    pub(super) fn custom_texture_descriptor_set(
        &mut self,
        textures: &[&VulkanTexture],
        edge_clamp: bool,
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet), VulkanError> {
        let layout = self.texture_ds_layouts[textures.len()];
        let (pool, set) = self.allocate_descriptor_set(layout)?;
        let sampler = if edge_clamp {
            self.samplers[&(TextureFilter::Linear, TextureFilter::Linear)]
        } else {
            self.custom_sampler
        };
        let image_infos: Vec<_> = textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(texture.0.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let writes: Vec<_> = image_infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };
        Ok((pool, set))
    }

    pub(super) fn device(&self) -> &Arc<Device> {
        &self.device
    }

    fn acquire_params_ring(&mut self) -> Result<ParamsRing, VulkanError> {
        if !self.params_rings.is_empty() {
            // Reusing a mapping requires positive proof of completion. A failed timeline query
            // must never authorize overwriting it.
            let completed = self.device.completed_point()?;
            if let Some(index) = self
                .params_rings
                .iter()
                .position(|(point, _)| *point <= completed)
            {
                let (_, mut ring) = self.params_rings.swap_remove(index);
                ring.used = 0;
                return Ok(ring);
            }
        }

        let (buffer, memory, ptr) =
            self.create_host_buffer(u64::from(PARAMS_RING_SIZE), vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let (ds_pool, ds) = match self.allocate_descriptor_set(self.params_ds_layout) {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe {
                    self.device.raw.destroy_buffer(buffer, None);
                    self.device.raw.free_memory(memory, None);
                }
                return Err(err);
            }
        };
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(u64::from(PARAMS_RANGE))];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&buffer_info);
        unsafe { self.device.raw.update_descriptor_sets(&[write], &[]) };

        Ok(ParamsRing {
            buffer,
            memory,
            ptr: ptr.cast(),
            ds_pool,
            ds,
            capacity: PARAMS_RING_SIZE,
            used: 0,
        })
    }

    fn recycle_params_ring(&mut self, point: u64, ring: ParamsRing) {
        if self.params_rings.len() < MAX_CACHED_PARAMS_RINGS {
            self.params_rings.push((point, ring));
        } else {
            // Bound retained memory without blocking a frame that needs more storage.
            ring.defer_destroy(&self.device, point);
        }
    }

    /// Processes deferred cleanup and recycles completed command buffers.
    fn cleanup(&mut self) {
        let completed = match self.device.completed_point() {
            Ok(completed) => completed,
            Err(err) => {
                // Failed queries (including OOM) do not prove GPU completion. Retain both
                // command buffers and deferred resources until completion can be confirmed.
                warn!(?err, "Unable to query Vulkan completion for resource cleanup");
                return;
            }
        };
        let mut i = 0;
        while i < self.in_flight.len() {
            if self.in_flight[i].point <= completed {
                let in_flight = self.in_flight.swap_remove(i);
                for cb in in_flight.command_buffers {
                    unsafe {
                        let _ = self
                            .device
                            .raw
                            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty());
                    }
                    self.free_command_buffers.push(cb);
                }
            } else {
                i += 1;
            }
        }
        self.device.process_cleanup(completed);
        self.dmabuf_textures.retain(|(weak, _)| !weak.is_gone());
        self.render_buffers.retain(|(weak, _)| !weak.is_gone());
    }

    fn acquire_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanError> {
        if let Some(cb) = self.free_command_buffers.pop() {
            return Ok(cb);
        }
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(4);
        let mut cbs = unsafe { self.device.raw.allocate_command_buffers(&allocate_info) }?;
        let cb = cbs.pop().unwrap();
        self.free_command_buffers.extend(cbs);
        Ok(cb)
    }

    /// Creates the binary semaphore signaled alongside the next timeline point, if sync_file
    /// export is supported.
    fn create_export_semaphore(&self) -> Option<vk::Semaphore> {
        self.device.external_semaphore_fd.as_ref()?;
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        unsafe { self.device.raw.create_semaphore(&create_info, None) }.ok()
    }

    /// Submits command buffers, signaling the next timeline point.
    ///
    /// Consumes any pending waits. Returns the signaled point and the fence for it.
    pub(super) fn submit(
        &mut self,
        command_buffers: &[vk::CommandBuffer],
        extra_binary_waits: Vec<vk::Semaphore>,
        extra_timeline_waits: Vec<u64>,
        with_export: bool,
    ) -> Result<(u64, VulkanFence), VulkanError> {
        let point = self.timeline_point + 1;

        let mut binary_waits = std::mem::take(&mut self.pending_binary_waits);
        binary_waits.extend(extra_binary_waits);
        let mut timeline_waits = std::mem::take(&mut self.pending_timeline_waits);
        timeline_waits.extend(extra_timeline_waits);

        let export_semaphore = with_export.then(|| self.create_export_semaphore()).flatten();

        let mut wait_infos: Vec<vk::SemaphoreSubmitInfo<'_>> = Vec::new();
        for sem in &binary_waits {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        for wait_point in &timeline_waits {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.device.timeline)
                    .value(*wait_point)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        let mut signal_infos = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.device.timeline)
                .value(point)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        if let Some(sem) = export_semaphore {
            signal_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sem)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        let cb_infos: Vec<_> = command_buffers
            .iter()
            .map(|cb| vk::CommandBufferSubmitInfo::default().command_buffer(*cb))
            .collect();

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cb_infos)
            .signal_semaphore_infos(&signal_infos);

        let res = unsafe {
            self.device
                .raw
                .queue_submit2(self.queue, &[submit_info], vk::Fence::null())
        };

        if let Err(err) = res {
            if let Some(sem) = export_semaphore {
                unsafe { self.device.raw.destroy_semaphore(sem, None) };
            }
            for sem in binary_waits {
                unsafe { self.device.raw.destroy_semaphore(sem, None) };
            }
            return Err(err.into());
        }

        self.timeline_point = point;
        self.in_flight.push(InFlight {
            point,
            command_buffers: command_buffers.to_vec(),
        });
        // Imported wait semaphores can be destroyed once the submission completed.
        self.device.defer_destroy(
            point,
            binary_waits.into_iter().map(CleanupItem::Semaphore).collect(),
        );

        let fence = VulkanFence {
            device: self.device.clone(),
            point,
            binary: export_semaphore.map(|sem| Mutex::new(BinarySemaphore::Unexported(sem))),
        };
        Ok((point, fence))
    }

    /// Records and submits a one-shot command buffer.
    ///
    /// `cleanup` is destroyed once the submission completes. Returns the timeline point.
    fn submit_one_shot(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), VulkanError>,
        cleanup: Vec<CleanupItem>,
        binary_waits: Vec<vk::Semaphore>,
        with_export: bool,
    ) -> Result<(u64, VulkanFence), VulkanError> {
        self.cleanup();
        let cb = self.acquire_command_buffer()?;
        let raw = self.device.raw.clone();
        unsafe {
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(cb, &begin_info)?;
        }
        record(&raw, cb)?;
        unsafe { raw.end_command_buffer(cb) }?;

        let (point, fence) = self.submit(&[cb], binary_waits, Vec::new(), with_export)?;
        self.device.defer_destroy(point, cleanup);
        Ok((point, fence))
    }

    /// Handles a [`SyncPoint`] wait by scheduling it for the next submission if possible.
    fn handle_wait(&mut self, sync: &SyncPoint) -> Result<(), VulkanError> {
        if sync.is_reached() {
            return Ok(());
        }

        // Fences from this very renderer wait on our own timeline.
        if let Some(fence) = sync.get::<VulkanFence>() {
            if Arc::ptr_eq(&fence.device, &self.device) {
                self.pending_timeline_waits.push(fence.point);
                return Ok(());
            }
        }

        // Try to import a native fence as a temporary binary semaphore.
        if let Some(fd) = sync.export() {
            if let Some(sem) = self.import_sync_file_semaphore(fd) {
                self.pending_binary_waits.push(sem);
                return Ok(());
            }
        }

        // Fall back to a CPU wait.
        sync.wait().map_err(|_| VulkanError::SyncInterrupted)
    }

    /// Imports a sync_file fd as a temporary binary semaphore usable as a submission wait.
    fn import_sync_file_semaphore(&self, fd: std::os::unix::io::OwnedFd) -> Option<vk::Semaphore> {
        let ext = self.device.external_semaphore_fd.as_ref()?;
        let create_info = vk::SemaphoreCreateInfo::default();
        let sem = unsafe { self.device.raw.create_semaphore(&create_info, None) }.ok()?;
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(sem)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(fd.as_raw_fd());
        match unsafe { ext.import_semaphore_fd(&import_info) } {
            Ok(()) => {
                // SAFETY: on success ownership of the fd moved to Vulkan.
                std::mem::forget(fd);
                Some(sem)
            }
            Err(_) => {
                unsafe { self.device.raw.destroy_semaphore(sem, None) };
                None
            }
        }
    }

    /// Collects wait semaphores for the implicit fences of a dmabuf-backed image.
    ///
    /// `write` accesses wait for all prior accesses, reads only for prior writes. When the
    /// sync_file ioctls are unsupported this falls back to a bounded CPU wait.
    pub(super) fn implicit_acquire_waits(
        &mut self,
        fds: &[std::os::unix::io::OwnedFd],
        write: bool,
        waits: &mut Vec<vk::Semaphore>,
    ) {
        use crate::backend::allocator::dmabuf::{SyncFileFlags, export_sync_file};

        if self.implicit_interop == Some(false) || self.device.external_semaphore_fd.is_none() {
            wait_dmabuf_fds_blocking(fds, write);
            return;
        }

        let flags = if write {
            SyncFileFlags::READ | SyncFileFlags::WRITE
        } else {
            SyncFileFlags::READ
        };
        for fd in fds {
            match export_sync_file(fd.as_fd(), flags) {
                Ok(sync_file) => {
                    self.implicit_interop = Some(true);
                    if let Some(sem) = self.import_sync_file_semaphore(sync_file) {
                        waits.push(sem);
                    }
                }
                Err(err) => {
                    if self.implicit_interop.is_none() {
                        warn!(
                            ?err,
                            "dmabuf sync_file export unsupported, falling back to blocking waits"
                        );
                        self.implicit_interop = Some(false);
                    }
                    wait_dmabuf_fds_blocking(std::slice::from_ref(fd), write);
                }
            }
        }
    }

    /// Attaches a sync_file to the implicit fences of dmabuf-backed images.
    pub(super) fn implicit_release_import(
        &self,
        sync_file: &std::os::unix::io::OwnedFd,
        fds: &[std::os::unix::io::OwnedFd],
        write: bool,
    ) {
        use crate::backend::allocator::dmabuf::{SyncFileFlags, import_sync_file};

        if self.implicit_interop == Some(false) {
            return;
        }
        let flags = if write {
            SyncFileFlags::WRITE
        } else {
            SyncFileFlags::READ
        };
        for fd in fds {
            if let Err(err) = import_sync_file(fd.as_fd(), flags, sync_file.as_fd()) {
                trace!(?err, "Failed to import sync_file into dmabuf");
            }
        }
    }

    /// Returns (creating if necessary) the pipeline for the given key.
    pub(super) fn get_pipeline(
        &mut self,
        format: vk::Format,
        solid: bool,
        blend: bool,
    ) -> Result<vk::Pipeline, VulkanError> {
        let key = PipelineKey { format, solid, blend };
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(*pipeline);
        }

        let raw = &self.device.raw;
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert_module)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(if solid {
                    self.solid_frag_module
                } else {
                    self.tex_frag_module
                })
                .name(entry),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attachment = if blend {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        };
        let blend_attachments = [blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let formats = [format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);

        let pipeline =
            unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) }
                .map_err(|_| VulkanError::PipelineCreation)?
                .into_iter()
                .next()
                .ok_or(VulkanError::PipelineCreation)?;

        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Returns the descriptor set for sampling `texture` with the current filters.
    pub(super) fn texture_descriptor_set(
        &mut self,
        texture: &VulkanTexture,
    ) -> Result<vk::DescriptorSet, VulkanError> {
        let filters = (self.downscale_filter, self.upscale_filter);
        if let Some((_, set)) = texture.0.descriptor_sets.lock().unwrap().get(&filters) {
            return Ok(*set);
        }

        let (pool, set) = self.allocate_descriptor_set(self.ds_layout)?;
        let sampler = self.samplers[&filters];
        // Dmabuf-imported textures are moved to `SHADER_READ_ONLY_OPTIMAL` by the per-frame
        // foreign-queue acquire barrier; uploaded textures stay in that layout after upload.
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(texture.0.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);
        unsafe { self.device.raw.update_descriptor_sets(&[write], &[]) };

        texture
            .0
            .descriptor_sets
            .lock()
            .unwrap()
            .insert(filters, (pool, set));
        Ok(set)
    }

    pub(super) fn allocate_descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet), VulkanError> {
        // Cached texture sets do not enter this path. Serialize actual pool allocations with
        // timeline-completed frees, and only return capacity after vkFreeDescriptorSets succeeds.
        let mut pools = self.device.descriptor_pools.lock().unwrap();
        for (pool, free) in pools.iter_mut() {
            if *free > 0 {
                let layouts = [layout];
                let allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(&layouts);
                match unsafe { self.device.raw.allocate_descriptor_sets(&allocate_info) } {
                    Ok(sets) => {
                        *free -= 1;
                        return Ok((*pool, sets[0]));
                    }
                    // Pools contain both sampled-image and uniform-buffer sets, so one type
                    // can be exhausted before max_sets is reached. A later free may restore it.
                    Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => (),
                    Err(err) => return Err(err.into()),
                }
            }
        }

        // No free pool, create a new one.
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(DESCRIPTOR_POOL_SIZE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(64),
        ];
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(DESCRIPTOR_POOL_SIZE)
            .pool_sizes(&sizes);
        let pool = unsafe { self.device.raw.create_descriptor_pool(&create_info, None) }?;

        let layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = match unsafe { self.device.raw.allocate_descriptor_sets(&allocate_info) } {
            Ok(sets) => sets,
            Err(err) => {
                unsafe { self.device.raw.destroy_descriptor_pool(pool, None) };
                return Err(err.into());
            }
        };
        pools.push((pool, DESCRIPTOR_POOL_SIZE - 1));
        Ok((pool, sets[0]))
    }

    /// Imports a dmabuf as a `VkImage` with bound memory.
    fn import_dmabuf_image(
        &self,
        dmabuf: &Dmabuf,
        usage: vk::ImageUsageFlags,
        initial_layout: vk::ImageLayout,
    ) -> Result<(vk::Image, Vec<vk::DeviceMemory>), VulkanError> {
        let format = dmabuf.format();
        let info = self
            .formats
            .get(&format.code)
            .ok_or(VulkanError::UnsupportedFormat(format.code))?;
        let mapping = info.mapping;

        let for_render = usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        let modifiers = if for_render {
            &info.render_modifiers
        } else {
            &info.texture_modifiers
        };
        let modifier_info = modifiers
            .iter()
            .find(|m| m.modifier == format.modifier)
            .ok_or(VulkanError::UnsupportedModifier(format.code))?;

        let size = dmabuf.size();
        if size.w as u32 > modifier_info.max_extent.width || size.h as u32 > modifier_info.max_extent.height {
            return Err(VulkanError::DmabufImport("dmabuf too large"));
        }
        if dmabuf.num_planes() != modifier_info.plane_count as usize {
            return Err(VulkanError::DmabufImport("unexpected plane count"));
        }

        let plane_layouts: Vec<vk::SubresourceLayout> = dmabuf
            .offsets()
            .zip(dmabuf.strides())
            .map(|(offset, stride)| vk::SubresourceLayout {
                offset: offset as u64,
                size: 0,
                row_pitch: stride as u64,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();

        let mut modifier_create_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(Into::<u64>::into(format.modifier))
            .plane_layouts(&plane_layouts);
        let mut external_create_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(mapping.vk)
            .extent(vk::Extent3D {
                width: size.w as u32,
                height: size.h as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(initial_layout)
            .push_next(&mut modifier_create_info)
            .push_next(&mut external_create_info);

        let raw = &self.device.raw;
        let image = unsafe { raw.create_image(&create_info, None) }?;

        // Import the memory of the first plane; all planes of a non-disjoint image must refer
        // to the same underlying memory.
        let fd = dmabuf
            .handles()
            .next()
            .ok_or(VulkanError::DmabufImport("dmabuf without planes"))?;
        let fd = fd.try_clone_to_owned().map_err(|_| {
            unsafe { raw.destroy_image(image, None) };
            VulkanError::DmabufImport("failed to duplicate dmabuf fd")
        })?;

        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            self.device.external_memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_props,
            )
        }
        .map_err(|err| {
            unsafe { raw.destroy_image(image, None) };
            VulkanError::from(err)
        })?;

        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        let type_bits = requirements.memory_type_bits & fd_props.memory_type_bits;
        let memory_type = self
            .device
            .find_memory_type(type_bits, vk::MemoryPropertyFlags::empty())
            .ok_or_else(|| {
                unsafe { raw.destroy_image(image, None) };
                VulkanError::NoMemoryType
            })?;

        let raw_fd = fd.into_raw_fd();
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(raw_fd);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        let memory = match unsafe { raw.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                // On failure ownership of the fd stays with us.
                unsafe {
                    let _ = std::os::unix::io::OwnedFd::from_raw_fd(raw_fd);
                    raw.destroy_image(image, None);
                }
                return Err(err.into());
            }
        };

        if let Err(err) = unsafe { raw.bind_image_memory(image, memory, 0) } {
            unsafe {
                raw.free_memory(memory, None);
                raw.destroy_image(image, None);
            }
            return Err(err.into());
        }

        Ok((image, vec![memory]))
    }

    fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        has_alpha: bool,
    ) -> Result<vk::ImageView, VulkanError> {
        let components = if has_alpha {
            vk::ComponentMapping::default()
        } else {
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::ONE,
            }
        };
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(components)
            .subresource_range(color_subresource_range());
        Ok(unsafe { self.device.raw.create_image_view(&create_info, None) }?)
    }

    /// Creates an optimally tiled image with fresh device-local memory.
    fn create_image(
        &self,
        mapping: FormatMapping,
        size: Size<i32, BufferCoord>,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), VulkanError> {
        let raw = &self.device.raw;
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(mapping.vk)
            .extent(vk::Extent3D {
                width: size.w as u32,
                height: size.h as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { raw.create_image(&create_info, None) }?;

        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        let memory_type = self
            .device
            .find_memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                self.device
                    .find_memory_type(requirements.memory_type_bits, vk::MemoryPropertyFlags::empty())
            })
            .ok_or_else(|| {
                unsafe { raw.destroy_image(image, None) };
                VulkanError::NoMemoryType
            })?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { raw.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { raw.destroy_image(image, None) };
                return Err(err.into());
            }
        };
        if let Err(err) = unsafe { raw.bind_image_memory(image, memory, 0) } {
            unsafe {
                raw.free_memory(memory, None);
                raw.destroy_image(image, None);
            }
            return Err(err.into());
        }
        Ok((image, memory))
    }

    /// Creates a host-visible buffer with mapped memory.
    fn create_host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut std::ffi::c_void), VulkanError> {
        let raw = &self.device.raw;
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { raw.create_buffer(&create_info, None) }?;
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let memory_type = self
            .device
            .find_memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| {
                unsafe { raw.destroy_buffer(buffer, None) };
                VulkanError::NoMemoryType
            })?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { raw.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { raw.destroy_buffer(buffer, None) };
                return Err(err.into());
            }
        };
        let res = unsafe {
            raw.bind_buffer_memory(buffer, memory, 0)
                .and_then(|_| raw.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()))
        };
        match res {
            Ok(ptr) => Ok((buffer, memory, ptr)),
            Err(err) => {
                unsafe {
                    raw.free_memory(memory, None);
                    raw.destroy_buffer(buffer, None);
                }
                Err(err.into())
            }
        }
    }

    fn acquire_upload_buffer(&mut self, size: u64) -> Result<UploadBuffer, VulkanError> {
        if !self.upload_buffers.is_empty() {
            let completed = self.device.completed_point()?;
            let best = self
                .upload_buffers
                .iter()
                .enumerate()
                .filter(|(_, (point, buffer))| *point <= completed && buffer.capacity >= size)
                .min_by_key(|(_, (_, buffer))| buffer.capacity)
                .map(|(index, _)| index);
            if let Some(index) = best {
                return Ok(self.upload_buffers.swap_remove(index).1);
            }
            // Replace an undersized completed allocation when a surface grows. Otherwise
            // small idle buffers could fill the cache and prevent larger uploads being cached.
            if let Some(index) = self
                .upload_buffers
                .iter()
                .position(|(point, _)| *point <= completed)
            {
                let (point, buffer) = self.upload_buffers.swap_remove(index);
                buffer.defer_destroy(&self.device, point);
            }
        }
        let (buffer, memory, ptr) = self.create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        Ok(UploadBuffer {
            buffer,
            memory,
            ptr: ptr.cast(),
            capacity: size,
        })
    }

    fn recycle_upload_buffer(&mut self, point: u64, buffer: UploadBuffer) {
        let cached_bytes: u64 = self.upload_buffers.iter().map(|(_, b)| b.capacity).sum();
        if self.upload_buffers.len() < MAX_CACHED_UPLOAD_BUFFERS
            && buffer.capacity <= MAX_CACHED_UPLOAD_BYTES.saturating_sub(cached_bytes)
        {
            self.upload_buffers.push((point, buffer));
        } else {
            buffer.defer_destroy(&self.device, point);
        }
    }

    /// Uploads all damaged regions with one staging allocation and queue submission.
    #[profiling::function]
    fn upload_memory(
        &mut self,
        texture: &VulkanTexture,
        data: &[u8],
        data_stride_pixels: i32,
        damage: &[Rectangle<i32, BufferCoord>],
        first_upload: bool,
    ) -> Result<(), VulkanError> {
        let bpp = format::bytes_per_pixel(texture.0.format)
            .ok_or(VulkanError::UnsupportedFormat(texture.0.format))?;
        let full = Rectangle::from_size(texture.0.size);
        // The shaper operates in physical coordinates; these are the same pixel coordinates
        // as the source buffer, without a scale or transform. Its disjoint output also avoids
        // overlapping destination regions in a single Vulkan copy command.
        let mut regions: Vec<Rectangle<i32, Physical>> = damage
            .iter()
            .filter_map(|rect| rect.intersection(full))
            .map(|rect| Rectangle::new((rect.loc.x, rect.loc.y).into(), (rect.size.w, rect.size.h).into()))
            .collect();
        let mut shaper: super::damage::DamageShaper = Default::default();
        shaper.shape_damage(&mut regions);
        if regions.is_empty() {
            return Ok(());
        }

        let stride = usize::try_from(data_stride_pixels).map_err(|_| VulkanError::UnexpectedSize)?;
        let mut copies = Vec::with_capacity(regions.len());
        let mut needed = 0usize;
        for region in &regions {
            let width = region.size.w as usize;
            let height = region.size.h as usize;
            let end = (region.loc.y as usize + height - 1)
                .checked_mul(stride)
                .and_then(|offset| offset.checked_add(region.loc.x as usize + width))
                .and_then(|pixels| pixels.checked_mul(bpp));
            if end.is_none_or(|end| end > data.len()) || region.loc.x as usize + width > stride {
                return Err(VulkanError::UnexpectedSize);
            }
            copies.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(needed as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D {
                        x: region.loc.x,
                        y: region.loc.y,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: width as u32,
                        height: height as u32,
                        depth: 1,
                    }),
            );
            needed = width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(bpp))
                .and_then(|bytes| needed.checked_add(bytes))
                .ok_or(VulkanError::UnexpectedSize)?;
        }
        let staging = self.acquire_upload_buffer(needed as u64)?;
        for (region, copy) in regions.iter().zip(&copies) {
            let row_bytes = region.size.w as usize * bpp;
            for row in 0..region.size.h as usize {
                let src_offset = ((region.loc.y as usize + row) * stride + region.loc.x as usize) * bpp;
                // The source bounds were checked above, and the allocation fits all packed regions.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(src_offset),
                        staging.ptr.add(copy.buffer_offset as usize + row * row_bytes),
                        row_bytes,
                    );
                }
            }
        }

        let image = texture.0.image;
        let old_layout = if first_upload {
            vk::ImageLayout::UNDEFINED
        } else {
            *texture.0.layout.lock().unwrap()
        };

        let submission = self.submit_one_shot(
            |raw, cb| {
                unsafe {
                    image_barrier(
                        raw,
                        cb,
                        image,
                        old_layout,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    );
                    raw.cmd_copy_buffer_to_image(
                        cb,
                        staging.buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &copies,
                    );
                    image_barrier(
                        raw,
                        cb,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                    );
                }
                Ok(())
            },
            Vec::new(),
            Vec::new(),
            false,
        );
        let (point, _) = match submission {
            Ok(submission) => submission,
            Err(err) => {
                staging.defer_destroy(&self.device, self.timeline_point);
                return Err(err);
            }
        };
        self.recycle_upload_buffer(point, staging);

        *texture.0.layout.lock().unwrap() = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        texture.0.mark_used(point);
        Ok(())
    }
}

pub(super) fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn image_barrier(
    raw: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .subresource_range(color_subresource_range());
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    unsafe { raw.cmd_pipeline_barrier2(cb, &dependency) };
}

/// Queue family ownership transfer barrier from/to the foreign queue.
#[allow(clippy::too_many_arguments)]
pub(super) fn foreign_barrier<'a>(
    queue_family: u32,
    acquire: bool,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'a> {
    let mut barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(color_subresource_range());
    if acquire {
        barrier = barrier
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(queue_family)
            .dst_stage_mask(stage)
            .dst_access_mask(access);
    } else {
        barrier = barrier
            .src_queue_family_index(queue_family)
            .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .src_stage_mask(stage)
            .src_access_mask(access);
    }
    barrier
}

/// Duplicates all plane fds of a dmabuf for implicit sync interop.
fn dup_dmabuf_fds(dmabuf: &Dmabuf) -> Vec<OwnedFd> {
    dmabuf
        .handles()
        .filter_map(|fd| fd.try_clone_to_owned().ok())
        .collect()
}

/// Blocking fallback wait on the implicit fences of dmabuf fds, bounded at one second.
///
/// Polling a dmabuf waits for outstanding writes (`POLLIN`, needed before reading) or all
/// outstanding accesses (`POLLOUT`, needed before writing).
pub(super) fn wait_dmabuf_fds_blocking(fds: &[OwnedFd], write: bool) {
    use rustix::event::{PollFd, PollFlags, poll};

    let flags = if write { PollFlags::OUT } else { PollFlags::IN };
    for fd in fds {
        let mut poll_fds = [PollFd::new(fd, flags)];
        match poll(
            &mut poll_fds,
            Some(&rustix::time::Timespec {
                tv_sec: 1,
                tv_nsec: 0,
            }),
        ) {
            Ok(0) => warn!("Timed out waiting for dmabuf fence"),
            Ok(_) => {}
            Err(err) => warn!(?err, "Failed to poll dmabuf fence"),
        }
    }
}

fn create_shader_module(raw: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, VulkanError> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes)).map_err(|_| VulkanError::PipelineCreation)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { raw.create_shader_module(&create_info, None) }?)
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        let raw = &self.device.raw;
        unsafe {
            let _ = raw.device_wait_idle();
            for (_, pipeline) in self.pipelines.drain() {
                raw.destroy_pipeline(pipeline, None);
            }
            for (_, pipeline) in self.custom_pipelines.drain() {
                raw.destroy_pipeline(pipeline, None);
            }
            for (count, layout) in self.pipeline_layouts.iter().enumerate() {
                let _ = count;
                raw.destroy_pipeline_layout(*layout, None);
            }
            for (count, layout) in self.texture_ds_layouts.iter().enumerate() {
                if count != 1 {
                    raw.destroy_descriptor_set_layout(*layout, None);
                }
            }
            raw.destroy_descriptor_set_layout(self.ds_layout, None);
            raw.destroy_descriptor_set_layout(self.params_ds_layout, None);
            raw.destroy_sampler(self.custom_sampler, None);
            raw.destroy_shader_module(self.vert_module, None);
            raw.destroy_shader_module(self.tex_frag_module, None);
            raw.destroy_shader_module(self.solid_frag_module, None);
            for (_, sampler) in self.samplers.drain() {
                raw.destroy_sampler(sampler, None);
            }
            // Drop cached textures and buffers so their cleanup lands in the queue. Pools belong
            // to Device: external textures can outlive this renderer and release their sets later.
            self.dmabuf_textures.clear();
            self.render_buffers.clear();
            for (point, ring) in self.params_rings.drain(..) {
                ring.defer_destroy(&self.device, point);
            }
            for (point, buffer) in self.upload_buffers.drain(..) {
                buffer.defer_destroy(&self.device, point);
            }
            self.device.process_cleanup(u64::MAX);
            raw.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl RendererSuper for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanTexture;
    type Framebuffer<'buffer> = VulkanTarget<'buffer>;
    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<VulkanTexture> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    #[profiling::function]
    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut VulkanTarget<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<VulkanFrame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        VulkanFrame::new(self, framebuffer, output_size, dst_transform)
    }

    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.handle_wait(sync)
    }

    #[profiling::function]
    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.cleanup();
        Ok(())
    }
}

impl ImportMem for VulkanRenderer {
    #[instrument(level = "trace", parent = &self.span, skip(self, data))]
    #[profiling::function]
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<VulkanTexture, Self::Error> {
        if !format::MEM_FORMATS.contains(&format) {
            return Err(VulkanError::UnsupportedFormat(format));
        }
        let mapping = format::get_format_mapping(format).ok_or(VulkanError::UnsupportedFormat(format))?;
        let info = self
            .formats
            .get(&format)
            .ok_or(VulkanError::UnsupportedFormat(format))?;
        let max_extent = info
            .optimal_max_extent
            .ok_or(VulkanError::UnsupportedFormat(format))?;
        if size.w <= 0 || size.h <= 0 || size.w as u32 > max_extent.width || size.h as u32 > max_extent.height
        {
            return Err(VulkanError::UnexpectedSize);
        }
        let bpp = format::bytes_per_pixel(format).unwrap() as i32;
        if data.len() < (size.w * size.h * bpp) as usize {
            return Err(VulkanError::UnexpectedSize);
        }

        let (image, memory) = self.create_image(mapping, size, format::mem_texture_usage())?;
        let view = match self.create_image_view(image, mapping.vk, mapping.has_alpha) {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    self.device.raw.free_memory(memory, None);
                    self.device.raw.destroy_image(image, None);
                }
                return Err(err);
            }
        };

        let texture = VulkanTexture(Arc::new(InnerTexture {
            device: self.device.clone(),
            image,
            view,
            memories: vec![memory],
            format,
            vk_format: mapping.vk,
            size,
            has_alpha: mapping.has_alpha,
            y_inverted: flipped,
            dmabuf_imported: false,
            dmabuf_fds: Vec::new(),
            writable: true,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
            last_use: AtomicU64::new(0),
            descriptor_sets: Mutex::new(HashMap::new()),
        }));

        self.upload_memory(&texture, data, size.w, &[Rectangle::from_size(size)], true)?;
        Ok(texture)
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, data))]
    #[profiling::function]
    fn update_memory(
        &mut self,
        texture: &VulkanTexture,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        if !texture.0.writable {
            return Err(VulkanError::NotWritable);
        }
        let size = texture.0.size;
        if region.loc.x < 0
            || region.loc.y < 0
            || region.size.w <= 0
            || region.size.h <= 0
            || region.loc.x + region.size.w > size.w
            || region.loc.y + region.size.h > size.h
        {
            return Err(VulkanError::OutOfBounds);
        }
        let bpp = format::bytes_per_pixel(texture.0.format).unwrap() as i32;
        if data.len() < (size.w * size.h * bpp) as usize {
            return Err(VulkanError::UnexpectedSize);
        }

        self.upload_memory(texture, data, size.w, &[region], false)
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(format::MEM_FORMATS.iter().copied())
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn import_shm_buffer(
        &mut self,
        buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<VulkanTexture, Self::Error> {
        use crate::wayland::shm::{shm_format_to_fourcc, with_buffer_contents};

        // Cache the texture in the surface data so subsequent commits only upload damage.
        type CacheMap = HashMap<ContextId<VulkanTexture>, VulkanTexture>;

        let mut surface_lock = surface.as_ref().map(|surface_data| {
            surface_data
                .data_map
                .get_or_insert_threadsafe(|| Arc::new(Mutex::new(CacheMap::new())))
                .lock()
                .unwrap()
        });

        with_buffer_contents(buffer, |ptr, len, data| {
            let offset = data.offset;
            let width = data.width;
            let height = data.height;
            let stride = data.stride;
            let fourcc = shm_format_to_fourcc(data.format)
                .ok_or(VulkanError::UnsupportedWlPixelFormat(data.format))?;
            if !format::MEM_FORMATS.contains(&fourcc) {
                return Err(VulkanError::UnsupportedWlPixelFormat(data.format));
            }
            let bpp = format::bytes_per_pixel(fourcc).unwrap() as i32;
            if stride % bpp != 0 {
                return Err(VulkanError::UnexpectedSize);
            }
            assert!((offset + (height - 1) * stride + width * bpp) as usize <= len);

            // SAFETY: the shm handler guarantees ptr..ptr+len is valid while the closure runs.
            let data = unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), len - offset as usize) };
            let stride_pixels = stride / bpp;

            let id = self.context_id();
            let existing = surface_lock
                .as_ref()
                .and_then(|cache| cache.get(&id).cloned())
                .filter(|texture| texture.0.size == (width, height).into() && texture.0.format == fourcc);

            match existing {
                Some(texture) => {
                    let full = Rectangle::from_size((width, height).into());
                    if damage.is_empty() {
                        self.upload_memory(&texture, data, stride_pixels, &[full], false)?;
                    } else {
                        self.upload_memory(&texture, data, stride_pixels, damage, false)?;
                    }
                    Ok(texture)
                }
                None => {
                    let mapping =
                        format::get_format_mapping(fourcc).ok_or(VulkanError::UnsupportedFormat(fourcc))?;
                    let size = Size::from((width, height));
                    let (image, memory) = self.create_image(mapping, size, format::mem_texture_usage())?;
                    let view = match self.create_image_view(image, mapping.vk, mapping.has_alpha) {
                        Ok(view) => view,
                        Err(err) => {
                            unsafe {
                                self.device.raw.free_memory(memory, None);
                                self.device.raw.destroy_image(image, None);
                            }
                            return Err(err);
                        }
                    };
                    let texture = VulkanTexture(Arc::new(InnerTexture {
                        device: self.device.clone(),
                        image,
                        view,
                        memories: vec![memory],
                        format: fourcc,
                        vk_format: mapping.vk,
                        size,
                        has_alpha: mapping.has_alpha,
                        y_inverted: false,
                        dmabuf_imported: false,
                        dmabuf_fds: Vec::new(),
                        writable: true,
                        layout: Mutex::new(vk::ImageLayout::UNDEFINED),
                        last_use: AtomicU64::new(0),
                        descriptor_sets: Mutex::new(HashMap::new()),
                    }));
                    self.upload_memory(&texture, data, stride_pixels, &[Rectangle::from_size(size)], true)?;
                    if let Some(cache) = surface_lock.as_mut() {
                        cache.insert(id, texture.clone());
                    }
                    Ok(texture)
                }
            }
        })?
    }
}

impl ImportDma for VulkanRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        self.dmabuf_texture_formats.clone()
    }

    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<VulkanTexture, Self::Error> {
        if let Some((_, texture)) = self
            .dmabuf_textures
            .iter()
            .find(|(weak, _)| weak.upgrade().map(|buf| buf == *dmabuf).unwrap_or(false))
        {
            return Ok(texture.clone());
        }

        let format = dmabuf.format();
        let mapping =
            format::get_format_mapping(format.code).ok_or(VulkanError::UnsupportedFormat(format.code))?;

        let (image, memories) =
            self.import_dmabuf_image(dmabuf, format::TEXTURE_USAGE, vk::ImageLayout::UNDEFINED)?;
        let view = match self.create_image_view(image, mapping.vk, mapping.has_alpha) {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    for memory in memories {
                        self.device.raw.free_memory(memory, None);
                    }
                    self.device.raw.destroy_image(image, None);
                }
                return Err(err);
            }
        };

        let texture = VulkanTexture(Arc::new(InnerTexture {
            device: self.device.clone(),
            image,
            view,
            memories,
            format: format.code,
            vk_format: mapping.vk,
            size: dmabuf.size(),
            has_alpha: mapping.has_alpha,
            y_inverted: dmabuf.y_inverted(),
            dmabuf_imported: true,
            dmabuf_fds: dup_dmabuf_fds(dmabuf),
            writable: false,
            layout: Mutex::new(vk::ImageLayout::GENERAL),
            last_use: AtomicU64::new(0),
            descriptor_sets: Mutex::new(HashMap::new()),
        }));

        self.dmabuf_textures.push((dmabuf.weak(), texture.clone()));
        Ok(texture)
    }
}

#[cfg(feature = "wayland_frontend")]
#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
impl crate::backend::renderer::ImportEgl for VulkanRenderer {
    fn bind_wl_display(
        &mut self,
        _display: &wayland_server::DisplayHandle,
    ) -> Result<(), crate::backend::egl::Error> {
        // No wl_drm support on the Vulkan renderer; clients use dmabuf.
        Err(crate::backend::egl::Error::DisplayNotSupported)
    }

    fn unbind_wl_display(&mut self) {}

    fn egl_reader(&self) -> Option<&crate::backend::egl::display::EGLBufferReader> {
        None
    }

    fn import_egl_buffer(
        &mut self,
        _buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<VulkanTexture, VulkanError> {
        Err(VulkanError::EglUnsupported)
    }
}

impl ImportDmaWl for VulkanRenderer {}

impl Bind<Dmabuf> for VulkanRenderer {
    #[profiling::function]
    fn bind<'a>(&mut self, dmabuf: &'a mut Dmabuf) -> Result<VulkanTarget<'a>, VulkanError> {
        if let Some((_, buffer)) = self
            .render_buffers
            .iter()
            .find(|(weak, _)| weak.upgrade().map(|buf| buf == *dmabuf).unwrap_or(false))
        {
            return Ok(VulkanTarget(TargetInner::Dmabuf {
                buffer: buffer.clone(),
                _lifetime: std::marker::PhantomData,
            }));
        }

        let format = dmabuf.format();
        let mapping =
            format::get_format_mapping(format.code).ok_or(VulkanError::UnsupportedFormat(format.code))?;

        let (image, memories) =
            self.import_dmabuf_image(dmabuf, format::render_usage(), vk::ImageLayout::UNDEFINED)?;
        // Render targets are written through an identity view; alpha of X-formats is don't-care.
        let view = match self.create_image_view(image, mapping.vk, true) {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    for memory in memories {
                        self.device.raw.free_memory(memory, None);
                    }
                    self.device.raw.destroy_image(image, None);
                }
                return Err(err);
            }
        };

        let buffer = Arc::new(RenderBuffer {
            device: self.device.clone(),
            image,
            view,
            memories,
            format: format.code,
            vk_format: mapping.vk,
            size: dmabuf.size(),
            dmabuf_fds: dup_dmabuf_fds(dmabuf),
            transitioned: AtomicBool::new(false),
            last_use: AtomicU64::new(0),
        });
        self.render_buffers.push((dmabuf.weak(), buffer.clone()));

        Ok(VulkanTarget(TargetInner::Dmabuf {
            buffer,
            _lifetime: std::marker::PhantomData,
        }))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.dmabuf_render_formats.clone())
    }
}

impl Bind<VulkanTexture> for VulkanRenderer {
    fn bind<'a>(&mut self, texture: &'a mut VulkanTexture) -> Result<VulkanTarget<'a>, VulkanError> {
        if texture.0.dmabuf_imported {
            return Err(VulkanError::DmabufImport(
                "cannot render to imported dmabuf texture",
            ));
        }
        Ok(VulkanTarget(TargetInner::Texture {
            texture: texture.clone(),
            _lifetime: std::marker::PhantomData,
        }))
    }
}

impl Offscreen<VulkanTexture> for VulkanRenderer {
    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanTexture, VulkanError> {
        let mapping = format::get_format_mapping(format).ok_or(VulkanError::UnsupportedFormat(format))?;
        let info = self
            .formats
            .get(&format)
            .ok_or(VulkanError::UnsupportedFormat(format))?;
        if !info.optimal_render {
            return Err(VulkanError::UnsupportedFormat(format));
        }
        if size.w <= 0 || size.h <= 0 {
            return Err(VulkanError::UnexpectedSize);
        }

        let (image, memory) = self.create_image(mapping, size, format::offscreen_usage())?;
        // Sampled through a view with alpha forced to one for opaque formats; rendering uses
        // the same view, which is fine as the alpha channel of X-formats is don't-care.
        let view = match self.create_image_view(image, mapping.vk, mapping.has_alpha) {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    self.device.raw.free_memory(memory, None);
                    self.device.raw.destroy_image(image, None);
                }
                return Err(err);
            }
        };

        Ok(VulkanTexture(Arc::new(InnerTexture {
            device: self.device.clone(),
            image,
            view,
            memories: vec![memory],
            format,
            vk_format: mapping.vk,
            size,
            has_alpha: mapping.has_alpha,
            y_inverted: false,
            dmabuf_imported: false,
            dmabuf_fds: Vec::new(),
            writable: false,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
            last_use: AtomicU64::new(0),
            descriptor_sets: Mutex::new(HashMap::new()),
        })))
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanTextureMapping;

    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn copy_framebuffer(
        &mut self,
        target: &VulkanTarget<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        let size = Texture::size(target);
        let src_layout = match &target.0 {
            TargetInner::Dmabuf { .. } => vk::ImageLayout::GENERAL,
            TargetInner::Texture { texture, .. } => *texture.0.layout.lock().unwrap(),
        };
        let foreign = matches!(&target.0, TargetInner::Dmabuf { .. });
        let src_fds = target_dmabuf_fds(target);
        let mapping = self.copy_image_to_memory(
            target.image(),
            target.vk_format(),
            src_layout,
            foreign,
            src_fds,
            size,
            region,
            format,
        )?;
        target.mark_used(mapping.point);
        Ok(mapping)
    }

    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn copy_texture(
        &mut self,
        texture: &VulkanTexture,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        let src_layout = if texture.0.dmabuf_imported {
            vk::ImageLayout::GENERAL
        } else {
            *texture.0.layout.lock().unwrap()
        };
        let mapping = self.copy_image_to_memory(
            texture.0.image,
            texture.0.vk_format,
            src_layout,
            texture.0.dmabuf_imported,
            &texture.0.dmabuf_fds,
            texture.0.size,
            region,
            format,
        )?;
        texture.0.mark_used(mapping.point);
        Ok(mapping)
    }

    fn can_read_texture(&mut self, _texture: &VulkanTexture) -> Result<bool, Self::Error> {
        Ok(true)
    }

    #[profiling::function]
    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        // Wait for the copy to complete before exposing the data.
        let semaphores = [self.device.timeline];
        let points = [texture_mapping.point];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&points);
        unsafe { self.device.raw.wait_semaphores(&wait_info, u64::MAX) }?;

        let bpp = format::bytes_per_pixel(texture_mapping.format).unwrap();
        let len = texture_mapping.size.w as usize * texture_mapping.size.h as usize * bpp;
        // SAFETY: the buffer was created with at least `len` bytes and the copy completed.
        Ok(unsafe { std::slice::from_raw_parts(texture_mapping.ptr as *const u8, len) })
    }
}

impl VulkanRenderer {
    /// Copies a region of an image into a new host-visible buffer, converting formats via a
    /// blit through a temporary image if necessary.
    #[allow(clippy::too_many_arguments)]
    fn copy_image_to_memory(
        &mut self,
        src_image: vk::Image,
        src_format: vk::Format,
        src_layout: vk::ImageLayout,
        src_foreign: bool,
        src_fds: &[OwnedFd],
        src_size: Size<i32, BufferCoord>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<VulkanTextureMapping, VulkanError> {
        let dst_mapping = format::get_format_mapping(format).ok_or(VulkanError::UnsupportedFormat(format))?;
        if region.loc.x < 0
            || region.loc.y < 0
            || region.size.w <= 0
            || region.size.h <= 0
            || region.loc.x + region.size.w > src_size.w
            || region.loc.y + region.size.h > src_size.h
        {
            return Err(VulkanError::OutOfBounds);
        }

        let bpp = format::bytes_per_pixel(format).ok_or(VulkanError::UnsupportedFormat(format))? as u64;
        let buffer_size = region.size.w as u64 * region.size.h as u64 * bpp;
        let (buffer, memory, ptr) =
            self.create_host_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_DST)?;

        // If the format differs, blit through a temporary image to convert.
        let needs_convert = dst_mapping.vk != src_format;
        let temp = if needs_convert {
            let temp_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
            match self.create_image(dst_mapping, region.size, temp_usage) {
                Ok(temp) => Some(temp),
                Err(err) => {
                    unsafe {
                        self.device.raw.free_memory(memory, None);
                        self.device.raw.destroy_buffer(buffer, None);
                    }
                    return Err(err);
                }
            }
        } else {
            None
        };

        // Wait for outstanding implicit-sync writes before reading a dmabuf-backed source.
        let mut binary_waits = Vec::new();
        if !src_fds.is_empty() {
            self.implicit_acquire_waits(src_fds, false, &mut binary_waits);
        }

        let queue_family = self.device.queue_family;
        let result = self.submit_one_shot(
            |raw, cb| {
                unsafe {
                    // Make the source readable for transfer.
                    if src_foreign {
                        let barriers = [foreign_barrier(
                            queue_family,
                            true,
                            src_image,
                            vk::ImageLayout::GENERAL,
                            vk::ImageLayout::GENERAL,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                        )];
                        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                        raw.cmd_pipeline_barrier2(cb, &dependency);
                    } else if src_layout != vk::ImageLayout::GENERAL {
                        image_barrier(
                            raw,
                            cb,
                            src_image,
                            src_layout,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::PipelineStageFlags2::ALL_COMMANDS,
                            vk::AccessFlags2::MEMORY_WRITE,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                        );
                    }
                    let src_transfer_layout = if src_foreign || src_layout == vk::ImageLayout::GENERAL {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    };

                    let (copy_src_image, copy_src_layout, copy_region) = if let Some((temp_image, _)) = temp {
                        // Blit (with format conversion) into the temporary image.
                        image_barrier(
                            raw,
                            cb,
                            temp_image,
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::PipelineStageFlags2::NONE,
                            vk::AccessFlags2::NONE,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                        );
                        let blit = vk::ImageBlit::default()
                            .src_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(1),
                            )
                            .src_offsets([
                                vk::Offset3D {
                                    x: region.loc.x,
                                    y: region.loc.y,
                                    z: 0,
                                },
                                vk::Offset3D {
                                    x: region.loc.x + region.size.w,
                                    y: region.loc.y + region.size.h,
                                    z: 1,
                                },
                            ])
                            .dst_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(1),
                            )
                            .dst_offsets([
                                vk::Offset3D { x: 0, y: 0, z: 0 },
                                vk::Offset3D {
                                    x: region.size.w,
                                    y: region.size.h,
                                    z: 1,
                                },
                            ]);
                        raw.cmd_blit_image(
                            cb,
                            src_image,
                            src_transfer_layout,
                            temp_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[blit],
                            vk::Filter::NEAREST,
                        );
                        image_barrier(
                            raw,
                            cb,
                            temp_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                        );
                        (
                            temp_image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            Rectangle::from_size(region.size),
                        )
                    } else {
                        (src_image, src_transfer_layout, region)
                    };

                    let copy = vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D {
                            x: copy_region.loc.x,
                            y: copy_region.loc.y,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: copy_region.size.w as u32,
                            height: copy_region.size.h as u32,
                            depth: 1,
                        });
                    raw.cmd_copy_image_to_buffer(cb, copy_src_image, copy_src_layout, buffer, &[copy]);

                    // Restore the source image state.
                    if src_foreign {
                        let barriers = [foreign_barrier(
                            queue_family,
                            false,
                            src_image,
                            vk::ImageLayout::GENERAL,
                            vk::ImageLayout::GENERAL,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                        )];
                        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                        raw.cmd_pipeline_barrier2(cb, &dependency);
                    } else if src_layout != vk::ImageLayout::GENERAL {
                        image_barrier(
                            raw,
                            cb,
                            src_image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            src_layout,
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                            vk::PipelineStageFlags2::ALL_COMMANDS,
                            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                        );
                    }
                }
                Ok(())
            },
            {
                let mut cleanup = Vec::new();
                if let Some((temp_image, temp_memory)) = temp {
                    cleanup.push(CleanupItem::Image(temp_image));
                    cleanup.push(CleanupItem::Memory(temp_memory));
                }
                cleanup
            },
            binary_waits,
            !src_fds.is_empty(),
        );

        let (point, fence) = match result {
            Ok(res) => res,
            Err(err) => {
                unsafe {
                    self.device.raw.free_memory(memory, None);
                    self.device.raw.destroy_buffer(buffer, None);
                }
                return Err(err);
            }
        };

        // Mark our read on the source for implicit-sync producers reusing the buffer.
        if !src_fds.is_empty() {
            use super::sync::Fence as _;
            if let Some(sync_file) = fence.export() {
                self.implicit_release_import(&sync_file, src_fds, false);
            }
        }

        #[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
        {
            let mut reads = super::utils::buffer_read::BufferReadSet::default();
            reads.capture();
            if !reads.is_empty() {
                reads.publish(&SyncPoint::from(fence));
            }
        }

        Ok(VulkanTextureMapping {
            device: self.device.clone(),
            buffer,
            memory,
            ptr,
            format,
            size: region.size,
            point,
        })
    }
}

impl Blit for VulkanRenderer {
    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    fn blit(
        &mut self,
        from: &VulkanTarget<'_>,
        to: &mut VulkanTarget<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        if from.image() == to.image() {
            return Err(VulkanError::BlitSameImage);
        }
        let src_size = Texture::size(from);
        let dst_size = Texture::size(to);
        if src.loc.x < 0
            || src.loc.y < 0
            || src.loc.x + src.size.w > src_size.w
            || src.loc.y + src.size.h > src_size.h
            || dst.loc.x < 0
            || dst.loc.y < 0
            || dst.loc.x + dst.size.w > dst_size.w
            || dst.loc.y + dst.size.h > dst_size.h
        {
            return Err(VulkanError::OutOfBounds);
        }

        let vk_filter = match filter {
            TextureFilter::Linear => vk::Filter::LINEAR,
            TextureFilter::Nearest => vk::Filter::NEAREST,
        };

        let queue_family = self.device.queue_family;
        let src_state = target_transfer_state(from);
        let dst_state = target_transfer_state(to);
        let src_image = from.image();
        let dst_image = to.image();

        // Wait for outstanding implicit-sync accesses on dmabuf-backed targets.
        let mut binary_waits = Vec::new();
        let src_fds = target_dmabuf_fds(from);
        let dst_fds = target_dmabuf_fds(to);
        if !src_fds.is_empty() {
            self.implicit_acquire_waits(src_fds, false, &mut binary_waits);
        }
        if !dst_fds.is_empty() {
            self.implicit_acquire_waits(dst_fds, true, &mut binary_waits);
        }

        let (point, fence) = self.submit_one_shot(
            |raw, cb| {
                unsafe {
                    transfer_prepare(raw, cb, queue_family, src_image, &src_state, false);
                    transfer_prepare(raw, cb, queue_family, dst_image, &dst_state, true);

                    let blit = vk::ImageBlit::default()
                        .src_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .src_offsets([
                            vk::Offset3D {
                                x: src.loc.x,
                                y: src.loc.y,
                                z: 0,
                            },
                            vk::Offset3D {
                                x: src.loc.x + src.size.w,
                                y: src.loc.y + src.size.h,
                                z: 1,
                            },
                        ])
                        .dst_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .dst_offsets([
                            vk::Offset3D {
                                x: dst.loc.x,
                                y: dst.loc.y,
                                z: 0,
                            },
                            vk::Offset3D {
                                x: dst.loc.x + dst.size.w,
                                y: dst.loc.y + dst.size.h,
                                z: 1,
                            },
                        ]);
                    raw.cmd_blit_image(
                        cb,
                        src_image,
                        src_state.transfer_layout(false),
                        dst_image,
                        dst_state.transfer_layout(true),
                        &[blit],
                        vk_filter,
                    );

                    transfer_restore(raw, cb, queue_family, src_image, &src_state, false);
                    transfer_restore(raw, cb, queue_family, dst_image, &dst_state, true);
                }
                Ok(())
            },
            Vec::new(),
            binary_waits,
            true,
        )?;

        // Mark our accesses for implicit-sync consumers of the dmabufs.
        {
            use super::sync::Fence as _;
            if (!src_fds.is_empty() || !dst_fds.is_empty()) && fence.is_exportable() {
                if let Some(sync_file) = fence.export() {
                    self.implicit_release_import(&sync_file, src_fds, false);
                    self.implicit_release_import(&sync_file, dst_fds, true);
                }
            }
        }

        from.mark_used(point);
        to.mark_used(point);
        // A previously uninitialized offscreen target now holds defined contents in the
        // transfer layout.
        if let TargetInner::Texture { texture, .. } = &to.0 {
            let mut layout = texture.0.layout.lock().unwrap();
            if *layout == vk::ImageLayout::UNDEFINED {
                *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }
        }
        let sync = SyncPoint::from(fence);
        #[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
        {
            let mut reads = super::utils::buffer_read::BufferReadSet::default();
            reads.capture();
            reads.publish(&sync);
        }
        Ok(sync)
    }
}

/// How to bring a target image into a transfer-capable state and back.
#[derive(Debug)]
pub(super) struct TransferState {
    pub foreign: bool,
    pub layout: vk::ImageLayout,
}

impl TransferState {
    pub(super) fn transfer_layout(&self, write: bool) -> vk::ImageLayout {
        if self.foreign || self.layout == vk::ImageLayout::GENERAL {
            vk::ImageLayout::GENERAL
        } else if write {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        }
    }
}

/// The duplicated dmabuf plane fds of a target, if it is dmabuf-backed.
pub(super) fn target_dmabuf_fds<'a>(target: &'a VulkanTarget<'_>) -> &'a [OwnedFd] {
    match &target.0 {
        TargetInner::Dmabuf { buffer, .. } => &buffer.dmabuf_fds,
        TargetInner::Texture { .. } => &[],
    }
}

pub(super) fn target_transfer_state(target: &VulkanTarget<'_>) -> TransferState {
    match &target.0 {
        TargetInner::Dmabuf { .. } => TransferState {
            foreign: true,
            layout: vk::ImageLayout::GENERAL,
        },
        TargetInner::Texture { texture, .. } => TransferState {
            foreign: false,
            layout: *texture.0.layout.lock().unwrap(),
        },
    }
}

pub(super) unsafe fn transfer_prepare(
    raw: &ash::Device,
    cb: vk::CommandBuffer,
    queue_family: u32,
    image: vk::Image,
    state: &TransferState,
    write: bool,
) {
    let access = if write {
        vk::AccessFlags2::TRANSFER_WRITE
    } else {
        vk::AccessFlags2::TRANSFER_READ
    };
    unsafe {
        if state.foreign {
            let barriers = [foreign_barrier(
                queue_family,
                true,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TRANSFER,
                access,
            )];
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            raw.cmd_pipeline_barrier2(cb, &dependency);
        } else {
            image_barrier(
                raw,
                cb,
                image,
                state.layout,
                state.transfer_layout(write),
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                access,
            );
        }
    }
}

pub(super) unsafe fn transfer_restore(
    raw: &ash::Device,
    cb: vk::CommandBuffer,
    queue_family: u32,
    image: vk::Image,
    state: &TransferState,
    write: bool,
) {
    let access = if write {
        vk::AccessFlags2::TRANSFER_WRITE
    } else {
        vk::AccessFlags2::TRANSFER_READ
    };
    unsafe {
        if state.foreign {
            let barriers = [foreign_barrier(
                queue_family,
                false,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TRANSFER,
                access,
            )];
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            raw.cmd_pipeline_barrier2(cb, &dependency);
        } else {
            let restore_layout = if state.layout == vk::ImageLayout::UNDEFINED {
                // The image had no defined contents before; keep it in the transfer layout and
                // let the caller update the tracked layout.
                state.transfer_layout(write)
            } else {
                state.layout
            };
            if restore_layout != state.transfer_layout(write) {
                image_barrier(
                    raw,
                    cb,
                    image,
                    state.transfer_layout(write),
                    restore_layout,
                    vk::PipelineStageFlags2::TRANSFER,
                    access,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                );
            }
        }
    }
}
