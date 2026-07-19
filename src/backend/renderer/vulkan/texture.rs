//! Texture and render target types of the Vulkan renderer.

use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use ash::vk;

use super::{CleanupItem, Device};
use crate::{
    backend::{
        allocator::{Fourcc, dmabuf::Dmabuf},
        renderer::{Texture, TextureFilter, TextureMapping},
    },
    utils::{Buffer as BufferCoord, Size},
};

/// A handle to a Vulkan texture.
#[derive(Debug, Clone)]
pub struct VulkanTexture(pub(super) Arc<InnerTexture>);

#[derive(Debug)]
pub(super) struct InnerTexture {
    pub(super) device: Arc<Device>,
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    pub(super) memories: Vec<vk::DeviceMemory>,
    pub(super) format: Fourcc,
    pub(super) vk_format: vk::Format,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) has_alpha: bool,
    pub(super) y_inverted: bool,
    /// Whether the image was imported from a dmabuf and is owned by a foreign queue.
    pub(super) dmabuf_imported: bool,
    /// Whether the contents may be updated via [`ImportMem`](crate::backend::renderer::ImportMem).
    pub(super) writable: bool,
    /// Current image layout. Unused for dmabuf imports, which stay `GENERAL` under
    /// foreign-queue ownership.
    pub(super) layout: Mutex<vk::ImageLayout>,
    /// Timeline point of the last submitted use.
    pub(super) last_use: AtomicU64,
    /// Cached descriptor sets per (downscale, upscale) filter combination.
    pub(super) descriptor_sets:
        Mutex<HashMap<(TextureFilter, TextureFilter), (vk::DescriptorPool, vk::DescriptorSet)>>,
}

impl InnerTexture {
    pub(super) fn mark_used(&self, point: u64) {
        self.last_use.fetch_max(point, Ordering::AcqRel);
    }
}

impl Drop for InnerTexture {
    fn drop(&mut self) {
        let point = self.last_use.load(Ordering::Acquire);
        let mut items = vec![CleanupItem::ImageView(self.view), CleanupItem::Image(self.image)];
        for memory in self.memories.drain(..) {
            items.push(CleanupItem::Memory(memory));
        }
        for (_, (pool, set)) in self.descriptor_sets.get_mut().unwrap().drain() {
            items.push(CleanupItem::DescriptorSet(pool, set));
        }
        self.device.defer_destroy(point, items);
    }
}

impl VulkanTexture {
    /// Vulkan image of this texture.
    ///
    /// The image will become invalid when all handles to this texture are dropped.
    pub fn image(&self) -> vk::Image {
        self.0.image
    }

    /// Whether the texture is upside down.
    pub fn is_y_inverted(&self) -> bool {
        self.0.y_inverted
    }
}

impl Texture for VulkanTexture {
    fn width(&self) -> u32 {
        self.0.size.w as u32
    }

    fn height(&self) -> u32 {
        self.0.size.h as u32
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.0.size
    }

    fn format(&self) -> Option<Fourcc> {
        Some(self.0.format)
    }
}

/// A render buffer created for a bound [`Dmabuf`].
#[derive(Debug)]
pub(super) struct RenderBuffer {
    pub(super) device: Arc<Device>,
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    pub(super) memories: Vec<vk::DeviceMemory>,
    pub(super) format: Fourcc,
    pub(super) vk_format: vk::Format,
    pub(super) size: Size<i32, BufferCoord>,
    /// Whether the image was transitioned away from `PREINITIALIZED` at least once.
    pub(super) transitioned: AtomicBool,
    pub(super) last_use: AtomicU64,
}

impl RenderBuffer {
    pub(super) fn mark_used(&self, point: u64) {
        self.last_use.fetch_max(point, Ordering::AcqRel);
    }
}

impl Drop for RenderBuffer {
    fn drop(&mut self) {
        let point = self.last_use.load(Ordering::Acquire);
        let mut items = vec![CleanupItem::ImageView(self.view), CleanupItem::Image(self.image)];
        for memory in self.memories.drain(..) {
            items.push(CleanupItem::Memory(memory));
        }
        self.device.defer_destroy(point, items);
    }
}

/// A framebuffer target of the [`VulkanRenderer`](super::VulkanRenderer).
#[derive(Debug)]
pub struct VulkanTarget<'a>(pub(super) TargetInner<'a>);

#[derive(Debug)]
pub(super) enum TargetInner<'a> {
    /// Rendering into a bound dmabuf.
    Dmabuf {
        buffer: Arc<RenderBuffer>,
        _lifetime: PhantomData<&'a mut Dmabuf>,
    },
    /// Rendering into an offscreen texture.
    Texture {
        texture: VulkanTexture,
        _lifetime: PhantomData<&'a mut VulkanTexture>,
    },
}

impl VulkanTarget<'_> {
    pub(super) fn image(&self) -> vk::Image {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => buffer.image,
            TargetInner::Texture { texture, .. } => texture.0.image,
        }
    }

    pub(super) fn view(&self) -> vk::ImageView {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => buffer.view,
            TargetInner::Texture { texture, .. } => texture.0.view,
        }
    }

    pub(super) fn vk_format(&self) -> vk::Format {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => buffer.vk_format,
            TargetInner::Texture { texture, .. } => texture.0.vk_format,
        }
    }

    /// The image layout the target is kept in while rendering.
    pub(super) fn render_layout(&self) -> vk::ImageLayout {
        match &self.0 {
            TargetInner::Dmabuf { .. } => vk::ImageLayout::GENERAL,
            TargetInner::Texture { .. } => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    pub(super) fn mark_used(&self, point: u64) {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => buffer.mark_used(point),
            TargetInner::Texture { texture, .. } => texture.0.mark_used(point),
        }
    }
}

impl Texture for VulkanTarget<'_> {
    fn width(&self) -> u32 {
        self.size().w as u32
    }

    fn height(&self) -> u32 {
        self.size().h as u32
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => buffer.size,
            TargetInner::Texture { texture, .. } => texture.size(),
        }
    }

    fn format(&self) -> Option<Fourcc> {
        match &self.0 {
            TargetInner::Dmabuf { buffer, .. } => Some(buffer.format),
            TargetInner::Texture { texture, .. } => Texture::format(texture),
        }
    }
}

/// A texture mapping of the [`VulkanRenderer`](super::VulkanRenderer).
#[derive(Debug)]
pub struct VulkanTextureMapping {
    pub(super) device: Arc<Device>,
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) ptr: *mut std::ffi::c_void,
    pub(super) format: Fourcc,
    pub(super) size: Size<i32, BufferCoord>,
    /// Timeline point of the copy; mapping contents are valid once reached.
    pub(super) point: u64,
}

// SAFETY: the mapped pointer is exclusively owned by the mapping.
unsafe impl Send for VulkanTextureMapping {}
unsafe impl Sync for VulkanTextureMapping {}

impl Drop for VulkanTextureMapping {
    fn drop(&mut self) {
        self.device.defer_destroy(
            self.point,
            vec![CleanupItem::Buffer(self.buffer), CleanupItem::Memory(self.memory)],
        );
    }
}

impl Texture for VulkanTextureMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn format(&self) -> Option<Fourcc> {
        Some(self.format)
    }
}

impl TextureMapping for VulkanTextureMapping {
    fn flipped(&self) -> bool {
        false
    }
}
