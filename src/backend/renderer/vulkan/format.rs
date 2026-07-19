//! Format tables and property queries for the Vulkan renderer.
//!
//! Unlike the [allocator format table](crate::backend::allocator::vulkan::format), the renderer
//! maps FourCC codes to `UNORM` (or float) Vulkan formats. The GLES renderer neither linearizes
//! texel data when sampling nor encodes when writing to the target — blending happens directly in
//! the (typically sRGB-encoded) framebuffer values. Using `UNORM` formats replicates that
//! behaviour exactly.

use ash::vk;

use crate::backend::{
    allocator::{Fourcc, Modifier},
    vulkan::PhysicalDevice,
};

/// Static mapping of a FourCC code to a Vulkan format.
#[derive(Debug, Clone, Copy)]
pub(super) struct FormatMapping {
    pub fourcc: Fourcc,
    pub vk: vk::Format,
    /// Whether the FourCC format has an alpha channel.
    ///
    /// Formats without alpha (`Xrgb8888`, ...) are sampled through an image view with the alpha
    /// component swizzled to one.
    pub has_alpha: bool,
}

const fn fm(fourcc: Fourcc, vk: vk::Format, has_alpha: bool) -> FormatMapping {
    FormatMapping {
        fourcc,
        vk,
        has_alpha,
    }
}

/// All format conversions known to the renderer.
///
/// Byte-order notes: FourCC codes describe little-endian packed values, non-`PACK` Vulkan formats
/// describe byte order. The `PACK32`/`PACK16` conversions are only valid on little-endian hosts.
pub(super) const KNOWN_FORMATS: &[FormatMapping] = &[
    fm(Fourcc::Argb8888, vk::Format::B8G8R8A8_UNORM, true),
    fm(Fourcc::Xrgb8888, vk::Format::B8G8R8A8_UNORM, false),
    fm(Fourcc::Abgr8888, vk::Format::R8G8B8A8_UNORM, true),
    fm(Fourcc::Xbgr8888, vk::Format::R8G8B8A8_UNORM, false),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Rgba8888, vk::Format::A8B8G8R8_UNORM_PACK32, true),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Rgbx8888, vk::Format::A8B8G8R8_UNORM_PACK32, false),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Argb2101010, vk::Format::A2R10G10B10_UNORM_PACK32, true),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Xrgb2101010, vk::Format::A2R10G10B10_UNORM_PACK32, false),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Abgr2101010, vk::Format::A2B10G10R10_UNORM_PACK32, true),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Xbgr2101010, vk::Format::A2B10G10R10_UNORM_PACK32, false),
    fm(Fourcc::Abgr16161616f, vk::Format::R16G16B16A16_SFLOAT, true),
    fm(Fourcc::Xbgr16161616f, vk::Format::R16G16B16A16_SFLOAT, false),
    #[cfg(target_endian = "little")]
    fm(Fourcc::Rgb565, vk::Format::R5G6B5_UNORM_PACK16, false),
];

/// Formats supported for memory (shm) imports.
pub(super) const MEM_FORMATS: &[Fourcc] = &[
    Fourcc::Abgr8888,
    Fourcc::Xbgr8888,
    Fourcc::Argb8888,
    Fourcc::Xrgb8888,
];

pub(super) fn get_format_mapping(fourcc: Fourcc) -> Option<FormatMapping> {
    KNOWN_FORMATS.iter().copied().find(|f| f.fourcc == fourcc)
}

/// Bytes per pixel of a mappable format.
pub(super) const fn bytes_per_pixel(fourcc: Fourcc) -> Option<usize> {
    match fourcc {
        Fourcc::Argb8888
        | Fourcc::Xrgb8888
        | Fourcc::Abgr8888
        | Fourcc::Xbgr8888
        | Fourcc::Rgba8888
        | Fourcc::Rgbx8888
        | Fourcc::Argb2101010
        | Fourcc::Xrgb2101010
        | Fourcc::Abgr2101010
        | Fourcc::Xbgr2101010 => Some(4),
        Fourcc::Abgr16161616f | Fourcc::Xbgr16161616f => Some(8),
        Fourcc::Rgb565 => Some(2),
        _ => None,
    }
}

/// Properties of one supported (format, modifier) pair.
#[derive(Debug, Clone, Copy)]
pub(super) struct ModifierInfo {
    pub modifier: Modifier,
    pub plane_count: u32,
    pub max_extent: vk::Extent2D,
}

/// Device-specific properties of a supported format.
#[derive(Debug, Clone)]
pub(super) struct FormatInfo {
    pub mapping: FormatMapping,
    /// Modifiers usable for sampling (dmabuf texture import).
    pub texture_modifiers: Vec<ModifierInfo>,
    /// Modifiers usable as a color attachment (dmabuf render target).
    pub render_modifiers: Vec<ModifierInfo>,
    /// Maximum extent for optimally tiled sampled images, if supported.
    pub optimal_max_extent: Option<vk::Extent2D>,
    /// Whether optimally tiled images support color attachment usage.
    pub optimal_render: bool,
}

pub(super) const TEXTURE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
pub(super) fn render_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
}
pub(super) fn mem_texture_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
}
pub(super) fn offscreen_usage() -> vk::ImageUsageFlags {
    render_usage() | vk::ImageUsageFlags::SAMPLED
}

const TEXTURE_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::SAMPLED_IMAGE;
const RENDER_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;

fn max_extent_for(
    phd: &PhysicalDevice,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    modifier: Option<u64>,
) -> Option<vk::Extent2D> {
    let mut modifier_info =
        vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default().sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(usage);
    if let Some(modifier) = modifier {
        modifier_info.drm_format_modifier = modifier;
        format_info = format_info.push_next(&mut modifier_info);
    }

    let mut properties = vk::ImageFormatProperties2::default();
    let res = unsafe {
        phd.instance()
            .handle()
            .get_physical_device_image_format_properties2(phd.handle(), &format_info, &mut properties)
    };

    res.ok().map(|_| vk::Extent2D {
        width: properties.image_format_properties.max_extent.width,
        height: properties.image_format_properties.max_extent.height,
    })
}

/// Queries the device-specific properties for a format mapping.
pub(super) fn query_format_info(phd: &PhysicalDevice, mapping: FormatMapping) -> FormatInfo {
    let instance = phd.instance().handle();

    // Query the number of supported modifiers, then the properties.
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut format_properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        instance.get_physical_device_format_properties2(phd.handle(), mapping.vk, &mut format_properties);
    }

    let mut modifier_properties =
        vec![vk::DrmFormatModifierPropertiesEXT::default(); modifier_list.drm_format_modifier_count as usize];
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifier_properties);
    let mut format_properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        instance.get_physical_device_format_properties2(phd.handle(), mapping.vk, &mut format_properties);
    }
    let optimal_features = format_properties.format_properties.optimal_tiling_features;

    let mut texture_modifiers = Vec::new();
    let mut render_modifiers = Vec::new();
    for props in modifier_properties {
        let features = props.drm_format_modifier_tiling_features;
        let modifier = Modifier::from(props.drm_format_modifier);

        if features.contains(TEXTURE_FEATURES) {
            if let Some(max_extent) = max_extent_for(
                phd,
                mapping.vk,
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                TEXTURE_USAGE,
                Some(props.drm_format_modifier),
            ) {
                texture_modifiers.push(ModifierInfo {
                    modifier,
                    plane_count: props.drm_format_modifier_plane_count,
                    max_extent,
                });
            }
        }

        if features.contains(RENDER_FEATURES) {
            if let Some(max_extent) = max_extent_for(
                phd,
                mapping.vk,
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                render_usage(),
                Some(props.drm_format_modifier),
            ) {
                render_modifiers.push(ModifierInfo {
                    modifier,
                    plane_count: props.drm_format_modifier_plane_count,
                    max_extent,
                });
            }
        }
    }

    let optimal_max_extent = optimal_features
        .contains(TEXTURE_FEATURES)
        .then(|| {
            max_extent_for(
                phd,
                mapping.vk,
                vk::ImageTiling::OPTIMAL,
                mem_texture_usage(),
                None,
            )
        })
        .flatten();
    let optimal_render = optimal_features.contains(RENDER_FEATURES)
        && max_extent_for(phd, mapping.vk, vk::ImageTiling::OPTIMAL, offscreen_usage(), None).is_some();

    FormatInfo {
        mapping,
        texture_modifiers,
        render_modifiers,
        optimal_max_extent,
        optimal_render,
    }
}
