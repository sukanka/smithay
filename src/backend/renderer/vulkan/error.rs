//! Error types for the Vulkan renderer.

use ash::vk;

use crate::backend::{SwapBuffersError, allocator::Fourcc};

/// Error returned by the [`VulkanRenderer`](super::VulkanRenderer) and [`VulkanFrame`](super::VulkanFrame).
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// A Vulkan API call returned an error
    #[error("Vulkan API error: {0}")]
    Vk(#[from] vk::Result),
    /// The physical device does not support a required Vulkan version
    #[error("The physical device does not support Vulkan {0}.{1}")]
    UnsupportedVersion(u32, u32),
    /// A required device extension is not supported
    #[error("Missing required device extension: {0}")]
    MissingExtension(&'static str),
    /// No suitable queue family was found
    #[error("The physical device has no graphics queue family")]
    NoGraphicsQueue,
    /// No suitable memory type was found for an allocation
    #[error("No suitable memory type for allocation")]
    NoMemoryType,
    /// The given buffer has an unsupported pixel format
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedFormat(Fourcc),
    /// The given buffer has an unsupported format modifier
    #[error("Unsupported format modifier for format {0:?}")]
    UnsupportedModifier(Fourcc),
    /// The given wl_shm format is not supported
    #[error("Unsupported wl_shm format: {0:?}")]
    #[cfg(feature = "wayland_frontend")]
    UnsupportedWlPixelFormat(wayland_server::protocol::wl_shm::Format),
    /// The given buffer was not accessible
    #[error("Error accessing the buffer: {0:?}")]
    #[cfg(feature = "wayland_frontend")]
    BufferAccessError(#[from] crate::wayland::shm::BufferAccessError),
    /// The provided buffer's size did not match the requested one
    #[error("The buffer is too small for the given dimensions")]
    UnexpectedSize,
    /// The requested operation is out of bounds of the target
    #[error("The requested region is out of bounds")]
    OutOfBounds,
    /// The texture cannot be updated (e.g. imported from a dmabuf)
    #[error("The texture is not writable")]
    NotWritable,
    /// Source and destination of a blit are the same
    #[error("Source and destination of the blit are the same image")]
    BlitSameImage,
    /// A custom pass referenced a texture that is not renderer-owned
    #[error("Custom passes require renderer-owned textures")]
    ForeignTextureInPass,
    /// Waiting for a sync point was interrupted
    #[error("Waiting for a sync point was interrupted")]
    SyncInterrupted,
    /// A shader module or pipeline could not be created
    #[error("Failed to create a shader module or pipeline")]
    PipelineCreation,
    /// A custom shader failed to compile
    #[error("Failed to compile a custom shader: {0}")]
    ShaderCompile(String),
    /// The dmabuf could not be imported
    #[error("Failed to import the dmabuf: {0}")]
    DmabufImport(&'static str),
}

impl From<VulkanError> for SwapBuffersError {
    #[inline]
    fn from(err: VulkanError) -> SwapBuffersError {
        match err {
            // Unrecoverable device or setup failures.
            VulkanError::Vk(vk::Result::ERROR_DEVICE_LOST)
            | VulkanError::Vk(vk::Result::ERROR_INITIALIZATION_FAILED) => {
                SwapBuffersError::ContextLost(Box::new(err))
            }
            x @ VulkanError::UnsupportedVersion(..)
            | x @ VulkanError::MissingExtension(_)
            | x @ VulkanError::NoGraphicsQueue
            | x @ VulkanError::PipelineCreation => SwapBuffersError::ContextLost(Box::new(x)),
            // Everything else is specific to one operation or buffer.
            x => SwapBuffersError::TemporaryFailure(Box::new(x)),
        }
    }
}
