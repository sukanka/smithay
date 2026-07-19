//! [`Fence`] implementation on top of Vulkan timeline semaphores.

use std::{
    os::unix::io::{FromRawFd, OwnedFd},
    sync::{Arc, Mutex},
};

use ash::vk;

use super::{CleanupItem, Device};
use crate::backend::renderer::sync::{Fence, Interrupted};

/// A fence tracking a point on the internal timeline semaphore of a
/// [`VulkanRenderer`](super::VulkanRenderer) submission.
#[derive(Debug)]
pub struct VulkanFence {
    pub(super) device: Arc<Device>,
    /// Timeline point signaled when the submission completes.
    pub(super) point: u64,
    /// Binary semaphore signaled by the same submission, used for sync_file export.
    ///
    /// Exporting a `SYNC_FD` has wait semantics on the binary payload, so the export happens at
    /// most once and the resulting fd is cached and duplicated for subsequent exports.
    pub(super) binary: Option<Mutex<BinarySemaphore>>,
}

#[derive(Debug)]
pub(super) enum BinarySemaphore {
    /// The semaphore has a pending or reached signal operation and was not exported yet.
    Unexported(vk::Semaphore),
    /// The semaphore payload was exported into a sync_file.
    Exported(vk::Semaphore, OwnedFd),
    /// Export failed; the semaphore cannot be used further.
    Failed(vk::Semaphore),
}

impl BinarySemaphore {
    fn handle(&self) -> vk::Semaphore {
        match self {
            BinarySemaphore::Unexported(sem)
            | BinarySemaphore::Exported(sem, _)
            | BinarySemaphore::Failed(sem) => *sem,
        }
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        if let Some(binary) = self.binary.take() {
            let sem = binary.into_inner().unwrap().handle();
            self.device
                .defer_destroy(self.point, vec![CleanupItem::Semaphore(sem)]);
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        unsafe { self.device.raw.get_semaphore_counter_value(self.device.timeline) }
            .map(|value| value >= self.point)
            .unwrap_or(true)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        let semaphores = [self.device.timeline];
        let points = [self.point];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&points);
        unsafe { self.device.raw.wait_semaphores(&wait_info, u64::MAX) }.map_err(|err| {
            tracing::warn!(?err, "Waiting for vulkan fence failed");
            Interrupted
        })
    }

    fn is_exportable(&self) -> bool {
        self.binary.is_some() && self.device.external_semaphore_fd.is_some()
    }

    fn export(&self) -> Option<OwnedFd> {
        let ext = self.device.external_semaphore_fd.as_ref()?;
        let mut guard = self.binary.as_ref()?.lock().unwrap();

        match &*guard {
            BinarySemaphore::Exported(_, fd) => fd.try_clone().ok(),
            BinarySemaphore::Failed(_) => None,
            BinarySemaphore::Unexported(sem) => {
                let sem = *sem;
                let info = vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(sem)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                match unsafe { ext.get_semaphore_fd(&info) } {
                    Ok(fd) => {
                        // SAFETY: on success ownership of the fd is transferred to us.
                        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
                        let clone = fd.try_clone().ok();
                        *guard = BinarySemaphore::Exported(sem, fd);
                        clone
                    }
                    Err(err) => {
                        tracing::warn!(?err, "Failed to export vulkan semaphore as sync_file");
                        *guard = BinarySemaphore::Failed(sem);
                        None
                    }
                }
            }
        }
    }
}
