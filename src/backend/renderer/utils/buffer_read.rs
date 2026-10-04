//! Per-submission Wayland read dependencies, including imports that render DMA shadows.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    os::fd::AsFd,
};

use super::Buffer;
use crate::backend::{
    allocator::dmabuf::{SyncFileFlags, import_sync_file},
    renderer::sync::SyncPoint,
};

thread_local! {
    static SOURCE: RefCell<Option<Buffer>> = const { RefCell::new(None) };
    #[cfg(test)]
    static PUBLISHED: RefCell<Vec<Vec<usize>>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn take_published() -> Vec<Vec<usize>> {
    PUBLISHED.with(|published| std::mem::take(&mut *published.borrow_mut()))
}

struct SourceGuard(Option<Buffer>);

impl Drop for SourceGuard {
    fn drop(&mut self) {
        SOURCE.with(|source| *source.borrow_mut() = self.0.take());
    }
}

pub(super) fn with_source<R>(buffer: &Buffer, f: impl FnOnce() -> R) -> R {
    let _guard = SourceGuard(SOURCE.with(|source| source.replace(Some(buffer.clone()))));
    f()
}

/// Buffers kept alive until the submission that read them has published its completion.
#[derive(Debug, Default)]
pub(crate) struct BufferReadSet {
    buffers: HashMap<usize, Buffer>,
}

impl BufferReadSet {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.buffers.len()
    }
    /// Call immediately before recording a texture read, after rejecting empty draws.
    pub(crate) fn capture(&mut self) {
        SOURCE.with(|source| {
            if let Some(buffer) = source.borrow().as_ref() {
                self.buffers
                    .entry(buffer.read_identity())
                    .or_insert_with(|| buffer.clone());
            }
        });
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    pub(crate) fn clear_completed(&mut self) {
        self.buffers.clear();
    }

    pub(crate) fn append(&mut self, other: &mut Self) {
        self.buffers.extend(other.buffers.drain());
    }

    /// Publish only this submission's reads. Every explicit commit keeps its own release
    /// point, while DMA-BUF reservation imports are deduplicated across aliases.
    pub(crate) fn publish(&mut self, sync: &SyncPoint) {
        if self.buffers.is_empty() {
            return;
        }
        #[cfg(test)]
        PUBLISHED.with(|published| {
            published
                .borrow_mut()
                .push(self.buffers.keys().copied().collect())
        });
        if !sync.is_reached() {
            let published = sync.export().is_some_and(|fence| {
                let mut imported = HashSet::new();
                let mut success = true;
                for buffer in self.buffers.values() {
                    // A failed merge leaves the earlier fence intact; waiting for this new
                    // submission below then preserves both contexts' read dependencies.
                    success &= buffer.add_release_fence(fence.as_fd()).is_ok();
                    if success {
                        if let Ok(dmabuf) = crate::wayland::dmabuf::get_dmabuf(buffer) {
                            if imported.insert(dmabuf.clone()) {
                                for plane in dmabuf.handles() {
                                    success &=
                                        import_sync_file(plane, SyncFileFlags::READ, fence.as_fd()).is_ok();
                                }
                            }
                        }
                    }
                }
                success
            });
            if !published {
                while sync.wait().is_err() {}
            }
        }
        self.buffers.clear();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::{
        allocator::{
            Fourcc, Modifier,
            dmabuf::{Dmabuf, DmabufFlags},
        },
        renderer::sync::{Fence, Interrupted},
    };
    use std::{
        os::fd::OwnedFd,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    use wayland_server::{Resource, protocol::wl_buffer::WlBuffer};

    pub(crate) struct State;
    impl<D: Send + Sync + 'static> wayland_server::Dispatch<WlBuffer, D> for State {
        fn request(
            _: &mut Self,
            _: &wayland_server::Client,
            _: &WlBuffer,
            _: <WlBuffer as Resource>::Request,
            _: &D,
            _: &wayland_server::DisplayHandle,
            _: &mut wayland_server::DataInit<'_, Self>,
        ) {
        }
    }

    pub(crate) fn buffer() -> (
        wayland_server::Display<State>,
        std::os::unix::net::UnixStream,
        Buffer,
    ) {
        buffer_with_data(())
    }

    fn buffer_with_data<D: Send + Sync + 'static>(
        data: D,
    ) -> (
        wayland_server::Display<State>,
        std::os::unix::net::UnixStream,
        Buffer,
    ) {
        let display = wayland_server::Display::new().unwrap();
        let (socket, server) = std::os::unix::net::UnixStream::pair().unwrap();
        let client = display.handle().insert_client(server, Arc::new(())).unwrap();
        let buffer = client
            .create_resource::<WlBuffer, D, State>(&display.handle(), 1, data)
            .unwrap();
        (display, socket, Buffer::with_implicit(buffer))
    }

    #[derive(Debug)]
    struct PendingFence {
        waits: Arc<AtomicUsize>,
        exportable: bool,
    }

    impl Fence for PendingFence {
        fn is_signaled(&self) -> bool {
            self.waits.load(Ordering::Relaxed) >= 3
        }
        fn wait(&self) -> Result<(), Interrupted> {
            if self.waits.fetch_add(1, Ordering::Relaxed) < 2 {
                Err(Interrupted)
            } else {
                Ok(())
            }
        }
        fn is_exportable(&self) -> bool {
            self.exportable
        }
        fn export(&self) -> Option<OwnedFd> {
            self.exportable
                .then(|| std::fs::File::open("/dev/null").unwrap().into())
        }
    }

    #[test]
    fn read_scopes_skip_unused_sources_deduplicate_draws_and_keep_distinct_commits() {
        let (_display, _socket, buffer) = buffer();
        let mut reads = BufferReadSet::default();
        buffer.with_read_source(|| {});
        reads.capture();
        assert!(reads.is_empty(), "scope/import alone is not a GPU read");
        buffer.with_read_source(|| {
            reads.capture();
            reads.capture();
        });
        assert_eq!(reads.len(), 1);
        let other_commit = Buffer::with_implicit((*buffer).clone());
        other_commit.with_read_source(|| reads.capture());
        assert_eq!(
            reads.len(),
            2,
            "distinct release points must never be deduplicated by wl_buffer id"
        );
    }

    #[test]
    fn nested_read_scopes_restore_source_even_when_an_inner_pass_panics() {
        let (_display, _socket, first) = buffer();
        let second = Buffer::with_implicit((*first).clone());
        let mut outer = BufferReadSet::default();
        let mut inner = BufferReadSet::default();
        first.with_read_source(|| {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                second.with_read_source(|| {
                    inner.capture();
                    panic!("failed inner pass")
                });
            }));
            outer.capture();
        });
        assert!(outer.buffers.contains_key(&first.read_identity()));
        assert!(!outer.buffers.contains_key(&second.read_identity()));
        assert!(inner.buffers.contains_key(&second.read_identity()));
        let mut later = BufferReadSet::default();
        later.capture();
        assert!(later.is_empty());
    }

    #[test]
    fn unexportable_submission_waits_through_interruptions_before_releasing_reads() {
        let (_display, _socket, buffer) = buffer();
        let mut reads = BufferReadSet::default();
        buffer.with_read_source(|| reads.capture());
        let waits = Arc::new(AtomicUsize::new(0));
        let sync = SyncPoint::from(PendingFence {
            waits: waits.clone(),
            exportable: false,
        });
        reads.publish(&sync);
        assert_eq!(waits.load(Ordering::Relaxed), 3);
        assert!(reads.is_empty());
    }

    #[test]
    fn completed_and_empty_submissions_do_not_export_or_wait() {
        let (_display, _socket, buffer) = buffer();
        let mut reads = BufferReadSet::default();
        let waits = Arc::new(AtomicUsize::new(0));
        let sync = SyncPoint::from(PendingFence {
            waits: waits.clone(),
            exportable: false,
        });
        reads.publish(&sync);
        assert_eq!(waits.load(Ordering::Relaxed), 0);
        buffer.with_read_source(|| reads.capture());
        waits.store(3, Ordering::Relaxed);
        reads.publish(&sync);
        assert_eq!(waits.load(Ordering::Relaxed), 3);
        assert!(reads.is_empty());
    }

    #[test]
    fn rejected_implicit_import_waits_before_releasing_buffers() {
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let mut builder = Dmabuf::builder((1, 1), Fourcc::Abgr8888, Modifier::Linear, DmabufFlags::empty());
        assert!(builder.add_plane(fd, 0, 4));
        let (_display, _socket, buffer) = buffer_with_data(builder.build().unwrap());
        let mut reads = BufferReadSet::default();
        buffer.with_read_source(|| reads.capture());
        let waits = Arc::new(AtomicUsize::new(0));
        let sync = SyncPoint::from(PendingFence {
            waits: waits.clone(),
            exportable: true,
        });
        reads.publish(&sync);
        assert_eq!(
            waits.load(Ordering::Relaxed),
            3,
            "ENOTTY must take the completion fallback"
        );
        assert!(reads.is_empty());
    }
}
