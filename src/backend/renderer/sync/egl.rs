use std::{os::unix::io::OwnedFd, time::Duration};

use crate::backend::{
    egl::fence::EGLFence,
    renderer::sync::{Fence, Interrupted, failed_wait},
};

impl Fence for EGLFence {
    fn wait(&self) -> Result<(), Interrupted> {
        self.client_wait(None, false).map(|_| ()).map_err(failed_wait)
    }

    fn is_exportable(&self) -> bool {
        self.is_native()
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export().ok()
    }

    fn is_signaled(&self) -> bool {
        self.client_wait(Some(Duration::ZERO), false).unwrap_or(false)
    }
}
