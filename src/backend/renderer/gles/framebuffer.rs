use std::{cell::RefCell, rc::Rc, sync::Weak};

use super::*;

// FBOs are small, but their attachments can keep deleted textures alive until cleanup.
const MAX_TEXTURE_FRAMEBUFFERS: usize = 32;

#[derive(Debug)]
pub(super) struct GlesFramebuffer {
    pub(super) fbo: ffi::types::GLuint,
    // Unlike textures, FBOs are not shared between EGL contexts. Never send them through
    // GlesCleanup, which another renderer in the share group may drain.
    deleted: Rc<RefCell<Vec<ffi::types::GLuint>>>,
}

impl Drop for GlesFramebuffer {
    fn drop(&mut self) {
        self.deleted.borrow_mut().push(self.fbo);
    }
}

#[derive(Debug, Default)]
pub(super) struct GlesFramebufferCache {
    entries: Vec<(Weak<()>, Rc<GlesFramebuffer>)>,
    scratch: Option<Rc<GlesFramebuffer>>,
    deleted: Rc<RefCell<Vec<ffi::types::GLuint>>>,
    #[cfg(test)]
    created: usize,
    #[cfg(test)]
    hits: usize,
}

impl GlesFramebufferCache {
    pub(super) fn create(&mut self, gl: &ffi::Gles2) -> Rc<GlesFramebuffer> {
        let mut fbo = 0;
        unsafe { gl.GenFramebuffers(1, &mut fbo) };
        #[cfg(test)]
        {
            self.created += 1;
        }
        Rc::new(GlesFramebuffer {
            fbo,
            deleted: self.deleted.clone(),
        })
    }

    pub(super) fn bind_texture(
        &mut self,
        gl: &ffi::Gles2,
        texture: &GlesTexture,
    ) -> Result<Rc<GlesFramebuffer>, GlesError> {
        self.entries.retain(|(identity, _)| identity.strong_count() != 0);
        let identity = Arc::downgrade(&texture.0.identity);
        if let Some(index) = self.entries.iter().position(|(key, _)| key.ptr_eq(&identity)) {
            // Move recent entries to the end so frequently used targets survive eviction.
            let entry = self.entries.remove(index);
            let framebuffer = entry.1.clone();
            self.entries.push(entry);
            #[cfg(test)]
            {
                self.hits += 1;
            }
            trace!(fbo = framebuffer.fbo, "Reusing texture framebuffer");
            return Ok(framebuffer);
        }

        let framebuffer = self.create(gl);
        unsafe {
            // FRAMEBUFFER binds both the draw and read targets on GLES 3, and is also
            // supported on GLES 2. Its single attachment is shared by both binding points.
            gl.BindFramebuffer(ffi::FRAMEBUFFER, framebuffer.fbo);
            gl.FramebufferTexture2D(
                ffi::FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                texture.tex_id(),
                0,
            );
            let status = gl.CheckFramebufferStatus(ffi::FRAMEBUFFER);
            gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
            if status != ffi::FRAMEBUFFER_COMPLETE {
                drop(framebuffer);
                self.drain(gl);
                return Err(GlesError::FramebufferBindingError);
            }
        }
        trace!(fbo = framebuffer.fbo, "Created texture framebuffer");
        if self.entries.len() == MAX_TEXTURE_FRAMEBUFFERS {
            self.entries.remove(0);
        }
        self.entries.push((identity, framebuffer.clone()));
        self.drain(gl);
        Ok(framebuffer)
    }

    pub(super) fn scratch(&mut self, gl: &ffi::Gles2) -> Rc<GlesFramebuffer> {
        if let Some(framebuffer) = &self.scratch {
            return framebuffer.clone();
        }
        let framebuffer = self.create(gl);
        self.scratch = Some(framebuffer.clone());
        framebuffer
    }

    pub(super) fn cleanup(&mut self, egl: &EGLContext, gl: &ffi::Gles2) -> Result<(), MakeCurrentError> {
        self.entries.retain(|(identity, _)| identity.strong_count() != 0);
        // Preserve the idle-GPU behavior: don't make the context current for an empty queue.
        if !self.deleted.borrow().is_empty() {
            unsafe { egl.make_current()? };
            self.drain(gl);
        }
        Ok(())
    }

    fn drain(&self, gl: &ffi::Gles2) {
        let mut deleted = self.deleted.borrow_mut();
        if !deleted.is_empty() {
            unsafe { gl.DeleteFramebuffers(deleted.len() as i32, deleted.as_ptr()) };
            deleted.clear();
        }
    }

    pub(super) fn clear(&mut self, gl: &ffi::Gles2) {
        self.entries.clear();
        self.scratch = None;
        self.drain(gl);
    }
}

#[cfg(test)]
#[path = "framebuffer_tests.rs"]
mod tests;
