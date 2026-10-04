use super::*;
use crate::backend::egl::EGLDisplay;

fn renderers(shared: bool) -> Option<Vec<GlesRenderer>> {
    let result = (|| -> Result<Vec<GlesRenderer>, String> {
        let mut devices = EGLDevice::enumerate()
            .map_err(|err| err.to_string())?
            .collect::<Vec<_>>();
        devices.sort_by_key(|device| !device.is_software());
        for device in devices {
            let Ok(display) = (unsafe { EGLDisplay::new(device) }) else {
                continue;
            };
            let Ok(context) = EGLContext::new(&display) else {
                continue;
            };
            let second = if shared {
                Some(EGLContext::new_shared(&display, &context).map_err(|err| err.to_string())?)
            } else {
                None
            };
            let mut renderers = vec![unsafe { GlesRenderer::new(context) }.map_err(|err| err.to_string())?];
            if let Some(context) = second {
                renderers.push(unsafe { GlesRenderer::new(context) }.map_err(|err| err.to_string())?);
            }
            return Ok(renderers);
        }
        Err("no usable EGL device".into())
    })();
    match result {
        Ok(renderers) => Some(renderers),
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_GLES").is_some_and(|v| !v.is_empty() && v != "0") {
                panic!("GLES framebuffer tests require a renderer: {err}");
            }
            eprintln!("skipping GLES framebuffer test: {err}");
            None
        }
    }
}

fn clear(renderer: &mut GlesRenderer, texture: &GlesTexture, color: [f32; 4]) {
    let target = renderer.bind_texture(texture).unwrap();
    target.0.make_current(&renderer.gl, &renderer.egl).unwrap();
    unsafe {
        renderer.gl.ClearColor(color[0], color[1], color[2], color[3]);
        renderer.gl.Clear(ffi::COLOR_BUFFER_BIT);
        renderer.gl.Flush();
    }
}

fn check_pixel(renderer: &mut GlesRenderer, texture: &GlesTexture, expected: [u8; 4]) {
    let mapping = renderer
        .copy_texture(texture, Rectangle::from_size((1, 1).into()), Fourcc::Abgr8888)
        .unwrap();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), expected);
    renderer
        .with_context(|gl| assert_eq!(unsafe { gl.GetError() }, ffi::NO_ERROR))
        .unwrap();
}

#[test]
fn repeated_texture_targets_create_one_framebuffer_and_preserve_pixels() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let mut texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    for index in 0..64 {
        let red = (index % 2) as f32;
        clear(renderer, &texture, [red, 0., 1., 1.]);
        check_pixel(renderer, &texture, [(red * 255.) as u8, 0, 255, 255]);
        assert!(
            texture.is_unique_reference(),
            "weak FBO identities must not force blur texture recreation"
        );
    }
    assert_eq!(
        renderer.texture_framebuffers.created, 1,
        "128 target binds allocate once"
    );
    assert_eq!(renderer.texture_framebuffers.hits, 127);
}

#[test]
fn destroyed_texture_reclaims_its_attachment_in_the_owning_context() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    drop(renderer.bind_texture(&texture).unwrap());
    let fbo = renderer.texture_framebuffers.entries[0].1.fbo;
    let texture_id = texture.tex_id();
    drop(texture);
    renderer.cleanup().unwrap();
    assert!(renderer.texture_framebuffers.entries.is_empty());
    unsafe {
        assert_eq!(renderer.gl.IsFramebuffer(fbo), ffi::FALSE);
        assert_eq!(renderer.gl.IsTexture(texture_id), ffi::FALSE);
    }
}

#[test]
fn readback_then_blit_reuses_the_color_attachment() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let source: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    let destination: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    clear(renderer, &source, [1., 0., 1., 1.]);
    clear(renderer, &destination, [0., 0., 0., 1.]);
    check_pixel(renderer, &source, [255, 0, 255, 255]);
    {
        let source = renderer.bind_texture(&source).unwrap();
        let mut destination = renderer.bind_texture(&destination).unwrap();
        let rect = Rectangle::from_size((1, 1).into());
        renderer
            .blit(&source, &mut destination, rect, rect, TextureFilter::Nearest)
            .unwrap()
            .wait()
            .unwrap();
    }
    check_pixel(renderer, &destination, [255, 0, 255, 255]);
    assert_eq!(renderer.texture_framebuffers.created, 2);
}

#[test]
fn texture_identity_prevents_numeric_name_reuse_from_hitting_the_cache() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let mut texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    clear(renderer, &texture, [1., 0., 0., 1.]);
    let first_fbo = renderer.texture_framebuffers.entries[0].1.fbo;
    // Model a new texture wrapper with the same GL numeric name without relying on a driver's
    // name allocator. Only the lifetime token changes, exactly as for a newly imported texture.
    Arc::get_mut(&mut texture.0).unwrap().identity = Arc::new(());
    clear(renderer, &texture, [0., 1., 0., 1.]);
    assert_eq!(renderer.texture_framebuffers.created, 2);
    assert_ne!(renderer.texture_framebuffers.entries[0].1.fbo, first_fbo);
    check_pixel(renderer, &texture, [0, 255, 0, 255]);
}

#[test]
fn eviction_is_bounded_and_does_not_delete_a_live_target() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let held_texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    let held_target = renderer.bind_texture(&held_texture).unwrap();
    let held_fbo = renderer.texture_framebuffers.entries[0].1.fbo;
    let mut textures = Vec::new();
    for _ in 0..MAX_TEXTURE_FRAMEBUFFERS + 8 {
        let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
        clear(renderer, &texture, [0., 0., 1., 1.]);
        textures.push(texture);
    }
    assert_eq!(
        renderer.texture_framebuffers.entries.len(),
        MAX_TEXTURE_FRAMEBUFFERS
    );
    assert_eq!(unsafe { renderer.gl.IsFramebuffer(held_fbo) }, ffi::TRUE);
    held_target.0.make_current(&renderer.gl, &renderer.egl).unwrap();
    assert_eq!(
        unsafe { renderer.gl.CheckFramebufferStatus(ffi::FRAMEBUFFER) },
        ffi::FRAMEBUFFER_COMPLETE
    );
    drop(held_target);
    renderer.cleanup().unwrap();
    assert_eq!(unsafe { renderer.gl.IsFramebuffer(held_fbo) }, ffi::FALSE);
    for texture in textures {
        check_pixel(renderer, &texture, [0, 0, 255, 255]);
    }
}

#[test]
fn shared_contexts_own_separate_framebuffers_and_cleanup_queues() {
    let Some(mut renderers) = renderers(true) else {
        return;
    };
    assert_eq!(renderers[0].context_id(), renderers[1].context_id());
    let texture: GlesTexture = renderers[0]
        .create_buffer(Fourcc::Abgr8888, (1, 1).into())
        .unwrap();
    for renderer in &mut renderers {
        drop(renderer.bind_texture(&texture).unwrap());
        assert_eq!(renderer.texture_framebuffers.created, 1);
    }
    let second_fbo = renderers[1].texture_framebuffers.entries[0].1.fbo;
    assert!(!Rc::ptr_eq(
        &renderers[0].texture_framebuffers.deleted,
        &renderers[1].texture_framebuffers.deleted
    ));
    drop(texture);
    renderers[0].cleanup().unwrap();
    // The other context still owns its FBO, even though shared texture cleanup has run.
    renderers[1]
        .with_context(|gl| assert_eq!(unsafe { gl.IsFramebuffer(second_fbo) }, ffi::TRUE))
        .unwrap();
    renderers[1].cleanup().unwrap();
    assert_eq!(unsafe { renderers[1].gl.IsFramebuffer(second_fbo) }, ffi::FALSE);
    drop(renderers.remove(0));
    let fresh: GlesTexture = renderers[0]
        .create_buffer(Fourcc::Abgr8888, (1, 1).into())
        .unwrap();
    clear(&mut renderers[0], &fresh, [1., 0., 1., 1.]);
    check_pixel(&mut renderers[0], &fresh, [255, 0, 255, 255]);
}

#[test]
fn renderbuffer_target_cleanup_cannot_delete_another_contexts_texture_fbo() {
    let Some(mut renderers) = renderers(true) else {
        return;
    };
    let texture: GlesTexture = renderers[0]
        .create_buffer(Fourcc::Abgr8888, (1, 1).into())
        .unwrap();
    let mut renderbuffer: GlesRenderbuffer = renderers[1]
        .create_buffer(Fourcc::Abgr8888, (1, 1).into())
        .unwrap();
    drop(renderers[0].bind_texture(&texture).unwrap());
    let texture_fbo = renderers[0].texture_framebuffers.entries[0].1.fbo;
    drop(renderers[1].bind(&mut renderbuffer).unwrap());
    assert_eq!(renderers[1].texture_framebuffers.deleted.borrow().len(), 1);
    renderers[0].cleanup().unwrap();
    renderers[0]
        .with_context(|gl| assert_eq!(unsafe { gl.IsFramebuffer(texture_fbo) }, ffi::TRUE))
        .unwrap();
    renderers[1].cleanup().unwrap();
    assert!(renderers[1].texture_framebuffers.deleted.borrow().is_empty());
    clear(&mut renderers[0], &texture, [1., 1., 1., 1.]);
    check_pixel(&mut renderers[0], &texture, [255, 255, 255, 255]);
}

#[test]
fn replacement_shared_context_starts_with_an_empty_framebuffer_cache() {
    let Some(mut renderers) = renderers(true) else {
        return;
    };
    let texture: GlesTexture = renderers[0]
        .create_buffer(Fourcc::Abgr8888, (1, 1).into())
        .unwrap();
    clear(&mut renderers[0], &texture, [0., 1., 1., 1.]);
    let context = EGLContext::new_shared(renderers[1].egl.display(), &renderers[1].egl).unwrap();
    drop(renderers.remove(0));
    let mut replacement = unsafe { GlesRenderer::new(context) }.unwrap();
    assert!(replacement.texture_framebuffers.entries.is_empty());
    assert_eq!(replacement.context_id(), renderers[0].context_id());
    check_pixel(&mut replacement, &texture, [0, 255, 255, 255]);
    assert_eq!(replacement.texture_framebuffers.created, 1);
}

#[test]
fn incomplete_attachment_is_not_cached_and_can_be_retried() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let name = renderer
        .with_context(|gl| unsafe {
            let mut name = 0;
            gl.GenTextures(1, &mut name);
            gl.BindTexture(ffi::TEXTURE_2D, name);
            gl.BindTexture(ffi::TEXTURE_2D, 0);
            name
        })
        .unwrap();
    let texture = unsafe { GlesTexture::from_raw(renderer, Some(ffi::RGBA8), false, name, (1, 1).into()) };
    assert!(renderer.bind_texture(&texture).is_err());
    assert!(renderer.texture_framebuffers.entries.is_empty());
    assert!(renderer.texture_framebuffers.deleted.borrow().is_empty());
    renderer
        .with_context(|gl| unsafe {
            gl.BindTexture(ffi::TEXTURE_2D, name);
            gl.TexImage2D(
                ffi::TEXTURE_2D,
                0,
                ffi::RGBA8 as i32,
                1,
                1,
                0,
                ffi::RGBA,
                ffi::UNSIGNED_BYTE,
                ptr::null(),
            );
            gl.BindTexture(ffi::TEXTURE_2D, 0);
        })
        .unwrap();
    clear(renderer, &texture, [1., 0., 0., 1.]);
    check_pixel(renderer, &texture, [255, 0, 0, 255]);
    assert_eq!(renderer.texture_framebuffers.created, 2);
}

#[test]
fn scratch_framebuffer_reuses_one_object_and_detaches_textures_even_on_error() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    for _ in 0..32 {
        let result = renderer
            .with_profiled_framebuffer(gpu_span_location!("scratch test"), |gl| unsafe {
                assert_eq!(
                    gl.CheckFramebufferStatus(ffi::FRAMEBUFFER),
                    ffi::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT
                );
                gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    texture.tex_id(),
                    0,
                );
                assert_eq!(
                    gl.CheckFramebufferStatus(ffi::FRAMEBUFFER),
                    ffi::FRAMEBUFFER_COMPLETE
                );
                gl.ClearColor(1., 1., 0., 1.);
                gl.Clear(ffi::COLOR_BUFFER_BIT);
                Err::<(), _>("callback error")
            })
            .unwrap();
        assert!(result.is_err());
    }
    assert_eq!(renderer.texture_framebuffers.created, 1);
    check_pixel(renderer, &texture, [255, 255, 0, 255]);
    let texture_id = texture.tex_id();
    drop(texture);
    renderer.cleanup().unwrap();
    assert_eq!(unsafe { renderer.gl.IsTexture(texture_id) }, ffi::FALSE);
}
