//! Texture fence ordering within one EGL command stream and between shared contexts.

use std::cell::RefCell;

use super::*;
use crate::backend::egl::{EGLDisplay, native::EGLSurfacelessDisplay};

#[derive(Debug, Default, Clone, Copy)]
struct Calls {
    polls: usize,
    waits: usize,
    fences: usize,
    deletes: usize,
    poll_result: ffi::types::GLenum,
}

thread_local! {
    static CALLS: RefCell<Calls> = const { RefCell::new(Calls {
        polls: 0,
        waits: 0,
        fences: 0,
        deletes: 0,
        poll_result: 0,
    }) };
}

extern "system" fn client_wait(
    _sync: ffi::types::GLsync,
    _flags: ffi::types::GLbitfield,
    _timeout: ffi::types::GLuint64,
) -> ffi::types::GLenum {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.polls += 1;
        calls.poll_result
    })
}

extern "system" fn server_wait(
    _sync: ffi::types::GLsync,
    _flags: ffi::types::GLbitfield,
    _timeout: ffi::types::GLuint64,
) {
    CALLS.with(|calls| calls.borrow_mut().waits += 1);
}

extern "system" fn fence(
    _condition: ffi::types::GLenum,
    _flags: ffi::types::GLbitfield,
) -> ffi::types::GLsync {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.fences += 1;
        // Opaque tokens used only by the mock entry points; never dereferenced.
        calls.fences as ffi::types::GLsync
    })
}

extern "system" fn delete(_sync: ffi::types::GLsync) {
    CALLS.with(|calls| calls.borrow_mut().deletes += 1);
}

fn mock_gl() -> Gles2 {
    CALLS.with(|calls| {
        *calls.borrow_mut() = Calls {
            poll_result: ffi::TIMEOUT_EXPIRED,
            ..Default::default()
        };
    });
    Gles2::load_with(|name| match name {
        "glClientWaitSync" => client_wait as *const () as *const _,
        "glWaitSync" => server_wait as *const () as *const _,
        "glFenceSync" => fence as *const () as *const _,
        "glDeleteSync" => delete as *const () as *const _,
        _ => std::ptr::null(),
    })
}

fn calls() -> Calls {
    CALLS.with(|calls| *calls.borrow())
}

#[test]
fn same_command_stream_skips_waits_but_keeps_latest_fences() {
    let gl = mock_gl();
    let stream = Arc::new(());
    let mut sync = TextureSync::default();
    sync.update_write(&gl, &stream);
    let first_write = sync.write_sync.lock().unwrap().as_ref().unwrap().sync;
    sync.wait_for_upload(&gl, &stream);
    assert_eq!(
        sync.write_sync.lock().unwrap().as_ref().unwrap().sync,
        first_write
    );

    sync.update_read(&gl, &stream);
    let first_read = sync.read_sync.lock().unwrap().as_ref().unwrap().sync;
    sync.update_read(&gl, &stream);
    assert_ne!(sync.read_sync.lock().unwrap().as_ref().unwrap().sync, first_read);
    sync.wait_for_all(&gl, &stream);
    sync.update_write(&gl, &stream);
    assert_ne!(
        sync.write_sync.lock().unwrap().as_ref().unwrap().sync,
        first_write
    );

    let calls = calls();
    assert_eq!(calls.polls, 0);
    assert_eq!(calls.waits, 0);
    assert_eq!(
        calls.fences, 4,
        "latest read and write fences must still be published"
    );
    assert_eq!(calls.deletes, 2, "replaced fences must still be destroyed");
}

#[test]
fn different_command_streams_preserve_upload_and_reader_dependencies() {
    let gl = mock_gl();
    let first = Arc::new(());
    let second = Arc::new(());
    let mut sync = TextureSync::default();

    sync.update_write(&gl, &first);
    sync.wait_for_upload(&gl, &second);
    sync.update_read(&gl, &second);
    sync.wait_for_all(&gl, &first);
    sync.update_write(&gl, &first);
    sync.update_read(&gl, &first);
    sync.wait_for_upload(&gl, &second);
    sync.update_read(&gl, &second);

    let calls = calls();
    assert_eq!(calls.polls, 3);
    assert_eq!(
        calls.waits, 5,
        "both upload waits and cross-context read chains remain"
    );
    assert_eq!(calls.fences, 5);
    assert!(Arc::ptr_eq(
        &sync.read_sync.lock().unwrap().as_ref().unwrap().command_stream,
        &second
    ));
    assert!(Arc::ptr_eq(
        &sync.write_sync.lock().unwrap().as_ref().unwrap().command_stream,
        &first
    ));
}

#[test]
fn replacing_another_contexts_write_fence_keeps_the_wait() {
    let gl = mock_gl();
    let mut sync = TextureSync::default();
    sync.update_write(&gl, &Arc::new(()));
    sync.update_write(&gl, &Arc::new(()));
    assert_eq!(calls().waits, 1);
    assert_eq!(calls().deletes, 1);
    assert_eq!(calls().fences, 2);
}

#[test]
fn completed_foreign_fence_is_deleted_without_a_server_wait() {
    let gl = mock_gl();
    let mut sync = TextureSync::default();
    sync.update_write(&gl, &Arc::new(()));
    CALLS.with(|calls| calls.borrow_mut().poll_result = ffi::CONDITION_SATISFIED);
    sync.wait_for_upload(&gl, &Arc::new(()));
    assert!(sync.write_sync.lock().unwrap().is_none());
    assert_eq!(calls().polls, 1);
    assert_eq!(calls().waits, 0);
    assert_eq!(calls().deletes, 1);
}

#[test]
fn fences_retain_context_identity_after_the_context_is_gone() {
    let gl = mock_gl();
    let original = Arc::new(());
    let weak = Arc::downgrade(&original);
    let mut sync = TextureSync::default();
    sync.update_write(&gl, &original);
    drop(original);
    let retained = weak
        .upgrade()
        .expect("a fence must keep its context generation alive");
    let replacement = Arc::new(());
    assert!(!Arc::ptr_eq(&retained, &replacement));
    sync.wait_for_upload(&gl, &replacement);
    assert_eq!(
        calls().waits,
        1,
        "a new context must not inherit the old context's ordering"
    );
}

fn renderers(shared: bool) -> Option<Vec<GlesRenderer>> {
    let result = (|| -> Result<Vec<GlesRenderer>, String> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.map_err(|err| err.to_string())?;
        let first = EGLContext::new(&display).map_err(|err| err.to_string())?;
        let second = if shared {
            Some(EGLContext::new_shared(&display, &first).map_err(|err| err.to_string())?)
        } else {
            None
        };
        let mut renderers = vec![unsafe { GlesRenderer::new(first) }.map_err(|err| err.to_string())?];
        if let Some(second) = second {
            renderers.push(unsafe { GlesRenderer::new(second) }.map_err(|err| err.to_string())?);
        }
        if renderers
            .iter()
            .any(|renderer| !renderer.capabilities.contains(&Capability::Fencing))
        {
            return Err("GLES 3 fencing is unavailable".into());
        }
        Ok(renderers)
    })();

    match result {
        Ok(renderers) => Some(renderers),
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_GLES")
                .is_some_and(|value| !value.is_empty() && value != "0")
            {
                panic!("GLES texture sync tests require a renderer: {err}");
            }
            eprintln!("skipping GLES texture sync test: {err}");
            None
        }
    }
}

fn sample(renderer: &mut GlesRenderer, source: &GlesTexture) -> GlesTexture {
    let mut destination = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    {
        let mut target = renderer.bind(&mut destination).unwrap();
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        let rect = Rectangle::from_size((1, 1).into());
        Frame::render_texture_from_to(
            &mut frame,
            source,
            Rectangle::from_size((1., 1.).into()),
            rect,
            &[rect],
            &[rect],
            Transform::Normal,
            1.,
        )
        .unwrap();
        // Keep submission/flush behavior intact; no CPU wait before the next writer.
        // Texture fences must order shared-context access without a CPU wait in this test.
        drop(frame.finish().unwrap());
    }
    destination
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
fn single_context_updates_and_samples_without_losing_pixels() {
    let Some(mut renderers) = renderers(false) else {
        return;
    };
    let renderer = &mut renderers[0];
    let source = renderer
        .import_memory(&[0, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
        .unwrap();
    let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let mut results = Vec::new();
    for color in colors.into_iter().cycle().take(24) {
        renderer
            .update_memory(&source, &color, Rectangle::from_size((1, 1).into()))
            .unwrap();
        results.push((sample(renderer, &source), color));
    }
    for (texture, expected) in results {
        check_pixel(renderer, &texture, expected);
    }
}

#[test]
fn shared_contexts_alternate_writes_and_reads_without_losing_pixels() {
    let Some(mut renderers) = renderers(true) else {
        return;
    };
    assert_eq!(
        renderers[0].context_id(),
        renderers[1].context_id(),
        "the texture share group is shared"
    );
    assert!(!Arc::ptr_eq(
        renderers[0].egl.command_stream_id(),
        renderers[1].egl.command_stream_id()
    ));
    let source = renderers[0]
        .import_memory(&[0, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
        .unwrap();
    let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let mut results = Vec::new();
    for (index, color) in colors.into_iter().cycle().take(24).enumerate() {
        let writer = index % 2;
        let reader = 1 - writer;
        renderers[writer]
            .update_memory(&source, &color, Rectangle::from_size((1, 1).into()))
            .unwrap();
        // Flushing the producer is necessary before a different context can wait on its
        // fence. The optimization must not replace or remove this cross-context requirement.
        renderers[writer]
            .with_context(|gl| unsafe { gl.Flush() })
            .unwrap();
        results.push((sample(&mut renderers[reader], &source), color));
    }
    // Defer readback until after all alternating submissions: a CPU wait between iterations
    // would mask missing read-before-overwrite dependencies.
    for (texture, expected) in results {
        check_pixel(&mut renderers[0], &texture, expected);
    }
}
