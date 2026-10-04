use std::{
    sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use super::*;
use crate::backend::egl::{EGLContext, native::EGLSurfacelessDisplay};

// Fake handles never invoke EGL. These tests drive the same cache/termination handshake
// deterministically, including the interval after the last Arc disappears but before Drop
// can acquire the cache mutex.
fn fake_handle(raw: usize) -> EGLDisplayHandle {
    EGLDisplayHandle {
        handle: raw as ffi::egl::types::EGLDisplay,
        should_terminate: false,
        _native: Box::new(()),
    }
}

#[test]
fn concurrent_live_handles_share_one_display_generation() {
    let cache = Arc::new(DisplayCache::default());
    let barrier = Arc::new(Barrier::new(8));
    let creates = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let cache = cache.clone();
            let barrier = barrier.clone();
            let creates = creates.clone();
            std::thread::spawn(move || {
                barrier.wait();
                cache.get_or_create(1usize as _, || {
                    creates.fetch_add(1, Ordering::SeqCst);
                    fake_handle(1)
                })
            })
        })
        .collect();
    let handles: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
    assert_eq!(creates.load(Ordering::SeqCst), 1);
    assert!(handles.iter().all(|handle| Arc::ptr_eq(handle, &handles[0])));
}

#[test]
fn replacement_waits_for_the_previous_generations_termination() {
    let cache = Arc::new(DisplayCache::default());
    let previous = cache.get_or_create(1usize as _, || fake_handle(1));
    let previous_weak = Arc::downgrade(&previous);
    let previous_identity = Arc::as_ptr(&previous);
    drop(previous);
    assert!(previous_weak.upgrade().is_none());

    let terminated = Arc::new(AtomicUsize::new(0));
    let worker = {
        let cache = cache.clone();
        let terminated = terminated.clone();
        std::thread::spawn(move || {
            cache.get_or_create(1usize as _, || {
                assert_eq!(
                    terminated.load(Ordering::SeqCst),
                    1,
                    "creating before old termination can invalidate the new display"
                );
                fake_handle(1)
            })
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while cache.waiters.load(Ordering::SeqCst) == 0 {
        assert!(
            Instant::now() < deadline,
            "replacement must reach the termination wait"
        );
        std::thread::yield_now();
    }
    // This is the same critical section that EGLDisplayHandle::drop uses for eglTerminate.
    cache.terminate(1usize as _, previous_identity, || {
        assert_eq!(cache.waiters.load(Ordering::SeqCst), 1);
        terminated.store(1, Ordering::SeqCst);
    });
    let replacement = worker.join().unwrap();
    assert_ne!(Arc::as_ptr(&replacement), previous_identity);
    assert_eq!(cache.waiters.load(Ordering::SeqCst), 0);
    let shared = cache.get_or_create(1usize as _, || panic!("the replacement is still alive"));
    assert!(Arc::ptr_eq(&replacement, &shared));
}

#[test]
fn parallel_surfaceless_display_and_context_lifetimes_remain_valid() {
    let probe = (|| -> Result<(), Error> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay)? };
        let _context = EGLContext::new(&display)?;
        Ok(())
    })();
    if let Err(err) = probe {
        if std::env::var_os("SMITHAY_TEST_REQUIRE_GLES").is_some_and(|v| !v.is_empty() && v != "0") {
            panic!("parallel EGL lifetime test requires a usable display: {err}");
        }
        tracing::warn!("skipping parallel EGL lifetime test: {err}");
        return;
    }

    let barrier = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|worker| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for iteration in 0..32 {
                    let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.unwrap();
                    let context = EGLContext::new(&display).unwrap();
                    unsafe { context.make_current() }.unwrap();
                    assert!(
                        !unsafe { ffi::egl::QueryString(display.display.handle, ffi::egl::VERSION as i32) }
                            .is_null()
                    );
                    if (worker + iteration) % 3 == 0 {
                        std::thread::yield_now();
                    }
                    context.unbind().unwrap();
                    drop(context);
                    drop(display);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}
