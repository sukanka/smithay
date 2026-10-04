use super::*;

#[test]
fn overlay_failure_preserves_primary_content_when_restoring_composition() {
    use crate::backend::renderer::element::solid::{SolidColorBuffer, SolidColorRenderElement};
    let primary_buffer = SolidColorBuffer::new((800, 600), [0., 1., 0., 1.]);
    let overlay_buffer = SolidColorBuffer::new((100, 100), [1., 0., 0., 1.]);
    let primary = SolidColorRenderElement::from_buffer(&primary_buffer, (0, 0), 1., 1., Kind::Unspecified);
    let overlay = SolidColorRenderElement::from_buffer(&overlay_buffer, (0, 0), 1., 1., Kind::Unspecified);
    let mut candidate = Some(&primary);
    let mut removed = vec![(0, &overlay)];
    let mut states = RenderElementStates::default();
    states
        .states
        .insert(primary.id().clone(), RenderElementState::zero_copy(800 * 600));
    restore_primary_for_composition(&mut candidate, 1, &mut removed, &mut states);
    assert!(candidate.is_none());
    assert!(!states.states.contains_key(primary.id()));
    removed.sort_by_key(|(z, _)| *z);
    assert_eq!(
        removed.iter().map(|(_, e)| e.id()).collect::<Vec<_>>(),
        [overlay.id(), primary.id()]
    );
    // If the failed-plane loop already removed primary, do not draw it twice.
    restore_primary_for_composition(&mut candidate, 1, &mut removed, &mut states);
    assert_eq!(removed.len(), 2);
}

type TestFrame = CompositorFrameState<
    GbmAllocator<crate::backend::drm::DrmDeviceFd>,
    GbmFramebufferExporter<crate::backend::drm::DrmDeviceFd>,
>;

fn frame(presentation_state: Option<FramePresentationState>) -> TestFrame {
    FrameState {
        planes: SmallVec::new(),
        async_flip_failed: false,
        presentation_state,
    }
}

#[test]
fn plane_test_reuse_survives_completed_frame_handoff() {
    for mode in [PresentationMode::VSync, PresentationMode::Async] {
        for vrr in [false, true] {
            let state = FramePresentationState { mode, vrr };
            let mut current = frame(None);
            assert!(!current.can_skip_plane_test(true, true, state));

            // A compositor rendering only after vblank has no pending frame when assigning the
            // next frame's planes. Moving the accepted frame to current must retain eligibility.
            for _ in 0..1000 {
                let mut pending = Some(frame(Some(state)));
                std::mem::swap(&mut pending.take().unwrap(), &mut current);
                assert!(pending.is_none());
                assert!(current.can_skip_plane_test(true, true, state));
            }
        }
    }
}

#[test]
fn plane_test_reuse_requires_compatible_partial_update() {
    let state = FramePresentationState {
        mode: PresentationMode::VSync,
        vrr: false,
    };
    let previous = frame(Some(state));
    // Geometry, format, or color pipeline changes make the plane incompatible. A reset or
    // connector modeset disallows partial updates even if the plane itself is unchanged.
    for (compatible, partial) in [(false, true), (true, false), (false, false)] {
        assert!(!previous.can_skip_plane_test(compatible, partial, state));
    }
    assert!(previous.can_skip_plane_test(true, true, state));
}

#[test]
fn plane_test_reuse_compares_accepted_mode_after_async_fallback() {
    let accepted = FramePresentationState {
        mode: PresentationMode::VSync,
        vrr: false,
    };
    let previous = frame(Some(accepted));
    assert!(!previous.can_skip_plane_test(
        true,
        true,
        FramePresentationState {
            mode: PresentationMode::Async,
            ..accepted
        },
    ));
    assert!(previous.can_skip_plane_test(true, true, accepted));
}

#[test]
fn vrr_changes_require_plane_test_without_a_modeset() {
    for mode in [PresentationMode::VSync, PresentationMode::Async] {
        for vrr in [false, true] {
            let old_state = FramePresentationState { mode, vrr };
            let new_state = FramePresentationState { mode, vrr: !vrr };
            let mut previous = frame(Some(old_state));
            // Some drivers let use_vrr() change the staged state without commit_pending().
            assert!(!previous.can_skip_plane_test(true, true, new_state));
            previous.presentation_state = Some(new_state);
            assert!(previous.can_skip_plane_test(true, true, new_state));
        }
    }
}

#[test]
fn invalidated_presentation_state_cannot_reuse_plane_test() {
    let state = FramePresentationState {
        mode: PresentationMode::VSync,
        vrr: true,
    };
    let mut previous = frame(Some(state));
    previous.presentation_state = None;
    assert!(!previous.can_skip_plane_test(true, true, state));
}

#[test]
fn damage_clip_coordinates_preserve_crop_scale_and_outward_rounding() {
    let src = Rectangle::new((2.25, 3.5).into(), (20., 30.).into());
    let dst = Rectangle::from_size((40, 60).into());
    let damage = Rectangle::new((4, 6).into(), (8, 10).into());
    assert_eq!(
        damage_clip_rects(src, dst, Transform::Normal, Transform::Normal, [damage]),
        vec![[4, 6, 9, 12]],
    );

    let src = Rectangle::from_size((40., 60.).into());
    let damage = Rectangle::new((4, 6).into(), (8, 10).into());
    assert_eq!(
        damage_clip_rects(src, dst, Transform::Normal, Transform::_180, [damage]),
        vec![[28, 44, 36, 54]],
    );
}

#[test]
fn damage_clip_cache_reuses_exact_encoded_rectangles() {
    let mut cache = DamageClipsCache::<Arc<usize>>::default();
    let mut creates = 0;
    let mut create = |_: &mut [[i32; 4]]| {
        creates += 1;
        Ok(Arc::new(creates))
    };
    let rects = vec![[0, 0, 1920, 1080]];
    let first = cache.get_or_create(rects.clone(), &mut create).unwrap().unwrap();
    for _ in 0..1000 {
        let next = cache.get_or_create(rects.clone(), &mut create).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &next));
    }
    assert!(cache.get_or_create(Vec::new(), &mut create).unwrap().is_none());
    let changed = cache
        .get_or_create(vec![[1, 0, 1920, 1080]], &mut create)
        .unwrap()
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &changed));
    assert_eq!(creates, 2);
}

#[test]
fn damage_clip_cache_eviction_keeps_in_flight_values_alive() {
    let mut cache = DamageClipsCache::<Arc<()>>::default();
    let first = cache
        .get_or_create(vec![[0, 0, 1, 1]], |_| Ok(Arc::new(())))
        .unwrap()
        .unwrap();
    let weak = Arc::downgrade(&first);
    for size in 2..=5 {
        cache
            .get_or_create(vec![[0, 0, size, size]], |_| Ok(Arc::new(())))
            .unwrap();
    }
    assert!(
        weak.upgrade().is_some(),
        "the in-flight frame still owns the evicted blob"
    );
    drop(first);
    assert!(
        weak.upgrade().is_none(),
        "the bounded cache must have evicted the oldest blob"
    );

    let mut large_creates = 0;
    for _ in 0..2 {
        cache
            .get_or_create(vec![[0, 0, 1, 1]; 65], |_| {
                large_creates += 1;
                Ok(Arc::new(()))
            })
            .unwrap();
    }
    assert_eq!(
        large_creates, 2,
        "oversized damage must not be retained in the cache"
    );
}

#[test]
fn failed_damage_blob_creation_is_retried() {
    let mut cache = DamageClipsCache::<usize>::default();
    let rects = vec![[0, 0, 1, 1]];
    assert!(
        cache
            .get_or_create(rects.clone(), |_| Err(std::io::ErrorKind::OutOfMemory.into()))
            .is_err()
    );
    let value = cache.get_or_create(rects, |_| Ok(7)).unwrap();
    assert_eq!(value, Some(7));
}

struct BufferTestState;

impl wayland_server::Dispatch<WlBuffer, ()> for BufferTestState {
    fn request(
        _: &mut Self,
        _: &wayland_server::Client,
        _: &WlBuffer,
        _: <WlBuffer as Resource>::Request,
        _: &(),
        _: &wayland_server::DisplayHandle,
        _: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

fn buffer_client() -> (
    wayland_server::Display<BufferTestState>,
    std::os::unix::net::UnixStream,
    wayland_server::Client,
) {
    let display = wayland_server::Display::new().unwrap();
    let (client_socket, server_socket) = std::os::unix::net::UnixStream::pair().unwrap();
    let client = display
        .handle()
        .insert_client(server_socket, Arc::new(()))
        .unwrap();
    (display, client_socket, client)
}

fn buffer_key(buffer: &WlBuffer, allow_opaque_fallback: bool) -> ElementFramebufferCacheKey {
    ElementFramebufferCacheKey {
        buffer: ElementFramebufferCacheBuffer::Wayland(buffer.downgrade()),
        allow_opaque_fallback,
    }
}

#[derive(Debug)]
struct TestFramebuffer {
    handle: framebuffer::Handle,
    drops: Arc<std::sync::atomic::AtomicUsize>,
}

impl AsRef<framebuffer::Handle> for TestFramebuffer {
    fn as_ref(&self) -> &framebuffer::Handle {
        &self.handle
    }
}

impl Framebuffer for TestFramebuffer {
    fn format(&self) -> DrmFormat {
        DrmFormat {
            code: DrmFourcc::Xrgb8888,
            modifier: DrmModifier::Linear,
        }
    }
}

impl Drop for TestFramebuffer {
    fn drop(&mut self) {
        self.drops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn cached_framebuffer(
    id: u32,
    drops: &Arc<std::sync::atomic::AtomicUsize>,
) -> CachedDrmFramebuffer<TestFramebuffer> {
    CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(TestFramebuffer {
        handle: drm::control::from_u32(id).unwrap(),
        drops: drops.clone(),
    }))
}

#[test]
fn retained_framebuffer_survives_composited_frames_without_reusing_plane_failures() {
    let (display, _socket, client) = buffer_client();
    let handle = display.handle();
    let buffer = client
        .create_resource::<WlBuffer, (), BufferTestState>(&handle, 1, ())
        .unwrap();
    let key = buffer_key(&buffer, false);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fb = cached_framebuffer(1, &drops);
    let mut previous = ElementState {
        instances: SmallVec::new(),
        fb_cache: ElementFramebufferCache::default(),
    };
    previous.fb_cache.insert(key.clone(), Ok(fb.clone()));
    let mut retained = RetainedFramebufferCache::default();
    retained.insert(key.clone(), fb.clone());
    drop(fb);
    // Notification/animation frames do not visit element_config(), so the element state
    // (including all its failed-plane decisions) is dropped, while the import survives.
    drop(previous);
    for _ in 0..1000 {
        retained.cleanup();
    }
    let recovered = retained
        .get(&key)
        .expect("the same live buffer must not require another import");
    assert_eq!(u32::from(*recovered.as_ref()), 1);
    assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 0);

    assert!(
        retained.get(&buffer_key(&buffer, true)).is_none(),
        "opaque fallback is part of the key"
    );
    let other = client
        .create_resource::<WlBuffer, (), BufferTestState>(&handle, 1, ())
        .unwrap();
    assert!(
        retained.get(&buffer_key(&other, false)).is_none(),
        "same-format resources are distinct"
    );
    let mut other_device = RetainedFramebufferCache::<TestFramebuffer>::default();
    assert!(
        other_device.get(&key).is_none(),
        "imports are local to one compositor/device"
    );
}

#[test]
fn retained_framebuffer_is_removed_when_weak_wayland_resource_dies() {
    let (display, _socket, client) = buffer_client();
    let handle = display.handle();
    let buffer = client
        .create_resource::<WlBuffer, (), BufferTestState>(&handle, 1, ())
        .unwrap();
    let key = buffer_key(&buffer, false);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut retained = RetainedFramebufferCache::default();
    retained.insert(key.clone(), cached_framebuffer(1, &drops));
    handle
        .backend_handle()
        .destroy_object::<BufferTestState>(&buffer.id())
        .unwrap();
    assert!(!key.is_alive());
    assert!(retained.get(&key).is_none());
    retained.cleanup();
    assert!(retained.entries.is_empty());
    assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn retained_framebuffer_cache_is_bounded_and_keeps_in_flight_handles_alive() {
    let (display, _socket, client) = buffer_client();
    let handle = display.handle();
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first = cached_framebuffer(1, &drops);
    let mut retained = RetainedFramebufferCache::default();
    let buffer = client
        .create_resource::<WlBuffer, (), BufferTestState>(&handle, 1, ())
        .unwrap();
    let first_key = buffer_key(&buffer, false);
    retained.insert(first_key.clone(), first.clone());
    for id in 2..=MAX_RETAINED_FRAMEBUFFERS as u32 + 1 {
        let buffer = client
            .create_resource::<WlBuffer, (), BufferTestState>(&handle, 1, ())
            .unwrap();
        retained.insert(buffer_key(&buffer, false), cached_framebuffer(id, &drops));
    }
    assert_eq!(retained.entries.len(), MAX_RETAINED_FRAMEBUFFERS);
    assert!(retained.get(&first_key).is_none());
    assert_eq!(
        drops.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "an in-flight frame still owns the evicted FB"
    );
    drop(first);
    assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 1);
    retained.entries.clear();
    assert_eq!(
        drops.load(std::sync::atomic::Ordering::Relaxed),
        MAX_RETAINED_FRAMEBUFFERS + 1
    );
}

fn placeholder_frame() -> (
    wayland_server::Display<BufferTestState>,
    std::os::unix::net::UnixStream,
    FrameState<Dmabuf, TestFramebuffer>,
) {
    let (display, socket, client) = buffer_client();
    let buffer = client
        .create_resource::<WlBuffer, (), BufferTestState>(&display.handle(), 1, ())
        .unwrap();
    let buffer = crate::backend::renderer::utils::Buffer::with_implicit(buffer);
    let claims = crate::backend::drm::device::PlaneClaimStorage::default();
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut frame = FrameState {
        planes: SmallVec::new(),
        async_flip_failed: false,
        presentation_state: None,
    };
    for index in 1..=3 {
        let plane = drm::control::from_u32(index).unwrap();
        frame.planes.push((
            plane,
            PlaneState {
                skip: false,
                needs_test: false,
                element_state: Some(PlaneElementState {
                    id: Id::new(),
                    commit: CommitCounter::default(),
                    z_index: index as usize,
                    cursor_size: None,
                    cursor_post_blend: false,
                }),
                config: Some(PlaneConfig {
                    properties: PlaneProperties {
                        src: Rectangle::from_size((800, 600).into()).to_f64(),
                        dst: Rectangle::from_size((800, 600).into()),
                        transform: Transform::Normal,
                        alpha: 1.,
                        format: DrmFormat {
                            code: DrmFourcc::Xrgb8888,
                            modifier: DrmModifier::Linear,
                        },
                    },
                    buffer: DrmScanoutBuffer {
                        buffer: ScanoutBuffer::Wayland(buffer.clone()),
                        fb: cached_framebuffer(index, &drops),
                    },
                    damage_clips: None,
                    plane_claim: claims.claim(plane, drm::control::from_u32(20).unwrap()).unwrap(),
                    sync: None,
                    color_pipeline: None,
                }),
            },
        ));
    }
    frame
        .planes
        .push((drm::control::from_u32(4).unwrap(), PlaneState::default()));
    (display, socket, frame)
}

#[test]
fn replacing_placeholder_retests_the_complete_plane_combination() {
    let (_display, _socket, mut next) = placeholder_frame();
    let previous = FrameState {
        planes: next.planes.clone(),
        async_flip_failed: false,
        presentation_state: None,
    };
    // All cursor/overlay tests succeeded against the old direct-scanout primary.
    // The new composition target changes format and has no direct-scanout element.
    let primary = &mut next.planes[0].1;
    primary.config.as_mut().unwrap().properties.format.code = DrmFourcc::Xrgb2101010;
    primary.element_state = None;
    assert!(
        !next.needs_complete_test(&previous, true),
        "the pre-fix flags incorrectly skipped validation"
    );
    next.mark_configured_planes_for_test();
    assert!(next.needs_complete_test(&previous, true));
    assert!(next.planes[..3].iter().all(|(_, state)| state.needs_test));
    assert!(
        !next.planes[3].1.needs_test,
        "unused planes need no new validation"
    );
    // Modesets/resets still force full validation independently of these flags.
    assert!(next.needs_complete_test(&previous, false));
}

#[test]
fn failed_placeholder_combination_falls_back_even_previously_accepted_elements() {
    for composed_primary in [false, true] {
        let (_display, _socket, mut next) = placeholder_frame();
        if composed_primary {
            next.planes[0].1.element_state = None;
        }
        // The individual assignment tests cleared all needs_test flags. A later
        // full-frame test now fails with the placeholder or its replacement.
        assert!(next.planes.iter().all(|(_, state)| !state.needs_test));
        next.mark_element_planes_for_fallback();
        let removed: Vec<_> = next
            .planes
            .iter()
            .filter(|(_, state)| state.needs_test)
            .map(|(plane, _)| u32::from(*plane))
            .collect();
        assert_eq!(
            removed,
            if composed_primary {
                vec![2, 3]
            } else {
                vec![1, 2, 3]
            }
        );
        assert!(!next.planes[3].1.needs_test);
    }
}
