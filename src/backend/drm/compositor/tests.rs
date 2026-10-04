use super::*;

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
