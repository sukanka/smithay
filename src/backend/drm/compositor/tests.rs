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
