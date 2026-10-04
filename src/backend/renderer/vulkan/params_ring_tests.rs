//! Real Vulkan tests for parameter storage reuse and the lifetime of recorded draws.
//!
//! Set `SMITHAY_TEST_REQUIRE_VULKAN=1` to require a compatible hardware or software device.

use std::sync::atomic::Ordering;

use ash::vk;

use super::super::descriptor_pool_tests::renderer;
use super::super::{
    CleanupItem, CustomUniform, CustomUniformDecl, CustomUniformKind, CustomUniformValue,
    DESCRIPTOR_POOL_SIZE, MAX_CACHED_PARAMS_RINGS, PARAMS_RANGE, VulkanPixelProgram, VulkanRenderer,
    VulkanTexture, uniform_block_glsl,
};
use crate::{
    backend::{
        allocator::Fourcc,
        renderer::{Bind, ExportMem, Frame, Offscreen, Renderer},
    },
    utils::{Rectangle, Transform},
};

/// Only used when no queue submissions are pending. Allocations use a future point to
/// exercise the reuse boundary deterministically, then host-signalling completes it.
fn signal_unused_point(renderer: &mut VulkanRenderer, point: u64) {
    assert!(renderer.in_flight.is_empty());
    let signal = vk::SemaphoreSignalInfo::default()
        .semaphore(renderer.device.timeline)
        .value(point);
    unsafe { renderer.device.raw.signal_semaphore(&signal) }.unwrap();
    renderer.timeline_point = point;
}

fn color_program(renderer: &mut VulkanRenderer) -> VulkanPixelProgram {
    let uniforms = [CustomUniformDecl {
        name: "color".into(),
        kind: CustomUniformKind::Vec4,
    }];
    let source = format!(
        "#version 450\n{}\nlayout(location = 0) out vec4 output_color;\n\
         void main() {{ output_color = color; }}\n",
        uniform_block_glsl(&uniforms)
    );
    renderer
        .compile_custom_pixel_shader(&source, &uniforms, &[])
        .unwrap()
}

fn pixels(renderer: &mut VulkanRenderer, texture: &VulkanTexture, size: (i32, i32)) -> Vec<u8> {
    let mapping = renderer
        .copy_texture(texture, Rectangle::from_size(size.into()), Fourcc::Abgr8888)
        .unwrap();
    renderer.map_texture(&mapping).unwrap().to_vec()
}

#[test]
fn parameter_rings_reuse_only_completed_timeline_points() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut pending = renderer.acquire_params_ring().unwrap();
    let pending_buffer = pending.buffer;
    let pending_set = pending.ds;
    pending.used = PARAMS_RANGE;
    renderer.recycle_params_ring(1, pending);

    // No GPU submission can complete point 1 yet. Acquisition must allocate rather
    // than wait for it or overwrite its parameters.
    let mut available = renderer.acquire_params_ring().unwrap();
    assert_ne!(available.buffer, pending_buffer);
    let available_buffer = available.buffer;
    available.used = PARAMS_RANGE;
    renderer.recycle_params_ring(0, available);
    let reused = renderer.acquire_params_ring().unwrap();
    assert_eq!(reused.buffer, available_buffer);
    assert_eq!(reused.used, 0);
    reused.defer_destroy(&renderer.device, 0);

    signal_unused_point(&mut renderer, 1);
    let completed = renderer.acquire_params_ring().unwrap();
    assert_eq!(completed.buffer, pending_buffer);
    assert_eq!(completed.ds, pending_set);
    assert_eq!(completed.used, 0);
    renderer.recycle_params_ring(1, completed);
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[test]
fn actual_vulkan_draws_and_readback_publish_their_own_wayland_reads() {
    use crate::backend::renderer::{
        ImportMem,
        utils::buffer_read::{take_published, tests::buffer},
    };
    let Some(mut renderer) = renderer() else { return };
    let (_display, _socket, buffer) = buffer();
    let source = renderer
        .import_memory(&[255, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
        .unwrap();
    let mut destination: VulkanTexture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    take_published();
    {
        let mut target = renderer.bind(&mut destination).unwrap();
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        let rect = Rectangle::from_size((1, 1).into());
        buffer
            .with_read_source(|| {
                frame.render_texture_from_to(
                    &source,
                    Rectangle::from_size((1., 1.).into()),
                    rect,
                    &[],
                    &[],
                    Transform::Normal,
                    1.,
                )
            })
            .unwrap();
        assert_eq!(frame.read_buffers.len(), 0);
        for _ in 0..4 {
            buffer
                .with_read_source(|| {
                    frame.render_texture_from_to(
                        &source,
                        Rectangle::from_size((1., 1.).into()),
                        rect,
                        &[rect],
                        &[],
                        Transform::Normal,
                        1.,
                    )
                })
                .unwrap();
        }
        assert_eq!(frame.read_buffers.len(), 1);
        drop(frame);
    }
    assert_eq!(pixels(&mut renderer, &destination, (1, 1)), [255, 0, 0, 255]);
    assert_eq!(
        take_published().len(),
        1,
        "offscreen/drop publishes without an output submission"
    );
    let mapping = buffer
        .with_read_source(|| {
            renderer.copy_texture(&source, Rectangle::from_size((1, 1).into()), Fourcc::Abgr8888)
        })
        .unwrap();
    assert_eq!(
        take_published().len(),
        1,
        "readback publishes before exposing mapped bytes"
    );
    assert_eq!(renderer.map_texture(&mapping).unwrap(), [255, 0, 0, 255]);
}

#[test]
fn parameter_ring_cache_is_bounded_and_released_with_renderer() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let count = MAX_CACHED_PARAMS_RINGS + 3;
    let rings = (0..count)
        .map(|_| renderer.acquire_params_ring().unwrap())
        .collect::<Vec<_>>();
    for ring in rings {
        renderer.recycle_params_ring(1, ring);
    }
    assert_eq!(renderer.params_rings.len(), MAX_CACHED_PARAMS_RINGS);

    renderer.cleanup();
    assert_eq!(
        renderer.device.descriptor_pools.lock().unwrap()[0].1,
        DESCRIPTOR_POOL_SIZE - count as u32,
        "overflow blocks must not be destroyed before their timeline point"
    );
    signal_unused_point(&mut renderer, 1);
    renderer.cleanup();
    assert_eq!(
        renderer.device.descriptor_pools.lock().unwrap()[0].1,
        DESCRIPTOR_POOL_SIZE - MAX_CACHED_PARAMS_RINGS as u32
    );

    let device = renderer.device.clone();
    drop(renderer);
    assert!(device.cleanup.lock().unwrap().is_empty());
    assert_eq!(device.descriptor_pools.lock().unwrap()[0].1, DESCRIPTOR_POOL_SIZE);
}

#[test]
fn parameter_uniforms_change_across_frames_without_reallocation() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let program = color_program(&mut renderer);
    let mut texture = renderer.create_buffer(Fourcc::Abgr8888, (2, 2).into()).unwrap();
    let mut first_allocation = None;
    let colors = [
        ([1., 0., 0., 1.], [255, 0, 0, 255]),
        ([0., 1., 0., 1.], [0, 255, 0, 255]),
        ([0., 0., 1., 1.], [0, 0, 255, 255]),
    ];

    for (color, pixel) in colors.into_iter().cycle().take(24) {
        let sync = {
            let mut target = renderer.bind(&mut texture).unwrap();
            let mut frame = renderer
                .render(&mut target, (2, 2).into(), Transform::Normal)
                .unwrap();
            let rect = Rectangle::from_size((2, 2).into());
            frame
                .render_custom(
                    &program,
                    rect,
                    &[rect],
                    &[CustomUniform {
                        name: "color",
                        value: CustomUniformValue::Vec4(color),
                    }],
                    &[],
                    1.,
                )
                .unwrap();
            frame.finish().unwrap()
        };
        sync.wait().unwrap();
        assert_eq!(renderer.params_rings.len(), 1);
        let allocation = (renderer.params_rings[0].1.buffer, renderer.params_rings[0].1.ds);
        assert_eq!(*first_allocation.get_or_insert(allocation), allocation);
        for actual in pixels(&mut renderer, &texture, (2, 2)).chunks_exact(4) {
            assert_eq!(
                actual, pixel,
                "a recycled ring must contain this frame's uniforms"
            );
        }
    }
}

#[test]
fn parameter_ring_rollover_preserves_earlier_draws_in_the_same_frame() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let program = color_program(&mut renderer);
    let mut texture = renderer.create_buffer(Fourcc::Abgr8888, (2, 1).into()).unwrap();
    let sync = {
        let mut target = renderer.bind(&mut texture).unwrap();
        let mut frame = renderer
            .render(&mut target, (2, 1).into(), Transform::Normal)
            .unwrap();
        let pixel = Rectangle::from_size((1, 1).into());
        frame
            .render_custom(
                &program,
                pixel,
                &[pixel],
                &[CustomUniform {
                    name: "color",
                    value: CustomUniformValue::Vec4([1., 0., 0., 1.]),
                }],
                &[],
                1.,
            )
            .unwrap();
        let first_buffer = frame.params_ring.as_ref().unwrap().buffer;
        let layout = frame.renderer.pipeline_layouts[0];
        while frame.params_ring.as_ref().unwrap().buffer == first_buffer {
            frame
                .bind_params_raw(&[0; PARAMS_RANGE as usize], layout)
                .unwrap();
        }
        assert_eq!(frame.retired_params_rings.len(), 1);
        frame
            .render_custom(
                &program,
                Rectangle::new((1, 0).into(), (1, 1).into()),
                &[pixel],
                &[CustomUniform {
                    name: "color",
                    value: CustomUniformValue::Vec4([0., 1., 0., 1.]),
                }],
                &[],
                1.,
            )
            .unwrap();
        frame.finish().unwrap()
    };
    sync.wait().unwrap();
    assert_eq!(renderer.params_rings.len(), 2);
    assert_eq!(
        pixels(&mut renderer, &texture, (2, 1)),
        [255, 0, 0, 255, 0, 255, 0, 255]
    );
}

#[test]
fn unfinished_frame_drop_submits_and_recycles_parameter_storage() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let program = color_program(&mut renderer);
    let mut texture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    {
        let mut target = renderer.bind(&mut texture).unwrap();
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        let rect = Rectangle::from_size((1, 1).into());
        frame
            .render_custom(
                &program,
                rect,
                &[rect],
                &[CustomUniform {
                    name: "color",
                    value: CustomUniformValue::Vec4([1., 0., 0., 1.]),
                }],
                &[],
                1.,
            )
            .unwrap();
        // Drop must submit the commands, preserving the ring until the GPU has read it.
    }
    assert_eq!(renderer.params_rings.len(), 1);
    assert_eq!(renderer.params_rings[0].0, renderer.timeline_point);
    assert_eq!(pixels(&mut renderer, &texture, (1, 1)), [255, 0, 0, 255]);
}

#[test]
fn failed_frame_leftovers_are_retired_instead_of_recycled() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut texture = renderer.create_buffer(Fourcc::Abgr8888, (1, 1).into()).unwrap();
    let buffer;
    {
        let mut target = renderer.bind(&mut texture).unwrap();
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        let layout = frame.renderer.pipeline_layouts[0];
        frame.bind_params_raw(&[0; 16], layout).unwrap();
        buffer = frame.params_ring.as_ref().unwrap().buffer;

        // Reproduce the ownership state after finish_internal fails before queue submission:
        // finalization has started, but the unsubmitted frame still owns its ring. Do not
        // inject a driver error or submit invalid commands just to exercise this drop path.
        unsafe {
            frame.renderer.device.raw.cmd_end_rendering(frame.cb);
            frame.renderer.device.raw.end_command_buffer(frame.cb).unwrap();
        }
        frame.finished.store(true, Ordering::SeqCst);
    }
    assert!(renderer.params_rings.is_empty());
    assert!(
        renderer
            .device
            .cleanup
            .lock()
            .unwrap()
            .iter()
            .any(|(_, item)| { matches!(item, CleanupItem::Buffer(retired) if *retired == buffer) })
    );
    renderer.cleanup();
    assert_eq!(
        renderer.device.descriptor_pools.lock().unwrap()[0].1,
        DESCRIPTOR_POOL_SIZE
    );
}
