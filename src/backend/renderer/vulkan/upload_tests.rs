//! Upload batching and staging lifetime tests on a hardware or software Vulkan device.

use super::*;

fn renderer() -> Option<VulkanRenderer> {
    static LOGGING: std::sync::Once = std::sync::Once::new();
    LOGGING.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_test_writer()
            .try_init();
    });
    super::descriptor_pool_tests::renderer()
}

#[test]
fn damaged_rows_share_one_submission_and_preserve_stride() {
    let Some(mut renderer) = renderer() else { return };
    let size = Size::from((16, 8));
    let initial = [0, 0, 0, 255].repeat(16 * 8);
    let texture = renderer
        .import_memory(&initial, Fourcc::Abgr8888, size, false)
        .unwrap();
    let mut data = vec![0xDD; 20 * 8 * 4];
    for y in 0..8 {
        data[y * 80..y * 80 + 64].copy_from_slice(&initial[y * 64..(y + 1) * 64]);
    }
    let damage = [
        Rectangle::new((-1, 1).into(), (4, 2).into()),
        Rectangle::new((1, 2).into(), (3, 2).into()),
        Rectangle::new((13, 6).into(), (5, 3).into()),
    ];
    let mut expected = initial;
    for rect in damage {
        let rect = rect.intersection(Rectangle::from_size(size)).unwrap();
        for y in rect.loc.y..rect.loc.y + rect.size.h {
            for x in rect.loc.x..rect.loc.x + rect.size.w {
                let pixel = [x as u8 * 10, y as u8 * 20, 55, 255];
                let source = (y as usize * 20 + x as usize) * 4;
                data[source..source + 4].copy_from_slice(&pixel);
                let dest = (y as usize * 16 + x as usize) * 4;
                expected[dest..dest + 4].copy_from_slice(&pixel);
            }
        }
    }
    let before = renderer.timeline_point;
    renderer
        .upload_memory(&texture, &data, 20, &damage, false)
        .unwrap();
    assert_eq!(renderer.timeline_point, before + 1);
    let mapping = renderer
        .copy_texture(&texture, Rectangle::from_size(size), Fourcc::Abgr8888)
        .unwrap();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), expected);

    let before = renderer.timeline_point;
    let outside = Rectangle::new((20, 20).into(), (2, 2).into());
    renderer
        .upload_memory(&texture, &data, 20, &[outside], false)
        .unwrap();
    assert_eq!(renderer.timeline_point, before);
}

#[test]
fn staging_reuse_waits_for_completion_and_handles_growth() {
    let Some(mut renderer) = renderer() else { return };
    let pending = renderer.acquire_upload_buffer(64).unwrap();
    let pending_handle = pending.buffer;
    renderer.recycle_upload_buffer(1, pending);
    let available = renderer.acquire_upload_buffer(64).unwrap();
    assert_ne!(available.buffer, pending_handle);
    let available_handle = available.buffer;
    renderer.recycle_upload_buffer(0, available);
    let reused = renderer.acquire_upload_buffer(32).unwrap();
    assert_eq!(reused.buffer, available_handle);
    renderer.recycle_upload_buffer(0, reused);

    let larger = renderer.acquire_upload_buffer(128).unwrap();
    assert_eq!(larger.capacity, 128);
    assert_eq!(
        renderer.upload_buffers.len(),
        1,
        "discard only the completed smaller buffer"
    );
    renderer.recycle_upload_buffer(0, larger);

    let signal = vk::SemaphoreSignalInfo::default()
        .semaphore(renderer.device.timeline)
        .value(1);
    unsafe { renderer.device.raw.signal_semaphore(&signal) }.unwrap();
    renderer.timeline_point = 1;
    let completed = renderer.acquire_upload_buffer(64).unwrap();
    assert_eq!(
        completed.buffer, pending_handle,
        "prefer the smallest completed allocation"
    );
    renderer.recycle_upload_buffer(1, completed);
}

#[test]
fn staging_cache_is_bounded_and_released_with_renderer() {
    let Some(mut renderer) = renderer() else { return };
    let buffers = (0..MAX_CACHED_UPLOAD_BUFFERS + 2)
        .map(|_| renderer.acquire_upload_buffer(64).unwrap())
        .collect::<Vec<_>>();
    for buffer in buffers {
        renderer.recycle_upload_buffer(1, buffer);
    }
    assert_eq!(renderer.upload_buffers.len(), MAX_CACHED_UPLOAD_BUFFERS);
    renderer.cleanup();
    assert!(!renderer.device.cleanup.lock().unwrap().is_empty());
    let device = renderer.device.clone();
    drop(renderer);
    assert!(device.cleanup.lock().unwrap().is_empty());
}

#[test]
fn staging_cache_limits_retained_bytes() {
    let Some(mut renderer) = renderer() else { return };
    let capacity = MAX_CACHED_UPLOAD_BYTES / 3 + 4;
    let buffers = (0..3)
        .map(|_| renderer.acquire_upload_buffer(capacity).unwrap())
        .collect::<Vec<_>>();
    for buffer in buffers {
        renderer.recycle_upload_buffer(1, buffer);
    }
    assert_eq!(renderer.upload_buffers.len(), 2);
    assert!(
        renderer
            .upload_buffers
            .iter()
            .map(|(_, buffer)| buffer.capacity)
            .sum::<u64>()
            <= MAX_CACHED_UPLOAD_BYTES
    );
}

#[test]
fn short_upload_data_is_rejected_before_submission() {
    let Some(mut renderer) = renderer() else { return };
    let texture = renderer
        .import_memory(&[0; 16], Fourcc::Abgr8888, (2, 2).into(), false)
        .unwrap();
    let before = renderer.timeline_point;
    let full = Rectangle::from_size((2, 2).into());
    assert!(matches!(
        renderer.upload_memory(&texture, &[0; 12], 2, &[full], false),
        Err(VulkanError::UnexpectedSize)
    ));
    assert_eq!(renderer.timeline_point, before);
}
