//! Real Vulkan descriptor allocation tests; no display or DRM scanout is required.
//!
//! Set `SMITHAY_TEST_REQUIRE_VULKAN=1` to fail instead of skipping when no compatible
//! hardware or software Vulkan renderer is available.

use super::*;
use crate::backend::vulkan::Instance;

pub(super) fn renderer() -> Option<VulkanRenderer> {
    let result = (|| {
        let instance = Instance::new(Version::VERSION_1_3, None).map_err(|err| err.to_string())?;
        let devices = PhysicalDevice::enumerate(&instance).map_err(|err| err.to_string())?;
        let mut errors = Vec::new();
        for device in devices {
            match VulkanRenderer::new(&device) {
                Ok(renderer) => return Ok(renderer),
                Err(err) => errors.push(format!("{}: {err}", device.name())),
            }
        }
        Err(format!("no compatible Vulkan renderer: {}", errors.join("; ")))
    })();

    match result {
        Ok(renderer) => Some(renderer),
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_VULKAN")
                .is_some_and(|value| !value.is_empty() && value != "0")
            {
                panic!("{err}");
            }
            eprintln!("skipping Vulkan descriptor pool test: {err}");
            None
        }
    }
}

#[test]
fn descriptor_pools_reuse_completed_sets_with_a_live_texture() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let texture = renderer.create_buffer(Fourcc::Argb8888, (1, 1).into()).unwrap();
    let cached_set = renderer.texture_descriptor_set(&texture).unwrap();

    // Keep a texture set alive while retiring thousands of transient sets from the same pool.
    // This rules out resetting the pool wholesale when temporary sets finish.
    let pool = renderer.device.descriptor_pools.lock().unwrap()[0].0;
    for point in 1..=64 {
        let mut pending = Vec::new();
        for _ in 0..32 {
            for layout in [renderer.params_ds_layout, renderer.texture_ds_layouts[2]] {
                let (allocated_pool, set) = renderer.allocate_descriptor_set(layout).unwrap();
                assert_eq!(allocated_pool, pool, "freed pool capacity must be reused");
                pending.push(CleanupItem::DescriptorSet(allocated_pool, set));
            }
        }
        renderer.device.defer_destroy(point, pending);

        // No commands use these sets; host-signalling the real timeline below gives a
        // deterministic completion boundary without blocking a GPU queue in this test.
        renderer.cleanup();
        {
            let pools = renderer.device.descriptor_pools.lock().unwrap();
            assert_eq!(pools.len(), 1);
            assert_eq!(
                pools[0].1,
                DESCRIPTOR_POOL_SIZE - 65,
                "in-flight sets must stay reserved"
            );
        }
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(renderer.device.timeline)
            .value(point);
        unsafe { renderer.device.raw.signal_semaphore(&signal) }.unwrap();
        renderer.timeline_point = point;
        renderer.cleanup();
        {
            let pools = renderer.device.descriptor_pools.lock().unwrap();
            assert_eq!(pools.len(), 1, "pool count must remain bounded across frames");
            assert_eq!(pools[0].1, DESCRIPTOR_POOL_SIZE - 1);
        }
        assert_eq!(renderer.texture_descriptor_set(&texture).unwrap(), cached_set);
    }

    drop(texture);
    renderer.cleanup();
    let pools = renderer.device.descriptor_pools.lock().unwrap();
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].1, DESCRIPTOR_POOL_SIZE);
}

#[test]
fn descriptor_pools_outlive_renderer_until_external_texture_is_dropped() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let texture = renderer.create_buffer(Fourcc::Argb8888, (1, 1).into()).unwrap();
    renderer.texture_descriptor_set(&texture).unwrap();
    let device = renderer.device.clone();
    let weak_device = Arc::downgrade(&device);

    drop(renderer);
    {
        let pools = device.descriptor_pools.lock().unwrap();
        assert_eq!(pools.len(), 1, "an external texture still owns a descriptor set");
        assert_eq!(pools[0].1, DESCRIPTOR_POOL_SIZE - 1);
    }
    drop(texture);
    device.process_cleanup(device.completed_point().unwrap());
    assert_eq!(device.descriptor_pools.lock().unwrap()[0].1, DESCRIPTOR_POOL_SIZE);
    drop(device);
    assert!(weak_device.upgrade().is_none());
}

#[test]
fn last_texture_drop_cleans_up_descriptor_pool_without_renderer() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let texture = renderer.create_buffer(Fourcc::Argb8888, (1, 1).into()).unwrap();
    renderer.texture_descriptor_set(&texture).unwrap();
    let device = Arc::downgrade(&renderer.device);

    drop(renderer);
    assert!(device.upgrade().is_some());
    // No renderer or external Device owner remains to run cleanup for this descriptor set.
    // Dropping the final texture must free the set before destroying its pool and device.
    drop(texture);
    assert!(device.upgrade().is_none());
}
