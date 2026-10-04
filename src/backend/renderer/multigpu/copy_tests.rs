//! Exercise the actual CPU copy path with renderer-owned mappings and textures.
use super::*;
use crate::backend::egl::{EGLContext, EGLDisplay, native::EGLSurfacelessDisplay};
use crate::backend::renderer::gles::GlesRenderer;
use std::{convert::Infallible, marker::PhantomData};

#[derive(Debug)]
struct TestDevice<R: Renderer + fmt::Debug>(R);
struct TestApi<R: Renderer + fmt::Debug>(PhantomData<R>);

impl<R: Renderer + fmt::Debug> ApiDevice for TestDevice<R> {
    type Renderer = R;
    fn renderer(&self) -> &R {
        &self.0
    }
    fn renderer_mut(&mut self) -> &mut R {
        &mut self.0
    }
    fn allocator(&mut self) -> &mut dyn Allocator<Buffer = Dmabuf, Error = AnyError> {
        panic!("CPU copies must not allocate DMA buffers")
    }
    fn node(&self) -> &DrmNode {
        panic!("CPU copies do not inspect DRM nodes")
    }
    fn can_do_cross_device_imports(&self) -> bool {
        false
    }
}

impl<R: Renderer + fmt::Debug> GraphicsApi for TestApi<R> {
    type Device = TestDevice<R>;
    type Error = Infallible;
    fn enumerate(&self, _: &mut Vec<Self::Device>) -> Result<(), Self::Error> {
        Ok(())
    }
    fn identifier() -> &'static str {
        "cpu-copy-test"
    }
}

fn gles() -> Option<GlesRenderer> {
    let result = (|| -> Result<_, String> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.map_err(|err| err.to_string())?;
        let context = EGLContext::new(&display).map_err(|err| err.to_string())?;
        unsafe { GlesRenderer::new(context) }.map_err(|err| err.to_string())
    })();
    required_renderer(result, "GLES")
}

fn required_renderer<R>(result: Result<R, String>, api: &str) -> Option<R> {
    match result {
        Ok(renderer) => Some(renderer),
        Err(err) => {
            if std::env::var_os(format!("SMITHAY_TEST_REQUIRE_{api}"))
                .is_some_and(|v| !v.is_empty() && v != "0")
            {
                panic!("CPU copy test requires {api}: {err}");
            }
            tracing::warn!("skipping {api} CPU copy test: {err}");
            None
        }
    }
}

#[cfg(feature = "renderer_vulkan")]
fn vulkan() -> Option<crate::backend::renderer::vulkan::VulkanRenderer> {
    use crate::backend::renderer::vulkan::VulkanRenderer;
    use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};
    let result = (|| -> Result<_, String> {
        let instance = Instance::new(Version::VERSION_1_3, None).map_err(|err| err.to_string())?;
        let devices = PhysicalDevice::enumerate(&instance).map_err(|err| err.to_string())?;
        for device in devices {
            if let Ok(renderer) = VulkanRenderer::new(&device) {
                return Ok(renderer);
            }
        }
        Err("No compatible Vulkan renderer".into())
    })();
    required_renderer(result, "VULKAN")
}

fn cpu_copy_roundtrip<R, T>(source: R, target: T)
where
    R: Renderer + fmt::Debug + ExportMem + ImportMem + Bind<Dmabuf>,
    T: Renderer + fmt::Debug + ExportMem + ImportMem,
    R::Error: 'static,
    T::Error: 'static,
{
    let mut source = TestDevice(source);
    let mut target = TestDevice(target);
    let size = (6, 5).into();
    let full = Rectangle::from_size(size);
    let format = Fourcc::Abgr8888;
    let mut expected: Vec<u8> = (0..30).flat_map(|index| [index as u8, 80, 120, 255]).collect();
    let src_texture = source.0.import_memory(&expected, format, size, false).unwrap();
    let mut slot = None;
    mem_copy::<TestApi<R>, TestApi<T>>(&src_texture, None, &mut slot, &mut source, &mut target).unwrap();
    let regions = [
        Rectangle::new((1, 1).into(), (2, 2).into()),
        Rectangle::new((4, 3).into(), (1, 1).into()),
    ];
    for region in regions {
        for y in region.loc.y..region.loc.y + region.size.h {
            for x in region.loc.x..region.loc.x + region.size.w {
                let offset = ((y * 6 + x) * 4) as usize;
                expected[offset..offset + 4].copy_from_slice(&[200, x as u8, y as u8, 255]);
            }
        }
        source.0.update_memory(&src_texture, &expected, region).unwrap();
    }
    mem_copy::<TestApi<R>, TestApi<T>>(&src_texture, Some(&regions), &mut slot, &mut source, &mut target)
        .unwrap();
    {
        let texture = slot.as_ref().unwrap().1.as_ref().unwrap();
        let mapping = target.0.copy_texture(texture, full, format).unwrap();
        assert_eq!(target.0.map_texture(&mapping).unwrap(), expected);
    }
    // Early mappings and a newly read disjoint region must compose in the same
    // full-image staging buffer without corrupting pixels outside either one.
    expected[(7 * 4)..(8 * 4)].copy_from_slice(&[1, 2, 3, 255]);
    source
        .0
        .update_memory(&src_texture, &expected, regions[0])
        .unwrap();
    let early = source.0.copy_texture(&src_texture, regions[0], format).unwrap();
    slot.as_mut().unwrap().0 = Some(vec![(Box::new(early), regions[0])]);
    expected[(22 * 4)..(23 * 4)].copy_from_slice(&[4, 5, 6, 255]);
    source
        .0
        .update_memory(&src_texture, &expected, regions[1])
        .unwrap();
    mem_copy::<TestApi<R>, TestApi<T>>(&src_texture, Some(&regions), &mut slot, &mut source, &mut target)
        .unwrap();
    mem_copy::<TestApi<R>, TestApi<T>>(&src_texture, Some(&[]), &mut slot, &mut source, &mut target).unwrap();
    let texture = slot.as_ref().unwrap().1.as_ref().unwrap();
    let mapping = target.0.copy_texture(texture, full, format).unwrap();
    assert_eq!(target.0.map_texture(&mapping).unwrap(), expected);
}

#[test]
fn gles_cpu_copy_handles_disjoint_regions_and_early_mappings() {
    let (Some(source), Some(target)) = (gles(), gles()) else {
        return;
    };
    cpu_copy_roundtrip(source, target);
}

#[cfg(feature = "renderer_vulkan")]
#[test]
fn cpu_copy_between_gles_and_vulkan_preserves_pixels() {
    let (Some(source), Some(target)) = (gles(), vulkan()) else {
        return;
    };
    cpu_copy_roundtrip(source, target);
    let (Some(source), Some(target)) = (vulkan(), gles()) else {
        return;
    };
    cpu_copy_roundtrip(source, target);
}

#[test]
fn opaque_partial_shadow_draw_matches_clear_then_draw() {
    use crate::backend::egl::{EGLContext, EGLDisplay, native::EGLSurfacelessDisplay};
    use crate::backend::renderer::{ExportMem, ImportMem, gles::GlesRenderer};
    let renderer = (|| -> Result<_, String> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.map_err(|err| err.to_string())?;
        let context = EGLContext::new(&display).map_err(|err| err.to_string())?;
        unsafe { GlesRenderer::new(context) }.map_err(|err| err.to_string())
    })();
    let mut renderer = match renderer {
        Ok(renderer) => renderer,
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_GLES").is_some_and(|v| !v.is_empty() && v != "0") {
                panic!("opaque shadow test requires GLES: {err}");
            }
            tracing::warn!("skipping opaque shadow test: {err}");
            return;
        }
    };
    let size = (6, 5).into();
    let data: Vec<_> = (0..30)
        .flat_map(|index| [(index * 7) as u8, 128, 50, 0])
        .collect();
    let source = renderer
        .import_memory(&data, Fourcc::Xbgr8888, size, false)
        .unwrap();
    let initial = [40u8, 80, 100, 255].repeat(30);
    let mut outputs = Vec::new();
    let damage = [
        Rectangle::new((1, 1).into(), (2, 2).into()),
        Rectangle::new((4, 3).into(), (1, 1).into()),
    ];
    for clear in [true, false] {
        let mut destination = renderer
            .import_memory(&initial, Fourcc::Abgr8888, size, false)
            .unwrap();
        {
            let mut target = renderer.bind(&mut destination).unwrap();
            let mut frame = renderer
                .render(&mut target, (6, 5).into(), Transform::Normal)
                .unwrap();
            if clear {
                frame.clear(Color32F::TRANSPARENT, &damage).unwrap();
            }
            Frame::render_texture_from_to(
                &mut frame,
                &source,
                Rectangle::from_size(size).to_f64(),
                Rectangle::from_size((6, 5).into()),
                &damage,
                &[],
                Transform::Normal,
                1.,
            )
            .unwrap();
            frame.finish().unwrap().wait().unwrap();
        }
        let mapping = renderer
            .copy_texture(&destination, Rectangle::from_size(size), Fourcc::Abgr8888)
            .unwrap();
        outputs.push(renderer.map_texture(&mapping).unwrap().to_vec());
    }
    assert_eq!(
        outputs[0], outputs[1],
        "opaque partial draws must fully replace damaged pixels"
    );
    assert_ne!(outputs[0], initial, "the test must draw actual pixels");
    for (index, pixel) in outputs[0].chunks_exact(4).enumerate() {
        let point = ((index % 6) as i32, (index / 6) as i32);
        let flipped_point = ((index % 6) as i32, 4 - (index / 6) as i32);
        if !damage
            .iter()
            .any(|rect| rect.contains(point) || rect.contains(flipped_point))
        {
            assert_eq!(pixel, &initial[index * 4..index * 4 + 4]);
        }
    }
}

#[cfg(feature = "backend_gbm")]
#[test]
fn full_dma_shadow_updates_rotate_and_partial_updates_preserve_the_current_slot() {
    use super::gbm::GbmGlesBackend;
    use crate::backend::allocator::gbm::GbmDevice;
    type Api = GbmGlesBackend<GlesRenderer, Arc<std::fs::File>>;

    let mut devices = None;
    let mut errors = Vec::new();
    for entry in std::fs::read_dir("/dev/dri").into_iter().flatten().flatten() {
        if !entry.file_name().to_string_lossy().starts_with("renderD") {
            continue;
        }
        let result = (|| -> Result<_, String> {
            let file = Arc::new(
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(entry.path())
                    .map_err(|e| e.to_string())?,
            );
            let node = DrmNode::from_path(entry.path()).map_err(|e| e.to_string())?;
            let gbm = GbmDevice::new(file).map_err(|e| e.to_string())?;
            let mut first = Api::default();
            let mut second = Api::default();
            first.add_node(node, gbm.clone()).map_err(|e| e.to_string())?;
            second.add_node(node, gbm).map_err(|e| e.to_string())?;
            let mut source = Vec::new();
            let mut target = Vec::new();
            first.enumerate(&mut source).map_err(|e| e.to_string())?;
            second.enumerate(&mut target).map_err(|e| e.to_string())?;
            let (Some(source), Some(target)) = (source.pop(), target.pop()) else {
                return Err("no renderers".into());
            };
            if !source.should_do_cross_device_exports() || !target.can_do_cross_device_imports() {
                return Err("DMA import/export disabled by renderer quirks".into());
            }
            Ok((first, second, source, target))
        })();
        match result {
            Ok(pair) => {
                devices = Some(pair);
                break;
            }
            Err(err) => errors.push(err),
        }
    }
    let Some((_source_api, _target_api, mut source, mut target)) = devices else {
        if std::env::var_os("SMITHAY_TEST_REQUIRE_DMA").is_some_and(|v| !v.is_empty() && v != "0") {
            panic!(
                "DMA shadow test requires GBM/EGL render devices: {}",
                errors.join("; ")
            );
        }
        tracing::warn!("skipping DMA shadow test: {}", errors.join("; "));
        return;
    };
    let size = (8, 8).into();
    let full = Rectangle::from_size(size);
    let partial = Rectangle::new((2, 3).into(), (3, 2).into());
    let initial = [20, 40, 60, 255].repeat(64);
    let src_texture = source
        .renderer_mut()
        .import_memory(&initial, Fourcc::Abgr8888, size, false)
        .unwrap();
    let mut slot = None;
    #[derive(Debug)]
    struct ConsumerFence(Arc<std::sync::atomic::AtomicUsize>);
    impl sync::Fence for ConsumerFence {
        fn is_signaled(&self) -> bool {
            self.0.load(std::sync::atomic::Ordering::Relaxed) > 0
        }
        fn wait(&self) -> Result<(), sync::Interrupted> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        fn is_exportable(&self) -> bool {
            false
        }
        fn export(&self) -> Option<std::os::fd::OwnedFd> {
            None
        }
    }
    let consumer_waits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut first_buffer = None;
    let mut second_buffer = None;
    for iteration in 0..5 {
        let mut expected = [20 + iteration as u8, 40, 60, 255].repeat(64);
        let damage = if iteration == 2 {
            expected = [21, 40, 60, 255].repeat(64);
            for y in 3..5 {
                for x in 2..5 {
                    expected[(y * 8 + x) * 4..(y * 8 + x) * 4 + 4].copy_from_slice(&[200, 100, 50, 255]);
                }
            }
            partial
        } else {
            full
        };
        source
            .renderer_mut()
            .update_memory(&src_texture, &expected, damage)
            .unwrap();
        texture_copy::<Api, Api>(&mut source, &mut target, &src_texture, &mut slot, Some(&[damage])).unwrap();
        let Some(GpuSingleTexture::Dma {
            texture,
            dmabuf,
            sync,
            spare,
            ..
        }) = slot.as_mut()
        else {
            panic!("expected DMA copy, got {slot:?}");
        };
        if let Some(producer) = sync.take() {
            target.renderer_mut().wait(&producer).unwrap();
        }
        let copied = texture
            .downcast_ref::<crate::backend::renderer::gles::GlesTexture>()
            .unwrap();
        let mapping = target
            .renderer_mut()
            .copy_texture(copied, full, Fourcc::Abgr8888)
            .unwrap();
        assert_eq!(target.renderer_mut().map_texture(&mapping).unwrap(), expected);
        // map_texture blocks until this read completes, so the slot is now safe
        // for its next producer; per-slot unresolved waits are covered separately.
        *sync = Some(SyncPoint::signaled());
        match iteration {
            0 => {
                first_buffer = Some(dmabuf.clone());
                assert!(spare.is_none());
            }
            1 => {
                second_buffer = Some(dmabuf.clone());
                assert_ne!(Some(&*dmabuf), first_buffer.as_ref());
                spare.as_mut().unwrap().2 = Some(ConsumerFence(consumer_waits.clone()).into());
            }
            2 => {
                assert_eq!(
                    Some(&*dmabuf),
                    second_buffer.as_ref(),
                    "partial update must remain in the current slot"
                );
                assert_eq!(consumer_waits.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
            3 => {
                assert_eq!(
                    Some(&*dmabuf),
                    first_buffer.as_ref(),
                    "full update should reuse the first slot"
                );
                assert_eq!(consumer_waits.load(std::sync::atomic::Ordering::Relaxed), 1);
            }
            4 => assert_eq!(Some(&*dmabuf), second_buffer.as_ref()),
            _ => unreachable!(),
        }
    }
}
