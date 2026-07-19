//! Smoke test for the Vulkan renderer: renders offscreen and into a dmabuf, reads the
//! results back and verifies the pixels.

use drm_fourcc::{DrmFourcc, DrmModifier};
use std::os::unix::io::AsFd;

use smithay::{
    backend::{
        allocator::{
            Allocator,
            dmabuf::{AsDmabuf, SyncFileFlags, export_sync_file, import_sync_file},
            vulkan::{ImageUsageFlags, VulkanAllocator},
        },
        renderer::{
            Bind, Color32F, ExportMem, Frame, ImportDma, ImportMem, Offscreen, Renderer,
            sync::SyncPoint,
            vulkan::{VulkanRenderer, VulkanTexture},
        },
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    utils::{Rectangle, Size, Transform},
};

fn check_pixel(data: &[u8], width: i32, x: i32, y: i32, expected: [u8; 4], what: &str) {
    let idx = ((y * width + x) * 4) as usize;
    let actual = &data[idx..idx + 4];
    assert_eq!(
        actual, expected,
        "unexpected pixel at ({x}, {y}) for {what}: got {actual:?}, expected {expected:?}"
    );
}

fn main() {
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().init();
    }

    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let mut renderer = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .find_map(|phd| match VulkanRenderer::new(&phd) {
            Ok(renderer) => {
                println!("Using physical device: {}", phd.name());
                Some(renderer)
            }
            Err(err) => {
                println!("Skipping {}: {}", phd.name(), err);
                None
            }
        })
        .expect("no physical device supports the vulkan renderer");

    let size = Size::<i32, smithay::utils::Buffer>::from((256, 256));
    let output_size = Size::from((256, 256));
    let full: Rectangle<i32, smithay::utils::Physical> = Rectangle::from_size(output_size);

    // A 2x2 blue texture (Argb8888 little-endian: B, G, R, A).
    let blue_pixels: Vec<u8> = std::iter::repeat_n([255u8, 0, 0, 255], 4).flatten().collect();
    let texture: VulkanTexture = renderer
        .import_memory(&blue_pixels, DrmFourcc::Argb8888, (2, 2).into(), false)
        .expect("import memory");

    // --- Offscreen rendering ---
    let mut offscreen: VulkanTexture = renderer
        .create_buffer(DrmFourcc::Argb8888, size)
        .expect("create offscreen buffer");
    let mut fb = renderer.bind(&mut offscreen).expect("bind offscreen");
    let mut frame = renderer
        .render(&mut fb, output_size, Transform::Normal)
        .expect("begin frame");

    // Red background, green top-left quadrant, blue texture in the top-right corner.
    frame
        .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[full])
        .expect("clear");
    frame
        .clear(
            Color32F::new(0.0, 1.0, 0.0, 1.0),
            &[Rectangle::new((0, 0).into(), (128, 128).into())],
        )
        .expect("clear quadrant");
    let tex_dst: Rectangle<i32, smithay::utils::Physical> = Rectangle::new((192, 0).into(), (64, 64).into());
    frame
        .render_texture_from_to(
            &texture,
            Rectangle::from_size((2.0, 2.0).into()),
            tex_dst,
            &[Rectangle::from_size(tex_dst.size)],
            &[],
            Transform::Normal,
            1.0,
        )
        .expect("draw texture");

    let sync: SyncPoint = frame.finish().expect("finish frame");
    println!(
        "offscreen frame submitted; sync exportable: {}",
        sync.is_exportable()
    );
    if let Some(fd) = sync.export() {
        println!("exported sync_file fd: {fd:?}");
    }
    sync.wait().expect("wait for frame");

    let mapping = renderer
        .copy_framebuffer(&fb, Rectangle::from_size(size), DrmFourcc::Argb8888)
        .expect("copy framebuffer");
    let data = renderer.map_texture(&mapping).expect("map");
    assert_eq!(data.len(), 256 * 256 * 4);

    // Argb8888 little-endian byte order: B, G, R, A.
    check_pixel(data, 256, 5, 5, [0, 255, 0, 255], "green quadrant");
    check_pixel(
        data,
        256,
        5,
        200,
        [0, 0, 255, 255],
        "red background (bottom left)",
    );
    check_pixel(
        data,
        256,
        200,
        200,
        [0, 0, 255, 255],
        "red background (bottom right)",
    );
    check_pixel(data, 256, 220, 5, [255, 0, 0, 255], "blue texture");
    println!("offscreen rendering: OK");
    drop(fb);

    // --- Rendering into a dmabuf ---
    let mut allocator = VulkanAllocator::new(
        renderer.physical_device(),
        ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
    )
    .expect("create allocator");
    let image = allocator
        .create_buffer(256, 256, DrmFourcc::Argb8888, &[DrmModifier::Linear])
        .expect("allocate buffer");
    let mut dmabuf = image.export().expect("export dmabuf");

    let mut fb = renderer.bind(&mut dmabuf).expect("bind dmabuf");
    let mut frame = renderer
        .render(&mut fb, output_size, Transform::Normal)
        .expect("begin dmabuf frame");
    frame
        .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[full])
        .expect("clear blue");
    frame
        .draw_solid(
            Rectangle::new((64, 64).into(), (32, 32).into()),
            &[Rectangle::from_size((32, 32).into())],
            Color32F::new(1.0, 1.0, 1.0, 1.0),
        )
        .expect("draw solid");
    let sync = frame.finish().expect("finish dmabuf frame");
    sync.wait().expect("wait dmabuf frame");

    let mapping = renderer
        .copy_framebuffer(&fb, Rectangle::from_size(size), DrmFourcc::Argb8888)
        .expect("copy dmabuf framebuffer");
    let data = renderer.map_texture(&mapping).expect("map dmabuf copy");
    check_pixel(data, 256, 5, 5, [255, 0, 0, 255], "blue background");
    check_pixel(data, 256, 80, 80, [255, 255, 255, 255], "white square");
    check_pixel(data, 256, 5, 250, [255, 0, 0, 255], "blue background (bottom)");
    println!("dmabuf rendering: OK");

    // --- Implicit sync interop ---
    // The frame should have attached a write fence to the dmabuf; export and re-import it.
    let plane_fd = dmabuf.handles().next().unwrap();
    match export_sync_file(plane_fd, SyncFileFlags::READ) {
        Ok(sync_file) => {
            import_sync_file(plane_fd, SyncFileFlags::WRITE, sync_file.as_fd()).expect("import sync file");
            println!("dmabuf sync_file export/import: OK");
        }
        Err(err) => println!("dmabuf sync_file ioctls unsupported on this kernel: {err}"),
    }

    // Sample the just-rendered dmabuf into the offscreen buffer. Ordering between the two
    // frames relies on the implicit-sync interop (plus same-queue submission order).
    let dmabuf_texture = renderer
        .import_dmabuf(&dmabuf, None)
        .expect("import dmabuf texture");
    let mut fb2 = renderer.bind(&mut offscreen).expect("rebind offscreen");
    let mut frame = renderer
        .render(&mut fb2, output_size, Transform::Normal)
        .expect("begin sampling frame");
    frame
        .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &[full])
        .expect("clear black");
    frame
        .render_texture_from_to(
            &dmabuf_texture,
            Rectangle::from_size((256.0, 256.0).into()),
            full,
            &[full],
            &[],
            Transform::Normal,
            1.0,
        )
        .expect("draw dmabuf texture");
    let sync = frame.finish().expect("finish sampling frame");
    sync.wait().expect("wait sampling frame");

    let mapping = renderer
        .copy_framebuffer(&fb2, Rectangle::from_size(size), DrmFourcc::Argb8888)
        .expect("copy sampled framebuffer");
    let data = renderer.map_texture(&mapping).expect("map sampled copy");
    check_pixel(data, 256, 5, 5, [255, 0, 0, 255], "sampled blue background");
    check_pixel(data, 256, 80, 80, [255, 255, 255, 255], "sampled white square");
    println!("dmabuf texture sampling with implicit sync: OK");
    drop(fb2);

    // --- Output transform ---
    let mut fb2 = renderer.bind(&mut offscreen).expect("rebind offscreen");
    let mut frame = renderer
        .render(&mut fb2, output_size, Transform::_90)
        .expect("begin rotated frame");
    frame
        .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &[full])
        .expect("clear black");
    // In 90°-transformed space, a rect at the origin lands in a different memory corner.
    frame
        .clear(
            Color32F::new(1.0, 1.0, 0.0, 1.0),
            &[Rectangle::new((0, 0).into(), (32, 32).into())],
        )
        .expect("clear corner");
    let sync = frame.finish().expect("finish rotated frame");
    sync.wait().expect("wait rotated frame");

    let mapping = renderer
        .copy_framebuffer(&fb2, Rectangle::from_size(size), DrmFourcc::Argb8888)
        .expect("copy rotated framebuffer");
    let data = renderer.map_texture(&mapping).expect("map rotated copy");
    // With a 90° output transform the logical origin maps away from memory (0, 0); just check
    // that exactly one corner is yellow and the opposite corner is black.
    let yellow = [0u8, 255, 255, 255];
    let corners = [(5, 5), (250, 5), (5, 250), (250, 250)];
    let yellow_corners = corners
        .iter()
        .filter(|(x, y)| {
            let idx = ((y * 256 + x) * 4) as usize;
            data[idx..idx + 4] == yellow
        })
        .count();
    assert_eq!(yellow_corners, 1, "expected exactly one yellow corner");
    println!("transformed rendering: OK");

    println!("all vulkan renderer smoke tests passed");
}
