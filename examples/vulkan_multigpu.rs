//! Smoke test for the multi-gpu Vulkan backend: renders offscreen through a
//! `MultiRenderer<VulkanBackend, VulkanBackend>` and verifies the readback.

use std::{fs::OpenOptions, os::unix::io::OwnedFd};

use drm_fourcc::DrmFourcc;
use smithay::{
    backend::{
        allocator::gbm::GbmDevice,
        drm::{DrmDeviceFd, DrmNode},
        renderer::{
            Bind, Color32F, ExportMem, Frame, Offscreen, Renderer,
            multigpu::{GpuManager, vulkan::VulkanBackend},
        },
    },
    utils::{DeviceFd, Rectangle, Size, Transform},
};

fn main() {
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().init();
    }

    // Find the first render node.
    let path = (128..192)
        .map(|minor| format!("/dev/dri/renderD{minor}"))
        .find(|path| std::path::Path::new(path).exists())
        .expect("no render node found");
    println!("Using render node: {path}");

    let node = DrmNode::from_path(&path).expect("create drm node");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open render node");
    let fd = DrmDeviceFd::new(DeviceFd::from(Into::<OwnedFd>::into(file)));
    let gbm = GbmDevice::new(fd).expect("create gbm device");

    let mut backend = VulkanBackend::default();
    backend.add_node(node, gbm);

    let mut gpus: GpuManager<VulkanBackend<DrmDeviceFd>> =
        GpuManager::new(backend).expect("create gpu manager");
    let mut renderer = gpus.single_renderer(&node).expect("create renderer");

    let size = Size::<i32, smithay::utils::Buffer>::from((128, 128));
    let output_size = Size::from((128, 128));
    let full: Rectangle<i32, smithay::utils::Physical> = Rectangle::from_size(output_size);

    let mut offscreen = renderer
        .create_buffer(DrmFourcc::Argb8888, size)
        .expect("create offscreen buffer");
    let mut fb = renderer.bind(&mut offscreen).expect("bind offscreen");
    let mut frame = renderer
        .render(&mut fb, output_size, Transform::Normal)
        .expect("begin frame");
    frame
        .clear(Color32F::new(0.0, 1.0, 0.0, 1.0), &[full])
        .expect("clear green");
    frame
        .draw_solid(
            Rectangle::new((32, 32).into(), (16, 16).into()),
            &[Rectangle::from_size((16, 16).into())],
            Color32F::new(1.0, 0.0, 1.0, 1.0),
        )
        .expect("draw solid");
    let sync = frame.finish().expect("finish frame");
    sync.wait().expect("wait frame");

    let mapping = renderer
        .copy_framebuffer(&fb, Rectangle::from_size(size), DrmFourcc::Argb8888)
        .expect("copy framebuffer");
    let data = renderer.map_texture(&mapping).expect("map");

    // Argb8888 little-endian byte order: B, G, R, A.
    let check = |x: i32, y: i32, expected: [u8; 4], what: &str| {
        let idx = ((y * 128 + x) * 4) as usize;
        assert_eq!(
            &data[idx..idx + 4],
            expected,
            "unexpected pixel at ({x}, {y}) for {what}"
        );
    };
    check(5, 5, [0, 255, 0, 255], "green background");
    check(40, 40, [255, 0, 255, 255], "magenta square");
    check(120, 120, [0, 255, 0, 255], "green background (bottom right)");

    println!("multigpu vulkan rendering: OK");
}
