//! Inject failures at the Vulkan dispatch boundary, exercising the real fence and
//! read-publication paths. Successful waits still complete a real, unused timeline
//! point; a device-lost result is never used as evidence of completion.

use std::{
    cell::RefCell,
    collections::VecDeque,
    ffi::{CStr, c_char, c_int},
    os::fd::{AsRawFd, IntoRawFd},
    time::Instant,
};

use super::*;
use crate::backend::renderer::{
    sync::SyncPoint,
    vulkan::{VulkanRenderer, descriptor_pool_tests::renderer},
};

#[derive(Clone, Copy)]
enum WaitAction {
    Fail(vk::Result),
    SignalThenWait,
}

#[derive(Debug, PartialEq, Eq)]
enum Call {
    Query,
    Export,
    Wait,
}

struct Faults {
    raw: ash::Device,
    query: Result<u64, vk::Result>,
    export: vk::Result,
    waits: VecDeque<WaitAction>,
    calls: Vec<Call>,
}

thread_local! {
    // Every test owns a separate device. Thread-local dispatch scripts prevent
    // failures from leaking to other tests running in parallel.
    static FAULTS: RefCell<Option<Faults>> = const { RefCell::new(None) };
}

unsafe extern "system" fn query_counter(
    _device: vk::Device,
    _semaphore: vk::Semaphore,
    value: *mut u64,
) -> vk::Result {
    let result = FAULTS.with(|faults| {
        let mut faults = faults.borrow_mut();
        let faults = faults.as_mut().unwrap();
        faults.calls.push(Call::Query);
        faults.query
    });
    match result {
        Ok(counter) => {
            unsafe { value.write(counter) };
            vk::Result::SUCCESS
        }
        Err(err) => err,
    }
}

unsafe extern "system" fn wait_semaphores(
    _device: vk::Device,
    info: *const vk::SemaphoreWaitInfo<'_>,
    timeout: u64,
) -> vk::Result {
    let (action, raw) = FAULTS.with(|faults| {
        let mut faults = faults.borrow_mut();
        let faults = faults.as_mut().unwrap();
        faults.calls.push(Call::Wait);
        (
            faults
                .waits
                .pop_front()
                .unwrap_or(WaitAction::Fail(vk::Result::ERROR_DEVICE_LOST)),
            faults.raw.clone(),
        )
    });
    match action {
        WaitAction::Fail(err) => err,
        WaitAction::SignalThenWait => {
            // This timeline has no queued work. Only this explicit host signal
            // establishes completion, after the preceding injected wait error.
            let info = unsafe { &*info };
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(unsafe { *info.p_semaphores })
                .value(unsafe { *info.p_values });
            match unsafe { raw.signal_semaphore(&signal) }
                .and_then(|()| unsafe { raw.wait_semaphores(info, timeout) })
            {
                Ok(()) => vk::Result::SUCCESS,
                Err(err) => err,
            }
        }
    }
}

unsafe extern "system" fn export_semaphore(
    _device: vk::Device,
    _info: *const vk::SemaphoreGetFdInfoKHR<'_>,
    output: *mut c_int,
) -> vk::Result {
    let result = FAULTS.with(|faults| {
        let mut faults = faults.borrow_mut();
        let faults = faults.as_mut().unwrap();
        faults.calls.push(Call::Export);
        faults.export
    });
    if result == vk::Result::SUCCESS {
        // Successful exports only exercise VulkanFence's fd ownership/cache in
        // isolation. Publication tests either must not export (no recipient) or
        // inject an export error, so no kernel sync-file ioctl consumes this fd.
        let Ok(file) = std::fs::File::open("/dev/null") else {
            return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
        };
        unsafe { output.write(file.into_raw_fd()) };
    }
    result
}

unsafe extern "system" fn get_device_proc_addr(
    _device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if unsafe { CStr::from_ptr(name) } == c"vkGetSemaphoreFdKHR" {
        Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetSemaphoreFdKHR, unsafe extern "system" fn()>(export_semaphore)
        })
    } else {
        None
    }
}

fn inject(
    renderer: &mut VulkanRenderer,
    query: Result<u64, vk::Result>,
    export: vk::Result,
    waits: impl IntoIterator<Item = WaitAction>,
) {
    let device = Arc::get_mut(&mut renderer.device).unwrap();
    let raw = device.raw.clone();
    let mut functions = raw.fp_v1_2().clone();
    functions.get_semaphore_counter_value = query_counter;
    functions.wait_semaphores = wait_semaphores;
    device.raw = ash::Device::from_parts_1_3(
        raw.handle(),
        raw.fp_v1_0().clone(),
        raw.fp_v1_1().clone(),
        functions,
        raw.fp_v1_3().clone(),
    );
    // Only the extension loader uses this synthetic instance. All allocation,
    // timeline signalling, successful waiting and destruction use the real device.
    let instance = unsafe {
        ash::Instance::load_with(
            |name| {
                if name == c"vkGetDeviceProcAddr" {
                    get_device_proc_addr as *const () as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                }
            },
            vk::Instance::null(),
        )
    };
    device.external_semaphore_fd = Some(ash::khr::external_semaphore_fd::Device::new(&instance, &raw));
    FAULTS.with(|faults| {
        *faults.borrow_mut() = Some(Faults {
            raw,
            query,
            export,
            waits: waits.into_iter().collect(),
            calls: Vec::new(),
        });
    });
}

fn fence(renderer: &VulkanRenderer) -> VulkanFence {
    let binary = unsafe {
        renderer
            .device
            .raw
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
    }
    .unwrap();
    VulkanFence {
        device: renderer.device.clone(),
        point: 1,
        binary: Some(Mutex::new(BinarySemaphore::Unexported(binary))),
    }
}

#[test]
fn failed_query_still_exports_and_caches_binary_payload() {
    let Some(mut renderer) = renderer() else { return };
    inject(
        &mut renderer,
        Err(vk::Result::ERROR_DEVICE_LOST),
        vk::Result::SUCCESS,
        [],
    );
    let sync = SyncPoint::from(fence(&renderer));
    assert!(!sync.is_reached(), "a failed query is not completion");
    let first = sync.export().expect("query errors must not suppress exports");
    // A later consumer duplicates the existing export, rather than exporting
    // the binary payload a second time.
    let second = sync.export().expect("the exported payload must remain available");
    assert_ne!(
        first.as_raw_fd(),
        second.as_raw_fd(),
        "exports own distinct fd duplicates"
    );
    FAULTS.with(|faults| {
        assert_eq!(
            faults.borrow().as_ref().unwrap().calls,
            [Call::Query, Call::Export]
        );
    });
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[test]
fn failed_query_without_release_recipient_waits_for_real_completion() {
    use crate::backend::renderer::utils::buffer_read::{BufferReadSet, tests::buffer};

    let Some(mut renderer) = renderer() else { return };
    inject(
        &mut renderer,
        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
        vk::Result::SUCCESS,
        [
            WaitAction::Fail(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            WaitAction::SignalThenWait,
        ],
    );
    let sync = SyncPoint::from(fence(&renderer));
    let (_display, _socket, buffer) = buffer();
    let mut reads = BufferReadSet::default();
    buffer.with_read_source(|| reads.capture());
    assert_eq!(reads.len(), 1, "unknown implicit buffers retain their reads");
    let before = Instant::now();
    reads.publish(&sync);
    assert!(before.elapsed() >= FAILED_WAIT_RETRY_DELAY);
    assert!(reads.is_empty());
    FAULTS.with(|faults| {
        let faults = faults.borrow();
        let faults = faults.as_ref().unwrap();
        assert_eq!(
            faults.calls,
            [Call::Query, Call::Wait, Call::Wait],
            "an export with no recipient would not protect the buffer"
        );
        assert_eq!(
            unsafe { faults.raw.get_semaphore_counter_value(renderer.device.timeline) }.unwrap(),
            1,
            "the original timeline, not an unconsumed exported fd, completes the read"
        );
    });
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[test]
fn failed_query_and_export_wait_for_real_completion_before_releasing_reads() {
    use crate::backend::{
        allocator::{
            Fourcc, Modifier,
            dmabuf::{Dmabuf, DmabufFlags},
        },
        renderer::utils::buffer_read::{BufferReadSet, tests::buffer_with_data},
    };

    let Some(mut renderer) = renderer() else { return };
    inject(
        &mut renderer,
        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
        vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        [
            WaitAction::Fail(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            WaitAction::SignalThenWait,
        ],
    );
    let sync = SyncPoint::from(fence(&renderer));
    // A DMA-BUF provides a real publication destination kind, so this exercises
    // the export-failure fallback rather than the no-recipient wait above. The
    // placeholder fd never reaches a kernel import: export is forced to fail.
    let mut dma = Dmabuf::builder((1, 1), Fourcc::Abgr8888, Modifier::Linear, DmabufFlags::empty());
    let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
    assert!(dma.add_plane(fd, 0, 4));
    let (_display, _socket, buffer) = buffer_with_data(dma.build().unwrap());
    let mut reads = BufferReadSet::default();
    buffer.with_read_source(|| reads.capture());
    let before = Instant::now();
    reads.publish(&sync);
    assert!(before.elapsed() >= FAILED_WAIT_RETRY_DELAY);
    assert!(reads.is_empty());
    assert!(sync.export().is_none(), "a failed export is not retried");
    FAULTS.with(|faults| {
        let faults = faults.borrow();
        let faults = faults.as_ref().unwrap();
        assert_eq!(faults.calls, [Call::Query, Call::Export, Call::Wait, Call::Wait]);
        assert_eq!(
            unsafe { faults.raw.get_semaphore_counter_value(renderer.device.timeline) }.unwrap(),
            1,
            "reads are released only after the real timeline point completes"
        );
    });
}

#[test]
fn device_lost_waits_remain_unresolved_and_do_not_busy_spin() {
    let Some(mut renderer) = renderer() else { return };
    inject(
        &mut renderer,
        Err(vk::Result::ERROR_DEVICE_LOST),
        vk::Result::ERROR_DEVICE_LOST,
        [WaitAction::Fail(vk::Result::ERROR_DEVICE_LOST); 2],
    );
    let sync = SyncPoint::from(fence(&renderer));
    let before = Instant::now();
    assert_eq!(sync.wait(), Err(Interrupted));
    assert_eq!(sync.wait(), Err(Interrupted));
    assert!(before.elapsed() >= FAILED_WAIT_RETRY_DELAY * 2);
    assert!(!sync.is_reached());
    FAULTS.with(|faults| {
        let faults = faults.borrow();
        let faults = faults.as_ref().unwrap();
        assert_eq!(faults.calls, [Call::Wait, Call::Wait, Call::Query]);
        assert_eq!(
            unsafe { faults.raw.get_semaphore_counter_value(renderer.device.timeline) }.unwrap(),
            0,
        );
    });
}
