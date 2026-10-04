//! Render-node-only tests using Vulkan-produced sync files whose completion is controlled
//! by host-set events in the GPU command stream. No DRM master, modeset or display is used.
//!
//! `SMITHAY_TEST_REQUIRE_SYNCOBJ=1` makes missing hardware a failure;
//! `SMITHAY_TEST_SYNCOBJ_RENDER_NODE` selects a render node instead of automatic discovery.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use ash::{khr, vk};

use super::*;
use crate::backend::drm::{DrmNode, NodeType};
use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};
use crate::utils::DeviceFd;

/// Owns a separate queue which cannot finish a fence until its gate is explicitly signalled.
/// Drop releases all gates before waiting, including during a failed assertion.
struct SyncFiles {
    _instance: Instance,
    device: ash::Device,
    external: khr::external_semaphore_fd::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    gates: Arc<Mutex<Vec<vk::Event>>>,
    signals: Vec<vk::Semaphore>,
    point: u64,
    timed_out: Arc<AtomicBool>,
    stage: Arc<Mutex<&'static str>>,
    watchdog: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

impl SyncFiles {
    fn new() -> Result<Self, String> {
        let instance = Instance::new(Version::VERSION_1_2, None).map_err(|err| err.to_string())?;
        let devices = PhysicalDevice::enumerate(&instance).map_err(|err| err.to_string())?;
        let mut errors = Vec::new();
        for physical in devices {
            if physical.api_version() < Version::VERSION_1_2
                || !physical.has_device_extension(khr::external_semaphore_fd::NAME)
            {
                continue;
            }
            let raw = instance.handle();
            let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let mut properties = vk::ExternalSemaphoreProperties::default();
            unsafe {
                raw.get_physical_device_external_semaphore_properties(
                    physical.handle(),
                    &info,
                    &mut properties,
                )
            };
            if !properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
            {
                continue;
            }
            let families = unsafe { raw.get_physical_device_queue_family_properties(physical.handle()) };
            let Some(family) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            else {
                continue;
            };
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family as u32)
                .queue_priorities(&[1.])];
            let extensions = [khr::external_semaphore_fd::NAME.as_ptr()];
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extensions);
            let device = match unsafe { raw.create_device(physical.handle(), &info, None) } {
                Ok(device) => device,
                Err(err) => {
                    errors.push(format!("{}: {err}", physical.name()));
                    continue;
                }
            };
            let queue = unsafe { device.get_device_queue(family as u32, 0) };
            let external = khr::external_semaphore_fd::Device::new(raw, &device);
            let mut files = Self {
                _instance: instance.clone(),
                device,
                external,
                queue,
                command_pool: vk::CommandPool::null(),
                gates: Arc::new(Mutex::new(Vec::new())),
                signals: Vec::new(),
                point: 0,
                timed_out: Arc::new(AtomicBool::new(false)),
                stage: Arc::new(Mutex::new("creating command pool")),
                watchdog: None,
            };
            files.start_watchdog();
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family as u32)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            files.command_pool =
                unsafe { files.device.create_command_pool(&info, None) }.map_err(|err| err.to_string())?;
            return Ok(files);
        }
        Err(format!(
            "no Vulkan device can export sync files: {}",
            errors.join("; ")
        ))
    }

    fn pending(&mut self) -> (u64, OwnedFd) {
        self.check_watchdog();
        self.set_stage("creating VkEvent and command buffer");
        let event = unsafe { self.device.create_event(&vk::EventCreateInfo::default(), None) }.unwrap();
        // Publish before submitting: even if a driver blocks inside queue_submit or export,
        // the watchdog can release this event. The mutex serializes all host event accesses.
        self.gates.lock().unwrap().push(event);
        self.check_watchdog();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&allocate) }.unwrap()[0];
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device.begin_command_buffer(command, &begin).unwrap();
            self.device.cmd_wait_events(
                command,
                &[event],
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ALL_COMMANDS,
                &[],
                &[],
                &[],
            );
            self.device.end_command_buffer(command).unwrap();
        }
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        let semaphore = unsafe { self.device.create_semaphore(&info, None) }.unwrap();
        self.signals.push(semaphore);
        self.point += 1;
        let signals = [semaphore];
        let commands = [command];
        // A future timeline wait can keep a native fence from being materialized by the
        // driver. Submit executable work instead, with its wait inside the command stream.
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&signals);
        self.set_stage("vkQueueSubmit waiting-event command buffer");
        unsafe { self.device.queue_submit(self.queue, &[submit], vk::Fence::null()) }.unwrap();
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        self.set_stage("vkGetSemaphoreFdKHR");
        let fd = unsafe { self.external.get_semaphore_fd(&info) }.unwrap();
        assert!(fd >= 0, "a gated, incomplete fence must have a real fd");
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        self.check_watchdog();
        self.set_stage("DRM syncobj operations and assertions");
        (self.point, fd)
    }

    fn signal(&self, point: u64) {
        self.check_watchdog();
        self.set_stage("host vkSetEvent");
        let gates = self.gates.lock().unwrap();
        unsafe { self.device.set_event(gates[point as usize - 1]) }.unwrap();
        drop(gates);
        self.set_stage("DRM syncobj operations and assertions");
    }

    fn set_stage(&self, stage: &'static str) {
        *self.stage.lock().unwrap() = stage;
    }

    fn check_watchdog(&self) {
        assert!(
            !self.timed_out.load(Ordering::Acquire),
            "syncobj test exceeded its GPU watchdog deadline"
        );
    }

    fn start_watchdog(&mut self) {
        let (stop, receiver) = mpsc::channel();
        let device = self.device.clone();
        let gates = self.gates.clone();
        let stage = self.stage.clone();
        let timed_out = self.timed_out.clone();
        let name = std::thread::current().name().unwrap_or("syncobj test").to_owned();
        let thread = std::thread::spawn(move || {
            if receiver.recv_timeout(Duration::from_secs(15)).is_ok() {
                return;
            }
            timed_out.store(true, Ordering::Release);
            eprintln!(
                "{name}: 15s watchdog timeout during {}; releasing all VkEvents",
                *stage.lock().unwrap()
            );
            for event in gates.lock().unwrap_or_else(|poison| poison.into_inner()).iter() {
                if let Err(err) = unsafe { device.set_event(*event) } {
                    eprintln!("{name}: watchdog could not set VkEvent: {err}");
                }
            }
            if receiver.recv_timeout(Duration::from_secs(5)).is_err() {
                eprintln!(
                    "{name}: still blocked during {} after releasing events; exiting failed test process",
                    *stage.lock().unwrap()
                );
                std::process::exit(1);
            }
        });
        self.watchdog = Some((stop, thread));
    }
}

impl Drop for SyncFiles {
    fn drop(&mut self) {
        self.set_stage("VkEvent release / vkDeviceWaitIdle during cleanup");
        unsafe {
            for event in self
                .gates
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .iter()
            {
                let _ = self.device.set_event(*event);
            }
            let _ = self.device.device_wait_idle();
            // ash::Device::clone only copies a handle. Join before destroying any object the
            // watchdog can access, even when an assertion or Vulkan call failed.
            if let Some((stop, thread)) = self.watchdog.take() {
                let _ = stop.send(());
                let _ = thread.join();
            }
            for semaphore in self.signals.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for event in self
                .gates
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .drain(..)
            {
                self.device.destroy_event(event, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
        if !std::thread::panicking() {
            self.check_watchdog();
        }
    }
}

fn open_render_node(path: &Path) -> Result<DrmDeviceFd, String> {
    let expected = DrmNode::from_path(path).map_err(|err| err.to_string())?;
    if expected.ty() != NodeType::Render {
        return Err("syncobj tests only open DRM render nodes".into());
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    let actual = DrmNode::from_file(&file).map_err(|err| err.to_string())?;
    if actual != expected {
        return Err("DRM node changed while opening it".into());
    }
    // Render nodes cannot acquire DRM master. No primary node is opened by this test.
    let fd: OwnedFd = file.into();
    let device = DrmDeviceFd::new(DeviceFd::from(fd));
    let probe = device.create_syncobj(false).map_err(|err| err.to_string())?;
    let result = device.syncobj_timeline_query(&[probe], &mut [0], false);
    let _ = device.destroy_syncobj(probe);
    result.map_err(|err| err.to_string())?;
    Ok(device)
}

fn setup() -> Option<(PathBuf, DrmDeviceFd, SyncFiles)> {
    let result = (|| {
        let paths = if let Some(path) = std::env::var_os("SMITHAY_TEST_SYNCOBJ_RENDER_NODE") {
            vec![PathBuf::from(path)]
        } else {
            let mut paths: Vec<_> = std::fs::read_dir("/dev/dri")
                .map_err(|err| err.to_string())?
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
                .map(|entry| entry.path())
                .collect();
            paths.sort();
            paths
        };
        let mut errors = Vec::new();
        for path in paths {
            match open_render_node(&path) {
                Ok(device) => return Ok((path, device, SyncFiles::new()?)),
                Err(err) => errors.push(format!("{}: {err}", path.display())),
            }
        }
        Err(format!(
            "no render node supports syncobj timelines: {}",
            errors.join("; ")
        ))
    })();
    match result {
        Ok(setup) => Some(setup),
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_SYNCOBJ")
                .is_some_and(|value| !value.is_empty() && value != "0")
            {
                panic!("{err}");
            }
            eprintln!("skipping syncobj scratch test: {err}");
            None
        }
    }
}

fn timeline(device: &DrmDeviceFd) -> DrmTimeline {
    let handle = device.create_syncobj(false).unwrap();
    let fd = device.syncobj_to_fd(handle, false).unwrap();
    device.destroy_syncobj(handle).unwrap();
    DrmTimeline::new(device, fd).unwrap()
}

fn point(timeline: &DrmTimeline, point: u64) -> DrmSyncPoint {
    DrmSyncPoint {
        timeline: timeline.clone(),
        point,
    }
}

fn poll(fd: &OwnedFd, timeout_ms: i32) -> bool {
    let mut pollfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
    assert!(result >= 0, "poll failed: {}", io::Error::last_os_error());
    assert_eq!(pollfd.revents & (libc::POLLNVAL | libc::POLLERR), 0);
    pollfd.revents & libc::POLLIN != 0
}

fn scratch(timeline: &DrmTimeline) -> Option<drm::control::syncobj::Handle> {
    timeline.0.dev_ctx.lock().unwrap().scratch
}

fn assert_destroyed(device: &DrmDeviceFd, handle: drm::control::syncobj::Handle) {
    assert!(device.syncobj_timeline_query(&[handle], &mut [0], false).is_err());
}

#[test]
fn scratch_reuse_preserves_incomplete_sync_files_and_timeline_points() {
    let Some((_, device, mut files)) = setup() else {
        return;
    };
    let timeline = timeline(&device);
    let mut first_scratch = None;
    for _ in 0..16 {
        let (a, source_a) = files.pending();
        let (b, source_b) = files.pending();
        let a = point(&timeline, a);
        let b = point(&timeline, b);
        a.import_sync_file(source_a.as_fd()).unwrap();
        let allocation = scratch(&timeline).unwrap();
        assert_eq!(*first_scratch.get_or_insert(allocation), allocation);
        let exported_a = a.export_sync_file().unwrap();
        b.import_sync_file(source_b.as_fd()).unwrap();
        let exported_b = b.export_sync_file().unwrap();
        assert_eq!(scratch(&timeline), Some(allocation));
        assert!(!poll(&exported_a, 0));
        assert!(!poll(&exported_b, 0));

        files.signal(a.point);
        assert!(
            poll(&exported_a, 5000),
            "old exported fence lost its original payload"
        );
        assert!(a.wait(0).is_ok());
        assert!(b.wait(0).is_err());
        assert!(
            !poll(&exported_b, 0),
            "replacing scratch prematurely signalled the new fence"
        );
        files.signal(b.point);
        assert!(poll(&exported_b, 5000));
        assert!(b.wait(0).is_ok());
    }
}

#[test]
fn merged_read_fences_keep_the_earlier_dependency_when_the_new_read_finishes_first() {
    let Some((_, _device, mut files)) = setup() else {
        return;
    };
    let (first, first_file) = files.pending();
    let (second, second_file) = files.pending();
    // Publish the later-to-complete read first, then add a new dependency which completes
    // earlier. Replacing the previous release fence would incorrectly lose second_file.
    let merged =
        crate::backend::renderer::sync::sync_file::merge(second_file.as_fd(), first_file.as_fd()).unwrap();
    assert!(!poll(&merged, 0));
    files.signal(first);
    assert!(poll(&first_file, 5000));
    assert!(
        !poll(&merged, 0),
        "release must still wait for the other context's read"
    );
    files.signal(second);
    assert!(poll(&merged, 5000));
}

#[test]
fn explicit_buffer_release_joins_each_read_submission() {
    let Some((_, device, mut files)) = setup() else {
        return;
    };
    let (_display, _socket, implicit) = crate::backend::renderer::utils::buffer_read::tests::buffer();
    let timeline = timeline(&device);
    let release = point(&timeline, 1);
    let acquire = point(&timeline, 0);
    let buffer =
        crate::backend::renderer::utils::Buffer::with_explicit((*implicit).clone(), acquire, release.clone());
    let (first, first_file) = files.pending();
    let (second, second_file) = files.pending();
    buffer.set_release_fence(second_file.as_fd());
    buffer.set_release_fence(first_file.as_fd());
    drop(buffer);
    let exported = release.export_sync_file().unwrap();
    files.signal(first);
    assert!(poll(&first_file, 5000));
    assert!(!poll(&exported, 0));
    files.signal(second);
    assert!(poll(&exported, 5000));
}

#[test]
fn scratch_errors_discard_old_payload_and_allow_recovery() {
    let Some((_, device, mut files)) = setup() else {
        return;
    };
    let timeline = timeline(&device);
    let (a, source_a) = files.pending();
    let a = point(&timeline, a);
    a.import_sync_file(source_a.as_fd()).unwrap();
    let exported_a = a.export_sync_file().unwrap();
    let failed = point(&timeline, a.point + 1);
    let invalid = std::fs::File::open("/dev/null").unwrap();
    assert!(failed.import_sync_file(invalid.as_fd()).is_err());
    assert_eq!(scratch(&timeline), None);
    assert!(
        failed.export_sync_file().is_err(),
        "failed import must not install the old fence"
    );
    assert_eq!(scratch(&timeline), None);
    assert!(!poll(&exported_a, 0));

    let (b, source_b) = files.pending();
    let b = point(&timeline, b);
    b.import_sync_file(source_b.as_fd()).unwrap();
    let exported_b = b.export_sync_file().unwrap();
    assert!(scratch(&timeline).is_some());
    files.signal(a.point);
    assert!(poll(&exported_a, 5000));
    assert!(!poll(&exported_b, 0));
    files.signal(b.point);
    assert!(poll(&exported_b, 5000));
}

#[test]
fn scratch_device_migration_and_drop_preserve_pending_fences() {
    let Some((path, old_device, mut files)) = setup() else {
        return;
    };
    let new_device = open_render_node(&path).unwrap();
    let timeline = timeline(&old_device);
    let (value, source) = files.pending();
    let pending = point(&timeline, value);
    pending.import_sync_file(source.as_fd()).unwrap();
    let old_export = pending.export_sync_file().unwrap();
    let event = pending.eventfd().unwrap();
    let (old_timeline, old_scratch) = {
        let ctx = timeline.0.dev_ctx.lock().unwrap();
        (ctx.syncobj, ctx.scratch.unwrap())
    };

    timeline.0.update_device(&new_device).unwrap();
    assert_eq!(scratch(&timeline), None);
    assert_destroyed(&old_device, old_timeline);
    assert_destroyed(&old_device, old_scratch);
    assert!(
        !poll(&event, 0),
        "migration must not release an incomplete eventfd blocker"
    );
    let new_export = pending.export_sync_file().unwrap();
    assert!(!poll(&new_export, 0));
    files.signal(value);
    assert!(poll(&old_export, 5000));
    assert!(poll(&new_export, 5000));
    assert!(poll(&event, 5000));
    let (new_timeline, new_scratch) = {
        let ctx = timeline.0.dev_ctx.lock().unwrap();
        (ctx.syncobj, ctx.scratch.unwrap())
    };
    drop(pending);
    drop(timeline);
    assert_destroyed(&new_device, new_timeline);
    assert_destroyed(&new_device, new_scratch);
}

#[test]
fn scratch_update_with_same_device_keeps_the_new_timeline_handle_alive() {
    let Some((_, device, mut files)) = setup() else {
        return;
    };
    let timeline = timeline(&device);
    let (value, source) = files.pending();
    let pending = point(&timeline, value);
    pending.import_sync_file(source.as_fd()).unwrap();
    let event = pending.eventfd().unwrap();
    let mut exports = vec![pending.export_sync_file().unwrap()];

    for _ in 0..8 {
        let (old_handle, old_scratch) = {
            let ctx = timeline.0.dev_ctx.lock().unwrap();
            (ctx.syncobj, ctx.scratch.unwrap())
        };
        // Unlike the migration test above, this is exactly the same DrmDeviceFd, including
        // the kernel handle namespace. Destroying the old context must not kill the new one.
        timeline.0.update_device(&device).unwrap();
        let new_handle = timeline.0.dev_ctx.lock().unwrap().syncobj;
        assert!(
            timeline.query_signalled_point().is_ok(),
            "same-device update invalidated the new handle: old={old_handle:?}, new={new_handle:?}"
        );
        if old_handle != new_handle {
            assert_destroyed(&device, old_handle);
        }
        assert_destroyed(&device, old_scratch);
        assert!(!poll(&event, 0));
        exports.push(pending.export_sync_file().unwrap());
        for fd in &exports {
            assert!(!poll(fd, 0));
        }
    }
    files.signal(value);
    for fd in &exports {
        assert!(poll(fd, 5000));
    }
    assert!(pending.wait(0).is_ok());
    assert!(poll(&event, 5000));
}

#[test]
fn scratch_invalidation_destroys_both_handles_and_rejects_further_use() {
    let Some((_, device, mut files)) = setup() else {
        return;
    };
    let timeline = timeline(&device);
    let (value, source) = files.pending();
    let pending = point(&timeline, value);
    pending.import_sync_file(source.as_fd()).unwrap();
    let exported = pending.export_sync_file().unwrap();
    let event = pending.eventfd().unwrap();
    assert!(!poll(&event, 0));
    let (handle, scratch) = {
        let ctx = timeline.0.dev_ctx.lock().unwrap();
        (ctx.syncobj, ctx.scratch.unwrap())
    };
    timeline.0.invalidate();
    assert!(
        poll(&event, 0),
        "invalidation must release the eventfd blocker immediately"
    );
    assert_destroyed(&device, handle);
    assert_destroyed(&device, scratch);
    assert!(pending.export_sync_file().is_err());
    assert!(pending.import_sync_file(source.as_fd()).is_err());
    assert!(!poll(&exported, 0));
    files.signal(value);
    assert!(poll(&exported, 5000));
}
