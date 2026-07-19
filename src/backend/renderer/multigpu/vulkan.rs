//! Implementation of the multi-gpu [`GraphicsApi`] using user provided GBM devices for
//! allocation and Vulkan for rendering.

use std::{
    collections::HashMap,
    fmt,
    os::unix::prelude::AsFd,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use ash::vk;
use tracing::warn;

use crate::backend::{
    SwapBuffersError,
    allocator::{
        Allocator,
        dmabuf::{AnyError, Dmabuf, DmabufAllocator},
        gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
    },
    drm::DrmNode,
    renderer::{
        RendererSuper,
        multigpu::{ApiDevice, Error as MultiError, GraphicsApi},
        vulkan::{VulkanError, VulkanRenderer},
    },
    vulkan::{Instance, InstanceError, PhysicalDevice, version::Version},
};

/// Errors raised by the [`VulkanBackend`]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Instance creation error
    #[error(transparent)]
    Instance(#[from] InstanceError),
    /// Vulkan api error
    #[error("Vulkan API error: {0}")]
    Vk(#[from] vk::Result),
    /// Renderer error
    #[error(transparent)]
    Renderer(#[from] VulkanError),
}

impl From<Error> for SwapBuffersError {
    #[inline]
    fn from(err: Error) -> SwapBuffersError {
        match err {
            x @ Error::Instance(_) | x @ Error::Vk(_) => SwapBuffersError::ContextLost(Box::new(x)),
            Error::Renderer(x) => x.into(),
        }
    }
}

/// A [`GraphicsApi`] utilizing user-provided GBM devices for allocation and Vulkan for
/// rendering.
pub struct VulkanBackend<A: AsFd + 'static> {
    devices: HashMap<DrmNode, GbmAllocator<A>>,
    instance: Mutex<Option<Instance>>,
    allocator_flags: GbmBufferFlags,
    needs_enumeration: AtomicBool,
}

impl<A: AsFd + fmt::Debug + 'static> fmt::Debug for VulkanBackend<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("devices", &self.devices.keys())
            .finish_non_exhaustive()
    }
}

impl<A: AsFd + 'static> Default for VulkanBackend<A> {
    #[inline]
    fn default() -> Self {
        VulkanBackend {
            devices: HashMap::new(),
            instance: Mutex::new(None),
            allocator_flags: GbmBufferFlags::RENDERING,
            needs_enumeration: AtomicBool::new(true),
        }
    }
}

impl<A: AsFd + Clone + Send + 'static> VulkanBackend<A> {
    /// Sets the default flags to use for allocating buffers via the [`GbmAllocator`]
    /// provided by these backends devices.
    ///
    /// Only affects nodes added via [`add_node`][Self::add_node] *after* calling this method.
    pub fn set_allocator_flags(&mut self, flags: GbmBufferFlags) {
        self.allocator_flags = flags;
    }

    /// Add a new GBM device for a given node to the api
    pub fn add_node(&mut self, node: DrmNode, gbm: GbmDevice<A>) {
        if self.devices.contains_key(&node) {
            return;
        }

        self.devices
            .insert(node, GbmAllocator::new(gbm, self.allocator_flags));
        self.needs_enumeration.store(true, Ordering::SeqCst);
    }

    /// Remove a given node from the api
    pub fn remove_node(&mut self, node: &DrmNode) {
        if self.devices.remove(node).is_some() {
            self.needs_enumeration.store(true, Ordering::SeqCst);
        }
    }
}

/// Returns whether a physical device drives the given DRM node.
fn phd_matches_node(phd: &PhysicalDevice, node: &DrmNode) -> bool {
    let matches = |result: Result<Option<DrmNode>, _>| {
        result
            .ok()
            .flatten()
            .is_some_and(|phd_node| phd_node.dev_id() == node.dev_id())
    };
    matches(phd.render_node()) || matches(phd.primary_node())
}

impl<A: AsFd + Clone + 'static> GraphicsApi for VulkanBackend<A> {
    type Device = VulkanDevice;
    type Error = Error;

    fn enumerate(&self, list: &mut Vec<Self::Device>) -> Result<(), Self::Error> {
        self.needs_enumeration.store(false, Ordering::SeqCst);

        // remove old stuff
        list.retain(|device| {
            self.devices
                .keys()
                .any(|node| device.node.dev_id() == node.dev_id())
        });

        let mut instance_guard = self.instance.lock().unwrap();
        if instance_guard.is_none() {
            *instance_guard = Some(Instance::new(Version::VERSION_1_3, None)?);
        }
        let instance = instance_guard.as_ref().unwrap();
        let physical_devices = PhysicalDevice::enumerate(instance)?.collect::<Vec<_>>();

        // add new stuff
        let new_renderers = self
            .devices
            .iter()
            .filter(|(node, _)| !list.iter().any(|device| device.node.dev_id() == node.dev_id()))
            .filter_map(|(node, gbm)| {
                let Some(phd) = physical_devices.iter().find(|phd| phd_matches_node(phd, node)) else {
                    warn!("Skipping node {node:?}: no matching vulkan physical device");
                    return None;
                };
                let renderer = match VulkanRenderer::new(phd) {
                    Ok(renderer) => renderer,
                    Err(err) => {
                        warn!("Skipping node {node:?}: {err}");
                        return None;
                    }
                };

                Some(VulkanDevice {
                    node: *node,
                    software: phd.ty() == vk::PhysicalDeviceType::CPU,
                    renderer,
                    allocator: Box::new(DmabufAllocator(gbm.clone())),
                })
            })
            .collect::<Vec<VulkanDevice>>();
        list.extend(new_renderers);

        // but don't replace already initialized renderers

        Ok(())
    }

    fn needs_enumeration(&self) -> bool {
        self.needs_enumeration.load(Ordering::Acquire)
    }

    fn identifier() -> &'static str {
        "gbm_vulkan"
    }
}

// TODO: Replace with specialization impl in multigpu/mod once possible
impl<T: GraphicsApi, A: AsFd + Clone + 'static> std::convert::From<VulkanError>
    for MultiError<VulkanBackend<A>, T>
where
    T::Error: 'static,
    <<T::Device as ApiDevice>::Renderer as RendererSuper>::Error: 'static,
{
    #[inline]
    fn from(err: VulkanError) -> MultiError<VulkanBackend<A>, T> {
        MultiError::Render(err)
    }
}

/// [`ApiDevice`] of the [`VulkanBackend`]
pub struct VulkanDevice {
    node: DrmNode,
    software: bool,
    renderer: VulkanRenderer,
    allocator: Box<dyn Allocator<Buffer = Dmabuf, Error = AnyError>>,
}

impl fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("node", &self.node)
            .field("renderer", &self.renderer)
            .finish_non_exhaustive()
    }
}

impl ApiDevice for VulkanDevice {
    type Renderer = VulkanRenderer;

    fn renderer(&self) -> &Self::Renderer {
        &self.renderer
    }
    fn renderer_mut(&mut self) -> &mut Self::Renderer {
        &mut self.renderer
    }
    fn allocator(&mut self) -> &mut dyn Allocator<Buffer = Dmabuf, Error = AnyError> {
        self.allocator.as_mut()
    }
    fn node(&self) -> &DrmNode {
        &self.node
    }
    fn can_do_cross_device_imports(&self) -> bool {
        !self.software
    }
}
