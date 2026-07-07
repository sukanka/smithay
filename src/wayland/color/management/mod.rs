//! Implementation of the wp-color-management-v1 protocol (the stabilized staging version
//! shipped in wayland-protocols).
//!
//! Clients use this protocol to describe the colorimetry of their surface contents (e.g.
//! BT.2020 primaries with the ST 2084 PQ transfer function for HDR video) by creating
//! parametric image descriptions and attaching them to a `wl_surface`. The attached
//! description is double-buffered surface state; the committed value can be read with
//! [`get_surface_description`]. What the compositor *does* with that information (HDR
//! signalling, color conversion, tone mapping) is entirely up to the compositor — see e.g.
//! [`ConnectorColorState`](crate::backend::drm::ConnectorColorState) for signalling HDR on a
//! DRM connector.
//!
//! Only *parametric* image descriptions with *named* transfer functions and primaries are
//! supported; ICC-file and Windows-scRGB descriptions are rejected with `unsupported_feature`.
//! The compositor chooses which transfer functions, primaries, features and rendering intents
//! to advertise when creating the [`ColorManagementState`].
//!
//! ## Usage
//!
//! Implement [`ColorManagementHandler`], create a [`ColorManagementState`] and route the
//! protocol objects with the crate-wide [`delegate_dispatch2!`](crate::delegate_dispatch2)
//! macro. In your rendering/output logic, read the committed description of relevant
//! surfaces with [`get_surface_description`].

use std::sync::Mutex;

use tracing::{debug, trace};
use wayland_protocols::wp::color_management::v1::server::{
    wp_color_management_output_v1::{self, WpColorManagementOutputV1},
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_management_surface_v1::{self, WpColorManagementSurfaceV1},
    wp_color_manager_v1::{self, WpColorManagerV1},
    wp_image_description_creator_params_v1::{self, WpImageDescriptionCreatorParamsV1},
    wp_image_description_info_v1::WpImageDescriptionInfoV1,
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use wayland_server::protocol::wl_output::WlOutput;
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, Weak};

use crate::output::Output;
use crate::wayland::compositor::{self, Cacheable};
use crate::wayland::{Dispatch2, GlobalDispatch2};

pub use wp_color_manager_v1::{Feature, Primaries, RenderIntent, TransferFunction};

const VERSION: u32 = 1;

/// A parsed, immutable parametric image description.
///
/// Only named transfer functions and primaries are representable, since those are the only
/// ones this implementation advertises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDescription {
    /// The transfer characteristics of the content.
    pub transfer: TransferFunction,
    /// The color primaries of the content.
    pub primaries: Primaries,
    /// Maximum content light level in cd/m², if the client provided it.
    pub max_cll: Option<u32>,
    /// Maximum frame-average light level in cd/m², if provided.
    pub max_fall: Option<u32>,
    /// Mastering display luminance as (min in 0.0001 cd/m², max in cd/m²), if provided.
    pub mastering_luminance: Option<(u32, u32)>,
    /// Content luminances as (min in 0.0001 cd/m², max in cd/m², reference white in cd/m²),
    /// if provided via `set_luminances`.
    pub luminances: Option<(u32, u32, u32)>,
}

impl ImageDescription {
    /// sRGB / sRGB — the default SDR description, also used for surfaces without an attached
    /// description.
    pub const SRGB: Self = Self {
        transfer: TransferFunction::Srgb,
        primaries: Primaries::Srgb,
        max_cll: None,
        max_fall: None,
        mastering_luminance: None,
        luminances: None,
    };

    /// Whether this description denotes HDR/wide-gamut content: an HDR transfer function
    /// (PQ or HLG) or BT.2020 primaries.
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer, TransferFunction::St2084Pq | TransferFunction::Hlg)
            || matches!(self.primaries, Primaries::Bt2020)
    }
}

/// Double-buffered per-surface color management state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorManagementSurfaceCachedState {
    /// The image description attached to the surface, if any.
    pub description: Option<ImageDescription>,
    /// The rendering intent the client prefers for mapping the surface to outputs.
    pub render_intent: RenderIntent,
}

impl Default for ColorManagementSurfaceCachedState {
    fn default() -> Self {
        Self {
            description: None,
            render_intent: RenderIntent::Perceptual,
        }
    }
}

impl Cacheable for ColorManagementSurfaceCachedState {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        *self
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        *into = self;
    }
}

/// Returns the committed image description and rendering intent of a surface.
pub fn get_surface_description(surface: &WlSurface) -> (Option<ImageDescription>, RenderIntent) {
    compositor::with_states(surface, |states| {
        let state = *states
            .cached_state
            .get::<ColorManagementSurfaceCachedState>()
            .current();
        (state.description, state.render_intent)
    })
}

/// Per-surface color-management bookkeeping in the surface's data map: enforces the
/// one-`wp_color_management_surface_v1`-per-surface rule and tracks the surface's feedback
/// objects for `preferred_changed`.
#[derive(Debug, Default)]
struct ColorManagementSurfaceData {
    attached: Mutex<bool>,
    feedbacks: Mutex<Vec<WpColorManagementSurfaceFeedbackV1>>,
    /// Identity of the last preferred description notified for this surface, for dedupe.
    last_preferred: Mutex<Option<u32>>,
}

/// User data of a `wp_image_description_v1`: the parsed description it represents.
#[derive(Debug)]
pub struct ImageDescriptionData {
    desc: ImageDescription,
}

impl ImageDescriptionData {
    /// The description this object represents.
    pub fn description(&self) -> ImageDescription {
        self.desc
    }
}

/// Accumulated parameters of a `wp_image_description_creator_params_v1`, validated on
/// `create`.
#[derive(Debug, Default)]
pub struct ImageDescriptionBuilder {
    transfer: Option<TransferFunction>,
    primaries: Option<Primaries>,
    max_cll: Option<u32>,
    max_fall: Option<u32>,
    mastering_luminance: Option<(u32, u32)>,
    luminances: Option<(u32, u32, u32)>,
}

/// Global data of `wp_color_manager_v1`, carrying the client visibility filter.
pub struct ColorManagementGlobalData {
    filter: Box<dyn Fn(&Client) -> bool + Send + Sync>,
}

impl std::fmt::Debug for ColorManagementGlobalData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColorManagementGlobalData")
            .finish_non_exhaustive()
    }
}

/// Handler trait for wp-color-management-v1.
pub trait ColorManagementHandler {
    /// Returns the [`ColorManagementState`].
    fn color_management_state(&mut self) -> &mut ColorManagementState;

    /// Called when a surface's *pending* image description changed (set or unset). The
    /// committed value becomes visible via [`get_surface_description`] after the next
    /// `wl_surface.commit`; compositors typically re-evaluate color handling for the
    /// surface's output on the next redraw.
    fn image_description_changed(&mut self, _surface: &WlSurface) {}

    /// The image description describing how the compositor presents the given output.
    ///
    /// Defaults to sRGB.
    fn description_for_output(&mut self, _output: &Output) -> ImageDescription {
        ImageDescription::SRGB
    }

    /// The image description the compositor would prefer the given surface to use, reported
    /// via the surface feedback object.
    ///
    /// Defaults to sRGB.
    fn preferred_description_for_surface(&mut self, _surface: &WlSurface) -> ImageDescription {
        ImageDescription::SRGB
    }

    /// Schedules sending the information events for `info` (describing `desc`), to run
    /// *after* the current request dispatch returns — see [`send_image_description_info`].
    /// This MUST be deferred (e.g. via an event loop idle callback):
    /// `wp_image_description_info_v1.done` is a destructor event, and destroying the object
    /// inside the very callback that created it corrupts wayland-backend's bookkeeping (it
    /// writes the new object's data after the callback returns, which would then be a
    /// use-after-free).
    fn schedule_image_description_info(&mut self, info: WpImageDescriptionInfoV1, desc: ImageDescription);
}

/// Sends the information events describing `desc` on `info`, terminating with the destructor
/// `done` event. Must be called *outside* the request callback that created `info` (e.g. from
/// an event-loop idle), via [`ColorManagementHandler::schedule_image_description_info`].
pub fn send_image_description_info(info: &WpImageDescriptionInfoV1, desc: &ImageDescription) {
    if !info.is_alive() {
        return;
    }
    info.primaries_named(desc.primaries);
    info.tf_named(desc.transfer);
    if let Some((min, max)) = desc.mastering_luminance {
        info.target_luminance(min, max);
    }
    if let Some(max_cll) = desc.max_cll {
        info.target_max_cll(max_cll);
    }
    if let Some(max_fall) = desc.max_fall {
        info.target_max_fall(max_fall);
    }
    info.done();
}

/// State of the wp-color-management-v1 global.
#[derive(Debug)]
pub struct ColorManagementState {
    supported_tfs: Vec<TransferFunction>,
    supported_primaries: Vec<Primaries>,
    supported_features: Vec<Feature>,
    supported_intents: Vec<RenderIntent>,
    /// Known distinct image descriptions; a description's identity is its index + 1.
    ///
    /// Identities must be stable so that the identity sent in `preferred_changed` matches the
    /// identity a subsequent `get_preferred` delivers via `ready`. The table grows
    /// monotonically with distinct descriptions, which is bounded in practice (clients create
    /// the same few descriptions).
    identities: Vec<ImageDescription>,
    /// Live `wp_color_management_output_v1` objects per output, for
    /// [`output_description_changed`](Self::output_description_changed).
    output_objects: Vec<WpColorManagementOutputV1>,
}

impl ColorManagementState {
    /// Creates a new wp-color-management-v1 global.
    ///
    /// The supported transfer functions, primaries, features and rendering intents are
    /// advertised to clients and validated in requests. [`Feature::Parametric`] is always
    /// advertised (this implementation is parametric-only); [`RenderIntent::Perceptual`]
    /// is always advertised as required by the protocol.
    ///
    /// The global is only visible to clients for which `filter` returns `true`.
    pub fn new<D, F>(
        display: &DisplayHandle,
        supported_tfs: impl IntoIterator<Item = TransferFunction>,
        supported_primaries: impl IntoIterator<Item = Primaries>,
        supported_features: impl IntoIterator<Item = Feature>,
        supported_intents: impl IntoIterator<Item = RenderIntent>,
        filter: F,
    ) -> Self
    where
        D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>,
        D: Dispatch<WpColorManagerV1, ()>,
        D: ColorManagementHandler,
        D: 'static,
        F: Fn(&Client) -> bool + Send + Sync + 'static,
    {
        let data = ColorManagementGlobalData {
            filter: Box::new(filter),
        };
        display.create_global::<D, WpColorManagerV1, _>(VERSION, data);

        let mut supported_features: Vec<Feature> = supported_features.into_iter().collect();
        if !supported_features.contains(&Feature::Parametric) {
            supported_features.push(Feature::Parametric);
        }
        let mut supported_intents: Vec<RenderIntent> = supported_intents.into_iter().collect();
        if !supported_intents.contains(&RenderIntent::Perceptual) {
            supported_intents.push(RenderIntent::Perceptual);
        }

        Self {
            supported_tfs: supported_tfs.into_iter().collect(),
            supported_primaries: supported_primaries.into_iter().collect(),
            supported_features,
            supported_intents,
            identities: Vec::new(),
            output_objects: Vec::new(),
        }
    }

    /// Returns the stable identity for a description, assigning a new one if it is not known
    /// yet.
    fn identity_for(&mut self, desc: ImageDescription) -> u32 {
        let index = match self.identities.iter().position(|d| *d == desc) {
            Some(index) => index,
            None => {
                self.identities.push(desc);
                self.identities.len() - 1
            }
        };
        index as u32 + 1
    }

    /// Notifies the given surface's feedback objects that the compositor's preferred image
    /// description for it changed.
    ///
    /// Deduplicated per surface: notifying the same description again is a no-op, so this is
    /// safe to call from a periodic refresh. Clients react by calling `get_preferred`, which
    /// routes through
    /// [`ColorManagementHandler::preferred_description_for_surface`] — that must already
    /// return the new description when this is called.
    pub fn preferred_changed(&mut self, surface: &WlSurface, desc: ImageDescription) {
        let identity = self.identity_for(desc);
        compositor::with_states(surface, |states| {
            let Some(data) = states.data_map.get::<ColorManagementSurfaceData>() else {
                return;
            };
            let mut last_preferred = data.last_preferred.lock().unwrap();
            if *last_preferred == Some(identity) {
                return;
            }
            *last_preferred = Some(identity);

            let mut feedbacks = data.feedbacks.lock().unwrap();
            feedbacks.retain(|feedback| feedback.is_alive());
            for feedback in feedbacks.iter() {
                feedback.preferred_changed(identity);
            }
        });
    }

    /// Notifies all `wp_color_management_output_v1` objects of the given output that its
    /// image description changed.
    ///
    /// Clients react by calling `get_image_description`, which routes through
    /// [`ColorManagementHandler::description_for_output`] — that must already return the new
    /// description when this is called.
    pub fn output_description_changed(&mut self, output: &Output) {
        self.output_objects.retain(|obj| obj.is_alive());
        for obj in &self.output_objects {
            let same_output = obj
                .data::<WlOutput>()
                .and_then(Output::from_resource)
                .is_some_and(|o| o == *output);
            if same_output {
                obj.image_description_changed();
            }
        }
    }
}

impl<D> GlobalDispatch2<WpColorManagerV1, D> for ColorManagementGlobalData
where
    D: Dispatch<WpColorManagerV1, ()>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn bind(
        &self,
        state: &mut D,
        _handle: &DisplayHandle,
        _client: &Client,
        manager: New<WpColorManagerV1>,
        data_init: &mut DataInit<'_, D>,
    ) {
        let manager = data_init.init(manager, ());

        let cm_state = state.color_management_state();
        for intent in &cm_state.supported_intents {
            manager.supported_intent(*intent);
        }
        for feature in &cm_state.supported_features {
            manager.supported_feature(*feature);
        }
        for tf in &cm_state.supported_tfs {
            manager.supported_tf_named(*tf);
        }
        for primaries in &cm_state.supported_primaries {
            manager.supported_primaries_named(*primaries);
        }
        manager.done();
    }

    fn can_view(&self, client: &Client) -> bool {
        (self.filter)(client)
    }
}

impl<D> Dispatch2<WpColorManagerV1, D> for ()
where
    D: Dispatch<WpColorManagementOutputV1, WlOutput>,
    D: Dispatch<WpColorManagementSurfaceV1, Weak<WlSurface>>,
    D: Dispatch<WpColorManagementSurfaceFeedbackV1, Weak<WlSurface>>,
    D: Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ImageDescriptionBuilder>>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &WpColorManagerV1,
        request: <WpColorManagerV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_manager_v1::Request;
        match request {
            Request::GetOutput { id, output } => {
                let obj = data_init.init(id, output);
                state.color_management_state().output_objects.push(obj);
            }
            Request::GetSurface { id, surface } => {
                let already_attached = compositor::with_states(&surface, |states| {
                    states
                        .data_map
                        .insert_if_missing(ColorManagementSurfaceData::default);
                    let data = states.data_map.get::<ColorManagementSurfaceData>().unwrap();
                    let mut attached = data.attached.lock().unwrap();
                    std::mem::replace(&mut *attached, true)
                });
                if already_attached {
                    resource.post_error(
                        wp_color_manager_v1::Error::SurfaceExists,
                        "surface already has a wp_color_management_surface_v1",
                    );
                    return;
                }
                data_init.init(id, surface.downgrade());
            }
            Request::GetSurfaceFeedback { id, surface } => {
                let feedback = data_init.init(id, surface.downgrade());
                // Track the feedback object so `preferred_changed` can reach it.
                compositor::with_states(&surface, |states| {
                    states
                        .data_map
                        .insert_if_missing(ColorManagementSurfaceData::default);
                    let data = states.data_map.get::<ColorManagementSurfaceData>().unwrap();
                    data.feedbacks.lock().unwrap().push(feedback);
                });
            }
            Request::CreateParametricCreator { obj } => {
                data_init.init(obj, Mutex::new(ImageDescriptionBuilder::default()));
            }
            Request::CreateIccCreator { .. } => {
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "ICC image descriptions are not supported",
                );
            }
            Request::CreateWindowsScrgb { .. } => {
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "Windows scRGB image descriptions are not supported",
                );
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch2<WpColorManagementOutputV1, D> for WlOutput
where
    D: Dispatch<WpImageDescriptionV1, ImageDescriptionData>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        _resource: &WpColorManagementOutputV1,
        request: <WpColorManagementOutputV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_output_v1::Request;
        match request {
            Request::GetImageDescription { image_description } => {
                let desc = Output::from_resource(self)
                    .map(|output| state.description_for_output(&output))
                    .unwrap_or(ImageDescription::SRGB);
                make_ready_description(state, image_description, desc, data_init);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch2<WpColorManagementSurfaceV1, D> for Weak<WlSurface>
where
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &WpColorManagementSurfaceV1,
        request: <WpColorManagementSurfaceV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_surface_v1::Request;
        match request {
            Request::SetImageDescription {
                image_description,
                render_intent,
            } => {
                let Ok(surface) = self.upgrade() else {
                    resource.post_error(
                        wp_color_management_surface_v1::Error::Inert,
                        "the underlying wl_surface was destroyed",
                    );
                    return;
                };

                let render_intent = match render_intent.into_result() {
                    Ok(intent) if state.color_management_state().supported_intents.contains(&intent) => {
                        intent
                    }
                    _ => {
                        resource.post_error(
                            wp_color_management_surface_v1::Error::RenderIntent,
                            "unsupported rendering intent",
                        );
                        return;
                    }
                };

                let Some(desc) = image_description.data::<ImageDescriptionData>().map(|d| d.desc) else {
                    resource.post_error(
                        wp_color_management_surface_v1::Error::ImageDescription,
                        "image description is not ready",
                    );
                    return;
                };

                if set_pending_description(&surface, Some(desc), render_intent) {
                    if desc.is_hdr() {
                        debug!(surface = ?surface.id(), ?desc, "client attached an HDR image description");
                    } else {
                        trace!(surface = ?surface.id(), ?desc, "client attached an image description");
                    }
                    state.image_description_changed(&surface);
                }
            }
            Request::UnsetImageDescription => {
                let Ok(surface) = self.upgrade() else {
                    resource.post_error(
                        wp_color_management_surface_v1::Error::Inert,
                        "the underlying wl_surface was destroyed",
                    );
                    return;
                };
                if set_pending_description(&surface, None, RenderIntent::Perceptual) {
                    state.image_description_changed(&surface);
                }
            }
            Request::Destroy => {}
            _ => {}
        }
    }

    fn destroyed(
        &self,
        state: &mut D,
        _client: wayland_server::backend::ClientId,
        _resource: &WpColorManagementSurfaceV1,
    ) {
        // Destroying the object does the same as unset_image_description, and allows
        // attaching a new wp_color_management_surface_v1 to the surface.
        if let Ok(surface) = self.upgrade() {
            let changed = compositor::with_states(&surface, |states| {
                if let Some(data) = states.data_map.get::<ColorManagementSurfaceData>() {
                    *data.attached.lock().unwrap() = false;
                }
                let mut guard = states.cached_state.get::<ColorManagementSurfaceCachedState>();
                let pending = guard.pending();
                let changed = pending.description.is_some();
                *pending = ColorManagementSurfaceCachedState::default();
                changed
            });
            if changed {
                state.image_description_changed(&surface);
            }
        }
    }
}

/// Stores a new pending image description on the surface. Returns whether the pending value
/// actually changed — clients (e.g. mpv) re-attach the same description every frame, and
/// callers only want to react/log on real changes.
fn set_pending_description(
    surface: &WlSurface,
    description: Option<ImageDescription>,
    render_intent: RenderIntent,
) -> bool {
    compositor::with_states(surface, |states| {
        let mut guard = states.cached_state.get::<ColorManagementSurfaceCachedState>();
        let pending = guard.pending();
        let new = ColorManagementSurfaceCachedState {
            description,
            render_intent,
        };
        if *pending == new {
            false
        } else {
            *pending = new;
            true
        }
    })
}

impl<D> Dispatch2<WpColorManagementSurfaceFeedbackV1, D> for Weak<WlSurface>
where
    D: Dispatch<WpImageDescriptionV1, ImageDescriptionData>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &WpColorManagementSurfaceFeedbackV1,
        request: <WpColorManagementSurfaceFeedbackV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_surface_feedback_v1::Request;
        match request {
            Request::GetPreferred { image_description }
            | Request::GetPreferredParametric { image_description } => {
                let Ok(surface) = self.upgrade() else {
                    resource.post_error(
                        wp_color_management_surface_feedback_v1::Error::Inert,
                        "the underlying wl_surface was destroyed",
                    );
                    return;
                };
                let desc = state.preferred_description_for_surface(&surface);
                make_ready_description(state, image_description, desc, data_init);
            }
            Request::Destroy => {}
            _ => {}
        }
    }

    fn destroyed(
        &self,
        _state: &mut D,
        _client: wayland_server::backend::ClientId,
        resource: &WpColorManagementSurfaceFeedbackV1,
    ) {
        if let Ok(surface) = self.upgrade() {
            compositor::with_states(&surface, |states| {
                if let Some(data) = states.data_map.get::<ColorManagementSurfaceData>() {
                    data.feedbacks.lock().unwrap().retain(|f| f != resource);
                }
            });
        }
    }
}

impl<D> Dispatch2<WpImageDescriptionCreatorParamsV1, D> for Mutex<ImageDescriptionBuilder>
where
    D: Dispatch<WpImageDescriptionV1, ImageDescriptionData>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &WpImageDescriptionCreatorParamsV1,
        request: <WpImageDescriptionCreatorParamsV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_image_description_creator_params_v1::{Error, Request};
        match request {
            Request::SetTfNamed { tf } => {
                let mut params = self.lock().unwrap();
                if params.transfer.is_some() {
                    resource.post_error(Error::AlreadySet, "transfer function already set");
                    return;
                }
                match tf
                    .into_result()
                    .ok()
                    .filter(|tf| state.color_management_state().supported_tfs.contains(tf))
                {
                    Some(tf) => params.transfer = Some(tf),
                    None => resource.post_error(Error::InvalidTf, "unsupported transfer function"),
                }
            }
            Request::SetPrimariesNamed { primaries } => {
                let mut params = self.lock().unwrap();
                if params.primaries.is_some() {
                    resource.post_error(Error::AlreadySet, "primaries already set");
                    return;
                }
                match primaries
                    .into_result()
                    .ok()
                    .filter(|p| state.color_management_state().supported_primaries.contains(p))
                {
                    Some(p) => params.primaries = Some(p),
                    None => resource.post_error(Error::InvalidPrimariesNamed, "unsupported primaries"),
                }
            }
            Request::SetMasteringLuminance { min_lum, max_lum } => {
                self.lock().unwrap().mastering_luminance = Some((min_lum, max_lum));
            }
            Request::SetMaxCll { max_cll } => {
                self.lock().unwrap().max_cll = Some(max_cll);
            }
            Request::SetMaxFall { max_fall } => {
                self.lock().unwrap().max_fall = Some(max_fall);
            }
            Request::SetLuminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                if !state
                    .color_management_state()
                    .supported_features
                    .contains(&Feature::SetLuminances)
                {
                    resource.post_error(Error::UnsupportedFeature, "set_luminances is not supported");
                    return;
                }
                self.lock().unwrap().luminances = Some((min_lum, max_lum, reference_lum));
            }
            Request::SetMasteringDisplayPrimaries { .. } => {
                // Accepted (if advertised) so HDR clients can convey mastering metadata
                // without erroring; the values are not used yet.
                if !state
                    .color_management_state()
                    .supported_features
                    .contains(&Feature::SetMasteringDisplayPrimaries)
                {
                    resource.post_error(
                        Error::UnsupportedFeature,
                        "set_mastering_display_primaries is not supported",
                    );
                }
            }
            Request::SetTfPower { .. } => {
                resource.post_error(Error::UnsupportedFeature, "set_tf_power is not supported");
            }
            Request::SetPrimaries { .. } => {
                resource.post_error(Error::UnsupportedFeature, "set_primaries is not supported");
            }
            Request::Create { image_description } => {
                let params = self.lock().unwrap();
                let (Some(transfer), Some(primaries)) = (params.transfer, params.primaries) else {
                    resource.post_error(
                        Error::IncompleteSet,
                        "transfer function and primaries are both required",
                    );
                    return;
                };
                let desc = ImageDescription {
                    transfer,
                    primaries,
                    max_cll: params.max_cll,
                    max_fall: params.max_fall,
                    mastering_luminance: params.mastering_luminance,
                    luminances: params.luminances,
                };
                drop(params);
                make_ready_description(state, image_description, desc, data_init);
            }
            _ => {}
        }
    }
}

impl<D> Dispatch2<WpImageDescriptionV1, D> for ImageDescriptionData
where
    D: Dispatch<WpImageDescriptionInfoV1, ()>,
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        _resource: &WpImageDescriptionV1,
        request: <WpImageDescriptionV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_image_description_v1::Request;
        match request {
            Request::GetInformation { information } => {
                // The actual events (ending in the destructor `done`) are sent deferred —
                // see the handler doc.
                let info = data_init.init(information, ());
                state.schedule_image_description_info(info, self.desc);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch2<WpImageDescriptionInfoV1, D> for ()
where
    D: ColorManagementHandler,
    D: 'static,
{
    fn request(
        &self,
        _state: &mut D,
        _client: &Client,
        _resource: &WpImageDescriptionInfoV1,
        _request: <WpImageDescriptionInfoV1 as Resource>::Request,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        // wp_image_description_info_v1 has no requests.
    }
}

/// Initializes a `wp_image_description_v1` carrying `desc` and immediately marks it ready.
fn make_ready_description<D>(
    state: &mut D,
    image_description: New<WpImageDescriptionV1>,
    desc: ImageDescription,
    data_init: &mut DataInit<'_, D>,
) where
    D: Dispatch<WpImageDescriptionV1, ImageDescriptionData> + ColorManagementHandler + 'static,
{
    let identity = state.color_management_state().identity_for(desc);
    let image = data_init.init(image_description, ImageDescriptionData { desc });
    image.ready(identity);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_stable_per_description() {
        let mut state = ColorManagementState {
            supported_tfs: Vec::new(),
            supported_primaries: Vec::new(),
            supported_features: Vec::new(),
            supported_intents: Vec::new(),
            identities: Vec::new(),
            output_objects: Vec::new(),
        };

        let srgb = ImageDescription::SRGB;
        let pq = ImageDescription {
            transfer: TransferFunction::St2084Pq,
            primaries: Primaries::Bt2020,
            max_cll: Some(800),
            max_fall: Some(400),
            mastering_luminance: None,
            luminances: None,
        };

        let a = state.identity_for(srgb);
        let b = state.identity_for(pq);
        assert_ne!(a, b);
        assert_ne!(a, 0, "identity 0 is reserved by the protocol");
        assert_ne!(b, 0);
        // The same description always maps to the same identity.
        assert_eq!(state.identity_for(srgb), a);
        assert_eq!(state.identity_for(pq), b);
        // A description differing only in metadata gets its own identity.
        let pq_brighter = ImageDescription {
            max_cll: Some(1000),
            ..pq
        };
        let c = state.identity_for(pq_brighter);
        assert_ne!(c, b);
        assert_eq!(state.identity_for(pq), b);
    }
}
