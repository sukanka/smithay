//! # EDID - Extended Display Identification Data
//!
//! This module is meant to help with extraction of EDID data from connectors
//!
//! ```no_run
//! # mod helpers { include!("./docs/doctest_helpers.rs"); };
//! # let drm_device: helpers::FakeDevice = todo!();
//! # let connector = todo!();
//! use smithay_drm_extras::display_info;
//!
//! let info = display_info::for_connector(&drm_device, connector).unwrap();
//!
//! println!("Monitor name: {:?}", info.model());
//! println!("Manufacturer name: {:?}", info.make());
//! ```

use drm::control::{Device as ControlDevice, connector};
use libdisplay_info::info::Info;

mod dolby_vision;
pub use dolby_vision::DolbyVisionCapabilities;

/// Try to read the [`Info`] from the connector EDID property
pub fn for_connector(device: &impl ControlDevice, connector: connector::Handle) -> Option<Info> {
    Info::parse_edid(&edid_for_connector(device, connector)?).ok()
}

/// Read the connector's Dolby Vision version 2 capabilities.
///
/// Returns `None` if the EDID is malformed or does not contain a supported Dolby
/// Vision block. This describes the sink, not the driver or active wire format.
pub fn get_dolby_vision_info(
    device: &impl ControlDevice,
    connector: connector::Handle,
) -> Option<DolbyVisionCapabilities> {
    DolbyVisionCapabilities::from_edid(&edid_for_connector(device, connector)?)
}

fn edid_for_connector(device: &impl ControlDevice, connector: connector::Handle) -> Option<Vec<u8>> {
    let props = device.get_properties(connector).ok()?;

    let (info, value) = props
        .into_iter()
        .filter_map(|(handle, value)| {
            let info = device.get_property(handle).ok()?;

            Some((info, value))
        })
        .find(|(info, _)| info.name().to_str() == Ok("EDID"))?;

    let blob = info.value_type().convert_value(value).as_blob()?;
    device.get_property_blob(blob).ok()
}
