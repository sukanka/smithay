//! Types for configuring the color pipeline of a connector.
//!
//! These types describe the wide-gamut / HDR signalling state of a DRM connector, i.e. the
//! values of the optional `Colorspace`, `HDR_OUTPUT_METADATA`, `NV_HDMI_VSIF_METADATA`
//! and `max bpc` connector
//! properties. They are used with [`DrmSurface::use_color_state`](super::DrmSurface::use_color_state)
//! (and the corresponding
//! [`DrmCompositor::use_color_state`](super::compositor::DrmCompositor::use_color_state)),
//! which stages the state so that it is applied as part of the *same* atomic commit as the
//! mode and plane state. Some drivers (notably nvidia) treat a `Colorspace` change as a full
//! modeset and misbehave when it is committed on its own, so smithay never issues standalone
//! connector-property commits for these.
//!
//! Whether a connector supports these properties can be queried with
//! [`DrmSurface::supported_colorspaces`](super::DrmSurface::supported_colorspaces),
//! [`DrmSurface::hdr_metadata_supported`](super::DrmSurface::hdr_metadata_supported),
//! [`DrmSurface::hdmi_vsif_supported`](super::DrmSurface::hdmi_vsif_supported) and
//! [`DrmSurface::max_bpc_range`](super::DrmSurface::max_bpc_range).
//!
//! Capabilities of the connected sink (whether it accepts a PQ EOTF, its desired luminance
//! range, BT.2020 signal support) should be read from its EDID, e.g. via
//! `smithay-drm-extras`' `display_info` module and libdisplay-info's
//! `Info::hdr_static_metadata()` / `Info::supported_signal_colorimetry()`.

/// Value of the `Colorspace` connector property.
///
/// This selects the colorimetry signalled to the sink in the AVI infoframe (HDMI) or MSA/SDP
/// (DisplayPort). It does not perform any color conversion; the submitted framebuffer contents
/// are expected to already be encoded in the signalled colorspace.
///
/// Only a portable subset of the kernel's colorspace values is exposed. The property's enum
/// values are driver-defined and are resolved by name at set time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Colorspace {
    /// The default colorspace of the connector (`Default`), typically sRGB/BT.709.
    #[default]
    Default,
    /// ITU-R BT.2020 RGB colorimetry (`BT2020_RGB`).
    Bt2020Rgb,
    /// ITU-R BT.2020 YCbCr colorimetry (`BT2020_YCC`).
    Bt2020Ycc,
    /// DCI-P3 RGB colorimetry with D65 white point (`DCI-P3_RGB_D65`).
    DciP3RgbD65,
    /// A colorspace not modelled by smithay.
    ///
    /// Only ever returned when reading back the current state of a connector; requesting it
    /// in [`use_color_state`](super::DrmSurface::use_color_state) fails.
    Unknown,
}

impl Colorspace {
    /// The kernel's name for this colorspace in the `Colorspace` property enum.
    ///
    /// Returns `None` for [`Colorspace::Unknown`].
    pub fn kernel_name(&self) -> Option<&'static str> {
        Some(match self {
            Colorspace::Default => "Default",
            Colorspace::Bt2020Rgb => "BT2020_RGB",
            Colorspace::Bt2020Ycc => "BT2020_YCC",
            Colorspace::DciP3RgbD65 => "DCI-P3_RGB_D65",
            Colorspace::Unknown => return None,
        })
    }

    pub(super) fn from_kernel_name(name: &str) -> Option<Self> {
        Some(match name {
            "Default" => Colorspace::Default,
            "BT2020_RGB" => Colorspace::Bt2020Rgb,
            "BT2020_YCC" => Colorspace::Bt2020Ycc,
            "DCI-P3_RGB_D65" => Colorspace::DciP3RgbD65,
            _ => return None,
        })
    }
}

/// Electro-optical transfer function of the HDR metadata infoframe, per CTA-861.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Eotf {
    /// Traditional gamma, SDR luminance range.
    TraditionalSdr,
    /// Traditional gamma, HDR luminance range.
    TraditionalHdr,
    /// SMPTE ST 2084, a.k.a. Perceptual Quantizer (PQ).
    SmpteSt2084,
    /// Hybrid Log-Gamma (HLG), per BT.2100.
    Hlg,
}

impl Eotf {
    fn to_raw(self) -> u8 {
        match self {
            Eotf::TraditionalSdr => 0,
            Eotf::TraditionalHdr => 1,
            Eotf::SmpteSt2084 => 2,
            Eotf::Hlg => 3,
        }
    }

    fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => Eotf::TraditionalSdr,
            1 => Eotf::TraditionalHdr,
            2 => Eotf::SmpteSt2084,
            3 => Eotf::Hlg,
            _ => return None,
        })
    }
}

/// A CIE 1931 xy chromaticity coordinate in CTA-861.3 units, i.e. the floating-point
/// coordinate scaled by 50000 (increments of 0.00002).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CtaCoordinate {
    /// The x coordinate, scaled by 50000.
    pub x: u16,
    /// The y coordinate, scaled by 50000.
    pub y: u16,
}

impl CtaCoordinate {
    /// Converts a floating point CIE 1931 xy coordinate into CTA-861.3 units.
    pub fn from_xy(x: f64, y: f64) -> Self {
        Self {
            x: (x * 50000.0).round() as u16,
            y: (y * 50000.0).round() as u16,
        }
    }

    /// The BT.2020 red primary (0.708, 0.292).
    pub const BT2020_RED: Self = Self { x: 35400, y: 14600 };
    /// The BT.2020 green primary (0.170, 0.797).
    pub const BT2020_GREEN: Self = Self { x: 8500, y: 39850 };
    /// The BT.2020 blue primary (0.131, 0.046).
    pub const BT2020_BLUE: Self = Self { x: 6550, y: 2300 };
    /// The D65 white point (0.3127, 0.3290).
    pub const D65_WHITE: Self = Self { x: 15635, y: 16450 };
}

/// Static HDR metadata (CTA-861.3 Static Metadata Type 1) for the `HDR_OUTPUT_METADATA`
/// connector property.
///
/// This describes the mastering display and content light levels of the submitted frames to
/// the sink. All fields use the raw infoframe units, so no rounding is hidden from callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HdrOutputMetadata {
    /// The electro-optical transfer function the frame contents are encoded with.
    pub eotf: Eotf,
    /// Chromaticity of the mastering display's red, green and blue primaries.
    pub display_primaries: [CtaCoordinate; 3],
    /// Chromaticity of the mastering display's white point.
    pub white_point: CtaCoordinate,
    /// Maximum luminance of the mastering display, in cd/m².
    pub max_display_mastering_luminance: u16,
    /// Minimum luminance of the mastering display, in 0.0001 cd/m² units.
    pub min_display_mastering_luminance: u16,
    /// Maximum content light level, in cd/m².
    pub max_cll: u16,
    /// Maximum frame-average light level, in cd/m².
    pub max_fall: u16,
}

impl HdrOutputMetadata {
    /// Convenience constructor for the most common HDR10-style signal: PQ transfer function
    /// with BT.2020 mastering primaries and a D65 white point.
    ///
    /// Luminance values should be clamped to what the sink advertises in its EDID HDR static
    /// metadata block.
    pub fn pq_bt2020(max_luminance: u16, min_luminance: u16, max_cll: u16, max_fall: u16) -> Self {
        Self {
            eotf: Eotf::SmpteSt2084,
            display_primaries: [
                CtaCoordinate::BT2020_RED,
                CtaCoordinate::BT2020_GREEN,
                CtaCoordinate::BT2020_BLUE,
            ],
            white_point: CtaCoordinate::D65_WHITE,
            max_display_mastering_luminance: max_luminance,
            min_display_mastering_luminance: min_luminance,
            max_cll,
            max_fall,
        }
    }
}

/// HDMI vendor-specific InfoFrame payload, including its three-byte IEEE OUI.
///
/// This is sent through NVIDIA's `NV_HDMI_VSIF_METADATA` connector property. The driver
/// supplies the InfoFrame header and checksum; they must not be included here. Payloads
/// contain 3 to 27 bytes, as required by that property's uAPI.
///
/// Setting an InfoFrame only changes signalling. The caller must separately produce pixels
/// in the corresponding format and verify the connected display's capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HdmiVsifMetadata {
    payload: [u8; 27],
    len: u8,
}

impl HdmiVsifMetadata {
    /// Copies a payload, returning `None` if its length is outside 3..=27 bytes.
    pub fn new(bytes: &[u8]) -> Option<Self> {
        if !(3..=27).contains(&bytes.len()) {
            return None;
        }
        let mut payload = [0; 27];
        payload[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            payload,
            len: bytes.len() as u8,
        })
    }

    /// Returns exactly the bytes sent to the driver, excluding unused storage.
    pub fn as_bytes(&self) -> &[u8] {
        &self.payload[..usize::from(self.len)]
    }

    /// Builds an HDMI Licensing VSIF announcing no additional video format.
    ///
    /// This is the standard four-byte payload produced by Linux's
    /// `hdmi_vendor_infoframe_pack()` with no HDMI VIC or 3D structure. It can be used
    /// temporarily on an SDR output to synchronize a driver's cached VSIF state.
    ///
    /// This sends a nonempty InfoFrame; it is not equivalent to clearing the connector
    /// property. Do not pair it with HDR10 static metadata: some hardware shares one slot
    /// between these InfoFrames. Finish synchronization by setting
    /// [`ConnectorColorState::hdmi_vsif`] to `None` before or together with enabling HDR10.
    pub const fn hdmi_no_additional_video_format() -> Self {
        let mut payload = [0; 27];
        payload[0] = 0x03;
        payload[1] = 0x0c;
        Self { payload, len: 4 }
    }

    /// Builds a minimal low-latency Dolby Vision signalling payload.
    ///
    /// The Dolby OUI is encoded least-significant byte first (`46 d0 00`); the next byte
    /// sets the low-latency and Dolby Vision signal flags. The remaining bytes are zero.
    /// This does not encode Dolby Vision RPU data, perform display mapping or convert pixels.
    pub const fn low_latency_dolby_vision() -> Self {
        let mut payload = [0; 27];
        payload[0] = 0x46;
        payload[1] = 0xd0;
        payload[3] = 0x03;
        Self { payload, len: 27 }
    }
}

/// Desired color pipeline configuration of a connector.
///
/// The default value describes plain SDR signalling: default colorspace, no HDR metadata or
/// vendor-specific InfoFrame, and the `max bpc` property left untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ConnectorColorState {
    /// The colorimetry to signal via the `Colorspace` property.
    pub colorspace: Colorspace,
    /// The HDR static metadata to signal via the `HDR_OUTPUT_METADATA` property.
    ///
    /// `None` disables the HDR infoframe (the property is set to no blob).
    pub hdr_metadata: Option<HdrOutputMetadata>,
    /// The HDMI vendor-specific InfoFrame sent via `NV_HDMI_VSIF_METADATA`.
    ///
    /// `None` clears this property on connectors exposing it. This does not affect vendor
    /// InfoFrames managed internally by the driver, such as the HDMI Forum VSIF.
    pub hdmi_vsif: Option<HdmiVsifMetadata>,
    /// The maximum bits per component to allow on the link via the `max bpc` property.
    ///
    /// `None` leaves the property at its current value.
    pub max_bpc: Option<u32>,
}

pub(super) mod ffi {
    //! Binary layout of the kernel's `HDR_OUTPUT_METADATA` blob
    //! (`struct hdr_output_metadata` in `include/uapi/drm/drm_mode.h`).

    /// Static Metadata Type 1 descriptor id (CTA-861.3).
    const HDMI_STATIC_METADATA_TYPE1: u32 = 0;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrColorPoint {
        pub x: u16,
        pub y: u16,
    }

    /// `struct hdr_metadata_infoframe`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrMetadataInfoframe {
        pub eotf: u8,
        pub metadata_type: u8,
        pub display_primaries: [HdrColorPoint; 3],
        pub white_point: HdrColorPoint,
        pub max_display_mastering_luminance: u16,
        pub min_display_mastering_luminance: u16,
        pub max_cll: u16,
        pub max_fall: u16,
    }

    /// `struct hdr_output_metadata`. Passed to the kernel verbatim as the blob contents,
    /// so the layout must match exactly.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrOutputMetadata {
        pub metadata_type: u32,
        pub hdmi_metadata_type1: HdrMetadataInfoframe,
        // The C ABI ends with two padding bytes. Make them actual initialized fields so
        // create_property_blob never reads Rust padding, and equal metadata has equal bytes.
        _reserved: [u8; 2],
    }

    // All 32 bytes are copied to the kernel. These sizes and offsets prove that the
    // integer fields plus explicit reserved bytes cover the layout without hidden padding.
    const _: () = {
        assert!(std::mem::size_of::<HdrColorPoint>() == 4);
        assert!(std::mem::size_of::<HdrMetadataInfoframe>() == 26);
        assert!(std::mem::offset_of!(HdrOutputMetadata, hdmi_metadata_type1) == 4);
        assert!(std::mem::offset_of!(HdrOutputMetadata, _reserved) == 30);
        assert!(std::mem::size_of::<HdrOutputMetadata>() == 32);
    };

    impl From<super::HdrOutputMetadata> for HdrOutputMetadata {
        fn from(meta: super::HdrOutputMetadata) -> Self {
            let coord = |c: super::CtaCoordinate| HdrColorPoint { x: c.x, y: c.y };
            HdrOutputMetadata {
                metadata_type: HDMI_STATIC_METADATA_TYPE1,
                hdmi_metadata_type1: HdrMetadataInfoframe {
                    eotf: meta.eotf.to_raw(),
                    metadata_type: HDMI_STATIC_METADATA_TYPE1 as u8,
                    display_primaries: [
                        coord(meta.display_primaries[0]),
                        coord(meta.display_primaries[1]),
                        coord(meta.display_primaries[2]),
                    ],
                    white_point: coord(meta.white_point),
                    max_display_mastering_luminance: meta.max_display_mastering_luminance,
                    min_display_mastering_luminance: meta.min_display_mastering_luminance,
                    max_cll: meta.max_cll,
                    max_fall: meta.max_fall,
                },
                _reserved: [0; 2],
            }
        }
    }

    impl HdrOutputMetadata {
        /// Parses the contents of an `HDR_OUTPUT_METADATA` blob.
        ///
        /// Returns `None` if the blob is too short, describes a different metadata type or
        /// uses an EOTF unknown to us.
        pub fn parse(bytes: &[u8]) -> Option<super::HdrOutputMetadata> {
            // The infoframe is valid without the trailing struct padding, so only require the
            // payload itself.
            const PAYLOAD_LEN: usize = 30;
            if bytes.len() < PAYLOAD_LEN {
                return None;
            }
            let u16_at = |off: usize| u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            let coord_at = |off: usize| super::CtaCoordinate {
                x: u16_at(off),
                y: u16_at(off + 2),
            };

            let metadata_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if metadata_type != HDMI_STATIC_METADATA_TYPE1 {
                return None;
            }

            Some(super::HdrOutputMetadata {
                eotf: super::Eotf::from_raw(bytes[4])?,
                display_primaries: [coord_at(6), coord_at(10), coord_at(14)],
                white_point: coord_at(18),
                max_display_mastering_luminance: u16_at(22),
                min_display_mastering_luminance: u16_at(24),
                max_cll: u16_at(26),
                max_fall: u16_at(28),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdmi_vsif_payload_boundaries() {
        for length in 0..=28 {
            let bytes = vec![0x46; length];
            let parsed = HdmiVsifMetadata::new(&bytes);
            assert_eq!(parsed.is_some(), (3..=27).contains(&length));
            if let Some(parsed) = parsed {
                assert_eq!(parsed.as_bytes(), bytes);
                assert_eq!(HdmiVsifMetadata::new(parsed.as_bytes()), Some(parsed));
            }
        }
    }

    #[test]
    fn low_latency_dolby_vision_payload() {
        // The payload used by the NVIDIA HDMI signalling probe: no header/checksum bytes.
        let metadata = HdmiVsifMetadata::low_latency_dolby_vision();
        assert_eq!(metadata.as_bytes().len(), 27);
        assert_eq!(&metadata.as_bytes()[..4], &[0x46, 0xd0, 0x00, 0x03]);
        assert_eq!(&metadata.as_bytes()[4..], &[0; 23]);
        assert_eq!(HdmiVsifMetadata::new(metadata.as_bytes()), Some(metadata));
        assert_eq!(ConnectorColorState::default().hdmi_vsif, None);
    }

    #[test]
    fn hdmi_no_additional_video_format_payload() {
        let metadata = HdmiVsifMetadata::hdmi_no_additional_video_format();
        // HDMI Licensing OUI, LSB first, followed by HDMI_Video_Format = 0.
        // No InfoFrame header, checksum, HDMI VIC or 3D fields belong in this blob.
        assert_eq!(metadata.as_bytes(), &[0x03, 0x0c, 0x00, 0x00]);
        assert_eq!(HdmiVsifMetadata::new(metadata.as_bytes()), Some(metadata));
        assert_ne!(metadata, HdmiVsifMetadata::low_latency_dolby_vision());
    }

    #[test]
    fn hdr_output_metadata_layout() {
        // The blob is passed to the kernel verbatim, so the layout must match
        // `struct hdr_output_metadata` exactly: 4-byte type + 26-byte infoframe, padded to 32.
        assert_eq!(std::mem::size_of::<ffi::HdrOutputMetadata>(), 32);
        assert_eq!(std::mem::offset_of!(ffi::HdrOutputMetadata, metadata_type), 0);
        assert_eq!(
            std::mem::offset_of!(ffi::HdrOutputMetadata, hdmi_metadata_type1),
            4
        );
        assert_eq!(std::mem::size_of::<ffi::HdrColorPoint>(), 4);
        assert_eq!(std::mem::offset_of!(ffi::HdrColorPoint, x), 0);
        assert_eq!(std::mem::offset_of!(ffi::HdrColorPoint, y), 2);
        assert_eq!(std::mem::size_of::<ffi::HdrMetadataInfoframe>(), 26);
        assert_eq!(std::mem::offset_of!(ffi::HdrMetadataInfoframe, eotf), 0);
        assert_eq!(std::mem::offset_of!(ffi::HdrMetadataInfoframe, metadata_type), 1);
        assert_eq!(
            std::mem::offset_of!(ffi::HdrMetadataInfoframe, display_primaries),
            2
        );
        assert_eq!(std::mem::offset_of!(ffi::HdrMetadataInfoframe, white_point), 14);
        assert_eq!(
            std::mem::offset_of!(ffi::HdrMetadataInfoframe, max_display_mastering_luminance),
            18
        );
        assert_eq!(
            std::mem::offset_of!(ffi::HdrMetadataInfoframe, min_display_mastering_luminance),
            20
        );
        assert_eq!(std::mem::offset_of!(ffi::HdrMetadataInfoframe, max_cll), 22);
        assert_eq!(std::mem::offset_of!(ffi::HdrMetadataInfoframe, max_fall), 24);

        let meta: ffi::HdrOutputMetadata = HdrOutputMetadata::pq_bt2020(500, 50, 400, 300).into();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const ffi::HdrOutputMetadata as *const u8,
                std::mem::size_of::<ffi::HdrOutputMetadata>(),
            )
        };

        // Includes the ABI's two trailing reserved bytes. No compiler padding is read.
        assert_eq!(
            bytes,
            &[
                0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x48, 0x8a, 0x08, 0x39, 0x34, 0x21, 0xaa, 0x9b, 0x96,
                0x19, 0xfc, 0x08, 0x13, 0x3d, 0x42, 0x40, 0xf4, 0x01, 0x32, 0x00, 0x90, 0x01, 0x2c, 0x01,
                0x00, 0x00,
            ]
        );

        // metadata_type: u32 = HDMI_STATIC_METADATA_TYPE1 (0)
        assert_eq!(&bytes[0..4], &0u32.to_le_bytes());
        // infoframe.eotf = SMPTE ST 2084 (PQ) = 2
        assert_eq!(bytes[4], 2);
        // infoframe.metadata_type = 0 (Static Metadata Type 1)
        assert_eq!(bytes[5], 0);
        // display_primaries[0] (red) = BT.2020 (0.708, 0.292) * 50000
        assert_eq!(&bytes[6..8], &35400u16.to_le_bytes());
        assert_eq!(&bytes[8..10], &14600u16.to_le_bytes());
        // display_primaries[1] (green) = (0.170, 0.797) * 50000
        assert_eq!(&bytes[10..12], &8500u16.to_le_bytes());
        assert_eq!(&bytes[12..14], &39850u16.to_le_bytes());
        // display_primaries[2] (blue) = (0.131, 0.046) * 50000
        assert_eq!(&bytes[14..16], &6550u16.to_le_bytes());
        assert_eq!(&bytes[16..18], &2300u16.to_le_bytes());
        // white_point = D65 (0.3127, 0.3290) * 50000
        assert_eq!(&bytes[18..20], &15635u16.to_le_bytes());
        assert_eq!(&bytes[20..22], &16450u16.to_le_bytes());
        // luminances and light levels
        assert_eq!(&bytes[22..24], &500u16.to_le_bytes());
        assert_eq!(&bytes[24..26], &50u16.to_le_bytes());
        assert_eq!(&bytes[26..28], &400u16.to_le_bytes());
        assert_eq!(&bytes[28..30], &300u16.to_le_bytes());
    }

    #[test]
    fn hdr_output_metadata_roundtrip() {
        let meta = HdrOutputMetadata::pq_bt2020(1000, 1, 800, 400);
        let raw: ffi::HdrOutputMetadata = meta.into();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &raw as *const ffi::HdrOutputMetadata as *const u8,
                std::mem::size_of::<ffi::HdrOutputMetadata>(),
            )
        };
        assert_eq!(ffi::HdrOutputMetadata::parse(bytes), Some(meta));
        assert_eq!(ffi::HdrOutputMetadata::parse(&bytes[..30]), Some(meta));
        assert_eq!(ffi::HdrOutputMetadata::parse(&bytes[..29]), None);
        let mut with_foreign_padding = bytes.to_vec();
        with_foreign_padding[30..].copy_from_slice(&[0xaa, 0xbb]);
        assert_eq!(ffi::HdrOutputMetadata::parse(&with_foreign_padding), Some(meta));
    }

    #[test]
    fn cta_coordinate_from_xy() {
        assert_eq!(CtaCoordinate::from_xy(0.708, 0.292), CtaCoordinate::BT2020_RED);
        assert_eq!(CtaCoordinate::from_xy(0.3127, 0.3290), CtaCoordinate::D65_WHITE);
    }

    #[test]
    fn colorspace_names_roundtrip() {
        for cs in [
            Colorspace::Default,
            Colorspace::Bt2020Rgb,
            Colorspace::Bt2020Ycc,
            Colorspace::DciP3RgbD65,
        ] {
            assert_eq!(Colorspace::from_kernel_name(cs.kernel_name().unwrap()), Some(cs));
        }
        assert_eq!(Colorspace::Unknown.kernel_name(), None);
    }
}
