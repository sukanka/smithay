//! Version 2 Dolby Vision CTA vendor-specific video data blocks.
//!
//! Keep this small parser independent of libdisplay-info 0.3's Dolby API so that
//! querying these capabilities does not raise the system library requirement.
//! The field encodings follow edid-decode's `cta_dolby_video`:
//! https://git.linuxtv.org/v4l-utils.git/tree/utils/edid-decode/parse-cta-block.cpp

/// Sink capabilities advertised by a version 2 Dolby Vision EDID block.
///
/// These are sink capabilities only. They do not prove that the GPU can send a
/// given wire format, or that a composited image has been mapped for Dolby Vision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DolbyVisionCapabilities {
    /// Dolby Vision vendor-specific video data block version (currently 2).
    pub version: u8,
    /// Supports low-latency Dolby Vision.
    pub supports_low_latency: bool,
    /// Supports standard Dolby Vision in addition to low-latency Dolby Vision.
    pub supports_standard: bool,
    /// Supports the low-latency HDMI interface.
    pub supports_low_latency_hdmi: bool,
    /// Advertises 10-bit 4:4:4 support. A 12-bit declaration does not imply this.
    pub supports_444_10bit: bool,
    /// Advertises 12-bit 4:4:4 support.
    pub supports_444_12bit: bool,
    /// Advertises 12-bit YCbCr 4:2:2 support.
    pub ycbcr422_12bit: bool,
    /// Minimum target luminance in cd/m² (nits), decoded from the 12-bit PQ value.
    pub target_min_luminance: f64,
    /// Maximum target luminance in cd/m² (nits), decoded from the 12-bit PQ value.
    pub target_max_luminance: f64,
    /// CIE 1931 xy chromaticity coordinates of the red, green and blue primaries.
    pub primaries: [[f64; 2]; 3],
}

impl DolbyVisionCapabilities {
    /// Extract Dolby Vision version 2 capabilities from a complete EDID blob.
    ///
    /// Validates the header, advertised length, all block checksums and CTA data
    /// block bounds. Unsupported Dolby block versions, malformed data, and
    /// duplicate version 2 Dolby blocks return `None`. This parser does not
    /// interpret HDMI Forum extension-count overrides or DisplayID containers.
    pub fn from_edid(data: &[u8]) -> Option<Self> {
        const HEADER: [u8; 8] = [0, 255, 255, 255, 255, 255, 255, 0];
        const BLOCK_SIZE: usize = 128;
        if data.len() < BLOCK_SIZE || data.len() > 256 * BLOCK_SIZE || data[..8] != HEADER {
            return None;
        }
        if data.len() != (usize::from(data[126]) + 1) * BLOCK_SIZE {
            return None;
        }
        if data
            .chunks_exact(BLOCK_SIZE)
            .any(|block| block.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) != 0)
        {
            return None;
        }

        let mut capabilities = None;
        for extension in data[BLOCK_SIZE..].chunks_exact(BLOCK_SIZE) {
            // CTA revision 3 introduced the extended data block tags.
            if extension[0] != 0x02 || extension[1] < 3 || extension[2] == 0 {
                continue;
            }
            let end = usize::from(extension[2]);
            if !(4..=127).contains(&end) {
                return None;
            }
            let mut offset = 4;
            while offset < end {
                let tag = extension[offset] >> 5;
                let length = usize::from(extension[offset] & 0x1f);
                let next = offset + 1 + length;
                if next > end {
                    return None;
                }
                let block = &extension[offset + 1..next];
                // Extended vendor-specific video tag, then Dolby's little-endian OUI.
                if tag == 7 && block.starts_with(&[0x01, 0x46, 0xd0, 0x00]) {
                    let payload = &block[4..];
                    let version = payload.first()? >> 5;
                    if version == 2 {
                        if capabilities.is_some() {
                            return None;
                        }
                        capabilities = Some(Self::from_v2_payload(payload)?);
                    }
                }
                offset = next;
            }
        }
        capabilities
    }

    fn from_v2_payload(payload: &[u8]) -> Option<Self> {
        let x = payload.get(..7)?;
        let interface = x[2] & 3;
        let support_444 = ((x[3] & 1) << 1) | (x[4] & 1);
        let primaries = [
            [
                0.625 + f64::from(x[5] >> 3) / 256.0,
                0.25 + f64::from(x[6] >> 3) / 256.0,
            ],
            [f64::from(x[3] >> 1) / 256.0, 0.5 + f64::from(x[4] >> 1) / 256.0],
            [
                0.125 + f64::from(x[5] & 7) / 256.0,
                0.03125 + f64::from(x[6] & 7) / 256.0,
            ],
        ];
        if primaries.iter().any(|[x, y]| *y <= 0.0 || x + y > 1.0) {
            return None;
        }
        Some(Self {
            version: 2,
            supports_low_latency: true,
            supports_standard: interface & 2 != 0,
            supports_low_latency_hdmi: interface & 1 != 0,
            // The fourth encoding is reserved; do not infer either bit depth.
            supports_444_10bit: support_444 == 1,
            supports_444_12bit: support_444 == 2,
            ycbcr422_12bit: x[0] & 1 != 0,
            target_min_luminance: pq_to_nits(20 * u16::from(x[1] >> 3)),
            target_max_luminance: pq_to_nits(2055 + 65 * u16::from(x[2] >> 3)),
            primaries,
        })
    }
}

fn pq_to_nits(code: u16) -> f64 {
    // SMPTE ST 2084 EOTF, normalized 12-bit code to absolute cd/m².
    let p = (f64::from(code) / 4095.0).powf(32.0 / 2523.0);
    let numerator = (p - 3424.0 / 4096.0).max(0.0);
    let denominator = 2413.0 / 128.0 - 2392.0 / 128.0 * p;
    10000.0 * (numerator / denominator).powf(16384.0 / 2610.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the public capability payload from the tested PG32UCDM3 is retained;
    // the synthetic base block contains no monitor identity or serial number.
    const DISPLAY_PAYLOAD: [u8; 7] = [0x4c, 0x0a, 0x65, 0x89, 0x62, 0x6e, 0x8f];

    fn checksum(block: &mut [u8]) {
        block[127] = 0u8.wrapping_sub(block[..127].iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)));
    }

    fn edid(payload: &[u8]) -> Vec<u8> {
        let mut edid = vec![0; 256];
        edid[..8].copy_from_slice(&[0, 255, 255, 255, 255, 255, 255, 0]);
        edid[18] = 1;
        edid[19] = 4;
        edid[126] = 1;
        checksum(&mut edid[..128]);
        let extension = &mut edid[128..];
        extension[0] = 2;
        extension[1] = 3;
        extension[2] = (9 + payload.len()) as u8;
        extension[4] = 0xe0 | (4 + payload.len()) as u8;
        extension[5..9].copy_from_slice(&[1, 0x46, 0xd0, 0]);
        extension[9..9 + payload.len()].copy_from_slice(payload);
        checksum(extension);
        edid
    }

    #[test]
    fn decodes_display_capabilities_and_physical_units() {
        let caps = DolbyVisionCapabilities::from_edid(&edid(&DISPLAY_PAYLOAD)).unwrap();
        assert_eq!(caps.version, 2);
        assert!(caps.supports_low_latency);
        assert!(caps.supports_low_latency_hdmi);
        assert!(!caps.supports_standard);
        assert!(caps.supports_444_12bit);
        assert!(!caps.supports_444_10bit);
        assert!(!caps.ycbcr422_12bit);
        assert!((caps.target_max_luminance - 578.3092796632134).abs() < 0.0001);
        assert!((caps.target_min_luminance - 0.0006366479051).abs() < 1e-12);
        assert_eq!(
            caps.primaries,
            [
                [0.67578125, 0.31640625],
                [0.265625, 0.69140625],
                [0.1484375, 0.05859375]
            ]
        );
    }

    #[test]
    fn handles_each_interface_and_444_encoding() {
        for interface in 0..4 {
            for support_444 in 0..4 {
                let mut payload = DISPLAY_PAYLOAD;
                payload[2] = (payload[2] & !3) | interface;
                payload[3] = (payload[3] & !1) | (support_444 >> 1);
                payload[4] = (payload[4] & !1) | (support_444 & 1);
                payload[0] |= 1;
                let caps = DolbyVisionCapabilities::from_edid(&edid(&payload)).unwrap();
                assert!(caps.supports_low_latency);
                assert_eq!(caps.supports_standard, interface & 2 != 0);
                assert_eq!(caps.supports_low_latency_hdmi, interface & 1 != 0);
                assert_eq!(caps.supports_444_10bit, support_444 == 1);
                assert_eq!(caps.supports_444_12bit, support_444 == 2);
                assert!(caps.ycbcr422_12bit);
            }
        }
    }

    #[test]
    fn rejects_unsupported_versions_and_truncated_payloads() {
        for version in [0, 1, 3, 4, 5, 6, 7] {
            let mut payload = DISPLAY_PAYLOAD;
            payload[0] = (payload[0] & 31) | (version << 5);
            assert!(DolbyVisionCapabilities::from_edid(&edid(&payload)).is_none());
        }
        for length in 0..DISPLAY_PAYLOAD.len() {
            assert!(DolbyVisionCapabilities::from_edid(&edid(&DISPLAY_PAYLOAD[..length])).is_none());
        }
    }

    #[test]
    fn rejects_truncated_oversized_and_bad_checksum_edids() {
        let valid = edid(&DISPLAY_PAYLOAD);
        for length in 0..valid.len() {
            assert!(DolbyVisionCapabilities::from_edid(&valid[..length]).is_none());
        }
        let mut damaged = valid.clone();
        damaged[0] = 1;
        checksum(&mut damaged[..128]);
        assert!(DolbyVisionCapabilities::from_edid(&damaged).is_none());
        for offset in [20, 200] {
            let mut damaged = valid.clone();
            damaged[offset] ^= 1;
            assert!(DolbyVisionCapabilities::from_edid(&damaged).is_none());
        }
        let mut damaged = valid.clone();
        damaged[126] = 2;
        checksum(&mut damaged[..128]);
        assert!(DolbyVisionCapabilities::from_edid(&damaged).is_none());
        let mut damaged = valid;
        damaged.resize(257 * 128, 0);
        assert!(DolbyVisionCapabilities::from_edid(&damaged).is_none());
    }

    #[test]
    fn rejects_invalid_cta_bounds_and_oui_matches_outside_video_blocks() {
        for end in [1, 2, 3, 5, 15, 128, 255] {
            let mut data = edid(&DISPLAY_PAYLOAD);
            data[130] = end;
            checksum(&mut data[128..]);
            assert!(DolbyVisionCapabilities::from_edid(&data).is_none());
        }
        for (index, value) in [
            (128, 0x70),
            (129, 2),
            (130, 0),
            (132, 0x6b),
            (133, 0),
            (134, 0x45),
        ] {
            let mut data = edid(&DISPLAY_PAYLOAD);
            data[index] = value;
            checksum(&mut data[128..]);
            assert!(DolbyVisionCapabilities::from_edid(&data).is_none());
        }
    }

    #[test]
    fn validates_blocks_after_a_match_and_rejects_duplicates() {
        let mut data = edid(&DISPLAY_PAYLOAD);
        data[130] = 17;
        data[144] = 0xff;
        checksum(&mut data[128..]);
        assert!(DolbyVisionCapabilities::from_edid(&data).is_none());

        let mut data = edid(&DISPLAY_PAYLOAD);
        data.extend_from_within(128..256);
        data[126] = 2;
        checksum(&mut data[..128]);
        assert!(DolbyVisionCapabilities::from_edid(&data).is_none());
    }

    #[test]
    fn rejects_nonphysical_chromaticities() {
        let mut payload = DISPLAY_PAYLOAD;
        payload[3] = 254;
        payload[4] = 254;
        assert!(DolbyVisionCapabilities::from_edid(&edid(&payload)).is_none());
    }
}
