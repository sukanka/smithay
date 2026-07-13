//! Discovery of per-plane color pipelines (`drm_colorop` objects).
//!
//! Kernels with the color pipeline API (Linux 6.19+) expose color hardware on a plane as an
//! optional `COLOR_PIPELINE` enum property, gated behind the
//! `DRM_CLIENT_CAP_PLANE_COLOR_PIPELINE` client capability. Each non-zero enum value is the
//! object id of the first `drm_colorop` in a pipeline; the colorops of a pipeline are chained
//! through their `NEXT` property and each describes one fixed-function color operation
//! (a named 1D curve, a 1D/3D LUT, a 3x4 matrix or a multiplier).
//!
//! This module performs *discovery* — it walks the advertised pipelines of a plane and
//! returns a description of the operations they can perform — and *resolution*: matching a
//! parametric [`ScanoutColorTransform`] against a discovered [`ColorPipeline`], producing the
//! concrete colorop property values to program. The programming itself happens as part of the
//! atomic plane state, see [`PlaneConfig::color_pipeline`](super::PlaneConfig::color_pipeline).
//!
//! Whether the capability could be enabled on a device is reported by
//! [`DrmDevice::plane_color_pipelines_supported`](super::DrmDevice::plane_color_pipelines_supported);
//! the pipelines of a plane are queried with
//! [`DrmDevice::plane_color_pipelines`](super::DrmDevice::plane_color_pipelines).

use std::collections::HashMap;
use std::io;
use std::num::NonZeroU32;

use drm::control::{Device as ControlDevice, RawResourceHandle, from_u32, plane, property};

use super::DrmDeviceFd;
use super::error::{AccessError, Error};
use crate::utils::DevPath;

use tracing::trace;

/// Maximum length of a colorop chain we are willing to walk, to guard against cyclic or
/// corrupted `NEXT` chains.
const MAX_PIPELINE_OPS: usize = 64;

/// A named transfer function supported by a [`ColorOpKind::Curve1D`] colorop.
///
/// The curves are defined by the kernel; see the `CURVE_1D_TYPE` colorop property. The `PQ 125`
/// variants use the gamescope/Windows-scRGB scale where 1.0 corresponds to 80 cd/m² and 125.0
/// to 10,000 cd/m².
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Curve1DType {
    /// The sRGB EOTF (`sRGB EOTF`), decoding to linear.
    SrgbEotf,
    /// The inverse sRGB EOTF (`sRGB Inverse EOTF`), encoding from linear.
    SrgbInvEotf,
    /// The PQ (ST 2084) EOTF scaled to \[0.0, 125.0\] (`PQ 125 EOTF`), decoding to linear.
    Pq125Eotf,
    /// The inverse of the scaled PQ EOTF (`PQ 125 Inverse EOTF`), encoding from linear.
    Pq125InvEotf,
    /// The inverse BT.2020 OETF (`BT.2020 Inverse OETF`), decoding to linear.
    Bt2020InvOetf,
    /// The BT.2020 OETF (`BT.2020 OETF`), encoding from linear.
    Bt2020Oetf,
    /// A pure 2.2 power-law EOTF (`Gamma 2.2`), decoding to linear.
    Gamma22,
    /// The inverse 2.2 power-law EOTF (`Gamma 2.2 Inverse`), encoding from linear.
    Gamma22Inv,
}

impl Curve1DType {
    /// The kernel's name for this curve in the `CURVE_1D_TYPE` property enum.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Curve1DType::SrgbEotf => "sRGB EOTF",
            Curve1DType::SrgbInvEotf => "sRGB Inverse EOTF",
            Curve1DType::Pq125Eotf => "PQ 125 EOTF",
            Curve1DType::Pq125InvEotf => "PQ 125 Inverse EOTF",
            Curve1DType::Bt2020InvOetf => "BT.2020 Inverse OETF",
            Curve1DType::Bt2020Oetf => "BT.2020 OETF",
            Curve1DType::Gamma22 => "Gamma 2.2",
            Curve1DType::Gamma22Inv => "Gamma 2.2 Inverse",
        }
    }

    fn from_kernel_name(name: &str) -> Option<Self> {
        Some(match name {
            "sRGB EOTF" => Curve1DType::SrgbEotf,
            "sRGB Inverse EOTF" => Curve1DType::SrgbInvEotf,
            "PQ 125 EOTF" => Curve1DType::Pq125Eotf,
            "PQ 125 Inverse EOTF" => Curve1DType::Pq125InvEotf,
            "BT.2020 Inverse OETF" => Curve1DType::Bt2020InvOetf,
            "BT.2020 OETF" => Curve1DType::Bt2020Oetf,
            "Gamma 2.2" => Curve1DType::Gamma22,
            "Gamma 2.2 Inverse" => Curve1DType::Gamma22Inv,
            _ => return None,
        })
    }
}

/// Interpolation used by a [`ColorOpKind::Lut1D`] colorop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lut1DInterpolation {
    /// Linear interpolation between LUT entries (`Linear`).
    Linear,
    /// An interpolation mode not modelled by smithay.
    Unknown,
}

/// Interpolation used by a [`ColorOpKind::Lut3D`] colorop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lut3DInterpolation {
    /// Tetrahedral interpolation (`Tetrahedral`).
    Tetrahedral,
    /// An interpolation mode not modelled by smithay.
    Unknown,
}

/// The operation a single colorop can perform, i.e. the value of its `TYPE` property together
/// with the type-specific capability properties.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorOpKind {
    /// A named 1D transfer function (`1D Curve`), selected via `CURVE_1D_TYPE`.
    Curve1D {
        /// The curves this colorop supports, together with the raw `CURVE_1D_TYPE` enum value
        /// used to select each of them.
        ///
        /// Curves unknown to smithay are omitted; an empty list means none of the advertised
        /// curves are usable.
        supported: Vec<(Curve1DType, u64)>,
    },
    /// A custom 1D LUT (`1D LUT`) uploaded via the `DATA` blob property.
    Lut1D {
        /// Number of entries of the LUT (the `SIZE` property).
        size: u32,
        /// Interpolation between LUT entries.
        interpolation: Lut1DInterpolation,
    },
    /// A 3x4 matrix (`3x4 Matrix`) applied to the pixel values, uploaded via `DATA`.
    Ctm3x4,
    /// A multiplier (`Multiplier`) applied to all pixel values, set via the `MULTIPLIER`
    /// fixed-point property.
    Multiplier,
    /// A 3D LUT (`3D LUT`) uploaded via `DATA`.
    Lut3D {
        /// Size of each dimension of the LUT cube (the `SIZE` property).
        size: u32,
        /// Interpolation between LUT entries.
        interpolation: Lut3DInterpolation,
    },
    /// A colorop type not modelled by smithay.
    ///
    /// Pipelines containing unknown, non-bypassable operations cannot be used safely and are
    /// skipped during discovery; unknown *bypassable* operations are kept so the rest of the
    /// pipeline remains usable.
    Unknown {
        /// The kernel's name for the colorop type.
        type_name: String,
    },
}

/// One color operation in a [`ColorPipeline`].
#[derive(Debug, Clone)]
pub struct ColorOp {
    /// The KMS object id of this colorop.
    pub id: u32,
    /// The operation this colorop performs.
    pub kind: ColorOpKind,
    /// Whether the colorop has a `BYPASS` property.
    ///
    /// Operations without one cannot be individually disabled: a user of the pipeline must
    /// program them (e.g. to an identity transform) whenever the pipeline is selected. Failing
    /// to account for non-bypassable operations is a known source of blank screens on some
    /// drivers.
    pub bypassable: bool,
    /// The atomic property handles of this colorop by name, for programming it as part of a
    /// plane update.
    pub(super) props: HashMap<String, property::Handle>,
}

/// A color pipeline advertised on a plane.
///
/// The pipeline processes plane pixels *before* blending, in the order of [`ops`](Self::ops).
#[derive(Debug, Clone)]
pub struct ColorPipeline {
    /// The value to set the plane's `COLOR_PIPELINE` property to in order to select this
    /// pipeline (the object id of the first colorop).
    pub id: u64,
    /// The color operations of this pipeline, in processing order.
    pub ops: Vec<ColorOp>,
}

/// Queries the color pipelines advertised on a plane.
///
/// Returns an empty list when the plane has no `COLOR_PIPELINE` property (either the kernel
/// predates the API, the client capability is not enabled, or the driver exposes no pipeline
/// on this plane, e.g. cursor planes on amdgpu).
pub(super) fn plane_color_pipelines<D>(dev: &D, plane: plane::Handle) -> Result<Vec<ColorPipeline>, Error>
where
    D: ControlDevice + DevPath,
{
    let props = dev.get_properties(plane).map_err(|source| {
        Error::Access(AccessError {
            errmsg: "Failed to get plane properties",
            dev: dev.dev_path(),
            source,
        })
    })?;

    let (prop_handles, _) = props.as_props_and_values();
    for prop in prop_handles {
        let Ok(info) = dev.get_property(*prop) else {
            continue;
        };
        if info.name().to_str() != Ok("COLOR_PIPELINE") {
            continue;
        }

        let property::ValueType::Enum(enum_values) = info.value_type() else {
            trace!(?plane, "COLOR_PIPELINE is not an enum property, ignoring");
            return Ok(Vec::new());
        };

        let (values, _) = enum_values.values();
        let mut pipelines = Vec::new();
        for &value in values {
            // 0 is the always-present "Bypass" entry, not a pipeline.
            if value == 0 {
                continue;
            }
            match walk_pipeline(dev, value) {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(err) => {
                    trace!(?plane, value, "skipping unusable color pipeline: {err:?}");
                }
            }
        }
        return Ok(pipelines);
    }

    Ok(Vec::new())
}

/// Walks the colorop chain starting at `first`, describing each operation.
fn walk_pipeline<D>(dev: &D, first: u64) -> Result<ColorPipeline, Error>
where
    D: ControlDevice + DevPath,
{
    let mut ops = Vec::new();
    let mut next = first as u32;

    while next != 0 {
        if ops.len() == MAX_PIPELINE_OPS || ops.iter().any(|op: &ColorOp| op.id == next) {
            return Err(Error::Access(AccessError {
                errmsg: "colorop NEXT chain is too long or cyclic",
                dev: dev.dev_path(),
                source: io::ErrorKind::InvalidData.into(),
            }));
        }
        let op = read_colorop(dev, next)?;
        next = op.next;
        ops.push(op.op);
    }

    Ok(ColorPipeline { id: first, ops })
}

struct ReadColorOp {
    op: ColorOp,
    next: u32,
}

/// Reads the properties of a single colorop object.
fn read_colorop<D>(dev: &D, id: u32) -> Result<ReadColorOp, Error>
where
    D: ControlDevice + DevPath,
{
    let mut prop_ids = Vec::new();
    let mut values = Vec::new();
    drm_ffi::mode::get_properties(
        dev.as_fd(),
        id,
        drm_ffi::DRM_MODE_OBJECT_COLOROP,
        Some(&mut prop_ids),
        Some(&mut values),
    )
    .map_err(|source| {
        Error::Access(AccessError {
            errmsg: "Failed to get colorop properties",
            dev: dev.dev_path(),
            source,
        })
    })?;

    let mut type_name = None;
    let mut bypassable = false;
    let mut next = 0u32;
    let mut curves = Vec::new();
    let mut size = 0u32;
    let mut lut1d_interpolation = Lut1DInterpolation::Unknown;
    let mut lut3d_interpolation = Lut3DInterpolation::Unknown;
    let mut props = HashMap::new();

    for (&prop_id, &value) in prop_ids.iter().zip(values.iter()) {
        let Some(handle) = from_u32::<property::Handle>(prop_id) else {
            continue;
        };
        let Ok(info) = dev.get_property(handle) else {
            continue;
        };
        let Ok(name) = info.name().to_str() else {
            continue;
        };
        props.insert(name.to_owned(), handle);

        match name {
            "TYPE" => {
                if let property::ValueType::Enum(enum_values) = info.value_type() {
                    type_name = enum_values
                        .get_value_from_raw_value(value)
                        .and_then(|v| v.name().to_str().ok())
                        .map(str::to_owned);
                }
            }
            "BYPASS" => bypassable = true,
            "NEXT" => next = value as u32,
            "CURVE_1D_TYPE" => {
                if let property::ValueType::Enum(enum_values) = info.value_type() {
                    let (_, entries) = enum_values.values();
                    curves.extend(entries.iter().filter_map(|entry| {
                        let curve = entry
                            .name()
                            .to_str()
                            .ok()
                            .and_then(Curve1DType::from_kernel_name)?;
                        Some((curve, entry.value()))
                    }));
                }
            }
            "SIZE" => size = value as u32,
            "LUT1D_INTERPOLATION" => {
                if let property::ValueType::Enum(enum_values) = info.value_type() {
                    if let Some(entry) = enum_values.get_value_from_raw_value(value) {
                        if entry.name().to_str() == Ok("Linear") {
                            lut1d_interpolation = Lut1DInterpolation::Linear;
                        }
                    }
                }
            }
            "LUT3D_INTERPOLATION" => {
                if let property::ValueType::Enum(enum_values) = info.value_type() {
                    if let Some(entry) = enum_values.get_value_from_raw_value(value) {
                        if entry.name().to_str() == Ok("Tetrahedral") {
                            lut3d_interpolation = Lut3DInterpolation::Tetrahedral;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let kind = match type_name.as_deref() {
        Some("1D Curve") => ColorOpKind::Curve1D { supported: curves },
        Some("1D LUT") => ColorOpKind::Lut1D {
            size,
            interpolation: lut1d_interpolation,
        },
        Some("3x4 Matrix") => ColorOpKind::Ctm3x4,
        Some("Multiplier") => ColorOpKind::Multiplier,
        Some("3D LUT") => ColorOpKind::Lut3D {
            size,
            interpolation: lut3d_interpolation,
        },
        other => {
            let type_name = other.unwrap_or("<missing TYPE>").to_owned();
            if !bypassable {
                return Err(Error::Access(AccessError {
                    errmsg: "pipeline contains an unknown non-bypassable colorop",
                    dev: dev.dev_path(),
                    source: io::ErrorKind::Unsupported.into(),
                }));
            }
            ColorOpKind::Unknown { type_name }
        }
    };

    Ok(ReadColorOp {
        op: ColorOp {
            id,
            kind,
            bypassable,
            props,
        },
        next,
    })
}

/// A parametric color transform to apply to a plane's pixels during scanout.
///
/// Semantically the transform is `encode(ctm × (multiplier × decode(pixel)))`: the pixel is
/// decoded to linear light, scaled, multiplied by a 3x4 matrix and re-encoded. Each stage is
/// optional; the default value is the identity transform (equivalent to selecting no pipeline
/// at all).
///
/// Linear-light values use the scale of the kernel's `PQ 125` curves: 1.0 corresponds to
/// 80 cd/m² and 125.0 to 10,000 cd/m² for the PQ curves, while the SDR curves map \[0, 1\]
/// electrical to \[0, 1\] linear.
///
/// A transform is turned into concrete colorop property values with [`Self::resolve`], or
/// applied automatically by
/// [`DrmCompositor::use_color_transforms`](super::compositor::DrmCompositor::use_color_transforms).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanoutColorTransform {
    /// The curve decoding the pixel values to linear light, or `None` if the content is
    /// already linear.
    pub decode: Option<Curve1DType>,
    /// A gain applied to the linear pixel values; 1.0 is the identity.
    pub multiplier: f64,
    /// A 3x4 matrix applied to the linear pixel values, in row-major order with the fourth
    /// column an offset (matching `struct drm_color_ctm_3x4`), or `None` for the identity.
    pub ctm: Option<[f64; 12]>,
    /// The curve re-encoding the linear pixel values, or `None` to keep them linear.
    pub encode: Option<Curve1DType>,
}

impl Default for ScanoutColorTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ScanoutColorTransform {
    /// The identity transform: pixels pass through unmodified (the plane's `COLOR_PIPELINE`
    /// is set to `Bypass`).
    pub const IDENTITY: Self = Self {
        decode: None,
        multiplier: 1.0,
        ctm: None,
        encode: None,
    };

    /// Whether this is the identity transform.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Resolves this transform against a color pipeline, producing the property values to
    /// program.
    ///
    /// Walks the pipeline's operations in order, assigning each stage of the transform to the
    /// first operation that can express it and bypassing all others. Operations without a
    /// `BYPASS` property are programmed to an identity where possible (matrix, multiplier);
    /// pipelines with other non-bypassable unused operations are rejected, as are pipelines
    /// that cannot express every stage.
    ///
    /// The `device` is used to create property blobs (e.g. the CTM matrix); their lifetime is
    /// tied to the returned value.
    ///
    /// Returns `None` if the pipeline cannot express the transform.
    pub fn resolve(&self, device: &DrmDeviceFd, pipeline: &ColorPipeline) -> Option<ResolvedColorPipeline> {
        let mut resolved = ResolvedColorPipeline {
            pipeline_id: pipeline.id,
            props: Vec::new(),
            blobs: Vec::new(),
        };

        // The remaining transform stages, in application order.
        let mut decode = self.decode;
        let mut multiplier = (self.multiplier != 1.0).then_some(self.multiplier);
        let mut ctm = self.ctm;
        let mut encode = self.encode;

        for op in &pipeline.ops {
            let mut used = false;

            match &op.kind {
                ColorOpKind::Curve1D { supported } => {
                    // A curve op can take the decode stage, or the encode stage once
                    // everything before it is placed.
                    let stage = if decode.is_some() {
                        &mut decode
                    } else if multiplier.is_none() && ctm.is_none() {
                        &mut encode
                    } else {
                        &mut None
                    };
                    if let Some(curve) = *stage {
                        if let Some(&(_, value)) = supported.iter().find(|(c, _)| *c == curve) {
                            resolved.set(op, "CURVE_1D_TYPE", value)?;
                            *stage = None;
                            used = true;
                        }
                    }
                }
                ColorOpKind::Multiplier => {
                    if decode.is_none() {
                        if let Some(gain) = multiplier {
                            resolved.set(op, "MULTIPLIER", to_s31_32(gain))?;
                            multiplier = None;
                            used = true;
                        }
                    }
                }
                ColorOpKind::Ctm3x4 => {
                    if decode.is_none() && (multiplier.is_some() || ctm.is_some()) {
                        // Fold a pending gain into the matrix: out = M × (g × in) scales the
                        // three input columns, leaving the offset column untouched.
                        let gain = multiplier.take().unwrap_or(1.0);
                        let mut matrix = ctm.take().unwrap_or(CTM_3X4_IDENTITY);
                        for row in 0..3 {
                            for col in 0..3 {
                                matrix[row * 4 + col] *= gain;
                            }
                        }
                        resolved.set_ctm(device, op, &matrix)?;
                        used = true;
                    }
                }
                ColorOpKind::Lut1D { .. } | ColorOpKind::Lut3D { .. } | ColorOpKind::Unknown { .. } => {}
            }

            if !used {
                resolved.bypass(device, op)?;
            }
        }

        // Every stage must have found an operation.
        if decode.is_some() || multiplier.is_some() || ctm.is_some() || encode.is_some() {
            return None;
        }

        Some(resolved)
    }
}

const CTM_3X4_IDENTITY: [f64; 12] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0,
];

/// Converts a floating point value to the kernel's S31.32 sign-magnitude fixed-point format.
fn to_s31_32(value: f64) -> u64 {
    let magnitude = (value.abs() * 4294967296.0).round() as u64;
    let magnitude = magnitude.min(i64::MAX as u64);
    ((value.is_sign_negative() as u64) << 63) | magnitude
}

/// `struct drm_color_ctm_3x4`: the contents of a 3x4 matrix colorop's `DATA` blob.
#[repr(C)]
struct CtmBlob {
    matrix: [u64; 12],
}

/// A property blob owned by a [`ResolvedColorPipeline`], destroyed when dropped.
#[derive(Debug)]
struct OwnedBlob {
    device: DrmDeviceFd,
    id: u64,
}

impl Drop for OwnedBlob {
    fn drop(&mut self) {
        // Nothing to be done if this fails.
        let _ = self.device.destroy_property_blob(self.id);
    }
}

/// A [`ScanoutColorTransform`] resolved against a specific [`ColorPipeline`]: the plane's
/// `COLOR_PIPELINE` value plus the property values of every colorop in the chain, ready to be
/// added to an atomic commit via [`PlaneConfig::color_pipeline`](super::PlaneConfig::color_pipeline).
///
/// Owns the property blobs (e.g. the CTM matrix) referenced by the values; they are destroyed
/// when the resolved pipeline is dropped, so it must be kept alive as long as a commit uses it.
#[derive(Debug)]
pub struct ResolvedColorPipeline {
    pipeline_id: u64,
    props: Vec<(RawResourceHandle, property::Handle, u64)>,
    #[allow(dead_code)] // Held to keep the kernel blobs alive.
    blobs: Vec<OwnedBlob>,
}

impl ResolvedColorPipeline {
    /// The value to set the plane's `COLOR_PIPELINE` property to.
    pub(super) fn pipeline_id(&self) -> u64 {
        self.pipeline_id
    }

    /// The colorop property values to add to the atomic commit.
    pub(super) fn props(&self) -> &[(RawResourceHandle, property::Handle, u64)] {
        &self.props
    }

    fn op_handle(op: &ColorOp) -> Option<RawResourceHandle> {
        NonZeroU32::new(op.id)
    }

    /// Programs a property of a used colorop, un-bypassing it.
    fn set(&mut self, op: &ColorOp, prop: &str, value: u64) -> Option<()> {
        let handle = Self::op_handle(op)?;
        self.props.push((handle, *op.props.get(prop)?, value));
        if op.bypassable {
            self.props.push((handle, *op.props.get("BYPASS")?, 0));
        }
        Some(())
    }

    /// Programs the `DATA` blob of a 3x4 matrix colorop.
    fn set_ctm(&mut self, device: &DrmDeviceFd, op: &ColorOp, matrix: &[f64; 12]) -> Option<()> {
        let blob = CtmBlob {
            matrix: matrix.map(to_s31_32),
        };
        let property::Value::Blob(id) = device.create_property_blob(&blob).ok()? else {
            return None;
        };
        self.blobs.push(OwnedBlob {
            device: device.clone(),
            id,
        });
        self.set(op, "DATA", id)
    }

    /// Bypasses an unused colorop, or programs it to an identity if it cannot be bypassed.
    fn bypass(&mut self, device: &DrmDeviceFd, op: &ColorOp) -> Option<()> {
        if op.bypassable {
            let handle = Self::op_handle(op)?;
            self.props.push((handle, *op.props.get("BYPASS")?, 1));
            return Some(());
        }
        match &op.kind {
            ColorOpKind::Multiplier => self.set(op, "MULTIPLIER", to_s31_32(1.0)),
            ColorOpKind::Ctm3x4 => self.set_ctm(device, op, &CTM_3X4_IDENTITY),
            // Curves and LUTs have no identity we can program without uploading data;
            // reject the pipeline.
            _ => None,
        }
    }
}

impl PartialEq for ResolvedColorPipeline {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline_id == other.pipeline_id && self.props == other.props
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s31_32_encoding() {
        assert_eq!(to_s31_32(0.0), 0);
        assert_eq!(to_s31_32(1.0), 1 << 32);
        assert_eq!(to_s31_32(2.5375), (2.5375f64 * 4294967296.0).round() as u64);
        // Sign-magnitude: -1.0 is the magnitude of 1.0 with the sign bit set.
        assert_eq!(to_s31_32(-1.0), (1 << 63) | (1 << 32));
        assert_eq!(to_s31_32(-0.5), (1 << 63) | (1 << 31));
    }

    #[test]
    fn identity_transform() {
        assert!(ScanoutColorTransform::IDENTITY.is_identity());
        assert!(ScanoutColorTransform::default().is_identity());
        assert!(
            !ScanoutColorTransform {
                multiplier: 2.0,
                ..Default::default()
            }
            .is_identity()
        );
    }
}
