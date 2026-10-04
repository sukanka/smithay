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
use std::os::unix::io::AsFd;
use std::sync::Arc;

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

    /// Evaluates the curve for a single channel value, in the value scale of the kernel's
    /// curves (see [`ScanoutColorTransform`]): the `PQ 125` EOTF maps \[0, 1\] to \[0, 125\] and
    /// its inverse maps \[0, 125\] back to \[0, 1\]; every other curve maps \[0, 1\] to \[0, 1\].
    /// Inputs outside the domain are clamped.
    pub fn eval(&self, x: f64) -> f64 {
        // SMPTE ST 2084
        const M1: f64 = 2610. / 16384.;
        const M2: f64 = 2523. / 4096. * 128.;
        const C1: f64 = 3424. / 4096.;
        const C2: f64 = 2413. / 4096. * 32.;
        const C3: f64 = 2392. / 4096. * 32.;
        // ITU-R BT.2020
        const ALPHA: f64 = 1.09929682680944;
        const BETA: f64 = 0.018053968510807;

        match self {
            Curve1DType::SrgbEotf => {
                let x = x.clamp(0., 1.);
                if x <= 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            }
            Curve1DType::SrgbInvEotf => {
                let x = x.clamp(0., 1.);
                if x <= 0.0031308 {
                    x * 12.92
                } else {
                    1.055 * x.powf(1. / 2.4) - 0.055
                }
            }
            Curve1DType::Pq125Eotf => {
                let p = x.clamp(0., 1.).powf(1. / M2);
                125. * ((p - C1).max(0.) / (C2 - C3 * p)).powf(1. / M1)
            }
            Curve1DType::Pq125InvEotf => {
                let y = (x / 125.).clamp(0., 1.).powf(M1);
                ((C1 + C2 * y) / (1. + C3 * y)).powf(M2)
            }
            Curve1DType::Bt2020InvOetf => {
                let x = x.clamp(0., 1.);
                if x < 4.5 * BETA {
                    x / 4.5
                } else {
                    ((x + ALPHA - 1.) / ALPHA).powf(1. / 0.45)
                }
            }
            Curve1DType::Bt2020Oetf => {
                let x = x.clamp(0., 1.);
                if x < BETA {
                    4.5 * x
                } else {
                    ALPHA * x.powf(0.45) - (ALPHA - 1.)
                }
            }
            Curve1DType::Gamma22 => x.clamp(0., 1.).powf(2.2),
            Curve1DType::Gamma22Inv => x.clamp(0., 1.).powf(1. / 2.2),
        }
    }

    /// The largest value [`Self::eval`] returns.
    fn output_max(&self) -> f64 {
        match self {
            Curve1DType::Pq125Eotf => 125.,
            _ => 1.,
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
    /// Searches for an assignment of the transform's stages to the pipeline's operations, in
    /// order. Named curves take the decode and encode stages; a 1D LUT can take the decode
    /// stage too, filled with the curve (which makes SDR decodes possible on pipelines whose
    /// curve ops only offer PQ, like nvidia's). The gain goes to a multiplier or is folded
    /// into a matrix, and is folded into a decode LUT when that keeps the LUT's output within
    /// \[0, 1\]. Among all working assignments the one using the fewest LUTs is picked, as
    /// named curves are exact and cheap.
    ///
    /// Unused operations are bypassed; operations without a `BYPASS` property are programmed
    /// to an identity where possible (matrix, multiplier, 1D LUT). Pipelines with other
    /// non-bypassable unused operations are rejected, as are pipelines that cannot express
    /// every stage.
    ///
    /// The `device` is used to create property blobs (matrices and LUTs); their lifetime is
    /// tied to the returned value.
    ///
    /// Returns `None` if the pipeline cannot express the transform.
    pub fn resolve(&self, device: &DrmDeviceFd, pipeline: &ColorPipeline) -> Option<ResolvedColorPipeline> {
        let plan = self.plan(pipeline)?;

        let mut resolved = ResolvedColorPipeline {
            pipeline_id: pipeline.id,
            props: Vec::new(),
            blobs: Vec::new(),
            post_blend: None,
        };
        for (op, plan) in pipeline.ops.iter().zip(plan) {
            match plan {
                OpPlan::Bypass => resolved.bypass(op)?,
                OpPlan::Curve(value) => resolved.set(op, "CURVE_1D_TYPE", value)?,
                OpPlan::Multiplier(gain) => resolved.set(op, "MULTIPLIER", to_s31_32(gain))?,
                OpPlan::Ctm(matrix) => resolved.set_ctm(device, op, &matrix)?,
                OpPlan::LutDecode { curve, gain } => {
                    resolved.set_lut1d(device, op, |u| curve.eval(u) * gain)?
                }
                OpPlan::LutIdentity => resolved.set_lut1d(device, op, |u| u)?,
            }
        }

        Some(resolved)
    }

    /// Finds how to program every op of `pipeline` for this transform, without touching the
    /// device. See [`Self::resolve`].
    fn plan(&self, pipeline: &ColorPipeline) -> Option<Vec<OpPlan>> {
        let stages = PendingStages {
            decode: self.decode,
            gain: self.multiplier,
            ctm: self.ctm,
            encode: self.encode,
        };
        let mut best = None;
        let mut current = Vec::with_capacity(pipeline.ops.len());
        plan_ops(&pipeline.ops, stages, 0, &mut current, &mut best);
        best.map(|(_, plan)| plan)
    }
}

/// How a single colorop is programmed by a resolved transform.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OpPlan {
    Bypass,
    /// Select a named curve, by its raw `CURVE_1D_TYPE` value.
    Curve(u64),
    Multiplier(f64),
    Ctm([f64; 12]),
    /// Fill a 1D LUT with `curve(u) * gain`.
    LutDecode {
        curve: Curve1DType,
        gain: f64,
    },
    /// Fill a non-bypassable 1D LUT with the identity.
    LutIdentity,
}

/// The stages of a [`ScanoutColorTransform`] that are not placed on an op yet.
#[derive(Debug, Clone, Copy)]
struct PendingStages {
    decode: Option<Curve1DType>,
    /// A gain still to be applied to the (decoded) values; 1.0 once placed. A decode LUT can
    /// leave a gain behind when the decoded range doesn't fit its \[0, 1\] output.
    gain: f64,
    ctm: Option<[f64; 12]>,
    encode: Option<Curve1DType>,
}

impl PendingStages {
    fn gain_placed(&self) -> bool {
        (self.gain - 1.0).abs() < 1e-9
    }

    fn is_done(&self) -> bool {
        self.decode.is_none() && self.gain_placed() && self.ctm.is_none() && self.encode.is_none()
    }
}

/// Depth-first search over the ways to program `ops`, keeping the cheapest complete plan.
fn plan_ops(
    ops: &[ColorOp],
    stages: PendingStages,
    cost: u32,
    current: &mut Vec<OpPlan>,
    best: &mut Option<(u32, Vec<OpPlan>)>,
) {
    if best.as_ref().is_some_and(|(best_cost, _)| *best_cost <= cost) {
        return;
    }
    let Some((op, rest)) = ops.split_first() else {
        if stages.is_done() {
            *best = Some((cost, current.clone()));
        }
        return;
    };
    for (plan, next, op_cost) in op_options(op, stages) {
        current.push(plan);
        plan_ops(rest, next, cost + op_cost, current, best);
        current.pop();
    }
}

/// The ways a single op can be programmed given the pending stages, as (plan, remaining
/// stages, cost), placing stages before bypassing so that equal-cost plans use the earliest
/// ops.
fn op_options(op: &ColorOp, stages: PendingStages) -> Vec<(OpPlan, PendingStages, u32)> {
    let mut options = Vec::with_capacity(2);
    match &op.kind {
        ColorOpKind::Curve1D { supported } => {
            let find = |curve: Curve1DType| supported.iter().find(|(c, _)| *c == curve).map(|(_, v)| *v);
            if let Some(value) = stages.decode.and_then(find) {
                options.push((
                    OpPlan::Curve(value),
                    PendingStages {
                        decode: None,
                        ..stages
                    },
                    0,
                ));
            } else if stages.decode.is_none() && stages.gain_placed() && stages.ctm.is_none() {
                // A curve op takes the encode stage once everything before it is placed.
                if let Some(value) = stages.encode.and_then(find) {
                    options.push((
                        OpPlan::Curve(value),
                        PendingStages {
                            encode: None,
                            ..stages
                        },
                        0,
                    ));
                }
            }
            if op.bypassable {
                options.push((OpPlan::Bypass, stages, 0));
            }
        }
        ColorOpKind::Multiplier => {
            if stages.decode.is_none() && !stages.gain_placed() {
                options.push((
                    OpPlan::Multiplier(stages.gain),
                    PendingStages { gain: 1.0, ..stages },
                    0,
                ));
            }
            if op.bypassable {
                options.push((OpPlan::Bypass, stages, 0));
            } else {
                options.push((OpPlan::Multiplier(1.0), stages, 0));
            }
        }
        ColorOpKind::Ctm3x4 => {
            if stages.decode.is_none() && (stages.ctm.is_some() || !stages.gain_placed()) {
                // Fold a pending gain into the matrix: out = M × (g × in) scales the three
                // input columns, leaving the offset column untouched.
                let mut matrix = stages.ctm.unwrap_or(CTM_3X4_IDENTITY);
                for row in 0..3 {
                    for col in 0..3 {
                        matrix[row * 4 + col] *= stages.gain;
                    }
                }
                options.push((
                    OpPlan::Ctm(matrix),
                    PendingStages {
                        gain: 1.0,
                        ctm: None,
                        ..stages
                    },
                    0,
                ));
            }
            if op.bypassable {
                options.push((OpPlan::Bypass, stages, 0));
            } else {
                options.push((OpPlan::Ctm(CTM_3X4_IDENTITY), stages, 0));
            }
        }
        ColorOpKind::Lut1D { size, .. } if *size >= 2 => {
            if let Some(curve) = stages.decode {
                // The LUT output is limited to [0, 1]: fold the gain in if the decoded range
                // still fits, otherwise normalize and leave the rest of the gain pending.
                let range = curve.output_max() * stages.gain;
                let (lut_gain, remaining) = if range <= 1.0 {
                    (stages.gain, 1.0)
                } else {
                    (1.0 / curve.output_max(), range)
                };
                options.push((
                    OpPlan::LutDecode {
                        curve,
                        gain: lut_gain,
                    },
                    PendingStages {
                        decode: None,
                        gain: remaining,
                        ..stages
                    },
                    1,
                ));
            }
            if op.bypassable {
                options.push((OpPlan::Bypass, stages, 0));
            } else {
                options.push((OpPlan::LutIdentity, stages, 1));
            }
        }
        ColorOpKind::Lut1D { .. } | ColorOpKind::Lut3D { .. } | ColorOpKind::Unknown { .. } => {
            if op.bypassable {
                options.push((OpPlan::Bypass, stages, 0));
            }
        }
    }
    options
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

/// Creates a property blob from plain integer data.
fn create_blob<T: Copy>(device: &DrmDeviceFd, data: &[T]) -> Option<OwnedBlob> {
    // SAFETY: only used with slices of plain unsigned integers, which have no padding.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_ptr() as *mut u8, std::mem::size_of_val(data)) };
    let blob = drm_ffi::mode::create_property_blob(device.as_fd(), bytes).ok()?;
    Some(OwnedBlob {
        device: device.clone(),
        id: u64::from(blob.blob_id),
    })
}

/// A property blob owned by a [`ResolvedColorPipeline`], destroyed when dropped.
#[derive(Debug)]
pub(super) struct OwnedBlob {
    device: DrmDeviceFd,
    id: u64,
}

impl Drop for OwnedBlob {
    fn drop(&mut self) {
        // Nothing to be done if this fails.
        let _ = self.device.destroy_property_blob(self.id);
    }
}

/// An encode applied after blending, on the CRTC's `GAMMA_LUT`, instead of on each plane.
///
/// Plane color pipelines on some hardware (nvidia) can decode, scale and convert the gamut of
/// plane contents, but always end in linear light: they cannot apply the final encode to the
/// output's signal. When a single plane makes up the whole output, that encode can move behind
/// blending instead. The plane is then programmed to output *normalized* linear light, where
/// 1.0 corresponds to `linear_max` (in the linear scale of [`ScanoutColorTransform`], i.e.
/// 1.0 = 80 cd/m² for the PQ curves), and the CRTC gamma LUT maps that to `encode(u ×
/// linear_max)`.
///
/// Note that a gamma LUT indexed by linear light has little precision near black; the
/// smaller `linear_max`, the better (e.g. the output's peak luminance rather than 10,000
/// cd/m²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostBlendEncode {
    /// The curve encoding the blended, linear values to the output's signal.
    pub encode: Curve1DType,
    /// The linear value that the plane output 1.0 stands for.
    pub linear_max: f64,
}

impl PostBlendEncode {
    /// Creates the `GAMMA_LUT` blob (`struct drm_color_lut` entries) for this encode.
    pub(super) fn create_gamma_lut(&self, device: &DrmDeviceFd, size: u32) -> Option<Arc<OwnedBlob>> {
        if size < 2 {
            return None;
        }
        let data = (0..size)
            .flat_map(|i| {
                let u = f64::from(i) / f64::from(size - 1);
                let value = to_unorm16(self.encode.eval(u * self.linear_max));
                [value, value, value, 0]
            })
            .collect::<Vec<u16>>();
        create_blob(device, &data).map(Arc::new)
    }
}

fn to_unorm16(value: f64) -> u16 {
    (value.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16
}

fn to_unorm32(value: f64) -> u32 {
    (value.clamp(0.0, 1.0) * f64::from(u32::MAX)).round() as u32
}

/// A [`ScanoutColorTransform`] resolved against a specific [`ColorPipeline`]: the plane's
/// `COLOR_PIPELINE` value plus the property values of every colorop in the chain, ready to be
/// added to an atomic commit via [`PlaneConfig::color_pipeline`](super::PlaneConfig::color_pipeline).
///
/// It may also carry the CRTC `GAMMA_LUT` value of a [`PostBlendEncode`] (or its reset), which
/// has to change atomically with the plane.
///
/// Owns the property blobs (matrices and LUTs) referenced by the values; they are destroyed
/// when the resolved pipeline is dropped, so it must be kept alive as long as a commit uses it.
#[derive(Debug)]
pub struct ResolvedColorPipeline {
    pipeline_id: u64,
    props: Vec<(RawResourceHandle, property::Handle, u64)>,
    #[allow(dead_code)] // Held to keep the kernel blobs alive.
    blobs: Vec<Arc<OwnedBlob>>,
    /// `Some(true)` if this carries a post-blend encode on the CRTC, `Some(false)` if it
    /// resets the CRTC gamma LUT.
    post_blend: Option<bool>,
}

/// Retains immutable base + CRTC gamma combinations across frames. Strong identities
/// prevent blob or allocation address reuse from aliasing an earlier combination; the
/// small bound also limits how many obsolete pipelines and LUTs can remain alive.
#[derive(Debug, Default)]
pub(super) struct GammaPipelineCache {
    entries: Vec<GammaPipelineEntry>,
}

#[derive(Debug)]
struct GammaPipelineEntry {
    base: Option<Arc<ResolvedColorPipeline>>,
    crtc: RawResourceHandle,
    property: property::Handle,
    lut: Option<Arc<OwnedBlob>>,
    resolved: Arc<ResolvedColorPipeline>,
}

impl GammaPipelineCache {
    const CAPACITY: usize = 8;

    pub(super) fn resolve(
        &mut self,
        base: Option<&Arc<ResolvedColorPipeline>>,
        crtc: RawResourceHandle,
        property: property::Handle,
        lut: Option<&Arc<OwnedBlob>>,
    ) -> Arc<ResolvedColorPipeline> {
        fn same<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
            match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
        }
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.crtc == crtc
                && entry.property == property
                && same(entry.base.as_ref(), base)
                && same(entry.lut.as_ref(), lut)
        }) {
            let entry = self.entries.remove(index);
            let resolved = entry.resolved.clone();
            self.entries.push(entry);
            return resolved;
        }
        let resolved = Arc::new(ResolvedColorPipeline::with_gamma_lut(
            base.map(AsRef::as_ref),
            crtc,
            property,
            lut,
        ));
        if self.entries.len() == Self::CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push(GammaPipelineEntry {
            base: base.cloned(),
            crtc,
            property,
            lut: lut.cloned(),
            resolved: resolved.clone(),
        });
        resolved
    }
}

impl ResolvedColorPipeline {
    /// The value to set the plane's `COLOR_PIPELINE` property to.
    pub(super) fn pipeline_id(&self) -> u64 {
        self.pipeline_id
    }

    /// The colorop (and CRTC) property values to add to the atomic commit.
    pub(super) fn props(&self) -> &[(RawResourceHandle, property::Handle, u64)] {
        &self.props
    }

    /// Whether this carries a post-blend encode (`Some(true)`) or a reset of it
    /// (`Some(false)`) on the CRTC gamma LUT.
    pub(super) fn post_blend(&self) -> Option<bool> {
        self.post_blend
    }

    /// A copy of `base` (or of a bypassed pipeline if `None`) that also sets the CRTC's
    /// `GAMMA_LUT` to `lut`, or resets it to no LUT.
    pub(super) fn with_gamma_lut(
        base: Option<&ResolvedColorPipeline>,
        crtc: RawResourceHandle,
        gamma_lut_prop: property::Handle,
        lut: Option<&Arc<OwnedBlob>>,
    ) -> ResolvedColorPipeline {
        let mut resolved = match base {
            Some(base) => ResolvedColorPipeline {
                pipeline_id: base.pipeline_id,
                props: base.props.clone(),
                blobs: base.blobs.clone(),
                post_blend: None,
            },
            None => ResolvedColorPipeline {
                pipeline_id: 0,
                props: Vec::new(),
                blobs: Vec::new(),
                post_blend: None,
            },
        };
        resolved
            .props
            .push((crtc, gamma_lut_prop, lut.map_or(0, |lut| lut.id)));
        if let Some(lut) = lut {
            resolved.blobs.push(lut.clone());
        }
        resolved.post_blend = Some(lut.is_some());
        resolved
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
        self.blobs.push(Arc::new(OwnedBlob {
            device: device.clone(),
            id,
        }));
        self.set(op, "DATA", id)
    }

    /// Programs the `DATA` blob (`struct drm_color_lut32` entries) of a 1D LUT colorop with
    /// `f` sampled over \[0, 1\], and selects linear interpolation.
    fn set_lut1d(&mut self, device: &DrmDeviceFd, op: &ColorOp, f: impl Fn(f64) -> f64) -> Option<()> {
        let ColorOpKind::Lut1D { size, interpolation } = op.kind else {
            return None;
        };
        let data = (0..size)
            .flat_map(|i| {
                let value = to_unorm32(f(f64::from(i) / f64::from(size - 1)));
                [value, value, value, 0]
            })
            .collect::<Vec<u32>>();
        let blob = create_blob(device, &data)?;
        let id = blob.id;
        self.blobs.push(Arc::new(blob));
        if interpolation != Lut1DInterpolation::Linear {
            // Linear is the only interpolation the kernel defines; an unknown current value
            // is left alone.
            trace!(op = op.id, "1D LUT colorop with unknown interpolation");
        }
        self.set(op, "DATA", id)
    }

    /// Bypasses an unused colorop.
    fn bypass(&mut self, op: &ColorOp) -> Option<()> {
        let handle = Self::op_handle(op)?;
        self.props.push((handle, *op.props.get("BYPASS")?, 1));
        Some(())
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
    fn gamma_pipeline_cache_reuses_combinations_and_separates_targets() {
        let mut cache = GammaPipelineCache::default();
        let crtc = NonZeroU32::new(1).unwrap();
        let prop = from_u32::<property::Handle>(2).unwrap();
        let base = Arc::new(ResolvedColorPipeline {
            pipeline_id: 42,
            props: vec![(crtc, prop, 123)],
            blobs: Vec::new(),
            post_blend: None,
        });
        let reset = cache.resolve(Some(&base), crtc, prop, None);
        for _ in 0..1000 {
            assert!(Arc::ptr_eq(&reset, &cache.resolve(Some(&base), crtc, prop, None)));
        }
        assert_eq!(reset.pipeline_id(), 42);
        assert_eq!(reset.props(), &[(crtc, prop, 123), (crtc, prop, 0)]);
        assert_eq!(reset.post_blend(), Some(false));
        let other_crtc = NonZeroU32::new(3).unwrap();
        let other_prop = from_u32::<property::Handle>(4).unwrap();
        assert!(!Arc::ptr_eq(
            &reset,
            &cache.resolve(Some(&base), other_crtc, prop, None)
        ));
        assert!(!Arc::ptr_eq(
            &reset,
            &cache.resolve(Some(&base), crtc, other_prop, None)
        ));
        assert!(!Arc::ptr_eq(&reset, &cache.resolve(None, crtc, prop, None)));
        let replacement = Arc::new(ResolvedColorPipeline {
            pipeline_id: 43,
            props: Vec::new(),
            blobs: Vec::new(),
            post_blend: None,
        });
        assert!(!Arc::ptr_eq(
            &reset,
            &cache.resolve(Some(&replacement), crtc, prop, None)
        ));
    }

    #[test]
    fn gamma_pipeline_cache_preserves_inflight_lut_and_bounds_retention() {
        // No DRM device is needed: /dev/null rejects the harmless blob-destroy ioctl.
        let fd: std::os::fd::OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let device = DrmDeviceFd::new(fd.into());
        let crtc = NonZeroU32::new(1).unwrap();
        let prop = from_u32::<property::Handle>(2).unwrap();
        let mut cache = GammaPipelineCache::default();
        let lut = Arc::new(OwnedBlob {
            device: device.clone(),
            id: 11,
        });
        let weak_lut = Arc::downgrade(&lut);
        let inflight = cache.resolve(None, crtc, prop, Some(&lut));
        assert_eq!(inflight.post_blend(), Some(true));
        assert_eq!(inflight.props(), &[(crtc, prop, 11)]);
        assert!(Arc::ptr_eq(
            &inflight,
            &cache.resolve(None, crtc, prop, Some(&lut))
        ));
        assert!(!Arc::ptr_eq(&inflight, &cache.resolve(None, crtc, prop, None)));
        // Even the same numeric kernel blob ID cannot alias a distinct owned LUT.
        let replacement = Arc::new(OwnedBlob { device, id: 11 });
        assert!(!Arc::ptr_eq(
            &inflight,
            &cache.resolve(None, crtc, prop, Some(&replacement))
        ));
        drop(lut);
        for id in 10..30 {
            cache.resolve(None, NonZeroU32::new(id).unwrap(), prop, None);
        }
        assert_eq!(cache.entries.len(), GammaPipelineCache::CAPACITY);
        assert!(weak_lut.upgrade().is_some());
        drop(inflight);
        assert!(weak_lut.upgrade().is_none());
    }

    fn op(id: u32, kind: ColorOpKind, bypassable: bool) -> ColorOp {
        // resolve() looks up properties by name; hand every op the full set with arbitrary
        // (nonzero) handles.
        let props = ["BYPASS", "CURVE_1D_TYPE", "MULTIPLIER", "DATA"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    name.to_string(),
                    from_u32::<property::Handle>(1000 + id + i as u32).unwrap(),
                )
            })
            .collect();
        ColorOp {
            id,
            kind,
            bypassable,
            props,
        }
    }

    fn curve(id: u32, supported: &[Curve1DType], bypassable: bool) -> ColorOp {
        op(
            id,
            ColorOpKind::Curve1D {
                supported: supported.iter().map(|&c| (c, c as u64)).collect(),
            },
            bypassable,
        )
    }

    /// The pipelines advertised by nvidia-drm on a GeForce RTX 5070 Ti, as discovered via
    /// drm_info. Identical in shape on 610.43.03 and 615.71.09 (kernel 7.3), and on every
    /// primary and overlay plane (cursor planes advertise no `COLOR_PIPELINE`). The colorop
    /// object ids used here are the 610.43.03 primary plane's; they differ per plane and shift
    /// between driver versions (+6 on 615.71.09), so nothing may depend on them.
    ///
    /// The same driver exposes 1024-entry `GAMMA_LUT` and `DEGAMMA_LUT`, a `CTM` and the
    /// vendor `NV_CRTC_REGAMMA_LUT` on the CRTCs.
    ///
    /// "NVIDIA Full": 3x4 Matrix, 1D Curve {PQ 125 EOTF}, 1D LUT, Multiplier, 3x4 Matrix,
    /// 1D Curve {PQ 125 Inverse EOTF} (non-bypassable), 3x4 Matrix, 1D LUT, 3x4 Matrix,
    /// 1D Curve {PQ 125 EOTF} (non-bypassable), 3x4 Matrix.
    fn nvidia_full() -> ColorPipeline {
        ColorPipeline {
            id: 58,
            ops: vec![
                op(58, ColorOpKind::Ctm3x4, true),
                curve(63, &[Curve1DType::Pq125Eotf], true),
                op(
                    68,
                    ColorOpKind::Lut1D {
                        size: 1024,
                        interpolation: Lut1DInterpolation::Linear,
                    },
                    true,
                ),
                op(75, ColorOpKind::Multiplier, true),
                op(80, ColorOpKind::Ctm3x4, true),
                curve(85, &[Curve1DType::Pq125InvEotf], false),
                op(89, ColorOpKind::Ctm3x4, true),
                op(
                    94,
                    ColorOpKind::Lut1D {
                        size: 1024,
                        interpolation: Lut1DInterpolation::Linear,
                    },
                    true,
                ),
                op(101, ColorOpKind::Ctm3x4, true),
                curve(106, &[Curve1DType::Pq125Eotf], false),
                op(110, ColorOpKind::Ctm3x4, true),
            ],
        }
    }

    /// "NVIDIA Lite": 3x4 Matrix, 1D Curve {PQ 125 EOTF}, 1D LUT, Multiplier, 3x4 Matrix.
    fn nvidia_lite() -> ColorPipeline {
        ColorPipeline {
            id: 115,
            ops: vec![
                op(115, ColorOpKind::Ctm3x4, true),
                curve(120, &[Curve1DType::Pq125Eotf], true),
                op(
                    125,
                    ColorOpKind::Lut1D {
                        size: 1024,
                        interpolation: Lut1DInterpolation::Linear,
                    },
                    true,
                ),
                op(132, ColorOpKind::Multiplier, true),
                op(137, ColorOpKind::Ctm3x4, true),
            ],
        }
    }

    /// "NVIDIA FP Full": 3x4 Matrix, 3x4 Matrix, 1D Curve {PQ 125 Inverse EOTF}
    /// (non-bypassable), 3x4 Matrix, 1D LUT, 3x4 Matrix, 1D Curve {PQ 125 EOTF}
    /// (non-bypassable), 3x4 Matrix.
    fn nvidia_fp_full() -> ColorPipeline {
        ColorPipeline {
            id: 142,
            ops: vec![
                op(142, ColorOpKind::Ctm3x4, true),
                op(147, ColorOpKind::Ctm3x4, true),
                curve(152, &[Curve1DType::Pq125InvEotf], false),
                op(156, ColorOpKind::Ctm3x4, true),
                op(
                    161,
                    ColorOpKind::Lut1D {
                        size: 1024,
                        interpolation: Lut1DInterpolation::Linear,
                    },
                    true,
                ),
                op(168, ColorOpKind::Ctm3x4, true),
                curve(173, &[Curve1DType::Pq125Eotf], false),
                op(177, ColorOpKind::Ctm3x4, true),
            ],
        }
    }

    /// "NVIDIA FP Lite": 3x4 Matrix, 3x4 Matrix.
    fn nvidia_fp_lite() -> ColorPipeline {
        ColorPipeline {
            id: 182,
            ops: vec![
                op(182, ColorOpKind::Ctm3x4, true),
                op(187, ColorOpKind::Ctm3x4, true),
            ],
        }
    }

    fn nvidia_pipelines() -> Vec<ColorPipeline> {
        vec![nvidia_full(), nvidia_lite(), nvidia_fp_full(), nvidia_fp_lite()]
    }

    fn resolve_any(transform: &ScanoutColorTransform, pipelines: &[ColorPipeline]) -> bool {
        pipelines.iter().any(|p| transform.plan(p).is_some())
    }

    const BT709_TO_BT2020: [f64; 12] = [
        0.6274, 0.3293, 0.0433, 0.0, //
        0.0691, 0.9195, 0.0114, 0.0, //
        0.0164, 0.0880, 0.8956, 0.0,
    ];

    /// The transform shapes niri uses on HDR (PQ blend space) outputs are inexpressible on
    /// the nvidia-drm pipelines as long as they include the encode: the curve ops only offer
    /// the PQ 125 pair (no Gamma 2.2 / sRGB curves), and the trailing non-bypassable
    /// `PQ 125 Inverse EOTF` / `PQ 125 EOTF` pair means every pipeline ends in linear light,
    /// so a transform whose final stage is an encode can never resolve. The encode has to
    /// move behind blending instead, see [`PostBlendEncode`].
    #[test]
    fn nvidia_pipelines_reject_pq_blend_transforms() {
        let pipelines = nvidia_pipelines();

        // SDR content on an HDR output: gamma 2.2 decode, reference-white gain, gamut
        // conversion, PQ encode.
        let sdr_on_hdr = ScanoutColorTransform {
            decode: Some(Curve1DType::Gamma22),
            multiplier: 203. / 80.,
            ctm: Some(BT709_TO_BT2020),
            encode: Some(Curve1DType::Pq125InvEotf),
        };
        assert!(!resolve_any(&sdr_on_hdr, &pipelines));

        // PQ content needing a PQ round-trip (e.g. non-BT.2020 container): rejected because
        // the trailing non-bypassable PQ 125 EOTF cannot be bypassed or used.
        let pq_reencode = ScanoutColorTransform {
            decode: Some(Curve1DType::Pq125Eotf),
            multiplier: 1.0,
            ctm: None,
            encode: Some(Curve1DType::Pq125InvEotf),
        };
        assert!(!resolve_any(&pq_reencode, &pipelines));

        // HDR content on an SDR output: PQ decode, gain, gamma 2.2 encode.
        let hdr_on_sdr = ScanoutColorTransform {
            decode: Some(Curve1DType::Pq125Eotf),
            multiplier: 80. / 203.,
            ctm: None,
            encode: Some(Curve1DType::Gamma22Inv),
        };
        assert!(!resolve_any(&hdr_on_sdr, &pipelines));
    }

    /// Transforms that end in linear light (no encode stage) fit the "NVIDIA Lite" pipeline:
    /// nvidia planes decode and gain before blending, and the wire encode has to happen after
    /// blending, on the CRTC `GAMMA_LUT`. KWin relies on exactly this split on nvidia: for a
    /// single fullscreen scanout layer it retargets the plane at a normalized linear
    /// intermediate and merges the trailing encode into the CRTC gamma LUT (KWin commit
    /// 9ca199df4d, "offload a single fullscreen layer's trailing encode to the output
    /// post-blend pipeline").
    #[test]
    fn nvidia_lite_accepts_linear_output_transforms() {
        let decode_and_gain = ScanoutColorTransform {
            decode: Some(Curve1DType::Pq125Eotf),
            multiplier: 2.0,
            ctm: None,
            encode: None,
        };
        let plan = decode_and_gain.plan(&nvidia_lite()).unwrap();
        // The named curve is preferred over filling the LUT with PQ.
        assert_eq!(
            plan,
            [
                OpPlan::Bypass,
                OpPlan::Curve(Curve1DType::Pq125Eotf as u64),
                OpPlan::Bypass,
                OpPlan::Multiplier(2.0),
                OpPlan::Bypass,
            ]
        );
        // The Full pipeline still rejects it: its trailing non-bypassable PQ pair is not
        // recognized as an identity.
        assert!(!resolve_any(&decode_and_gain, &[nvidia_full()]));
    }

    /// SDR content retargeted at a normalized linear intermediate (the plane half of a
    /// post-blend encode) resolves on "NVIDIA Lite" by decoding gamma 2.2 in the 1D LUT, as
    /// the pipeline has no SDR curve.
    #[test]
    fn nvidia_lite_decodes_sdr_in_lut() {
        let linear_max = 1000. / 80.;
        let sdr_linear = ScanoutColorTransform {
            decode: Some(Curve1DType::Gamma22),
            multiplier: 203. / 80. / linear_max,
            ctm: Some(BT709_TO_BT2020),
            encode: None,
        };
        let plan = sdr_linear.plan(&nvidia_lite()).unwrap();
        // The gain (0.2) keeps the decoded range within [0, 1], so it is folded into the LUT
        // and the matrix at the end only converts the gamut.
        assert_eq!(
            plan,
            [
                OpPlan::Bypass,
                OpPlan::Bypass,
                OpPlan::LutDecode {
                    curve: Curve1DType::Gamma22,
                    gain: 203. / 80. / linear_max,
                },
                OpPlan::Bypass,
                OpPlan::Ctm(BT709_TO_BT2020),
            ]
        );

        // With a gain above 1 the LUT normalizes and a later op applies the gain.
        let sdr_gain = ScanoutColorTransform {
            decode: Some(Curve1DType::Gamma22),
            multiplier: 2.5,
            ctm: None,
            encode: None,
        };
        let plan = sdr_gain.plan(&nvidia_lite()).unwrap();
        assert_eq!(
            plan[2],
            OpPlan::LutDecode {
                curve: Curve1DType::Gamma22,
                gain: 1.0
            }
        );
        assert_eq!(plan[3], OpPlan::Multiplier(2.5));

        // PQ content with a gamut conversion, normalized: named PQ curve, gain, matrix.
        let pq_linear = ScanoutColorTransform {
            decode: Some(Curve1DType::Pq125Eotf),
            multiplier: 1. / linear_max,
            ctm: Some(BT709_TO_BT2020),
            encode: None,
        };
        let plan = pq_linear.plan(&nvidia_lite()).unwrap();
        assert_eq!(plan[1], OpPlan::Curve(Curve1DType::Pq125Eotf as u64));
        assert_eq!(plan[2], OpPlan::Bypass);
        assert_eq!(plan[3], OpPlan::Multiplier(1. / linear_max));
        assert_eq!(plan[4], OpPlan::Ctm(BT709_TO_BT2020));

        // "NVIDIA FP Lite" (two matrices) cannot decode at all.
        assert!(sdr_linear.plan(&nvidia_fp_lite()).is_none());
    }

    #[test]
    fn curves_round_trip() {
        let pairs = [
            (Curve1DType::SrgbEotf, Curve1DType::SrgbInvEotf),
            (Curve1DType::Pq125Eotf, Curve1DType::Pq125InvEotf),
            (Curve1DType::Bt2020InvOetf, Curve1DType::Bt2020Oetf),
            (Curve1DType::Gamma22, Curve1DType::Gamma22Inv),
        ];
        for (decode, encode) in pairs {
            for i in 0..=100 {
                let x = f64::from(i) / 100.;
                let y = encode.eval(decode.eval(x));
                assert!((x - y).abs() < 1e-6, "{decode:?}/{encode:?} at {x}: {y}");
            }
        }
        // PQ 125: 1.0 = 80 cd/m², 125.0 = 10,000 cd/m².
        assert!((Curve1DType::Pq125Eotf.eval(1.0) - 125.).abs() < 1e-9);
        // 203 cd/m² encodes to ~0.58 in PQ.
        assert!((Curve1DType::Pq125InvEotf.eval(203. / 80.) - 0.5806).abs() < 1e-3);
    }

    /// Control: an AMD-style pipeline (every op bypassable, SDR + PQ curves available)
    /// resolves the same transforms the nvidia pipelines reject.
    #[test]
    fn bypassable_pipeline_accepts_pq_blend_transforms() {
        let all_curves = [
            Curve1DType::SrgbEotf,
            Curve1DType::SrgbInvEotf,
            Curve1DType::Pq125Eotf,
            Curve1DType::Pq125InvEotf,
            Curve1DType::Gamma22,
            Curve1DType::Gamma22Inv,
        ];
        let pipeline = ColorPipeline {
            id: 300,
            ops: vec![
                curve(300, &all_curves, true),
                op(310, ColorOpKind::Multiplier, true),
                curve(320, &all_curves, true),
            ],
        };

        let sdr_on_hdr = ScanoutColorTransform {
            decode: Some(Curve1DType::Gamma22),
            multiplier: 203. / 80.,
            ctm: None,
            encode: Some(Curve1DType::Pq125InvEotf),
        };
        assert!(resolve_any(&sdr_on_hdr, &[pipeline]));
    }

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
