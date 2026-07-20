//! Custom fragment shader programs for the Vulkan renderer.
//!
//! A [`VulkanPixelProgram`] draws rectangles with a caller-provided fragment shader,
//! compiled from GLSL at runtime. Uniform values are passed per draw through a std140
//! uniform block (set 1, binding 0) suballocated from the frame's parameter ring; the
//! block's GLSL text can be generated with [`uniform_block_glsl`] so that the layout is
//! guaranteed to match the serialization. Up to [`MAX_CUSTOM_TEXTURES`] textures are bound
//! as combined image samplers at set 0.
//!
//! The vertex stage is the renderer's built-in quad shader: `v_coords` (location 0)
//! interpolates from 0 to 1 across the destination rectangle, and the push-constant block
//! provides `data.rect_size_misc.z` (alpha) and `.w` (tint).

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ash::vk;

use super::{CleanupItem, Device, VulkanError};

/// Maximum number of textures a custom program can sample.
pub const MAX_CUSTOM_TEXTURES: usize = 2;

/// Maximum size in bytes of a custom program's uniform block.
///
/// This matches the range of the ring buffer descriptor.
pub const MAX_CUSTOM_PARAMS_SIZE: u32 = super::PARAMS_RANGE;

/// Type of a custom shader uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum CustomUniformKind {
    Float,
    Vec2,
    Vec3,
    Vec4,
    /// A 3x3 matrix; occupies three vec4-aligned columns in std140.
    Mat3,
}

impl CustomUniformKind {
    /// std140 base alignment.
    fn align(self) -> u32 {
        match self {
            CustomUniformKind::Float => 4,
            CustomUniformKind::Vec2 => 8,
            CustomUniformKind::Vec3 | CustomUniformKind::Vec4 | CustomUniformKind::Mat3 => 16,
        }
    }

    /// Size occupied in the buffer (std140).
    fn size(self) -> u32 {
        match self {
            CustomUniformKind::Float => 4,
            CustomUniformKind::Vec2 => 8,
            CustomUniformKind::Vec3 => 12,
            CustomUniformKind::Vec4 => 16,
            CustomUniformKind::Mat3 => 48,
        }
    }

    fn glsl_type(self) -> &'static str {
        match self {
            CustomUniformKind::Float => "float",
            CustomUniformKind::Vec2 => "vec2",
            CustomUniformKind::Vec3 => "vec3",
            CustomUniformKind::Vec4 => "vec4",
            CustomUniformKind::Mat3 => "mat3",
        }
    }
}

/// Declaration of a custom shader uniform.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct CustomUniformDecl {
    pub name: String,
    pub kind: CustomUniformKind,
}

/// A value for a custom shader uniform.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum CustomUniformValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    /// Column-major 3x3 matrix.
    Mat3([f32; 9]),
}

impl CustomUniformValue {
    fn write_std140(&self, out: &mut [u8], offset: u32) {
        let offset = offset as usize;
        let mut write = |floats: &[f32], at: usize| {
            for (i, value) in floats.iter().enumerate() {
                let bytes = value.to_le_bytes();
                out[at + i * 4..at + i * 4 + 4].copy_from_slice(&bytes);
            }
        };
        match self {
            CustomUniformValue::Float(v) => write(&[*v], offset),
            CustomUniformValue::Vec2(v) => write(v, offset),
            CustomUniformValue::Vec3(v) => write(v, offset),
            CustomUniformValue::Vec4(v) => write(v, offset),
            CustomUniformValue::Mat3(m) => {
                for col in 0..3 {
                    write(&m[col * 3..col * 3 + 3], offset + col * 16);
                }
            }
        }
    }
}

/// A named uniform value for a custom program draw.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub struct CustomUniform<'a> {
    pub name: &'a str,
    pub value: CustomUniformValue,
}

/// An owned named uniform value, for storing with a program override.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct OwnedCustomUniform {
    pub name: String,
    pub value: CustomUniformValue,
}

impl OwnedCustomUniform {
    /// Borrows this uniform for a draw call.
    pub fn borrow(&self) -> CustomUniform<'_> {
        CustomUniform {
            name: &self.name,
            value: self.value,
        }
    }
}

/// Computes the std140 offsets for an ordered list of declarations.
///
/// Returns (offsets, total size).
fn std140_layout(decls: &[CustomUniformDecl]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::with_capacity(decls.len());
    let mut offset = 0u32;
    for decl in decls {
        let align = decl.kind.align();
        offset = offset.div_ceil(align) * align;
        offsets.push(offset);
        offset += decl.kind.size();
    }
    (offsets, offset.div_ceil(16) * 16)
}

/// Generates the GLSL text of the uniform block matching the serialization of the given
/// (ordered) declarations, as an instance-less block making every member visible under its
/// own name.
pub fn uniform_block_glsl(decls: &[CustomUniformDecl]) -> String {
    let mut out = String::from("layout(std140, set = 1, binding = 0) uniform NiriCustom {\n");
    for decl in decls {
        out.push_str("    ");
        out.push_str(decl.kind.glsl_type());
        out.push(' ');
        out.push_str(&decl.name);
        out.push_str(";\n");
    }
    out.push_str("};\n");
    out
}

/// Generates the GLSL declarations for the samplers of a custom program.
pub fn texture_bindings_glsl(names: &[&str]) -> String {
    let mut out = String::new();
    for (i, name) in names.iter().enumerate() {
        out.push_str(&format!(
            "layout(set = 0, binding = {i}) uniform sampler2D {name};\n"
        ));
    }
    out
}

static PROGRAM_ID: AtomicUsize = AtomicUsize::new(1);

/// A custom fragment shader program of the [`VulkanRenderer`](super::VulkanRenderer).
#[derive(Debug, Clone)]
pub struct VulkanPixelProgram(pub(super) Arc<ProgramInner>);

impl PartialEq for VulkanPixelProgram {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Debug)]
pub(super) struct ProgramInner {
    pub(super) device: Arc<Device>,
    pub(super) id: usize,
    pub(super) module: vk::ShaderModule,
    /// Uniforms with their std140 offsets.
    pub(super) uniforms: Vec<(CustomUniformDecl, u32)>,
    pub(super) block_size: u32,
    pub(super) texture_names: Vec<String>,
}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // The module is only needed for (future) pipeline creation; cached pipelines stay
        // valid without it.
        self.device
            .defer_destroy(0, vec![CleanupItem::ShaderModule(self.module)]);
    }
}

impl VulkanPixelProgram {
    /// Serializes the uniform values into the program's std140 block.
    ///
    /// Uniforms that were declared but not provided stay zero; provided uniforms that were
    /// not declared are ignored (matching GL, where unknown names resolve to no location).
    pub(super) fn serialize_uniforms(&self, uniforms: &[CustomUniform<'_>]) -> Vec<u8> {
        let mut out = vec![0u8; self.0.block_size as usize];
        for (decl, offset) in &self.0.uniforms {
            if let Some(uniform) = uniforms.iter().find(|u| u.name == decl.name) {
                uniform.value.write_std140(&mut out, *offset);
            }
        }
        out
    }
}

pub(super) fn compile_program(
    device: &Arc<Device>,
    compiler: &shaderc::Compiler,
    src: &str,
    uniforms: &[CustomUniformDecl],
    texture_names: &[&str],
) -> Result<VulkanPixelProgram, VulkanError> {
    if texture_names.len() > MAX_CUSTOM_TEXTURES {
        return Err(VulkanError::PipelineCreation);
    }

    let (offsets, block_size) = std140_layout(uniforms);
    if block_size > MAX_CUSTOM_PARAMS_SIZE {
        return Err(VulkanError::PipelineCreation);
    }

    let artifact = compiler
        .compile_into_spirv(src, shaderc::ShaderKind::Fragment, "custom.frag", "main", None)
        .map_err(|err| {
            tracing::warn!("error compiling custom fragment shader: {err}");
            VulkanError::ShaderCompile(err.to_string())
        })?;

    let code_bytes = artifact.as_binary_u8();
    let code = ash::util::read_spv(&mut std::io::Cursor::new(code_bytes))
        .map_err(|_| VulkanError::PipelineCreation)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    let module = unsafe { device.raw.create_shader_module(&create_info, None) }?;

    Ok(VulkanPixelProgram(Arc::new(ProgramInner {
        device: device.clone(),
        id: PROGRAM_ID.fetch_add(1, Ordering::Relaxed),
        module,
        uniforms: uniforms.iter().cloned().zip(offsets).collect(),
        block_size,
        texture_names: texture_names.iter().map(|s| s.to_string()).collect(),
    })))
}
