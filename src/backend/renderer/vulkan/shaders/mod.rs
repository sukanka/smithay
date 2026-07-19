//! Precompiled SPIR-V shaders for the Vulkan renderer.
//!
//! The `.spv` binaries are checked in so that building smithay does not require a GLSL
//! compiler. To regenerate them after changing the GLSL sources run (from this directory):
//!
//! ```sh
//! glslc -O quad.vert -o quad.vert.spv
//! glslc -O texture.frag -o texture.frag.spv
//! glslc -O solid.frag -o solid.frag.spv
//! ```

pub(super) const QUAD_VERT: &[u8] = include_bytes!("quad.vert.spv");
pub(super) const TEXTURE_FRAG: &[u8] = include_bytes!("texture.frag.spv");
pub(super) const SOLID_FRAG: &[u8] = include_bytes!("solid.frag.spv");
