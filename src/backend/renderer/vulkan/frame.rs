//! Frame handling of the Vulkan renderer.

use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use glam::{Mat3, Vec2};
use tracing::warn;

use super::{
    PushConstants, VulkanError, VulkanRenderer, VulkanTexture, color_subresource_range, foreign_barrier,
    image_barrier, target_transfer_state, texture::TargetInner, transfer_prepare, transfer_restore,
};
use crate::{
    backend::renderer::{
        BlitFrame, Color32F, ContextId, DebugFlags, Frame, FrameContext, Texture, TextureFilter,
        sync::SyncPoint,
    },
    utils::{Buffer as BufferCoord, Physical, Point, Rectangle, Size, Transform},
};

use super::VulkanTarget;

/// A rendering frame of the [`VulkanRenderer`].
pub struct VulkanFrame<'frame, 'buffer> {
    pub(super) renderer: &'frame mut VulkanRenderer,
    pub(super) target: &'frame mut VulkanTarget<'buffer>,
    /// Render command buffer; recording inside an active dynamic rendering scope unless
    /// temporarily suspended for a blit.
    cb: vk::CommandBuffer,
    projection: Mat3,
    transform: Transform,
    size: Size<i32, Physical>,
    /// Target size in buffer coordinates (pre-transform).
    buffer_size: Size<i32, Physical>,
    /// Dmabuf-imported textures used this frame, needing foreign queue acquire/release.
    foreign_textures: Vec<VulkanTexture>,
    /// All textures used this frame, kept alive until submission.
    used_textures: Vec<VulkanTexture>,
    bound_pipeline: vk::Pipeline,
    /// Layout of an offscreen target texture before this frame started.
    target_initial_layout: vk::ImageLayout,
    finished: AtomicBool,
}

impl std::fmt::Debug for VulkanFrame<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanFrame")
            .field("transform", &self.transform)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<'frame, 'buffer> VulkanFrame<'frame, 'buffer> {
    pub(super) fn new(
        renderer: &'frame mut VulkanRenderer,
        target: &'frame mut VulkanTarget<'buffer>,
        mut output_size: Size<i32, Physical>,
        transform: Transform,
    ) -> Result<Self, VulkanError> {
        renderer.cleanup();

        // Create the pipelines for this format upfront to avoid surprises mid-frame.
        let format = target.vk_format();
        for solid in [false, true] {
            for blend in [false, true] {
                renderer.get_pipeline(format, solid, blend)?;
            }
        }

        let cb = renderer.acquire_command_buffer()?;
        let raw = renderer.device().raw.clone();

        let buffer_size = output_size;
        unsafe {
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(cb, &begin_info)?;

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(output_size.w as f32)
                .height(output_size.h as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            raw.cmd_set_viewport(cb, 0, &[viewport]);
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: output_size.w as u32,
                    height: output_size.h as u32,
                },
            };
            raw.cmd_set_scissor(cb, 0, &[scissor]);

            begin_rendering(&raw, cb, target, buffer_size);
        }

        // Handle the width/height swap when the output is rotated by 90°/270°.
        if let Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270 = transform {
            std::mem::swap(&mut output_size.w, &mut output_size.h);
        }

        // Replicates the GLES renderer's projection setup. The final mapping — compositor
        // pixel (0, 0) to the top-left of buffer memory — is identical in Vulkan since its
        // NDC is y-down while framebuffer row 0 is the first row in memory.
        let mut renderer_mat = Mat3::IDENTITY;
        let t = Mat3::IDENTITY;
        let x = 2.0 / (output_size.w as f32);
        let y = 2.0 / (output_size.h as f32);

        // Rotation & Reflection
        renderer_mat.x_axis.x = x * t.x_axis.x;
        renderer_mat.y_axis.x = x * t.x_axis.y;
        renderer_mat.x_axis.y = y * -t.y_axis.x;
        renderer_mat.y_axis.y = y * -t.y_axis.y;

        // Translation
        renderer_mat.z_axis.x = -(1.0f32.copysign(renderer_mat.x_axis.x + renderer_mat.y_axis.x));
        renderer_mat.z_axis.y = -(1.0f32.copysign(renderer_mat.x_axis.y + renderer_mat.y_axis.y));

        let flip180 = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]);
        let projection = flip180 * transform.matrix() * renderer_mat;

        let target_initial_layout = match &target.0 {
            TargetInner::Texture { texture, .. } => *texture.0.layout.lock().unwrap(),
            TargetInner::Dmabuf { .. } => vk::ImageLayout::GENERAL,
        };

        Ok(VulkanFrame {
            renderer,
            target,
            cb,
            projection,
            transform,
            size: output_size,
            buffer_size,
            foreign_textures: Vec::new(),
            used_textures: Vec::new(),
            bound_pipeline: vk::Pipeline::null(),
            target_initial_layout,
            finished: AtomicBool::new(false),
        })
    }

    fn bind_pipeline(&mut self, pipeline: vk::Pipeline) {
        if self.bound_pipeline != pipeline {
            unsafe {
                self.renderer.device().raw.cmd_bind_pipeline(
                    self.cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );
            }
            self.bound_pipeline = pipeline;
        }
    }

    /// Records one draw per damage rect with the given base push constants.
    ///
    /// `rects` are clamped against `dest` and offset by `rect_offset`.
    fn draw_rects(
        &mut self,
        base: &PushConstants,
        dest: Rectangle<i32, Physical>,
        rect_offset: Point<i32, Physical>,
        rects: &[Rectangle<i32, Physical>],
    ) {
        let raw = self.renderer.device().raw.clone();
        for rect in rects {
            let rect_constrained_loc = rect.loc.constrain(Rectangle::from_size(dest.size));
            let rect_clamped_size = rect
                .size
                .clamp((0, 0), (dest.size.to_point() - rect_constrained_loc).to_size());
            if rect_clamped_size.w <= 0 || rect_clamped_size.h <= 0 {
                continue;
            }

            let mut pc = *base;
            pc.pos_off_rect[2] = (rect_offset.x + rect_constrained_loc.x) as f32;
            pc.pos_off_rect[3] = (rect_offset.y + rect_constrained_loc.y) as f32;
            pc.rect_size_misc[0] = rect_clamped_size.w as f32;
            pc.rect_size_misc[1] = rect_clamped_size.h as f32;

            unsafe {
                raw.cmd_push_constants(
                    self.cb,
                    self.renderer.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc.as_bytes(),
                );
                raw.cmd_draw(self.cb, 4, 1, 0, 0);
            }
        }
    }

    fn draw_solid_internal(
        &mut self,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
        blend: bool,
    ) -> Result<(), VulkanError> {
        if damage.is_empty() {
            return Ok(());
        }

        let pipeline = self.renderer.get_pipeline(self.target.vk_format(), true, blend)?;
        self.bind_pipeline(pipeline);

        let (mat, off) = decompose(&self.projection);
        let pc = PushConstants {
            mat_pos: mat,
            pos_off_rect: [off[0], off[1], 0.0, 0.0],
            rect_size_misc: [0.0, 0.0, 1.0, 0.0],
            color: [color.r(), color.g(), color.b(), color.a()],
            ..Default::default()
        };

        // Solid vertices are in absolute coordinates: offset the damage rects by dest.loc.
        self.draw_rects(&pc, dest, dest.loc, damage);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_texture_internal(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        transform: Transform,
        alpha: f32,
    ) -> Result<(), VulkanError> {
        if damage.is_empty() {
            return Ok(());
        }

        let tex_size = texture.size();
        if src.size.is_empty() || tex_size.is_empty() {
            return Ok(());
        }

        // Track usage for keep-alive and foreign queue transfer.
        if texture.0.dmabuf_imported
            && !self
                .foreign_textures
                .iter()
                .any(|tex| tex.0.image == texture.0.image)
        {
            self.foreign_textures.push(texture.clone());
        }
        self.used_textures.push(texture.clone());

        let mut tex_mat = build_texture_mat(src, dest, tex_size, transform);
        if texture.0.y_inverted {
            tex_mat = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]) * tex_mat;
        }

        let pos_mat =
            self.projection * Mat3::from_translation(Vec2::new(dest.loc.x as f32, dest.loc.y as f32));
        let (mat_pos, pos_off) = decompose(&pos_mat);
        let (mat_uv, uv_off) = decompose(&tex_mat);

        let tint = if self.renderer.debug_flags.contains(DebugFlags::TINT) {
            1.0
        } else {
            0.0
        };
        let pc = PushConstants {
            mat_pos,
            pos_off_rect: [pos_off[0], pos_off[1], 0.0, 0.0],
            rect_size_misc: [0.0, 0.0, alpha, tint],
            mat_uv,
            uv_off: [uv_off[0], uv_off[1], 0.0, 0.0],
            color: [0.0; 4],
        };

        let ds = self.renderer.texture_descriptor_set(texture)?;

        // Split the damage into opaque and non-opaque regions to disable blending where
        // possible, mirroring the GLES renderer.
        let mut opaque_damage: Vec<Rectangle<i32, Physical>> = Vec::new();
        let mut non_opaque_damage: Vec<Rectangle<i32, Physical>> = Vec::new();

        let is_implicit_opaque = !texture.0.has_alpha && alpha == 1f32;
        if is_implicit_opaque {
            opaque_damage.extend_from_slice(damage);
        } else if alpha != 1f32 || opaque_regions.is_empty() {
            non_opaque_damage.extend_from_slice(damage);
        } else {
            non_opaque_damage.extend_from_slice(damage);
            opaque_damage.extend_from_slice(damage);
            non_opaque_damage =
                Rectangle::subtract_rects_many_in_place(non_opaque_damage, opaque_regions.iter().copied());
            opaque_damage =
                Rectangle::subtract_rects_many_in_place(opaque_damage, non_opaque_damage.iter().copied());
        }

        let format = self.target.vk_format();
        let raw = self.renderer.device().raw.clone();
        for (blend, damage) in [(true, &non_opaque_damage), (false, &opaque_damage)] {
            if damage.is_empty() {
                continue;
            }
            let pipeline = self.renderer.get_pipeline(format, false, blend)?;
            self.bind_pipeline(pipeline);
            unsafe {
                raw.cmd_bind_descriptor_sets(
                    self.cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.renderer.pipeline_layout,
                    0,
                    &[ds],
                    &[],
                );
            }
            // Texture vertices are in dest-local coordinates.
            self.draw_rects(&pc, dest, Point::default(), damage);
        }

        Ok(())
    }

    fn finish_internal(&mut self) -> Result<SyncPoint, VulkanError> {
        if self.finished.swap(true, Ordering::SeqCst) {
            return Ok(SyncPoint::signaled());
        }

        let raw = self.renderer.device().raw.clone();
        let queue_family = self.renderer.device().queue_family;

        unsafe {
            raw.cmd_end_rendering(self.cb);

            // Release foreign resources and finalize target layout at the end of the render
            // command buffer.
            let mut release_barriers: Vec<vk::ImageMemoryBarrier2<'_>> = Vec::new();
            for texture in &self.foreign_textures {
                release_barriers.push(foreign_barrier(
                    queue_family,
                    false,
                    texture.0.image,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                ));
            }
            match &self.target.0 {
                TargetInner::Dmabuf { buffer, .. } => {
                    release_barriers.push(foreign_barrier(
                        queue_family,
                        false,
                        buffer.image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ));
                }
                TargetInner::Texture { texture, .. } => {
                    release_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .image(texture.0.image)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                            .subresource_range(color_subresource_range()),
                    );
                    *texture.0.layout.lock().unwrap() = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                }
            }
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&release_barriers);
            raw.cmd_pipeline_barrier2(self.cb, &dependency);

            raw.end_command_buffer(self.cb)?;
        }

        // Record the acquire barriers into a separate command buffer executed first.
        let pre_cb = self.renderer.acquire_command_buffer()?;
        unsafe {
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(pre_cb, &begin_info)?;

            let mut acquire_barriers: Vec<vk::ImageMemoryBarrier2<'_>> = Vec::new();
            for texture in &self.foreign_textures {
                acquire_barriers.push(foreign_barrier(
                    queue_family,
                    true,
                    texture.0.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                ));
            }
            match &self.target.0 {
                TargetInner::Dmabuf { buffer, .. } => {
                    let old_layout = if buffer.transitioned.swap(true, Ordering::AcqRel) {
                        vk::ImageLayout::GENERAL
                    } else {
                        // First use of this buffer: `PREINITIALIZED` keeps the (externally
                        // written) contents intact, unlike `UNDEFINED`.
                        vk::ImageLayout::PREINITIALIZED
                    };
                    acquire_barriers.push(foreign_barrier(
                        queue_family,
                        true,
                        buffer.image,
                        old_layout,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ));
                }
                TargetInner::Texture { texture, .. } => {
                    let old_layout = self.target_initial_layout;
                    acquire_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .image(texture.0.image)
                            .old_layout(old_layout)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_access_mask(
                                vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            )
                            .subresource_range(color_subresource_range()),
                    );
                }
            }
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&acquire_barriers);
            raw.cmd_pipeline_barrier2(pre_cb, &dependency);
            raw.end_command_buffer(pre_cb)?;
        }

        let (point, fence) = self
            .renderer
            .submit(&[pre_cb, self.cb], Vec::new(), Vec::new(), true)?;

        for texture in &self.used_textures {
            texture.0.mark_used(point);
        }
        self.target.mark_used(point);

        Ok(SyncPoint::from(fence))
    }
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = VulkanError;
    type TextureId = VulkanTexture;

    fn context_id(&self) -> ContextId<VulkanTexture> {
        self.renderer.context_id.clone()
    }

    #[profiling::function]
    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        if at.is_empty() {
            return Ok(());
        }
        // Like the GLES renderer, clearing is an unblended solid draw so it goes through the
        // same projection handling.
        self.draw_solid_internal(Rectangle::from_size(self.size), at, color, false)
    }

    #[profiling::function]
    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        self.draw_solid_internal(dst, damage, color, !color.is_opaque())
    }

    #[profiling::function]
    fn render_texture_from_to(
        &mut self,
        texture: &VulkanTexture,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        self.render_texture_internal(texture, src, dst, damage, opaque_regions, src_transform, alpha)
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.size
    }

    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.handle_wait(sync)
    }

    #[profiling::function]
    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        self.finish_internal()
    }
}

impl Drop for VulkanFrame<'_, '_> {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::SeqCst) {
            // The command buffer must not stay in the recording state; submit the frame.
            if let Err(err) = self.finish_internal() {
                warn!(?err, "Error dropping unfinished vulkan frame");
            }
        }
    }
}

impl<'buffer> BlitFrame<VulkanTarget<'buffer>> for VulkanFrame<'_, '_> {
    fn blit_to(
        &mut self,
        to: &mut VulkanTarget<'buffer>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.blit_internal(to, src, dst, filter, true)
    }

    fn blit_from(
        &mut self,
        from: &VulkanTarget<'buffer>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.blit_internal(from, src, dst, filter, false)
    }
}

impl VulkanFrame<'_, '_> {
    /// Blits between the bound target and another framebuffer, suspending the active dynamic
    /// rendering scope for the duration of the transfer.
    fn blit_internal(
        &mut self,
        other: &VulkanTarget<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
        to_other: bool,
    ) -> Result<SyncPoint, VulkanError> {
        if other.image() == self.target.image() {
            return Err(VulkanError::BlitSameImage);
        }

        let raw = self.renderer.device().raw.clone();
        let queue_family = self.renderer.device().queue_family;
        let vk_filter = match filter {
            TextureFilter::Linear => vk::Filter::LINEAR,
            TextureFilter::Nearest => vk::Filter::NEAREST,
        };

        // The bound target was already acquired for this frame; only a layout transition is
        // needed if it uses an optimal attachment layout.
        let bound_layout = self.target.render_layout();
        let other_state = target_transfer_state(other);

        unsafe {
            raw.cmd_end_rendering(self.cb);

            let (bound_write, other_write) = (!to_other, to_other);
            if bound_layout != vk::ImageLayout::GENERAL {
                image_barrier(
                    &raw,
                    self.cb,
                    self.target.image(),
                    bound_layout,
                    if bound_write {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    } else {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    },
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    if bound_write {
                        vk::AccessFlags2::TRANSFER_WRITE
                    } else {
                        vk::AccessFlags2::TRANSFER_READ
                    },
                );
            } else {
                // Contents were written as attachment; make them transfer-visible.
                image_barrier(
                    &raw,
                    self.cb,
                    self.target.image(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                );
            }
            transfer_prepare(
                &raw,
                self.cb,
                queue_family,
                other.image(),
                &other_state,
                other_write,
            );

            let bound_transfer_layout = if bound_layout == vk::ImageLayout::GENERAL {
                vk::ImageLayout::GENERAL
            } else if bound_write {
                vk::ImageLayout::TRANSFER_DST_OPTIMAL
            } else {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            };

            let (src_image, src_layout, dst_image, dst_layout) = if to_other {
                (
                    self.target.image(),
                    bound_transfer_layout,
                    other.image(),
                    other_state.transfer_layout(true),
                )
            } else {
                (
                    other.image(),
                    other_state.transfer_layout(false),
                    self.target.image(),
                    bound_transfer_layout,
                )
            };

            let blit = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D {
                        x: src.loc.x,
                        y: src.loc.y,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: src.loc.x + src.size.w,
                        y: src.loc.y + src.size.h,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D {
                        x: dst.loc.x,
                        y: dst.loc.y,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: dst.loc.x + dst.size.w,
                        y: dst.loc.y + dst.size.h,
                        z: 1,
                    },
                ]);
            raw.cmd_blit_image(
                self.cb,
                src_image,
                src_layout,
                dst_image,
                dst_layout,
                &[blit],
                vk_filter,
            );

            // Restore the bound target for attachment use.
            if bound_layout != vk::ImageLayout::GENERAL {
                image_barrier(
                    &raw,
                    self.cb,
                    self.target.image(),
                    bound_transfer_layout,
                    bound_layout,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
            } else {
                image_barrier(
                    &raw,
                    self.cb,
                    self.target.image(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
            }
            transfer_restore(&raw, self.cb, queue_family, other.image(), &other_state, to_other);

            begin_rendering(&raw, self.cb, self.target, self.buffer_size);
            self.bound_pipeline = vk::Pipeline::null();
            // Dynamic state persists across rendering scopes but re-set for clarity.
            let viewport = vk::Viewport::default()
                .width(self.buffer_size.w as f32)
                .height(self.buffer_size.h as f32)
                .max_depth(1.0);
            raw.cmd_set_viewport(self.cb, 0, &[viewport]);
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.buffer_size.w as u32,
                    height: self.buffer_size.h as u32,
                },
            };
            raw.cmd_set_scissor(self.cb, 0, &[scissor]);
        }

        if to_other {
            if let TargetInner::Texture { texture, .. } = &other.0 {
                let mut layout = texture.0.layout.lock().unwrap();
                if *layout == vk::ImageLayout::UNDEFINED {
                    *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                }
            }
        }

        // The blit completes with this frame's submission.
        Ok(SyncPoint::signaled())
    }
}

fn begin_rendering(
    raw: &ash::Device,
    cb: vk::CommandBuffer,
    target: &VulkanTarget<'_>,
    size: Size<i32, Physical>,
) {
    let color_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(target.view())
        .image_layout(target.render_layout())
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE);
    let color_attachments = [color_attachment];
    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: size.w as u32,
                height: size.h as u32,
            },
        })
        .layer_count(1)
        .color_attachments(&color_attachments);
    unsafe { raw.cmd_begin_rendering(cb, &rendering_info) };
}

/// Splits a 2D affine transform into its linear part and translation.
fn decompose(m: &Mat3) -> ([f32; 4], [f32; 2]) {
    (
        [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y],
        [m.z_axis.x, m.z_axis.y],
    )
}

// Copied from the GLES renderer to produce identical sampling behaviour.
fn build_texture_mat(
    src: Rectangle<f64, BufferCoord>,
    dest: Rectangle<i32, Physical>,
    texture: Size<i32, BufferCoord>,
    transform: Transform,
) -> Mat3 {
    let dst_src_size = transform.transform_size(src.size);
    let scale = dst_src_size.to_f64() / dest.size.to_f64();

    let mut tex_mat = Mat3::IDENTITY;

    // first bring the damage into src scale
    tex_mat = Mat3::from_scale(Vec2::new(scale.x as f32, scale.y as f32)) * tex_mat;

    // then compensate for the texture transform
    let transform_mat = transform.matrix();
    let translation = match transform {
        Transform::Normal => Mat3::IDENTITY,
        Transform::_90 => Mat3::from_translation(Vec2::new(0f32, dst_src_size.w as f32)),
        Transform::_180 => Mat3::from_translation(Vec2::new(dst_src_size.w as f32, dst_src_size.h as f32)),
        Transform::_270 => Mat3::from_translation(Vec2::new(dst_src_size.h as f32, 0f32)),
        Transform::Flipped => Mat3::from_translation(Vec2::new(dst_src_size.w as f32, 0f32)),
        Transform::Flipped90 => Mat3::IDENTITY,
        Transform::Flipped180 => Mat3::from_translation(Vec2::new(0f32, dst_src_size.h as f32)),
        Transform::Flipped270 => {
            Mat3::from_translation(Vec2::new(dst_src_size.h as f32, dst_src_size.w as f32))
        }
    };
    tex_mat = transform_mat * tex_mat;
    tex_mat = translation * tex_mat;

    // now we can add the src crop loc, the size already done implicit by the src size
    tex_mat = Mat3::from_translation(Vec2::new(src.loc.x as f32, src.loc.y as f32)) * tex_mat;

    // at last we have to normalize the values for UV space
    tex_mat = Mat3::from_scale(Vec2::new(
        (1.0f64 / texture.w as f64) as f32,
        (1.0f64 / texture.h as f64) as f32,
    )) * tex_mat;

    tex_mat
}

/// Guard type wrapping the underlying [`VulkanRenderer`] of a [`VulkanFrame`].
#[derive(Debug)]
pub struct VulkanFrameGuard<'a, 'frame> {
    renderer: &'a mut &'frame mut VulkanRenderer,
}

impl AsRef<VulkanRenderer> for VulkanFrameGuard<'_, '_> {
    fn as_ref(&self) -> &VulkanRenderer {
        self.renderer
    }
}

impl AsMut<VulkanRenderer> for VulkanFrameGuard<'_, '_> {
    fn as_mut(&mut self) -> &mut VulkanRenderer {
        self.renderer
    }
}

impl<'a, 'frame, 'buffer> FrameContext<'a, 'frame, 'buffer, VulkanRenderer> for VulkanFrame<'frame, 'buffer>
where
    'frame: 'a,
{
    type Guard = VulkanFrameGuard<'a, 'frame>;

    fn renderer(&'a mut self) -> Self::Guard {
        VulkanFrameGuard {
            renderer: &mut self.renderer,
        }
    }
}
