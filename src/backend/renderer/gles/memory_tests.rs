use super::*;
use crate::backend::egl::{EGLDisplay, native::EGLSurfacelessDisplay};

fn renderer() -> Option<GlesRenderer> {
    let result = (|| -> Result<_, String> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.map_err(|e| e.to_string())?;
        let context = EGLContext::new(&display).map_err(|e| e.to_string())?;
        unsafe { GlesRenderer::new(context) }.map_err(|e| e.to_string())
    })();
    match result {
        Ok(renderer) => Some(renderer),
        Err(err) => {
            if std::env::var_os("SMITHAY_TEST_REQUIRE_GLES").is_some_and(|v| !v.is_empty() && v != "0") {
                panic!("GLES memory test requires a renderer: {err}");
            }
            eprintln!("skipping GLES memory test: {err}");
            None
        }
    }
}

#[test]
fn packed_lengths_reject_invalid_or_overflowing_dimensions() {
    assert_eq!(packed_buffer_len((3, 5).into(), 8).unwrap(), 120);
    for size in [(0, 4), (-1, 4), (4, -1), (i32::MAX, i32::MAX)] {
        // Size::new rejects negative dimensions before the renderer can see them, but its
        // public fields may still be changed by callers.
        let mut dimensions: Size<i32, BufferCoord> = (1, 1).into();
        dimensions.w = size.0;
        dimensions.h = size.1;
        assert!(packed_buffer_len(dimensions, 8).is_err());
    }
}

#[test]
fn hdr_readback_maps_every_half_float_component() {
    let Some(mut renderer) = renderer() else { return };
    let pixel = [0x3800u16, 0x3c00, 0x4000, 0x3c00]; // 0.5, 1, 2, 1
    let bytes: Vec<_> = pixel
        .into_iter()
        .cycle()
        .take(24)
        .flat_map(u16::to_ne_bytes)
        .collect();
    let texture = renderer
        .import_memory(&bytes, Fourcc::Abgr16161616f, (3, 2).into(), false)
        .unwrap();
    let mapping = renderer
        .copy_texture(
            &texture,
            Rectangle::from_size((3, 2).into()),
            Fourcc::Abgr16161616f,
        )
        .unwrap();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), bytes);
    // Mapping an already mapped PBO must use the same format-specific length too.
    assert_eq!(renderer.map_texture(&mapping).unwrap().len(), 3 * 2 * 8);
    renderer
        .with_context(|gl| assert_eq!(unsafe { gl.GetError() }, ffi::NO_ERROR))
        .unwrap();
}

#[test]
fn odd_width_rgb_upload_uses_packed_rows() {
    let Some(mut renderer) = renderer() else { return };
    let name = renderer
        .with_context(|gl| unsafe {
            let mut name = 0;
            gl.GenTextures(1, &mut name);
            gl.BindTexture(ffi::TEXTURE_2D, name);
            gl.TexImage2D(
                ffi::TEXTURE_2D,
                0,
                ffi::RGB8 as i32,
                3,
                3,
                0,
                ffi::RGB,
                ffi::UNSIGNED_BYTE,
                ptr::null(),
            );
            gl.BindTexture(ffi::TEXTURE_2D, 0);
            name
        })
        .unwrap();
    let texture = unsafe { GlesTexture::from_raw(&renderer, Some(ffi::RGB8), true, name, (3, 3).into()) };
    let bytes: Vec<_> = (0..9).flat_map(|index| [(index * 20) as u8, 100, 200]).collect();
    renderer
        .update_memory(&texture, &bytes, Rectangle::from_size((3, 3).into()))
        .unwrap();
    let mapping = renderer
        .copy_texture(&texture, Rectangle::from_size((3, 3).into()), Fourcc::Abgr8888)
        .unwrap();
    let expected: Vec<_> = bytes
        .chunks_exact(3)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
        .collect();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), expected);
    renderer
        .with_context(|gl| assert_eq!(unsafe { gl.GetError() }, ffi::NO_ERROR))
        .unwrap();
}

#[test]
fn partial_upload_requires_full_staging_and_preserves_untouched_pixels() {
    let Some(mut renderer) = renderer() else { return };
    let size: Size<i32, BufferCoord> = (5, 4).into();
    let base = [10u8, 20, 30, 255].repeat(20);
    let texture = renderer
        .import_memory(&base, Fourcc::Abgr8888, size, false)
        .unwrap();
    let regions = [
        Rectangle::new((1, 1).into(), (2, 2).into()),
        Rectangle::new((4, 3).into(), (1, 1).into()),
    ];
    let mut expected = base.clone();
    let mut staging = base.clone();
    for (index, pixel) in staging.chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&[(index * 9) as u8, 200, 100, 255]);
    }
    for region in regions {
        assert!(matches!(
            renderer.update_memory(&texture, &[200; 16], region),
            Err(GlesError::UnexpectedSize)
        ));
        renderer.update_memory(&texture, &staging, region).unwrap();
        for y in region.loc.y..region.loc.y + region.size.h {
            for x in region.loc.x..region.loc.x + region.size.w {
                let offset = ((y * size.w + x) * 4) as usize;
                expected[offset..offset + 4].copy_from_slice(&staging[offset..offset + 4]);
            }
        }
    }
    for region in [
        Rectangle::new((-1, 0).into(), (1, 1).into()),
        Rectangle::new((4, 3).into(), (2, 1).into()),
        Rectangle::new((i32::MAX, 0).into(), (1, 1).into()),
    ] {
        assert!(matches!(
            renderer.update_memory(&texture, &staging, region),
            Err(GlesError::UnexpectedSize)
        ));
    }
    let mapping = renderer
        .copy_texture(&texture, Rectangle::from_size(size), Fourcc::Abgr8888)
        .unwrap();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), expected);
    renderer
        .with_context(|gl| assert_eq!(unsafe { gl.GetError() }, ffi::NO_ERROR))
        .unwrap();
}
