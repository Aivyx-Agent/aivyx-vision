//! A small JPEG of a generated image, for a model to look at.
//!
//! The products show a model the images its tools made (so it can check
//! them and try again), and a preview keeps that cheap: a model sees an
//! image at a few hundred pixels anyway, and every image stays in the
//! conversation for the rest of it.

use std::io::Cursor;
use std::path::Path;

use aivyx_vision_core::VisionError;
use image::ImageReader;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;

/// The longest side of a preview, in pixels.
pub const PREVIEW_MAX_EDGE: u32 = 768;

/// The largest file a preview is made from; generated images are far
/// smaller, and decoding is bounded by this.
const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;

/// A JPEG of the image at `path`, at most [`PREVIEW_MAX_EDGE`] on its
/// longest side (never enlarged), with its media type. PNG, JPEG and WebP
/// are read; anything else is an error.
pub fn model_preview(path: &Path) -> Result<(String, Vec<u8>), VisionError> {
    let io = |e: std::io::Error| VisionError::Io(format!("reading {path:?} for a preview: {e}"));
    let len = std::fs::metadata(path).map_err(io)?.len();
    if len > MAX_SOURCE_BYTES {
        return Err(VisionError::InvalidRequest(format!(
            "{path:?} is too large to preview ({len} bytes)"
        )));
    }
    let image = ImageReader::open(path)
        .map_err(io)?
        .with_guessed_format()
        .map_err(io)?
        .decode()
        .map_err(|e| {
            VisionError::InvalidRequest(format!("{path:?} is not a readable image: {e}"))
        })?;
    let image = if image.width().max(image.height()) > PREVIEW_MAX_EDGE {
        image.resize(PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE, FilterType::Triangle)
    } else {
        image
    };
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(Cursor::new(&mut jpeg), 85)
        .encode_image(&image.to_rgb8())
        .map_err(|e| VisionError::Internal(format!("encoding a preview of {path:?}: {e}")))?;
    Ok(("image/jpeg".to_string(), jpeg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(path: &Path, width: u32, height: u32) {
        image::RgbImage::from_pixel(width, height, image::Rgb([200, 150, 40]))
            .save(path)
            .unwrap();
    }

    #[test]
    fn a_large_image_shrinks_to_the_preview_edge_keeping_its_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.png");
        write_png(&path, 1536, 1152);
        let (media_type, jpeg) = model_preview(&path).unwrap();
        assert_eq!(media_type, "image/jpeg");
        let preview = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((preview.width(), preview.height()), (768, 576));
    }

    #[test]
    fn a_small_image_is_not_enlarged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.png");
        write_png(&path, 300, 200);
        let (_, jpeg) = model_preview(&path).unwrap();
        let preview = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((preview.width(), preview.height()), (300, 200));
    }

    #[test]
    fn a_non_image_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.glb");
        std::fs::write(&path, b"glTF\x02\x00\x00\x00").unwrap();
        let err = model_preview(&path).unwrap_err();
        assert!(matches!(err, VisionError::InvalidRequest(_)), "{err}");
    }
}
