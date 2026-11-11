//! Removes text metadata from a generated image before it is written.
//!
//! mold embeds the prompt and every generation setting in each image it
//! returns when `embed_metadata` is on, which is the default in its
//! container (`MOLD_EMBED_METADATA=1`; found in the 2026-11-11 live run:
//! `mold:prompt`, `mold:parameters` and more as PNG `tEXt` chunks). A
//! prompt can hold private details, and a generated image is often shared,
//! so the text goes before the file is written. The pixels are untouched.

/// PNG chunks that carry text or capture details, never pixels:
/// `tEXt`/`zTXt`/`iTXt` (text, where mold writes), `eXIf` and `tIME`.
const PNG_METADATA_CHUNKS: &[&[u8; 4]] = &[b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"];

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// The image without its text metadata, or `None` when it couldn't be
/// parsed (the caller keeps the original and tells the agent). Formats
/// other than PNG and JPEG are returned unchanged: mold only sends those
/// two unless asked for another.
pub(crate) fn strip_metadata(content_type: &str, bytes: &[u8]) -> Option<Vec<u8>> {
    match content_type {
        "image/png" => strip_png(bytes),
        "image/jpeg" => strip_jpeg(bytes),
        _ => Some(bytes.to_vec()),
    }
}

fn strip_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut rest = bytes.strip_prefix(PNG_SIGNATURE)?;
    let mut out = PNG_SIGNATURE.to_vec();
    loop {
        let length = u32::from_be_bytes(rest.get(..4)?.try_into().ok()?) as usize;
        let total = length.checked_add(12)?;
        let chunk = rest.get(..total)?;
        let kind: &[u8; 4] = chunk[4..8].try_into().ok()?;
        if !PNG_METADATA_CHUNKS.contains(&kind) {
            out.extend_from_slice(chunk);
        }
        rest = &rest[total..];
        if kind == b"IEND" {
            return Some(out);
        }
    }
}

/// JPEG segments that carry text, never pixels: APP1 (EXIF, XMP), APP13
/// (IPTC) and COM. Everything from the start of scan on is copied as is.
fn strip_jpeg(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut rest = bytes.strip_prefix(&[0xFF, 0xD8])?;
    let mut out = vec![0xFF, 0xD8];
    loop {
        let marker = match rest.get(..2)? {
            [0xFF, m] => *m,
            _ => return None,
        };
        if marker == 0xDA || marker == 0xD9 {
            out.extend_from_slice(rest);
            return Some(out);
        }
        let length = usize::from(u16::from_be_bytes(rest.get(2..4)?.try_into().ok()?));
        let segment = rest.get(..length.checked_add(2)?)?;
        if !matches!(marker, 0xE1 | 0xED | 0xFE) {
            out.extend_from_slice(segment);
        }
        rest = &rest[segment.len()..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
        chunk.extend_from_slice(kind);
        chunk.extend_from_slice(data);
        chunk.extend_from_slice(&[0, 0, 0, 0]);
        chunk
    }

    fn png(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        for (kind, data) in chunks {
            bytes.extend(png_chunk(kind, data));
        }
        bytes
    }

    /// The shape mold 0.37 sends (2026-11-11 live run): text chunks with
    /// the prompt between the header and the pixels.
    #[test]
    fn png_text_chunks_are_removed_and_the_pixels_kept() {
        let with_text = png(&[
            (b"IHDR", &[0; 13]),
            (b"tEXt", b"mold:prompt\0a private prompt"),
            (
                b"iTXt",
                b"mold:parameters\0\0\0\0\0{\"prompt\":\"a private prompt\"}",
            ),
            (b"IDAT", b"pixels"),
            (b"tIME", &[0; 7]),
            (b"IEND", b""),
        ]);
        let stripped = strip_metadata("image/png", &with_text).unwrap();
        assert_eq!(
            stripped,
            png(&[(b"IHDR", &[0; 13]), (b"IDAT", b"pixels"), (b"IEND", b"")])
        );
        assert!(!stripped.windows(7).any(|w| w == b"private"));
    }

    #[test]
    fn a_png_without_metadata_is_unchanged() {
        let plain = png(&[(b"IHDR", &[0; 13]), (b"IDAT", b"pixels"), (b"IEND", b"")]);
        assert_eq!(strip_metadata("image/png", &plain).unwrap(), plain);
    }

    #[test]
    fn a_truncated_png_is_reported_rather_than_guessed_at() {
        let mut broken = png(&[(b"IHDR", &[0; 13]), (b"tEXt", b"mold:prompt\0secret")]);
        broken.truncate(broken.len() - 3);
        assert_eq!(strip_metadata("image/png", &broken), None);
        assert_eq!(strip_metadata("image/png", b"not a png"), None);
    }

    #[test]
    fn jpeg_exif_xmp_and_comments_are_removed() {
        let segment = |marker: u8, data: &[u8]| {
            let mut s = vec![0xFF, marker];
            s.extend_from_slice(&((data.len() + 2) as u16).to_be_bytes());
            s.extend_from_slice(data);
            s
        };
        let scan = [0xFF, 0xDA, 0x00, 0x02, 1, 2, 3, 0xFF, 0xD9];
        let mut with_text = vec![0xFF, 0xD8];
        with_text.extend(segment(0xE0, b"JFIF\0"));
        with_text.extend(segment(0xE1, b"Exif\0\0a private prompt"));
        with_text.extend(segment(0xFE, b"a private prompt"));
        with_text.extend(segment(0xDB, &[0; 4]));
        with_text.extend_from_slice(&scan);

        let mut expected = vec![0xFF, 0xD8];
        expected.extend(segment(0xE0, b"JFIF\0"));
        expected.extend(segment(0xDB, &[0; 4]));
        expected.extend_from_slice(&scan);
        assert_eq!(strip_metadata("image/jpeg", &with_text).unwrap(), expected);
    }

    #[test]
    fn other_formats_pass_through() {
        assert_eq!(
            strip_metadata("image/webp", b"RIFF....").unwrap(),
            b"RIFF....".to_vec()
        );
    }
}
