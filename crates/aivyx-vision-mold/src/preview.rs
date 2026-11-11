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
/// are read; a binary glTF mesh (`.glb`) is rendered from two angles
/// ([`mesh_preview`]); anything else is an error.
pub fn model_preview(path: &Path) -> Result<(String, Vec<u8>), VisionError> {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("glb"))
    {
        return mesh_preview(path);
    }
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

/// The most triangles [`mesh_preview`] draws; generated meshes have about
/// 40,000.
const MAX_PREVIEW_TRIANGLES: usize = 2_000_000;

/// The side of each of the two views [`mesh_preview`] draws.
const MESH_VIEW: u32 = 384;

/// A JPEG of the mesh in the binary glTF at `path`, drawn twice side by
/// side (from the front-left and the back-right, a little above), grey and
/// flat-shaded on white, so a model can see the shape it made. Buffers
/// must be inside the file; textures are ignored.
pub fn mesh_preview(path: &Path) -> Result<(String, Vec<u8>), VisionError> {
    let io = |e: std::io::Error| VisionError::Io(format!("reading {path:?} for a preview: {e}"));
    let len = std::fs::metadata(path).map_err(io)?.len();
    if len > MAX_SOURCE_BYTES {
        return Err(VisionError::InvalidRequest(format!(
            "{path:?} is too large to preview ({len} bytes)"
        )));
    }
    let bytes = std::fs::read(path).map_err(io)?;
    let triangles = mesh_triangles(&bytes)
        .map_err(|e| VisionError::InvalidRequest(format!("{path:?}: {e}")))?;
    let mut canvas =
        image::RgbImage::from_pixel(MESH_VIEW * 2, MESH_VIEW, image::Rgb([255, 255, 255]));
    for (i, yaw) in [35f32, 215f32].into_iter().enumerate() {
        draw_view(
            &mut canvas,
            &triangles,
            yaw.to_radians(),
            (i as u32) * MESH_VIEW,
        );
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(Cursor::new(&mut jpeg), 85)
        .encode_image(&canvas)
        .map_err(|e| VisionError::Internal(format!("encoding a preview of {path:?}: {e}")))?;
    Ok(("image/jpeg".to_string(), jpeg))
}

type Triangle = [[f32; 3]; 3];

/// Every triangle of the default scene, in world space.
fn mesh_triangles(glb: &[u8]) -> Result<Vec<Triangle>, String> {
    let gltf = gltf::Gltf::from_slice(glb).map_err(|e| format!("not a readable glTF: {e}"))?;
    let blob = gltf.blob.as_deref();
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or("the glTF has no scene")?;
    let mut triangles = Vec::new();
    let mut stack: Vec<(gltf::Node, [[f32; 4]; 4])> =
        scene.nodes().map(|n| (n, identity())).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = mul(&parent, &node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = primitive.reader(|buffer| match buffer.source() {
                    gltf::buffer::Source::Bin => blob,
                    gltf::buffer::Source::Uri(_) => None,
                });
                let Some(positions) = reader.read_positions() else {
                    continue;
                };
                let points: Vec<[f32; 3]> = positions.map(|p| apply(&world, p)).collect();
                let indices: Vec<u32> = match reader.read_indices() {
                    Some(indices) => indices.into_u32().collect(),
                    None => (0..points.len() as u32).collect(),
                };
                for tri in indices.as_chunks::<3>().0 {
                    if triangles.len() >= MAX_PREVIEW_TRIANGLES {
                        return Err("the mesh has too many triangles to preview".to_string());
                    }
                    let get = |i: u32| points.get(i as usize).copied();
                    if let (Some(a), Some(b), Some(c)) = (get(tri[0]), get(tri[1]), get(tri[2])) {
                        triangles.push([a, b, c]);
                    }
                }
            }
        }
        stack.extend(node.children().map(|child| (child, world)));
    }
    if triangles.is_empty() {
        return Err("the glTF has no triangles".to_string());
    }
    Ok(triangles)
}

/// Draws the mesh into a `MESH_VIEW` square at `x0`: turned by `yaw`
/// around the vertical axis and tilted 20 degrees down, scaled to fit,
/// orthographic, with a depth buffer and a light from the viewer's upper
/// left.
fn draw_view(canvas: &mut image::RgbImage, triangles: &[Triangle], yaw: f32, x0: u32) {
    let pitch = 20f32.to_radians();
    let (sy, cy, sp, cp) = (yaw.sin(), yaw.cos(), pitch.sin(), pitch.cos());
    let view = |p: [f32; 3]| {
        let x = p[0] * cy + p[2] * sy;
        let z = -p[0] * sy + p[2] * cy;
        [x, p[1] * cp - z * sp, p[1] * sp + z * cp]
    };
    let viewed: Vec<[[f32; 3]; 3]> = triangles.iter().map(|t| t.map(view)).collect();
    let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
    for v in viewed.iter().flatten() {
        for k in 0..2 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    let size = MESH_VIEW as f32;
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(f32::EPSILON);
    let scale = size * 0.9 / span;
    let centre = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
    let to_px = |v: [f32; 3]| {
        [
            size / 2.0 + (v[0] - centre[0]) * scale,
            size / 2.0 - (v[1] - centre[1]) * scale,
            v[2],
        ]
    };
    let light = normalise([-0.4, 0.6, 1.0]);
    let mut depth = vec![f32::MIN; (MESH_VIEW * MESH_VIEW) as usize];
    for tri in &viewed {
        let n = normalise(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
        // Two-sided: generated meshes don't promise a winding order.
        let shade = 0.25 + 0.7 * dot(n, light).abs();
        let grey = (shade.clamp(0.0, 1.0) * 220.0) as u8;
        let [a, b, c] = tri.map(to_px);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area.abs() < f32::EPSILON {
            continue;
        }
        let lo_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as u32;
        let hi_x = a[0].max(b[0]).max(c[0]).ceil().min(size - 1.0) as u32;
        let lo_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as u32;
        let hi_y = a[1].max(b[1]).max(c[1]).ceil().min(size - 1.0) as u32;
        for py in lo_y..=hi_y {
            for px in lo_x..=hi_x {
                let (x, y) = (px as f32 + 0.5, py as f32 + 0.5);
                let w0 = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / area;
                let w1 = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
                let slot = &mut depth[(py * MESH_VIEW + px) as usize];
                if z > *slot {
                    *slot = z;
                    canvas.put_pixel(
                        x0 + px,
                        py,
                        image::Rgb([grey, grey, grey.saturating_add(8)]),
                    );
                }
            }
        }
    }
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Column-major 4x4 product, as glTF stores matrices.
fn mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, cell) in out_col.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn apply(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|row| m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt().max(f32::EPSILON);
    [v[0] / len, v[1] / len, v[2] / len]
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

    /// A binary glTF of a tetrahedron, translated by its node.
    fn tetrahedron_glb() -> Vec<u8> {
        let positions: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices: [u16; 12] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        let mut bin: Vec<u8> = positions
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        bin.extend(indices.iter().flat_map(|i| i.to_le_bytes()));
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "translation": [5.0, 0.0, 0.0]}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"byteLength": bin.len()}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 48},
                {"buffer": 0, "byteOffset": 48, "byteLength": 24}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"}
            ]
        });
        let mut json = serde_json::to_vec(&json).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut glb = b"glTF".to_vec();
        glb.extend(2u32.to_le_bytes());
        glb.extend((total as u32).to_le_bytes());
        glb.extend((json.len() as u32).to_le_bytes());
        glb.extend(b"JSON");
        glb.extend(json);
        glb.extend((bin.len() as u32).to_le_bytes());
        glb.extend(b"BIN\0");
        glb.extend(bin);
        glb
    }

    #[test]
    fn a_mesh_is_drawn_from_two_angles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tetra.glb");
        std::fs::write(&path, tetrahedron_glb()).unwrap();
        let triangles = mesh_triangles(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(triangles.len(), 4);
        assert!(
            triangles.iter().flatten().all(|p| p[0] >= 5.0),
            "node transform applied"
        );

        let (media_type, jpeg) = model_preview(&path).unwrap();
        assert_eq!(media_type, "image/jpeg");
        let preview = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(preview.dimensions(), (MESH_VIEW * 2, MESH_VIEW));
        // Both views drew something: each half has dark (shaded) pixels.
        for half in 0..2 {
            let drawn = (0..MESH_VIEW)
                .flat_map(|y| (0..MESH_VIEW).map(move |x| (x + half * MESH_VIEW, y)))
                .filter(|&(x, y)| preview.get_pixel(x, y)[0] < 200)
                .count();
            assert!(drawn > 1000, "view {half}: {drawn} pixels drawn");
        }
    }

    #[test]
    fn a_broken_mesh_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.glb");
        std::fs::write(&path, b"glTF\x02\x00\x00\x00").unwrap();
        let err = model_preview(&path).unwrap_err();
        assert!(matches!(err, VisionError::InvalidRequest(_)), "{err}");
    }

    #[test]
    fn a_non_image_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an image").unwrap();
        let err = model_preview(&path).unwrap_err();
        assert!(matches!(err, VisionError::InvalidRequest(_)), "{err}");
    }
}
