//! `GenerationProvider` implementation backed by `mold serve`: acquires
//! the GPU lock, calls mold's synchronous `/api/generate`, always
//! releases the lock (success or failure), writes the returned image
//! bytes to disk, and returns a `GeneratedAsset`. `generate_3d` returns
//! `VisionError::Unsupported` -- see this crate's own top-level doc
//! comment for why.
//!
//! `generate_image`'s own acquire/generate/release/write sequence runs on
//! an internally spawned task, detached from the caller's own future --
//! see `generate_image`'s doc comment for why a `Drop`-based release
//! guard would have been the wrong fix.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use aivyx_vision_core::{
    GeneratedAsset, GenerationProvider, ImageRequest, ThreeDRequest, VisionError,
};
use async_trait::async_trait;
use base64::Engine;

use crate::gpu_lock_client::{GpuLockClient, GpuLockClientError};
use crate::mold_client::{
    GenerateImageRequest, GenerateImageResponse, MoldClient, MoldClientError, MoldModel,
};

/// `mold serve`'s own HTTP read timeout -- generation is real, wall-clock
/// GPU work, not a quick API call. Matches `aivyx-broker/src/main.rs`'s
/// own `read_timeout` reasoning for the same class of problem (a cold
/// generation routinely exceeds simple defaults). Not yet configurable
/// via `MoldConfig` -- a fixed v1 default, revisit if a real workload
/// needs otherwise.
const MOLD_READ_TIMEOUT: Duration = Duration::from_secs(300);

pub struct MoldConfig {
    pub broker_url: String,
    pub mold_url: String,
    pub api_key: Option<String>,
    pub output_dir: PathBuf,
    /// The mold model to use (for example `flux2-klein:q8`). `None` uses
    /// the image model that is downloaded on the server; see
    /// [`resolve_model`].
    pub model: Option<String>,
}

pub struct MoldProvider {
    gpu_lock: GpuLockClient,
    mold: MoldClient,
    output_dir: PathBuf,
    model: Option<String>,
}

impl MoldProvider {
    pub fn new(config: MoldConfig) -> Result<Self, VisionError> {
        let http = build_http_client()?;

        // Trimmed here, once, rather than at each of the three `format!`
        // call sites inside `mold_client.rs`/`gpu_lock_client.rs` -- a
        // trailing slash on either configured URL would otherwise produce
        // a double-slash request path (e.g. `.../api/generate` becoming
        // `..//api/generate`), which may not route correctly.
        let mold_url = config.mold_url.trim_end_matches('/').to_string();
        let broker_url = config.broker_url.trim_end_matches('/').to_string();

        Ok(Self {
            gpu_lock: GpuLockClient::new(http.clone(), broker_url),
            mold: MoldClient::new(http, mold_url, config.api_key),
            output_dir: config.output_dir,
            model: config.model,
        })
    }
}

#[async_trait]
impl GenerationProvider for MoldProvider {
    /// Detached from the caller's own future on purpose. A `Drop`-based
    /// release guard would have been the *wrong* fix here: releasing the
    /// lock the instant the caller's future is dropped could free it
    /// before `mold serve` has actually finished the in-flight
    /// generation, letting a second caller start a second GPU job while
    /// the first is still running. Spawning the whole
    /// `acquire -> generate -> release -> write` sequence and only
    /// `.await`ing the `JoinHandle` here means a cancelled caller
    /// abandons *waiting on the result*, never the operation itself --
    /// the lock is only ever released after generation has genuinely
    /// finished, cancelled or not. See
    /// `docs/superpowers/plans/2026-09-19-gpu-lock-cancellation-safety.md`
    /// for the full design rationale, if useful.
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
        let gpu_lock = self.gpu_lock.clone();
        let mold = self.mold.clone();
        let output_dir = self.output_dir.clone();
        let configured_model = self.model.clone();

        let handle = tokio::spawn(async move {
            // Built (and the reference image read) before the lock is
            // taken: a bad path or an oversized file fails without a broker
            // round trip, and the lock is never held for local file I/O.
            let (mut mold_req, mut notes) =
                tokio::task::spawn_blocking(move || build_mold_request(&req))
                    .await
                    .map_err(|e| {
                        VisionError::Internal(format!("reading the reference image: {e}"))
                    })??;
            // mold 0.37 requires model, width, height and steps (found in the
            // 2026-11-11 live run); fill whatever the caller left unset from
            // the model's own defaults.
            let models = mold.models().await.map_err(map_mold_error)?;
            let (chosen, model_notes) = resolve_model(configured_model.as_deref(), &models)?;
            notes.extend(model_notes);
            notes.extend(fill_from_model(&mut mold_req, chosen));

            let lease = gpu_lock.acquire().await.map_err(map_gpu_lock_error)?;
            let result = mold.generate(mold_req).await.map_err(map_mold_error);
            if let Err(e) = gpu_lock.release(&lease).await {
                tracing::warn!(
                    error = %e,
                    "gpu-lock release failed after generation; aivyx-broker's reap_expired will eventually reclaim it"
                );
            }

            let response = result?;
            let mut asset =
                tokio::task::spawn_blocking(move || write_generated_asset(&output_dir, response))
                    .await
                    .map_err(|e| {
                        VisionError::Internal(format!("writing the generated asset: {e}"))
                    })??;
            notes.append(&mut asset.warnings);
            asset.warnings = notes;
            Ok(asset)
        });

        handle
            .await
            .map_err(|e| VisionError::Internal(format!("generation task panicked: {e}")))?
    }

    async fn generate_3d(&self, _req: ThreeDRequest) -> Result<GeneratedAsset, VisionError> {
        Err(VisionError::Unsupported(
            "3D generation via mold (Pass B is not yet implemented)",
        ))
    }
}

/// No redirects: a redirect would carry the `X-Api-Key` header (and the
/// prompt, on 307/308) to whatever host the endpoint names, and hand the
/// body of an arbitrary URL to the agent as "the generated image".
///
/// No proxy: `mold serve` and `aivyx-broker` are the operator's own local
/// services, and reqwest otherwise sends even loopback requests -- with the
/// prompt, the API key and any reference image -- to `HTTP_PROXY` /
/// `ALL_PROXY` when one is set (audit 2026-11-11 X1).
fn build_http_client() -> Result<reqwest::Client, VisionError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .read_timeout(MOLD_READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|e| VisionError::BackendUnreachable(format!("failed to build HTTP client: {e}")))
}

/// A reference image larger than this is refused rather than read whole.
const MAX_REFERENCE_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

/// Reads a reference image with a size cap. `path` must already be
/// validated by the caller (this does not restrict where it points).
fn read_reference_image(path: &Path) -> Result<Vec<u8>, VisionError> {
    use std::io::Read;
    let io = |e: std::io::Error| VisionError::Io(format!("reading reference image {path:?}: {e}"));
    let file = std::fs::File::open(path).map_err(io)?;
    let mut bytes = Vec::new();
    file.take(MAX_REFERENCE_IMAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() as u64 > MAX_REFERENCE_IMAGE_BYTES {
        return Err(VisionError::Io(format!(
            "reference image {path:?} is larger than the {MAX_REFERENCE_IMAGE_BYTES}-byte limit"
        )));
    }
    Ok(bytes)
}

/// Model families that make still images from a prompt. mold also serves
/// video (`ltx-video`, `ltx2`, `wan`), 3D (`hunyuan3d`), upscalers,
/// controlnets and prompt expanders, none of which `/api/generate` should
/// get here.
const IMAGE_FAMILIES: &[&str] = &[
    "flux",
    "flux2",
    "qwen-image",
    "qwen-image21",
    "sd15",
    "sd3",
    "sdxl",
    "wuerstchen",
    "z-image",
];

/// Picks the model to generate with, and any note for the agent about the
/// choice:
///
/// - a configured model must be one mold knows (it may not be downloaded
///   yet: mold pulls on first use, which is slow, so that gets a note);
/// - otherwise the downloaded image-family model; with several, the first
///   by name, with a note saying how to choose.
fn resolve_model<'a>(
    configured: Option<&str>,
    models: &'a [MoldModel],
) -> Result<(&'a MoldModel, Vec<String>), VisionError> {
    if let Some(name) = configured {
        let model =
            models
                .iter()
                .find(|m| m.name == name)
                .ok_or_else(|| VisionError::BackendError {
                    status: 404,
                    code: Some("MODEL_NOT_FOUND".to_string()),
                    message: format!("mold serve doesn't know the configured model {name:?}"),
                })?;
        let notes = if model.downloaded {
            Vec::new()
        } else {
            vec![format!(
                "mold had to download {name} before generating, so this took longer"
            )]
        };
        return Ok((model, notes));
    }
    let mut candidates: Vec<&MoldModel> = models
        .iter()
        .filter(|m| m.downloaded && IMAGE_FAMILIES.contains(&m.family.as_str()))
        .collect();
    candidates.sort_by(|a, b| a.name.cmp(&b.name));
    match candidates.as_slice() {
        [] => Err(VisionError::BackendError {
            status: 404,
            code: Some("MODEL_NOT_FOUND".to_string()),
            message: "mold serve has no image model downloaded; pull one (for example \
                      `mold pull flux2-klein:q8`) or configure the model to use"
                .to_string(),
        }),
        [only] => Ok((only, Vec::new())),
        [first, ..] => Ok((
            first,
            vec![format!(
                "several image models are downloaded on mold serve; used {} (configure the \
                 model to choose another)",
                first.name
            )],
        )),
    }
}

/// Fills the fields mold requires from `model`'s defaults, keeping
/// whatever the caller set, and fits the size to what mold accepts for the
/// model: a multiple of its `dimension_alignment`, and no more than its
/// `max_pixels` (mold refuses anything else). Returns a note for the agent
/// when the size changed.
fn fill_from_model(req: &mut GenerateImageRequest, model: &MoldModel) -> Vec<String> {
    req.model.get_or_insert_with(|| model.name.clone());
    if req.steps.is_none() {
        req.steps = model.default_steps;
    }
    let asked = (req.width, req.height);
    let align = model.dimension_alignment.filter(|a| *a > 1);
    let fit = |value: Option<u32>, default: Option<u32>| match (value, align) {
        (Some(v), Some(a)) => Some(((v + a / 2) / a).max(1) * a),
        (Some(v), None) => Some(v),
        (None, _) => default,
    };
    req.width = fit(req.width, model.default_width);
    req.height = fit(req.height, model.default_height);
    let show = |v: Option<u32>| v.map_or("default".to_string(), |v| v.to_string());

    if let (Some(w), Some(h), Some(max)) = (req.width, req.height, model.max_pixels)
        && u64::from(w) * u64::from(h) > max
    {
        // Scale both sides by the same factor, then round down to the grid,
        // so the result keeps the shape and stays under the limit.
        let scale = (max as f64 / (f64::from(w) * f64::from(h))).sqrt();
        let a = align.unwrap_or(1);
        let down = |v: u32| (((f64::from(v) * scale) as u32) / a).max(1) * a;
        req.width = Some(down(w));
        req.height = Some(down(h));
        return vec![format!(
            "{} allows at most {max} pixels: generated {}x{} instead of {}x{}",
            model.name,
            show(req.width),
            show(req.height),
            show(asked.0.or(Some(w))),
            show(asked.1.or(Some(h))),
        )];
    }

    let rounded_width = asked.0.is_some() && asked.0 != req.width;
    let rounded_height = asked.1.is_some() && asked.1 != req.height;
    if !rounded_width && !rounded_height {
        return Vec::new();
    }
    vec![format!(
        "{} needs sizes in multiples of {}: generated {}x{} instead of {}x{}",
        model.name,
        align.unwrap_or(1),
        show(req.width),
        show(req.height),
        show(asked.0),
        show(asked.1),
    )]
}

/// Turns the backend-agnostic request into mold's, with any note for the
/// agent. With a reference image, a missing size comes from the reference
/// so the result keeps the picture's shape: mold otherwise renders at the
/// model's default size (2026-11-11 live run: a 1024x768 reference came
/// back 1024x1024 with its layout lost).
fn build_mold_request(
    req: &ImageRequest,
) -> Result<(GenerateImageRequest, Vec<String>), VisionError> {
    let mut notes = Vec::new();
    if let Some(strength) = req.strength
        && !(0.0..=1.0).contains(&strength)
    {
        return Err(VisionError::InvalidRequest(format!(
            "strength must be between 0.0 and 1.0, got {strength}"
        )));
    }
    let prompt = match &req.style_hint {
        Some(hint) => format!("{}, {hint}", req.prompt),
        None => req.prompt.clone(),
    };
    let (mut width, mut height) = (req.width, req.height);
    let mut strength = req.strength;
    let source_image = match &req.reference_image {
        Some(path) => {
            let bytes = read_reference_image(path)?;
            if width.is_none() || height.is_none() {
                match imagesize::blob_size(&bytes) {
                    Ok(size) if size.width > 0 && size.height > 0 => {
                        (width, height) = size_from_reference(
                            (width, height),
                            (size.width as u64, size.height as u64),
                        );
                    }
                    _ => notes.push(
                        "couldn't read the reference image's size; used the model's default \
                         size, so the result may be a different shape"
                            .to_string(),
                    ),
                }
            }
            Some(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        None => {
            if strength.take().is_some() {
                notes.push(
                    "strength only applies to image-to-image (with a reference image); ignored"
                        .to_string(),
                );
            }
            None
        }
    };
    Ok((
        GenerateImageRequest {
            prompt,
            width,
            height,
            seed: req.seed,
            source_image,
            strength,
            ..Default::default()
        },
        notes,
    ))
}

/// Fills a missing width or height from the reference image's size: both
/// when neither is set, otherwise the missing side in the reference's
/// proportions.
fn size_from_reference(
    asked: (Option<u32>, Option<u32>),
    (ref_w, ref_h): (u64, u64),
) -> (Option<u32>, Option<u32>) {
    let clamp = |v: u64| u32::try_from(v.max(1)).unwrap_or(u32::MAX);
    match asked {
        (None, None) => (Some(clamp(ref_w)), Some(clamp(ref_h))),
        (Some(w), None) => (Some(w), Some(clamp(u64::from(w) * ref_h / ref_w))),
        (None, Some(h)) => (Some(clamp(u64::from(h) * ref_w / ref_h)), Some(h)),
        both => both,
    }
}

/// Writes the image as a new file only its owner can read: generated images
/// can show private content. A directory this creates is owner-only too; an
/// existing one keeps its permissions (in aivyx-coder it is the project's
/// own `assets/generated/`). Unix only; elsewhere the platform defaults
/// apply.
fn write_generated_asset(
    output_dir: &Path,
    response: GenerateImageResponse,
) -> Result<GeneratedAsset, VisionError> {
    let mut dir = std::fs::DirBuilder::new();
    dir.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut dir, 0o700);
    dir.create(output_dir)
        .map_err(|e| VisionError::Io(format!("creating output dir {output_dir:?}: {e}")))?;
    let ext = extension_for_content_type(&response.content_type);
    let path = output_dir.join(format!("{}.{ext}", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    // mold embeds the prompt and settings in the image by default; a
    // prompt can be private and the file is often shared (live run
    // 2026-11-11), so they don't reach the file.
    let mut warnings = response.warnings;
    let bytes = match crate::metadata::strip_metadata(&response.content_type, &response.bytes) {
        Some(bytes) => bytes,
        None => {
            warnings.push(
                "couldn't read the image to remove its embedded metadata; the file may \
                 contain the prompt"
                    .to_string(),
            );
            response.bytes
        }
    };
    let write = |options: &std::fs::OpenOptions| -> std::io::Result<()> {
        use std::io::Write;
        options.open(&path)?.write_all(&bytes)
    };
    write(&options)
        .map_err(|e| VisionError::Io(format!("writing generated asset {path:?}: {e}")))?;
    Ok(GeneratedAsset {
        path,
        backend: "mold".to_string(),
        seed_used: response.seed_used,
        generated_at: SystemTime::now(),
        warnings,
    })
}

fn extension_for_content_type(content_type: &str) -> &'static str {
    match content_type {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        _ => "bin",
    }
}

/// Only used for `acquire()`'s own error path. `GpuLockClientError` is
/// one enum shared with `release()` (whose failures are handled
/// separately, as a logged warning -- see `generate_image` above), so
/// this match still covers `ReleaseRejected` for exhaustiveness even
/// though `acquire()` never actually produces it.
fn map_gpu_lock_error(e: GpuLockClientError) -> VisionError {
    match e {
        GpuLockClientError::Timeout => VisionError::GpuLockTimeout,
        GpuLockClientError::Transport(msg) => {
            VisionError::BackendUnreachable(format!("aivyx-broker: {msg}"))
        }
        GpuLockClientError::ReleaseRejected(msg) => {
            VisionError::BackendUnreachable(format!("aivyx-broker: {msg}"))
        }
        GpuLockClientError::ResponseTooLarge(cap) => VisionError::BackendUnreachable(format!(
            "aivyx-broker: response exceeded the {cap}-byte limit"
        )),
    }
}

fn map_mold_error(e: MoldClientError) -> VisionError {
    match e {
        MoldClientError::Transport(msg) => {
            VisionError::BackendUnreachable(format!("mold serve: {msg}"))
        }
        MoldClientError::Timeout => VisionError::GenerationTimeout,
        MoldClientError::ModelNotFound(msg) => VisionError::BackendError {
            status: 404,
            code: Some("MODEL_NOT_FOUND".to_string()),
            message: msg,
        },
        MoldClientError::QueueFull => VisionError::BackendError {
            status: 503,
            code: Some("QUEUE_FULL".to_string()),
            message: "mold serve's own generation queue is full".to_string(),
        },
        MoldClientError::BackendError {
            status,
            code,
            message,
        } => VisionError::BackendError {
            status,
            code,
            message,
        },
        MoldClientError::ResponseTooLarge(cap) => VisionError::BackendUnreachable(format!(
            "mold serve: response exceeded the {cap}-byte limit"
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn an_oversized_reference_image_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.png");
        let f = std::fs::File::create(&big).unwrap();
        f.set_len(MAX_REFERENCE_IMAGE_BYTES + 1).unwrap();
        let err = read_reference_image(&big).unwrap_err().to_string();
        assert!(err.contains("limit"), "{err}");
        let ok = dir.path().join("ok.png");
        std::fs::write(&ok, b"png").unwrap();
        assert_eq!(read_reference_image(&ok).unwrap(), b"png");
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed_and_the_api_key_goes_nowhere_else() {
        let mold = MockServer::start().await;
        let elsewhere = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("location", format!("{}/api/generate", elsewhere.uri())),
            )
            .mount(&mold)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1, 2, 3]))
            .expect(0)
            .mount(&elsewhere)
            .await;
        let client = MoldClient::new(
            build_http_client().unwrap(),
            mold.uri(),
            Some("secret-key".to_string()),
        );
        let result = client
            .generate(GenerateImageRequest {
                prompt: "x".to_string(),
                ..Default::default()
            })
            .await;
        assert!(
            result.is_err(),
            "a redirect must not count as a generated image"
        );
        assert!(elsewhere.received_requests().await.unwrap().is_empty());
    }

    async fn test_provider(
        broker: &MockServer,
        mold: &MockServer,
        output_dir: PathBuf,
    ) -> MoldProvider {
        // Every generation asks mold for its models first (to fill the
        // fields mold requires); one downloaded image model, as in the
        // 2026-11-11 live run.
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"name": "flux2-klein:q8", "family": "flux2", "downloaded": true,
                 "default_steps": 4, "default_width": 1024, "default_height": 1024},
                {"name": "ltx2:q8", "family": "ltx2", "downloaded": true,
                 "default_steps": 8, "default_width": 768, "default_height": 512}
            ])))
            .mount(mold)
            .await;
        MoldProvider::new(MoldConfig {
            broker_url: broker.uri(),
            mold_url: mold.uri(),
            api_key: None,
            output_dir,
            model: None,
        })
        .unwrap()
    }

    fn sample_image_request() -> ImageRequest {
        ImageRequest {
            prompt: "anything".to_string(),
            reference_image: None,
            width: None,
            height: None,
            seed: None,
            strength: None,
            style_hint: None,
        }
    }

    #[tokio::test]
    async fn generate_image_success_writes_a_file_and_returns_a_generated_asset() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();

        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-1"})),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&broker)
            .await;
        let png_bytes = vec![1, 2, 3, 4];
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(png_bytes.clone())
                    .insert_header("content-type", "image/png")
                    .insert_header("x-mold-seed-used", "7"),
            )
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;
        let asset = provider
            .generate_image(ImageRequest {
                prompt: "a red circle".to_string(),
                ..sample_image_request()
            })
            .await
            .unwrap();

        assert_eq!(asset.backend, "mold");
        assert_eq!(asset.seed_used, Some(7));
        assert_eq!(std::fs::read(&asset.path).unwrap(), png_bytes);
        assert_eq!(asset.path.extension().unwrap(), "png");
    }

    #[tokio::test]
    async fn generate_image_releases_the_lease_even_when_generation_fails() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();

        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-2"})),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": "inference crashed",
                "code": "INFERENCE_ERROR"
            })))
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;
        let err = provider
            .generate_image(sample_image_request())
            .await
            .unwrap_err();

        assert!(matches!(err, VisionError::BackendError { status: 500, .. }));
        // The release mock's `.expect(1)` above is checked here, explicitly,
        // rather than relying on Drop timing.
        broker.verify().await;
    }

    #[tokio::test]
    async fn generate_image_returns_gpu_lock_timeout_and_never_calls_mold_when_the_lock_is_busy() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();

        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(503).set_body_json(
                    serde_json::json!({"error": "timed out waiting for the GPU lock"}),
                ),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;
        let err = provider
            .generate_image(sample_image_request())
            .await
            .unwrap_err();

        assert!(matches!(err, VisionError::GpuLockTimeout));
        mold.verify().await;
    }

    #[tokio::test]
    async fn generate_3d_returns_unsupported_without_touching_the_network() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();
        // No mocks registered on either server at all -- if generate_3d ever
        // made a real request, wiremock's default unmatched-request behavior
        // would surface it as a connection-level response mismatch, but we
        // don't even need that: the assertion below is on the error variant
        // alone, and this crate's own generate_3d body (Step 3) never
        // constructs a request in the first place.
        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;

        let err = provider
            .generate_3d(ThreeDRequest {
                prompt: "a small statue".to_string(),
                reference_image: None,
                style_hint: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, VisionError::Unsupported(_)));
    }

    #[test]
    fn build_mold_request_appends_the_style_hint_to_the_prompt() {
        let req = ImageRequest {
            prompt: "a mountain".to_string(),
            style_hint: Some("watercolor".to_string()),
            ..sample_image_request()
        };
        let (mold_req, _) = build_mold_request(&req).unwrap();
        assert_eq!(mold_req.prompt, "a mountain, watercolor");
    }

    #[test]
    fn build_mold_request_base64_encodes_the_reference_image_file() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("ref.png");
        std::fs::write(&image_path, [1u8, 2, 3, 4]).unwrap();

        let req = ImageRequest {
            prompt: "edit this".to_string(),
            reference_image: Some(image_path),
            ..sample_image_request()
        };
        let (mold_req, _) = build_mold_request(&req).unwrap();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(mold_req.source_image.unwrap())
            .unwrap();
        assert_eq!(decoded, vec![1, 2, 3, 4]);
    }

    /// A PNG this crate can parse: signature, header, end.
    fn tiny_png() -> Vec<u8> {
        let mut bytes = png_header(1, 1);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(b"IEND");
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes
    }

    /// The start of a PNG: enough for its size to be read.
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
        bytes
    }

    fn reference_request(dir: &Path, width: u32, height: u32) -> ImageRequest {
        let image_path = dir.join("ref.png");
        std::fs::write(&image_path, png_header(width, height)).unwrap();
        ImageRequest {
            prompt: "the same scene at night".to_string(),
            reference_image: Some(image_path),
            ..sample_image_request()
        }
    }

    /// 2026-11-11 live run: a 1024x768 reference with no size came back
    /// 1024x1024 (the model's default) and lost the picture's layout.
    #[test]
    fn build_mold_request_takes_the_size_from_the_reference_image() {
        let dir = tempfile::tempdir().unwrap();
        let (mold_req, notes) =
            build_mold_request(&reference_request(dir.path(), 1024, 768)).unwrap();
        assert_eq!((mold_req.width, mold_req.height), (Some(1024), Some(768)));
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn build_mold_request_keeps_the_reference_proportions_when_one_side_is_given() {
        let dir = tempfile::tempdir().unwrap();
        let req = ImageRequest {
            width: Some(600),
            ..reference_request(dir.path(), 1200, 800)
        };
        let (mold_req, _) = build_mold_request(&req).unwrap();
        assert_eq!((mold_req.width, mold_req.height), (Some(600), Some(400)));

        let req = ImageRequest {
            height: Some(400),
            ..reference_request(dir.path(), 1200, 800)
        };
        let (mold_req, _) = build_mold_request(&req).unwrap();
        assert_eq!((mold_req.width, mold_req.height), (Some(600), Some(400)));
    }

    #[test]
    fn build_mold_request_keeps_an_explicit_size_with_a_reference_image() {
        let dir = tempfile::tempdir().unwrap();
        let req = ImageRequest {
            width: Some(512),
            height: Some(512),
            ..reference_request(dir.path(), 1024, 768)
        };
        let (mold_req, _) = build_mold_request(&req).unwrap();
        assert_eq!((mold_req.width, mold_req.height), (Some(512), Some(512)));
    }

    #[test]
    fn build_mold_request_notes_a_reference_image_whose_size_cannot_be_read() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("ref.png");
        std::fs::write(&image_path, [1u8, 2, 3, 4]).unwrap();
        let req = ImageRequest {
            reference_image: Some(image_path),
            ..sample_image_request()
        };
        let (mold_req, notes) = build_mold_request(&req).unwrap();
        assert_eq!((mold_req.width, mold_req.height), (None, None));
        assert!(
            notes[0].contains("couldn't read the reference image's size"),
            "{notes:?}"
        );
    }

    #[test]
    fn build_mold_request_passes_strength_with_a_reference_image() {
        let dir = tempfile::tempdir().unwrap();
        let req = ImageRequest {
            strength: Some(0.9),
            ..reference_request(dir.path(), 1024, 768)
        };
        let (mold_req, notes) = build_mold_request(&req).unwrap();
        assert_eq!(mold_req.strength, Some(0.9));
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn build_mold_request_drops_strength_without_a_reference_image_and_says_so() {
        let req = ImageRequest {
            strength: Some(0.5),
            ..sample_image_request()
        };
        let (mold_req, notes) = build_mold_request(&req).unwrap();
        assert_eq!(mold_req.strength, None);
        assert!(notes[0].contains("strength"), "{notes:?}");
    }

    #[test]
    fn build_mold_request_refuses_a_strength_outside_zero_to_one() {
        for strength in [-0.1, 1.5, f32::NAN] {
            let req = ImageRequest {
                strength: Some(strength),
                ..sample_image_request()
            };
            let err = build_mold_request(&req).unwrap_err();
            assert!(
                matches!(err, VisionError::InvalidRequest(_)),
                "{strength}: {err}"
            );
        }
    }

    #[test]
    fn build_mold_request_errors_clearly_when_the_reference_image_cannot_be_read() {
        let req = ImageRequest {
            prompt: "edit this".to_string(),
            reference_image: Some(PathBuf::from("/nonexistent/path/ref.png")),
            ..sample_image_request()
        };
        let err = build_mold_request(&req).unwrap_err();
        assert!(matches!(err, VisionError::Io(_)));
    }

    #[tokio::test]
    async fn generate_image_releases_the_lease_even_when_the_caller_cancels_mid_generation() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();

        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"lease_id": "l-cancel"})),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(150))
                    .set_body_bytes(vec![9, 9, 9])
                    .insert_header("content-type", "image/png"),
            )
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;

        // Simulates a real caller losing interest mid-generation (a timeout
        // wrapper, tokio::select!, a shutdown signal, ...): spawn the call on
        // its own task, then abort *that outer task* -- not anything inside
        // generate_image itself -- while mold's response is still delayed.
        let outer =
            tokio::spawn(async move { provider.generate_image(sample_image_request()).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        outer.abort();

        // Give generate_image's own internally-spawned task (independent of
        // the outer task we just aborted) time to run past mold's artificial
        // delay and release the lock for real.
        tokio::time::sleep(Duration::from_millis(300)).await;

        // The release mock's `.expect(1)` above is checked here, explicitly --
        // this is the whole point of the test: release must still happen even
        // though nothing is left awaiting generate_image's own result.
        broker.verify().await;
    }

    /// Audit 2026-11-11 L1: an unreadable reference image fails before the
    /// GPU lock is requested.
    #[tokio::test]
    async fn a_bad_reference_image_fails_without_touching_the_broker() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&broker)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;
        let err = provider
            .generate_image(ImageRequest {
                reference_image: Some(PathBuf::from("/nonexistent/ref.png")),
                ..sample_image_request()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, VisionError::Io(_)), "got: {err:?}");
        broker.verify().await;
    }

    /// Audit 2026-11-11 L3/L6: a new output directory and file are
    /// owner-only, and mold's request warning reaches the asset.
    #[tokio::test]
    async fn generated_files_are_owner_only_and_carry_molds_warning() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let parent = tempfile::tempdir().unwrap();
        let output_dir = parent.path().join("vision");
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-3"})),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(tiny_png())
                    .insert_header("content-type", "image/png")
                    .insert_header("x-mold-request-warning", "dimensions adjusted"),
            )
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.clone()).await;
        let asset = provider
            .generate_image(sample_image_request())
            .await
            .unwrap();
        assert_eq!(asset.warnings, vec!["dimensions adjusted".to_string()]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&asset.path), 0o600);
            assert_eq!(mode(&output_dir), 0o700);
        }
    }

    fn model(name: &str, family: &str, downloaded: bool) -> MoldModel {
        MoldModel {
            name: name.to_string(),
            family: family.to_string(),
            downloaded,
            default_steps: Some(4),
            default_width: Some(1024),
            default_height: Some(768),
            dimension_alignment: Some(16),
            max_pixels: Some(1_800_000),
        }
    }

    /// Audit 2026-11-11 M1: with no model configured, the downloaded image
    /// model is used; video, 3D and undownloaded models are skipped.
    #[test]
    fn resolve_model_picks_the_downloaded_image_model() {
        let models = [
            model("ltx2:q8", "ltx2", true),
            model("hunyuan3d:q8", "hunyuan3d", true),
            model("flux-schnell:q8", "flux", false),
            model("flux2-klein:q8", "flux2", true),
        ];
        let (chosen, notes) = resolve_model(None, &models).unwrap();
        assert_eq!(chosen.name, "flux2-klein:q8");
        assert!(notes.is_empty());
    }

    #[test]
    fn resolve_model_says_which_it_used_when_several_are_downloaded() {
        let models = [
            model("sdxl:q8", "sdxl", true),
            model("flux2-klein:q8", "flux2", true),
        ];
        let (chosen, notes) = resolve_model(None, &models).unwrap();
        assert_eq!(chosen.name, "flux2-klein:q8", "first by name");
        assert!(notes[0].contains("several image models"), "{notes:?}");
    }

    #[test]
    fn resolve_model_explains_when_no_image_model_is_downloaded() {
        let models = [
            model("ltx2:q8", "ltx2", true),
            model("flux2-klein:q8", "flux2", false),
        ];
        let err = resolve_model(None, &models).unwrap_err().to_string();
        assert!(err.contains("mold pull"), "{err}");
    }

    #[test]
    fn resolve_model_honours_and_checks_a_configured_model() {
        let models = [
            model("flux2-klein:q8", "flux2", true),
            model("sdxl:q8", "sdxl", false),
        ];
        let (chosen, notes) = resolve_model(Some("sdxl:q8"), &models).unwrap();
        assert_eq!(chosen.name, "sdxl:q8");
        assert!(notes[0].contains("download"), "{notes:?}");
        let err = resolve_model(Some("nope:q8"), &models)
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope:q8"), "{err}");
    }

    #[test]
    fn fill_from_model_keeps_the_callers_values() {
        let mut req = GenerateImageRequest {
            prompt: "x".to_string(),
            width: Some(512),
            ..Default::default()
        };
        let notes = fill_from_model(&mut req, &model("flux2-klein:q8", "flux2", true));
        assert_eq!(req.model.as_deref(), Some("flux2-klein:q8"));
        assert_eq!(
            (req.width, req.height, req.steps),
            (Some(512), Some(768), Some(4))
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    /// mold refuses sizes off the model's grid (2026-11-11 live run:
    /// 500x500 on flux2 was a 422), so they are rounded, with a note.
    #[test]
    fn fill_from_model_rounds_an_off_grid_size_and_says_so() {
        let mut req = GenerateImageRequest {
            prompt: "x".to_string(),
            width: Some(500),
            height: Some(500),
            ..Default::default()
        };
        let notes = fill_from_model(&mut req, &model("flux2-klein:q8", "flux2", true));
        assert_eq!((req.width, req.height), (Some(496), Some(496)));
        assert!(
            notes[0].contains("generated 496x496 instead of 500x500"),
            "{notes:?}"
        );
    }

    /// mold refuses anything over the model's pixel limit (2026-11-11 live
    /// run: 2048x2048 on flux2 was a 422, "exceeds the 1.8MP limit"), and
    /// a phone photo as a reference would be far over it.
    #[test]
    fn fill_from_model_scales_a_size_over_the_pixel_limit_down_and_says_so() {
        let mut req = GenerateImageRequest {
            prompt: "x".to_string(),
            width: Some(4032),
            height: Some(3024),
            ..Default::default()
        };
        let notes = fill_from_model(&mut req, &model("flux2-klein:q8", "flux2", true));
        assert_eq!((req.width, req.height), (Some(1536), Some(1152)));
        assert!(
            notes[0].contains("at most 1800000 pixels: generated 1536x1152 instead of 4032x3024"),
            "{notes:?}"
        );
    }

    /// Rounding to the grid alone can tip a size over the limit.
    #[test]
    fn fill_from_model_never_rounds_past_the_pixel_limit() {
        let model = MoldModel {
            max_pixels: Some(1000 * 1000),
            ..model("m", "flux2", true)
        };
        let mut req = GenerateImageRequest {
            prompt: "x".to_string(),
            width: Some(1000),
            height: Some(1000),
            ..Default::default()
        };
        fill_from_model(&mut req, &model);
        let (w, h) = (req.width.unwrap(), req.height.unwrap());
        assert!(w * h <= 1000 * 1000, "{w}x{h}");
        assert_eq!((w % 16, h % 16), (0, 0));
    }

    /// The fields mold 0.37 requires (model, width, height, steps) reach it.
    #[tokio::test]
    async fn generate_image_sends_every_field_mold_requires() {
        let broker = MockServer::start().await;
        let mold = MockServer::start().await;
        let output_dir = tempfile::tempdir().unwrap();
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-4"})),
            )
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&broker)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "prompt": "anything", "model": "flux2-klein:q8",
                "width": 1024, "height": 1024, "steps": 4
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![1, 2, 3])
                    .insert_header("content-type", "image/png"),
            )
            .expect(1)
            .mount(&mold)
            .await;

        let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;
        provider
            .generate_image(sample_image_request())
            .await
            .unwrap();
        mold.verify().await;
    }

    #[test]
    fn extension_for_content_type_maps_known_types_and_falls_back_to_bin() {
        assert_eq!(extension_for_content_type("image/png"), "png");
        assert_eq!(extension_for_content_type("image/jpeg"), "jpg");
        assert_eq!(
            extension_for_content_type("application/octet-stream"),
            "bin"
        );
    }
}
