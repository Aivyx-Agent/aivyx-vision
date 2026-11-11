//! `GenerationProvider` -- the backend-agnostic contract Aivyx-Vision's
//! image/3D generation backends implement, plus the request/response
//! types and error enum shared across all of them. No I/O, no backend
//! dependencies -- this crate must be safe for any backend crate
//! (`aivyx-vision-mold`, and later `aivyx-vision-comfyui`) to depend on
//! without pulling in an HTTP client or anything else backend-specific.
//! (The Aivyx-Vision v1 design, §4, has the full reasoning.)

use std::path::PathBuf;
use std::time::SystemTime;

use async_trait::async_trait;

/// The contract every Aivyx-Vision generation backend implements. A
/// backend that can't do one of the two domains (e.g. an image-only
/// backend) returns `VisionError::Unsupported` from that method rather
/// than the trait requiring universal capability -- matching
/// `aivyx-llm`'s `LlmProvider`'s own tolerance of partial implementation.
#[async_trait]
pub trait GenerationProvider: Send + Sync {
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError>;
    async fn generate_3d(&self, req: ThreeDRequest) -> Result<GeneratedAsset, VisionError>;
    /// Enlarges an existing image with the backend's upscaler, writing the
    /// result as a new file. Backends without one keep this default.
    async fn upscale_image(&self, _req: UpscaleRequest) -> Result<GeneratedAsset, VisionError> {
        Err(VisionError::Unsupported("upscaling"))
    }
}

/// An image to enlarge with [`GenerationProvider::upscale_image`].
#[derive(Debug, Clone, Default)]
pub struct UpscaleRequest {
    /// A local PNG or JPEG file.
    pub image: PathBuf,
}

/// Backend-agnostic image generation request. No mold-CLI-flag or
/// ComfyUI-workflow-JSON details belong here -- each backend crate
/// translates these fields into its own real request shape internally.
#[derive(Debug, Clone, Default)]
pub struct ImageRequest {
    pub prompt: String,
    /// A local file path to use for image-to-image generation, if the
    /// backend and model support it.
    pub reference_image: Option<PathBuf>,
    /// Output size. With a reference image and no size, the reference's
    /// own size is used (and with one side given, the other keeps the
    /// reference's proportions), so image-to-image keeps the picture's
    /// shape.
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<u64>,
    /// How far image-to-image may move away from the reference, from 0.0
    /// (keep it) to 1.0 (ignore it). `None` uses the backend's default.
    /// Ignored, with a warning, when there is no reference image.
    pub strength: Option<f32>,
    /// Inpainting: a local mask image the same shape as `reference_image`
    /// (white = repaint, black = keep). Needs `reference_image`.
    pub mask_image: Option<PathBuf>,
    /// A free-text style nudge (e.g. "watercolor", "pixel art"). Backends
    /// with no discrete style parameter fold this into the prompt they
    /// actually send.
    pub style_hint: Option<String>,
}

/// Backend-agnostic 3D generation request: a mesh from a reference image
/// of one object, or from a prompt (the backend makes the image first).
#[derive(Debug, Clone, Default)]
pub struct ThreeDRequest {
    pub prompt: String,
    /// A local PNG or JPEG of the object to turn into a mesh. Without one,
    /// the backend generates an image from `prompt` first.
    pub reference_image: Option<PathBuf>,
    pub style_hint: Option<String>,
    pub seed: Option<u64>,
}

/// The most variations [`generate_variations`] makes in one call: each is a
/// full generation holding the GPU.
pub const MAX_VARIATIONS: usize = 4;

/// Makes `count` (1 to [`MAX_VARIATIONS`]) images from one request, with
/// consecutive seeds so each can be reproduced: `req.seed`, or the seed the
/// first image used, plus 0, 1, 2... Generations run one after another;
/// every result is returned, failures included, so one failure doesn't
/// lose the others.
pub async fn generate_variations(
    provider: &dyn GenerationProvider,
    req: ImageRequest,
    count: usize,
) -> Result<Vec<Result<GeneratedAsset, VisionError>>, VisionError> {
    if !(1..=MAX_VARIATIONS).contains(&count) {
        return Err(VisionError::InvalidRequest(format!(
            "count must be between 1 and {MAX_VARIATIONS}, got {count}"
        )));
    }
    let mut results = Vec::with_capacity(count);
    let mut base = req.seed;
    for i in 0..count as u64 {
        let seed = base.map(|b| b.wrapping_add(i));
        let result = provider
            .generate_image(ImageRequest {
                seed,
                ..req.clone()
            })
            .await;
        if base.is_none()
            && let Ok(asset) = &result
        {
            base = asset.seed_used.map(|s| s.wrapping_sub(i));
        }
        results.push(result);
    }
    Ok(results)
}

/// The result of a successful generation: a file path plus metadata, not
/// raw bytes in memory -- outputs (images, meshes) can be large, and
/// every consumer of this type wants a path to hand to its own storage
/// policy, not a byte buffer to manage itself.
#[derive(Debug, Clone)]
pub struct GeneratedAsset {
    pub path: PathBuf,
    /// Which backend produced this (e.g. `"mold"`) -- rides the existing
    /// audit chain in each consuming product alongside the other
    /// metadata fields.
    pub backend: String,
    pub seed_used: Option<u64>,
    pub generated_at: SystemTime,
    /// Advisories the backend attached to an otherwise successful result,
    /// such as mold adjusting the requested size. Empty when there were
    /// none; a consumer should pass them on to the agent, which would
    /// otherwise report what it asked for rather than what it got.
    pub warnings: Vec<String>,
}

/// Everything that can go wrong generating an asset. Distinguishes
/// backend-unreachable, unsupported-domain, generation-timeout, and
/// GPU-lock-acquisition-timeout as separate variants deliberately -- the
/// operator-facing fix differs per case (start the backend vs. wait for
/// the GPU vs. switch backend), so a single opaque "generation failed" is
/// not acceptable. `#[non_exhaustive]` because a real backend crate may
/// need to add a variant this crate didn't anticipate (see `Io` and
/// `BackendError`, both added here already for exactly that reason).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VisionError {
    #[error("backend unreachable: {0}")]
    BackendUnreachable(String),
    #[error("this backend does not support {0}")]
    Unsupported(&'static str),
    #[error("generation timed out")]
    GenerationTimeout,
    #[error("timed out waiting for the GPU lock")]
    GpuLockTimeout,
    /// A backend server reached and responded, but rejected the request
    /// for its own typed reason (model not found, its own queue full,
    /// internal inference failure, ...). `code` carries the backend's own
    /// machine-readable error code when it has one.
    #[error("backend rejected the request: {message} (status {status}, code {code:?})")]
    BackendError {
        status: u16,
        code: Option<String>,
        message: String,
    },
    /// A *local* I/O failure -- reading a reference image from disk,
    /// writing a generated asset to disk -- distinct from a backend
    /// problem: this machine's filesystem, not the generation backend, is
    /// what's wrong.
    #[error("local I/O failed: {0}")]
    Io(String),
    /// The request itself is invalid (for example, a strength outside
    /// 0.0..=1.0), whatever backend it goes to.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// A failure inside this crate or its backend client that is neither
    /// the backend's nor the local filesystem's fault (for example, the
    /// generation task panicked).
    #[error("internal error: {0}")]
    Internal(String),
}

/// A test double for `GenerationProvider`: returns a canned result
/// (success or error) exactly once per call, and records the request it
/// was called with so a caller's tests can assert on it. Lets both
/// consuming products' tool-level tests exercise their own adapter logic
/// without a real backend running. Available in this crate's own tests
/// unconditionally; a consuming crate opts in via the `testing` Cargo
/// feature.
#[cfg(any(test, feature = "testing"))]
pub struct FakeGenerationProvider {
    image_result: std::sync::Mutex<Option<Result<GeneratedAsset, VisionError>>>,
    threed_result: std::sync::Mutex<Option<Result<GeneratedAsset, VisionError>>>,
    captured_image_request: std::sync::Mutex<Option<ImageRequest>>,
    captured_3d_request: std::sync::Mutex<Option<ThreeDRequest>>,
    upscale_result: std::sync::Mutex<Option<Result<GeneratedAsset, VisionError>>>,
    captured_upscale_request: std::sync::Mutex<Option<UpscaleRequest>>,
}

#[cfg(any(test, feature = "testing"))]
impl FakeGenerationProvider {
    pub fn with_upscale_result(result: Result<GeneratedAsset, VisionError>) -> Self {
        let fake = Self::with_image_result(Err(VisionError::Internal(
            "no image result configured".to_string(),
        )));
        *fake.upscale_result.lock().unwrap() = Some(result);
        fake
    }

    pub fn captured_upscale_request(&self) -> Option<UpscaleRequest> {
        self.captured_upscale_request.lock().unwrap().clone()
    }

    pub fn with_image_result(result: Result<GeneratedAsset, VisionError>) -> Self {
        Self {
            image_result: std::sync::Mutex::new(Some(result)),
            threed_result: std::sync::Mutex::new(None),
            captured_image_request: std::sync::Mutex::new(None),
            captured_3d_request: std::sync::Mutex::new(None),
            upscale_result: std::sync::Mutex::new(None),
            captured_upscale_request: std::sync::Mutex::new(None),
        }
    }

    pub fn with_3d_result(result: Result<GeneratedAsset, VisionError>) -> Self {
        Self {
            image_result: std::sync::Mutex::new(None),
            threed_result: std::sync::Mutex::new(Some(result)),
            captured_image_request: std::sync::Mutex::new(None),
            captured_3d_request: std::sync::Mutex::new(None),
            upscale_result: std::sync::Mutex::new(None),
            captured_upscale_request: std::sync::Mutex::new(None),
        }
    }

    pub fn captured_image_request(&self) -> Option<ImageRequest> {
        self.captured_image_request.lock().unwrap().clone()
    }

    pub fn captured_3d_request(&self) -> Option<ThreeDRequest> {
        self.captured_3d_request.lock().unwrap().clone()
    }
}

#[cfg(any(test, feature = "testing"))]
#[async_trait]
impl GenerationProvider for FakeGenerationProvider {
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
        *self.captured_image_request.lock().unwrap() = Some(req);
        self.image_result
            .lock()
            .unwrap()
            .take()
            .expect("FakeGenerationProvider.generate_image called with no image_result configured, or called more than once")
    }

    async fn generate_3d(&self, req: ThreeDRequest) -> Result<GeneratedAsset, VisionError> {
        *self.captured_3d_request.lock().unwrap() = Some(req);
        self.threed_result
            .lock()
            .unwrap()
            .take()
            .expect("FakeGenerationProvider.generate_3d called with no 3d_result configured, or called more than once")
    }

    async fn upscale_image(&self, req: UpscaleRequest) -> Result<GeneratedAsset, VisionError> {
        *self.captured_upscale_request.lock().unwrap() = Some(req);
        self.upscale_result
            .lock()
            .unwrap()
            .take()
            .expect("FakeGenerationProvider.upscale_image called with no upscale_result configured, or called more than once")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// Answers every call, using the requested seed or else 1000 + call
    /// number, and records the seeds it was asked for.
    struct SeedEcho {
        asked: std::sync::Mutex<Vec<Option<u64>>>,
        fail_call: Option<usize>,
    }

    #[async_trait]
    impl GenerationProvider for SeedEcho {
        async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
            let mut asked = self.asked.lock().unwrap();
            asked.push(req.seed);
            let call = asked.len() - 1;
            if self.fail_call == Some(call) {
                return Err(VisionError::GenerationTimeout);
            }
            Ok(GeneratedAsset {
                path: PathBuf::from(format!("/out/{call}.png")),
                backend: "echo".to_string(),
                seed_used: Some(req.seed.unwrap_or(1000 + call as u64)),
                generated_at: SystemTime::UNIX_EPOCH,
                warnings: Vec::new(),
            })
        }
        async fn generate_3d(&self, _: ThreeDRequest) -> Result<GeneratedAsset, VisionError> {
            Err(VisionError::Unsupported("3D"))
        }
    }

    fn echo(fail_call: Option<usize>) -> SeedEcho {
        SeedEcho {
            asked: std::sync::Mutex::new(Vec::new()),
            fail_call,
        }
    }

    #[tokio::test]
    async fn variations_use_consecutive_seeds_from_the_requested_one() {
        let provider = echo(None);
        let req = ImageRequest {
            prompt: "a lantern".to_string(),
            seed: Some(7),
            ..Default::default()
        };
        let results = generate_variations(&provider, req, 3).await.unwrap();
        assert_eq!(*provider.asked.lock().unwrap(), [Some(7), Some(8), Some(9)]);
        let seeds: Vec<_> = results
            .iter()
            .map(|r| r.as_ref().unwrap().seed_used)
            .collect();
        assert_eq!(seeds, [Some(7), Some(8), Some(9)]);
    }

    #[tokio::test]
    async fn without_a_seed_variations_continue_from_the_first_images_seed() {
        let provider = echo(None);
        let req = ImageRequest {
            prompt: "a lantern".to_string(),
            ..Default::default()
        };
        generate_variations(&provider, req, 3).await.unwrap();
        assert_eq!(
            *provider.asked.lock().unwrap(),
            [None, Some(1001), Some(1002)]
        );
    }

    #[tokio::test]
    async fn one_failed_variation_keeps_the_others() {
        let provider = echo(Some(1));
        let req = ImageRequest {
            prompt: "a lantern".to_string(),
            seed: Some(7),
            ..Default::default()
        };
        let results = generate_variations(&provider, req, 3).await.unwrap();
        assert!(results[0].is_ok() && results[1].is_err() && results[2].is_ok());
    }

    #[tokio::test]
    async fn upscaling_is_unsupported_unless_a_backend_provides_it() {
        let err = echo(None)
            .upscale_image(UpscaleRequest::default())
            .await
            .unwrap_err();
        assert!(
            matches!(err, VisionError::Unsupported("upscaling")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn variations_refuse_a_count_outside_one_to_four() {
        for count in [0, MAX_VARIATIONS + 1] {
            let err = generate_variations(&echo(None), ImageRequest::default(), count)
                .await
                .unwrap_err();
            assert!(matches!(err, VisionError::InvalidRequest(_)), "{count}");
        }
    }

    #[tokio::test]
    async fn generation_provider_is_usable_as_a_trait_object() {
        let provider: Arc<dyn GenerationProvider> = Arc::new(
            FakeGenerationProvider::with_image_result(Ok(GeneratedAsset {
                path: "/tmp/fake.png".into(),
                backend: "fake".to_string(),
                seed_used: Some(42),
                generated_at: std::time::SystemTime::now(),
                warnings: Vec::new(),
            })),
        );
        let asset = provider
            .generate_image(ImageRequest {
                prompt: "a red circle".to_string(),
                reference_image: None,
                width: None,
                height: None,
                seed: None,
                strength: None,
                mask_image: None,
                style_hint: None,
            })
            .await
            .unwrap();
        assert_eq!(asset.backend, "fake");
        assert_eq!(asset.seed_used, Some(42));
    }

    #[tokio::test]
    async fn fake_generation_provider_captures_the_image_request_it_was_called_with() {
        let provider = FakeGenerationProvider::with_image_result(Ok(GeneratedAsset {
            path: "/tmp/fake.png".into(),
            backend: "fake".to_string(),
            seed_used: None,
            generated_at: std::time::SystemTime::now(),
            warnings: Vec::new(),
        }));
        provider
            .generate_image(ImageRequest {
                prompt: "a blue square".to_string(),
                reference_image: None,
                width: Some(512),
                height: Some(512),
                seed: None,
                strength: None,
                mask_image: None,
                style_hint: Some("watercolor".to_string()),
            })
            .await
            .unwrap();
        let captured = provider
            .captured_image_request()
            .expect("must capture the request");
        assert_eq!(captured.prompt, "a blue square");
        assert_eq!(captured.style_hint.as_deref(), Some("watercolor"));
    }

    #[tokio::test]
    async fn fake_generation_provider_returns_the_configured_error() {
        let provider = FakeGenerationProvider::with_image_result(Err(VisionError::GpuLockTimeout));
        let err = provider
            .generate_image(ImageRequest {
                prompt: "anything".to_string(),
                reference_image: None,
                width: None,
                height: None,
                seed: None,
                strength: None,
                mask_image: None,
                style_hint: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, VisionError::GpuLockTimeout));
    }

    #[tokio::test]
    async fn fake_generation_provider_serves_3d_results_independently_of_image_results() {
        let provider = FakeGenerationProvider::with_3d_result(Ok(GeneratedAsset {
            path: "/tmp/fake.glb".into(),
            backend: "fake".to_string(),
            seed_used: None,
            generated_at: std::time::SystemTime::now(),
            warnings: Vec::new(),
        }));
        let asset = provider
            .generate_3d(ThreeDRequest {
                prompt: "a small statue".to_string(),
                reference_image: None,
                style_hint: None,
                seed: None,
            })
            .await
            .unwrap();
        assert_eq!(asset.path, std::path::PathBuf::from("/tmp/fake.glb"));
    }

    #[test]
    fn vision_error_messages_are_specific_enough_to_act_on() {
        assert_eq!(
            VisionError::GpuLockTimeout.to_string(),
            "timed out waiting for the GPU lock"
        );
        assert_eq!(
            VisionError::Unsupported("3D generation").to_string(),
            "this backend does not support 3D generation"
        );
        assert_eq!(
            VisionError::BackendError {
                status: 404,
                code: Some("MODEL_NOT_FOUND".to_string()),
                message: "no such model".to_string(),
            }
            .to_string(),
            "backend rejected the request: no such model (status 404, code Some(\"MODEL_NOT_FOUND\"))"
        );
    }
}
