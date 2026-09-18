//! `GenerationProvider` -- the backend-agnostic contract Aivyx-Vision's
//! image/3D generation backends implement, plus the request/response
//! types and error enum shared across all of them. No I/O, no backend
//! dependencies -- this crate must be safe for any backend crate
//! (`aivyx-vision-mold`, and later `aivyx-vision-comfyui`) to depend on
//! without pulling in an HTTP client or anything else backend-specific.
//! See `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §4 for the full design.

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
}

/// Backend-agnostic image generation request. No mold-CLI-flag or
/// ComfyUI-workflow-JSON details belong here -- each backend crate
/// translates these fields into its own real request shape internally.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    /// A local file path to use for image-to-image generation, if the
    /// backend and model support it.
    pub reference_image: Option<PathBuf>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<u64>,
    /// A free-text style nudge (e.g. "watercolor", "pixel art"). Backends
    /// with no discrete style parameter fold this into the prompt they
    /// actually send.
    pub style_hint: Option<String>,
}

/// Backend-agnostic 3D generation request. Deliberately minimal for now
/// -- the real shape (image-to-3D reference handling, mesh format knobs,
/// etc.) is plan-time work for the Pass B build that actually implements
/// `generate_3d` against a real backend.
#[derive(Debug, Clone)]
pub struct ThreeDRequest {
    pub prompt: String,
    pub reference_image: Option<PathBuf>,
    pub style_hint: Option<String>,
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
}

#[cfg(any(test, feature = "testing"))]
impl FakeGenerationProvider {
    pub fn with_image_result(result: Result<GeneratedAsset, VisionError>) -> Self {
        Self {
            image_result: std::sync::Mutex::new(Some(result)),
            threed_result: std::sync::Mutex::new(None),
            captured_image_request: std::sync::Mutex::new(None),
            captured_3d_request: std::sync::Mutex::new(None),
        }
    }

    pub fn with_3d_result(result: Result<GeneratedAsset, VisionError>) -> Self {
        Self {
            image_result: std::sync::Mutex::new(None),
            threed_result: std::sync::Mutex::new(Some(result)),
            captured_image_request: std::sync::Mutex::new(None),
            captured_3d_request: std::sync::Mutex::new(None),
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn generation_provider_is_usable_as_a_trait_object() {
        let provider: Arc<dyn GenerationProvider> = Arc::new(
            FakeGenerationProvider::with_image_result(Ok(GeneratedAsset {
                path: "/tmp/fake.png".into(),
                backend: "fake".to_string(),
                seed_used: Some(42),
                generated_at: std::time::SystemTime::now(),
            })),
        );
        let asset = provider
            .generate_image(ImageRequest {
                prompt: "a red circle".to_string(),
                reference_image: None,
                width: None,
                height: None,
                seed: None,
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
        }));
        provider
            .generate_image(ImageRequest {
                prompt: "a blue square".to_string(),
                reference_image: None,
                width: Some(512),
                height: Some(512),
                seed: None,
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
        }));
        let asset = provider
            .generate_3d(ThreeDRequest {
                prompt: "a small statue".to_string(),
                reference_image: None,
                style_hint: None,
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
