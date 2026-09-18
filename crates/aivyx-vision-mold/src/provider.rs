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
    GenerateImageRequest, GenerateImageResponse, MoldClient, MoldClientError,
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
}

pub struct MoldProvider {
    gpu_lock: GpuLockClient,
    mold: MoldClient,
    output_dir: PathBuf,
}

impl MoldProvider {
    pub fn new(config: MoldConfig) -> Result<Self, VisionError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .read_timeout(MOLD_READ_TIMEOUT)
            .build()
            .map_err(|e| {
                VisionError::BackendUnreachable(format!("failed to build HTTP client: {e}"))
            })?;

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

        let handle = tokio::spawn(async move {
            let lease = gpu_lock.acquire().await.map_err(map_gpu_lock_error)?;

            let mold_req = build_mold_request(&req);
            let result = match mold_req {
                Ok(mold_req) => mold.generate(mold_req).await.map_err(map_mold_error),
                Err(e) => Err(e),
            };

            if let Err(e) = gpu_lock.release(&lease).await {
                tracing::warn!(
                    error = %e,
                    "gpu-lock release failed after generation; aivyx-broker's reap_expired will eventually reclaim it"
                );
            }

            let response = result?;
            write_generated_asset(&output_dir, response)
        });

        handle.await.map_err(|e| {
            VisionError::BackendUnreachable(format!("generation task panicked: {e}"))
        })?
    }

    async fn generate_3d(&self, _req: ThreeDRequest) -> Result<GeneratedAsset, VisionError> {
        Err(VisionError::Unsupported(
            "3D generation via mold (Pass B is not yet implemented)",
        ))
    }
}

fn build_mold_request(req: &ImageRequest) -> Result<GenerateImageRequest, VisionError> {
    let prompt = match &req.style_hint {
        Some(hint) => format!("{}, {hint}", req.prompt),
        None => req.prompt.clone(),
    };
    let source_image = match &req.reference_image {
        Some(path) => {
            let bytes = std::fs::read(path)
                .map_err(|e| VisionError::Io(format!("reading reference image {path:?}: {e}")))?;
            Some(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        None => None,
    };
    Ok(GenerateImageRequest {
        prompt,
        width: req.width,
        height: req.height,
        seed: req.seed,
        source_image,
        ..Default::default()
    })
}

fn write_generated_asset(
    output_dir: &Path,
    response: GenerateImageResponse,
) -> Result<GeneratedAsset, VisionError> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| VisionError::Io(format!("creating output dir {output_dir:?}: {e}")))?;
    let ext = extension_for_content_type(&response.content_type);
    let path = output_dir.join(format!("{}.{ext}", uuid::Uuid::new_v4()));
    std::fs::write(&path, &response.bytes)
        .map_err(|e| VisionError::Io(format!("writing generated asset {path:?}: {e}")))?;
    Ok(GeneratedAsset {
        path,
        backend: "mold".to_string(),
        seed_used: response.seed_used,
        generated_at: SystemTime::now(),
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
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn test_provider(
        broker: &MockServer,
        mold: &MockServer,
        output_dir: PathBuf,
    ) -> MoldProvider {
        MoldProvider::new(MoldConfig {
            broker_url: broker.uri(),
            mold_url: mold.uri(),
            api_key: None,
            output_dir,
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
                "code": "INFERENCE_FAILURE"
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
            .respond_with(ResponseTemplate::new(503))
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
        let mold_req = build_mold_request(&req).unwrap();
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
        let mold_req = build_mold_request(&req).unwrap();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(mold_req.source_image.unwrap())
            .unwrap();
        assert_eq!(decoded, vec![1, 2, 3, 4]);
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
