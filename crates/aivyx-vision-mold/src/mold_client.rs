//! HTTP client against an operator-run `mold serve` instance's
//! synchronous `POST /api/generate` endpoint. Grounded directly against
//! mold's own real API reference (`https://utensils.io/mold/api/`,
//! confirmed 2026-09-18): JSON request in, raw image bytes out on `200`
//! with a `Content-Type` header and an `x-mold-seed-used` header, JSON
//! `{"error", "code"}` on failure. Only wraps the one synchronous
//! endpoint this crate's Pass A needs -- mold exposes many more (queue
//! management, gallery, mesh workflows, streaming/SSE generation) that
//! are out of scope here.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::body_limits::{self, CappedBodyError};

/// `mold serve` returns real generated image bytes on success -- large,
/// but bounded. 64 MiB comfortably covers legitimate PNG/JPEG/WebP output
/// at any size `MoldConfig`'s width/height/steps could reasonably
/// produce, while still bounding a misbehaving or malicious server's
/// response to a fixed, finite amount of memory rather than whatever it
/// chooses to send.
pub(crate) const MAX_IMAGE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// `POST /api/upscale` replies with JSON holding the image as an array of
/// byte values, about three times the PNG's size (2026-11-11 live run: a
/// 7.8 MB PNG came back as 25 MB). The input is capped at 1 megapixel, so
/// a 4x result is at most 16 megapixels.
pub(crate) const MAX_UPSCALE_RESPONSE_BYTES: usize = 256 * 1024 * 1024;

/// `mold serve`'s own error responses are small, hand-written JSON
/// (`{"error", "code"}`) -- nowhere near large enough to need anything
/// close to `MAX_IMAGE_RESPONSE_BYTES`.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// `GET /api/models` lists every model mold knows (about 180 in 0.37,
/// each with its generation profile): a few MiB at most.
const MAX_MODELS_BODY_BYTES: usize = 16 * 1024 * 1024;

/// How long `generate()` is allowed to take *overall* -- connecting,
/// sending the request, and reading the full response body -- before it
/// gives up, as a mechanism distinct from the HTTP client's own
/// `read_timeout` (`MOLD_READ_TIMEOUT` in `provider.rs`). `read_timeout`
/// only bounds a single read operation and resets after each one
/// succeeds, so a server that trickles a response a few bytes at a time
/// -- each individual read completing comfortably inside that per-read
/// window -- could otherwise hold a request open indefinitely without
/// ever tripping it. This is a true ceiling on the whole call instead,
/// using the same "generous enough for realistic local FLUX/SDXL
/// generation time" reasoning `MOLD_READ_TIMEOUT` already documents.
const DEFAULT_MOLD_OVERALL_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateImageRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Base64-encoded source image, for image-to-image generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_image: Option<String>,
    /// How far image-to-image may move from `source_image`, 0.0 to 1.0
    /// (mold's default is 0.75).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// Base64-encoded inpainting mask (white = repaint, black = keep);
    /// mold requires `source_image` with it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_image: Option<String>,
    /// `false` stops mold writing the prompt and settings into the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_metadata: Option<bool>,
    /// Hunyuan3D options; only sent to a 3D model (mold refuses them for
    /// an image model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshOptions>,
    /// `false` sends mold's own copy of the result straight to its trash
    /// (purged after `gallery.trash_retention_days`) instead of its
    /// gallery.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_to_gallery: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct GenerateImageResponse {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub seed_used: Option<u64>,
    /// mold's `x-mold-request-warning` (all advisories, for example a size
    /// adjusted to the model's grid), or its older
    /// `x-mold-dimension-warning` when only that is sent. mold says not to
    /// split the value on `;`, so it is kept whole.
    pub warnings: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum MoldClientError {
    #[error("could not reach mold serve: {0}")]
    Transport(String),
    #[error("generation timed out")]
    Timeout,
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("mold serve's own generation queue is full")]
    QueueFull,
    #[error("mold serve rejected the request (status {status}, code {code:?}): {message}")]
    BackendError {
        status: u16,
        code: Option<String>,
        message: String,
    },
    #[error("mold serve's response exceeded the {0}-byte limit")]
    ResponseTooLarge(usize),
}

fn map_capped_body_error(e: CappedBodyError) -> MoldClientError {
    match e {
        CappedBodyError::TooLarge(cap) => MoldClientError::ResponseTooLarge(cap),
        CappedBodyError::Transport(e) => classify_transport_error(e),
    }
}

/// Classifies a `reqwest::Error` from a mold serve request. A *connect*
/// timeout means the backend genuinely isn't reachable, so it stays
/// `Transport`; a timeout during the request/response itself (most
/// realistically `.bytes().await` stalling on a slow generation, since
/// `MoldProvider::new` sets a long `read_timeout` for exactly that case)
/// means the backend is reachable but generation didn't finish in time,
/// which is a distinct, separately-actionable failure.
fn classify_transport_error(e: reqwest::Error) -> MoldClientError {
    if e.is_timeout() && !e.is_connect() {
        MoldClientError::Timeout
    } else {
        MoldClientError::Transport(e.to_string())
    }
}

#[derive(Clone)]
pub struct MoldClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    overall_timeout: Duration,
}

impl MoldClient {
    pub fn new(http: reqwest::Client, base_url: String, api_key: Option<String>) -> Self {
        Self {
            http,
            base_url,
            api_key,
            overall_timeout: DEFAULT_MOLD_OVERALL_TIMEOUT,
        }
    }

    /// Overrides the default overall per-request deadline (see
    /// `DEFAULT_MOLD_OVERALL_TIMEOUT`). A seam for tests that need a
    /// short deadline to exercise that path deterministically and
    /// quickly, without waiting on the real multi-minute default;
    /// production callers should rarely need this.
    pub fn with_overall_timeout(mut self, overall_timeout: Duration) -> Self {
        self.overall_timeout = overall_timeout;
        self
    }

    pub async fn generate(
        &self,
        req: GenerateImageRequest,
    ) -> Result<GenerateImageResponse, MoldClientError> {
        match tokio::time::timeout(self.overall_timeout, self.generate_inner(req)).await {
            Ok(result) => result,
            Err(_) => Err(MoldClientError::Timeout),
        }
    }

    async fn generate_inner(
        &self,
        req: GenerateImageRequest,
    ) -> Result<GenerateImageResponse, MoldClientError> {
        let mut builder = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&req);
        if let Some(key) = &self.api_key {
            builder = builder.header("X-Api-Key", key);
        }
        let response = builder.send().await.map_err(classify_transport_error)?;

        let status = response.status();
        if status.is_success() {
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/octet-stream")
                .to_string();
            let seed_used = response
                .headers()
                .get("x-mold-seed-used")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let header = |name: &str| {
                response
                    .headers()
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            };
            let mut warnings: Vec<String> = ["x-mold-request-warning", "x-mold-dimension-warning"]
                .iter()
                .find_map(|name| header(name))
                .into_iter()
                .collect();
            // A mesh reply describes itself in headers (2026-11-11 live run).
            if let Some(faces) = header("x-mold-mesh-faces") {
                let textured = header("x-mold-mesh-textured").as_deref() == Some("true");
                warnings.push(format!(
                    "mesh: {faces} faces, {}",
                    if textured {
                        "textured"
                    } else {
                        "untextured (geometry only)"
                    }
                ));
            }
            let bytes = body_limits::read_capped_body(response, MAX_IMAGE_RESPONSE_BYTES)
                .await
                .map_err(map_capped_body_error)?;
            return Ok(GenerateImageResponse {
                bytes,
                content_type,
                seed_used,
                warnings,
            });
        }

        Err(error_from_response(response).await)
    }

    /// Every model `mold serve` knows (`GET /api/models`), downloaded or
    /// not, with its default size and step count.
    pub async fn models(&self) -> Result<Vec<MoldModel>, MoldClientError> {
        match tokio::time::timeout(self.overall_timeout, self.models_inner()).await {
            Ok(result) => result,
            Err(_) => Err(MoldClientError::Timeout),
        }
    }

    async fn models_inner(&self) -> Result<Vec<MoldModel>, MoldClientError> {
        let mut builder = self.http.get(format!("{}/api/models", self.base_url));
        if let Some(key) = &self.api_key {
            builder = builder.header("X-Api-Key", key);
        }
        let response = builder.send().await.map_err(classify_transport_error)?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }
        let bytes = body_limits::read_capped_body(response, MAX_MODELS_BODY_BYTES)
            .await
            .map_err(map_capped_body_error)?;
        serde_json::from_slice(&bytes).map_err(|e| MoldClientError::BackendError {
            status: 200,
            code: None,
            message: format!("mold serve's model list was not understood: {e}"),
        })
    }
}

/// `POST /api/upscale` (not in mold 0.37's OpenAPI document; shape found
/// in the 2026-11-11 live run).
#[derive(Debug, Clone, Serialize)]
pub struct UpscaleImageRequest {
    pub model: String,
    /// Base64-encoded PNG or JPEG.
    pub image: String,
    /// Smaller tiles use less GPU memory: the default ran out of memory
    /// beside a loaded LLM in the live run, 256 did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile_size: Option<u32>,
}

/// `GenerateRequest.mesh` in mold 0.37.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MeshOptions {
    /// Decimate to about this many triangles. mold's raw output is dense
    /// (375,000 faces, 8.7 MB, for one object in the 2026-11-11 live run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_faces: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct UpscaleImageResponse {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Deserialize)]
struct UpscaleBody {
    image: UpscaledImage,
}

#[derive(Deserialize)]
struct UpscaledImage {
    data: Vec<u8>,
    #[serde(default)]
    format: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
}

impl MoldClient {
    pub async fn upscale(
        &self,
        req: UpscaleImageRequest,
    ) -> Result<UpscaleImageResponse, MoldClientError> {
        match tokio::time::timeout(self.overall_timeout, self.upscale_inner(req)).await {
            Ok(result) => result,
            Err(_) => Err(MoldClientError::Timeout),
        }
    }

    async fn upscale_inner(
        &self,
        req: UpscaleImageRequest,
    ) -> Result<UpscaleImageResponse, MoldClientError> {
        let mut builder = self
            .http
            .post(format!("{}/api/upscale", self.base_url))
            .json(&req);
        if let Some(key) = &self.api_key {
            builder = builder.header("X-Api-Key", key);
        }
        let response = builder.send().await.map_err(classify_transport_error)?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }
        let bytes = body_limits::read_capped_body(response, MAX_UPSCALE_RESPONSE_BYTES)
            .await
            .map_err(map_capped_body_error)?;
        let body: UpscaleBody =
            serde_json::from_slice(&bytes).map_err(|e| MoldClientError::BackendError {
                status: 200,
                code: None,
                message: format!("mold serve's upscale reply was not understood: {e}"),
            })?;
        let content_type = match body.image.format.as_str() {
            "jpeg" | "jpg" => "image/jpeg",
            "webp" => "image/webp",
            _ => "image/png",
        };
        Ok(UpscaleImageResponse {
            bytes: body.image.data,
            content_type: content_type.to_string(),
            width: body.image.width,
            height: body.image.height,
        })
    }
}

/// One entry of `GET /api/models` -- only the fields this crate uses.
#[derive(Debug, Clone, Deserialize)]
pub struct MoldModel {
    pub name: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub downloaded: bool,
    pub default_steps: Option<u32>,
    pub default_width: Option<u32>,
    pub default_height: Option<u32>,
    /// Width and height must be multiples of this; mold refuses others
    /// with a 422 (2026-11-11 live run: 500x500 on flux2).
    pub dimension_alignment: Option<u32>,
    /// The largest width x height mold accepts for this model; over it is
    /// a 422 (2026-11-11 live run: 1800000 on flux2-klein).
    pub max_pixels: Option<u64>,
}

/// Turns a non-2xx response into a typed error. mold answers most errors
/// with JSON `{"error", "code"}`, but request-validation failures (a
/// missing field, for example) as plain text; that text is kept rather
/// than replaced with "(no error body)" (found in the 2026-11-11 live run).
async fn error_from_response(response: reqwest::Response) -> MoldClientError {
    #[derive(Deserialize)]
    struct ErrorBody {
        error: String,
        code: Option<String>,
    }
    let status = response.status().as_u16();
    let error_bytes = match body_limits::read_capped_body(response, MAX_ERROR_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(e) => return map_capped_body_error(e),
    };
    let body = serde_json::from_slice::<ErrorBody>(&error_bytes).unwrap_or_else(|_| {
        let text = String::from_utf8_lossy(&error_bytes);
        let text = text.trim();
        ErrorBody {
            error: if text.is_empty() {
                "(no error body)".to_string()
            } else {
                text.chars().take(300).collect()
            },
            code: None,
        }
    });

    match (status, body.code.as_deref()) {
        (404, Some("MODEL_NOT_FOUND" | "UNKNOWN_MODEL")) => {
            MoldClientError::ModelNotFound(body.error)
        }
        // mold 0.3x's Secure mode: plain HTTP is accepted only on its
        // private loopback engine, or when the server opts in. Found in
        // the 2026-11-11 live run against mold 0.37 in Docker.
        (426, _) => MoldClientError::BackendError {
            status: 426,
            code: body.code,
            message: format!(
                "mold serve only accepts plain-HTTP API calls on its own machine's loopback, \
                 or when started with --allow-direct-api (MOLD_ALLOW_DIRECT_API=1); a mold \
                 in Docker or on another machine needs that flag: {}",
                body.error
            ),
        },
        (401, _) => MoldClientError::BackendError {
            status: 401,
            code: body.code,
            message: format!(
                "mold serve rejected the API key (set the mold api_key to one of the \
                 server's MOLD_API_KEY values): {}",
                body.error
            ),
        },
        (503, Some("QUEUE_FULL")) => MoldClientError::QueueFull,
        _ => MoldClientError::BackendError {
            status,
            code: body.code,
            message: body.error,
        },
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn generate_returns_image_bytes_content_type_and_seed_used_on_success() {
        let mold = MockServer::start().await;
        let png_bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(png_bytes.clone())
                    .insert_header("content-type", "image/png")
                    .insert_header("x-mold-seed-used", "12345"),
            )
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let response = client
            .generate(GenerateImageRequest {
                prompt: "a red circle".to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(response.bytes, png_bytes);
        assert_eq!(response.content_type, "image/png");
        assert_eq!(response.seed_used, Some(12345));
    }

    #[tokio::test]
    async fn generate_sends_the_api_key_header_when_configured() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(header("x-api-key", "secret-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![1, 2, 3])
                    .insert_header("content-type", "image/png"),
            )
            .mount(&mold)
            .await;

        let client = MoldClient::new(
            reqwest::Client::new(),
            mold.uri(),
            Some("secret-key".to_string()),
        );
        client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn generate_maps_a_404_model_not_found_response_to_a_named_error() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "model flux-schnell:q8 is not loaded",
                "code": "MODEL_NOT_FOUND"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::ModelNotFound(msg) if msg.contains("flux-schnell")));
    }

    /// mold's API reference lists `UNKNOWN_MODEL` beside `MODEL_NOT_FOUND`
    /// for a model the host can't resolve (audit 2026-11-11 M1).
    #[tokio::test]
    async fn generate_maps_a_404_unknown_model_response_to_model_not_found() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "unknown model nope:q8",
                "code": "UNKNOWN_MODEL"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::ModelNotFound(msg) if msg.contains("nope")));
    }

    /// The exact body mold 0.37 sent in the 2026-11-11 live run.
    #[tokio::test]
    async fn generate_explains_molds_secure_connection_requirement_on_426() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(426).set_body_json(serde_json::json!({
                "error": "A secure Mold connection is required",
                "code": "SECURE_CONNECTION_REQUIRED"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(
            matches!(&err, MoldClientError::BackendError { status: 426, message, .. }
                if message.contains("--allow-direct-api")),
            "got: {err:?}"
        );
    }

    /// mold sends request-validation failures as plain text (2026-11-11
    /// live run); the text must reach the caller.
    #[tokio::test]
    async fn generate_keeps_a_plain_text_error_body() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(422).set_body_string(
                "Failed to deserialize the JSON body into the target type: missing field `steps`",
            ))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(
            matches!(&err, MoldClientError::BackendError { status: 422, message, .. }
                if message.contains("missing field `steps`")),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn generate_says_the_api_key_was_rejected_on_401() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": "missing X-Api-Key header",
                "code": "UNAUTHORIZED"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(
            matches!(&err, MoldClientError::BackendError { status: 401, message, .. }
                if message.contains("API key")),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn generate_keeps_molds_request_warning_whole() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![1, 2, 3])
                    .insert_header("content-type", "image/png")
                    .insert_header(
                        "x-mold-request-warning",
                        "dimensions adjusted from 1000x1000 to 1024x1024; steps clamped",
                    ),
            )
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let response = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            response.warnings,
            vec!["dimensions adjusted from 1000x1000 to 1024x1024; steps clamped".to_string()]
        );
    }

    #[tokio::test]
    async fn generate_maps_a_503_queue_full_response_to_a_named_error() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": "generation queue is full",
                "code": "QUEUE_FULL"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::QueueFull));
    }

    #[tokio::test]
    async fn generate_maps_an_unrecognized_error_response_to_the_generic_backend_error() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": "inference crashed",
                "code": "INFERENCE_ERROR"
            })))
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            MoldClientError::BackendError { status: 500, .. }
        ));
    }

    #[tokio::test]
    async fn generate_returns_transport_error_when_mold_is_unreachable() {
        let client = MoldClient::new(
            reqwest::Client::new(),
            "http://127.0.0.1:1".to_string(),
            None,
        );
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::Transport(_)));
    }

    #[tokio::test]
    async fn generate_returns_timeout_when_the_response_body_stalls_past_the_client_timeout() {
        let mold = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![1, 2, 3])
                    .insert_header("content-type", "image/png")
                    .set_delay(std::time::Duration::from_millis(200)),
            )
            .mount(&mold)
            .await;

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();
        let client = MoldClient::new(http, mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::Timeout));
    }

    #[tokio::test]
    async fn generate_fails_with_a_clear_error_when_the_response_body_exceeds_the_cap() {
        let mold = MockServer::start().await;
        // One byte over the cap, declared via Content-Length -- the fix
        // must reject this before ever buffering the whole thing.
        let oversized_body = vec![0u8; MAX_IMAGE_RESPONSE_BYTES + 1];
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(oversized_body)
                    .insert_header("content-type", "image/png"),
            )
            .mount(&mold)
            .await;

        let client = MoldClient::new(reqwest::Client::new(), mold.uri(), None);
        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            MoldClientError::ResponseTooLarge(cap) if cap == MAX_IMAGE_RESPONSE_BYTES
        ));
    }

    #[tokio::test]
    async fn generate_returns_timeout_when_the_overall_deadline_elapses_despite_no_single_stall() {
        // A hand-rolled TCP server rather than wiremock: the bug this is
        // regression-testing is specifically that a server trickling a
        // response a few bytes at a time -- each individual gap well
        // inside any per-read timeout -- could hold the request open
        // indefinitely, because only a per-read timeout existed. wiremock
        // has no primitive for streaming a body slowly one write at a
        // time, so this drives a raw socket directly.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut discard = [0u8; 1024];
            let _ = socket.read(&mut discard).await;

            let body = b"not-a-real-png-but-fine";
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: image/png\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            for chunk in body.chunks(4) {
                let _ = socket.write_all(chunk).await;
                let _ = socket.flush().await;
                // Each individual gap (30ms) is far below any reasonable
                // per-read timeout; only their sum (6 * 30ms = 180ms),
                // bounded by the overall deadline below, catches this.
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
        });

        let client = MoldClient::new(reqwest::Client::new(), format!("http://{addr}"), None)
            .with_overall_timeout(Duration::from_millis(80));

        let err = client
            .generate(GenerateImageRequest {
                prompt: "anything".to_string(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, MoldClientError::Timeout));
    }
}
