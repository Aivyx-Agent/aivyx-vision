//! HTTP client against an operator-run `mold serve` instance's
//! synchronous `POST /api/generate` endpoint. Grounded directly against
//! mold's own real API reference (`https://utensils.io/mold/api/`,
//! confirmed 2026-09-18): JSON request in, raw image bytes out on `200`
//! with a `Content-Type` header and an `x-mold-seed-used` header, JSON
//! `{"error", "code"}` on failure. Only wraps the one synchronous
//! endpoint this crate's Pass A needs -- mold exposes many more (queue
//! management, gallery, mesh workflows, streaming/SSE generation) that
//! are out of scope here.

use serde::{Deserialize, Serialize};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GenerateImageResponse {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub seed_used: Option<u64>,
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
}

impl MoldClient {
    pub fn new(http: reqwest::Client, base_url: String, api_key: Option<String>) -> Self {
        Self {
            http,
            base_url,
            api_key,
        }
    }

    pub async fn generate(
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
            let bytes = response
                .bytes()
                .await
                .map_err(classify_transport_error)?
                .to_vec();
            return Ok(GenerateImageResponse {
                bytes,
                content_type,
                seed_used,
            });
        }

        #[derive(Deserialize)]
        struct ErrorBody {
            error: String,
            code: Option<String>,
        }
        let body: ErrorBody = response.json().await.unwrap_or(ErrorBody {
            error: "(no error body)".to_string(),
            code: None,
        });

        match (status.as_u16(), body.code.as_deref()) {
            (404, Some("MODEL_NOT_FOUND")) => Err(MoldClientError::ModelNotFound(body.error)),
            (503, Some("QUEUE_FULL")) => Err(MoldClientError::QueueFull),
            _ => Err(MoldClientError::BackendError {
                status: status.as_u16(),
                code: body.code,
                message: body.error,
            }),
        }
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
                "code": "INFERENCE_FAILURE"
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
}
