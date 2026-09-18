//! HTTP client against `aivyx-broker`'s GPU lock (`POST /gpu-lock/acquire`
//! / `POST /gpu-lock/release`) -- the primitive `aivyx-vision-mold` uses
//! to keep its GPU-heavy generation calls from contending with local LLM
//! inference sharing the same machine. See `aivyx-broker/src/gpu_lock.rs`
//! for the lock's own implementation and
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5 for why this exists here
//! rather than reusing `aivyx-broker`'s llama-server-specific scheduling.

use serde::{Deserialize, Serialize};

/// An opaque handle to a held GPU lock, as issued by `aivyx-broker`. This
/// crate never parses or constructs the underlying UUID itself -- it only
/// ever round-trips the exact string the broker returned from `acquire`
/// back into `release`, so there is no reason to add a UUID dependency
/// here at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseId(pub(crate) String);

impl std::fmt::Display for LeaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuLockClientError {
    #[error("gpu-lock request to the broker failed: {0}")]
    Transport(String),
    #[error("timed out waiting for the GPU lock (broker returned 503)")]
    Timeout,
    #[error("broker rejected the release: {0}")]
    ReleaseRejected(String),
}

#[derive(Clone)]
pub struct GpuLockClient {
    http: reqwest::Client,
    broker_url: String,
}

impl GpuLockClient {
    pub fn new(http: reqwest::Client, broker_url: String) -> Self {
        Self { http, broker_url }
    }

    pub async fn acquire(&self) -> Result<LeaseId, GpuLockClientError> {
        let response = self
            .http
            .post(format!("{}/gpu-lock/acquire", self.broker_url))
            .send()
            .await
            .map_err(|e| GpuLockClientError::Transport(e.to_string()))?;

        if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            return Err(GpuLockClientError::Timeout);
        }
        if !response.status().is_success() {
            return Err(GpuLockClientError::Transport(format!(
                "unexpected status {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct AcquireResponse {
            lease_id: String,
        }
        let body: AcquireResponse = response
            .json()
            .await
            .map_err(|e| GpuLockClientError::Transport(e.to_string()))?;
        Ok(LeaseId(body.lease_id))
    }

    pub async fn release(&self, lease: &LeaseId) -> Result<(), GpuLockClientError> {
        #[derive(Serialize)]
        struct ReleaseRequest<'a> {
            lease_id: &'a str,
        }
        let response = self
            .http
            .post(format!("{}/gpu-lock/release", self.broker_url))
            .json(&ReleaseRequest { lease_id: &lease.0 })
            .send()
            .await
            .map_err(|e| GpuLockClientError::Transport(e.to_string()))?;

        if response.status().is_success() {
            return Ok(());
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(GpuLockClientError::ReleaseRejected(format!(
            "status {status}: {body}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn acquire_returns_the_lease_id_from_a_successful_broker_response() {
        let broker = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": "11111111-1111-1111-1111-111111111111"
            })))
            .mount(&broker)
            .await;

        let client = GpuLockClient::new(reqwest::Client::new(), broker.uri());
        let lease = client.acquire().await.unwrap();
        assert_eq!(lease.to_string(), "11111111-1111-1111-1111-111111111111");
    }

    #[tokio::test]
    async fn acquire_returns_timeout_when_the_broker_returns_503() {
        let broker = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": "timed out waiting for the GPU lock"
            })))
            .mount(&broker)
            .await;

        let client = GpuLockClient::new(reqwest::Client::new(), broker.uri());
        let err = client.acquire().await.unwrap_err();
        assert!(matches!(err, GpuLockClientError::Timeout));
    }

    #[tokio::test]
    async fn release_succeeds_and_sends_the_exact_lease_id_back() {
        let broker = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .and(body_json(
                serde_json::json!({"lease_id": "22222222-2222-2222-2222-222222222222"}),
            ))
            .respond_with(ResponseTemplate::new(200))
            .mount(&broker)
            .await;

        let client = GpuLockClient::new(reqwest::Client::new(), broker.uri());
        let lease = LeaseId("22222222-2222-2222-2222-222222222222".to_string());
        client.release(&lease).await.unwrap();
    }

    #[tokio::test]
    async fn release_of_an_unknown_lease_returns_a_clear_error_not_a_panic() {
        let broker = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/release"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "lease not found, already released, or expired"
            })))
            .mount(&broker)
            .await;

        let client = GpuLockClient::new(reqwest::Client::new(), broker.uri());
        let lease = LeaseId("33333333-3333-3333-3333-333333333333".to_string());
        let err = client.release(&lease).await.unwrap_err();
        assert!(matches!(err, GpuLockClientError::ReleaseRejected(_)));
    }

    #[tokio::test]
    async fn acquire_returns_transport_error_when_the_broker_is_unreachable() {
        // Port 1 on loopback is reserved and nothing listens there -- a
        // real, deterministic connection-refused, no live server needed.
        let client = GpuLockClient::new(reqwest::Client::new(), "http://127.0.0.1:1".to_string());
        let err = client.acquire().await.unwrap_err();
        assert!(matches!(err, GpuLockClientError::Transport(_)));
    }
}
