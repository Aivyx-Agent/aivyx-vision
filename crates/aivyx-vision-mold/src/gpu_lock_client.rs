//! HTTP client against `aivyx-broker`'s GPU lock (`POST /gpu-lock/acquire`
//! / `POST /gpu-lock/release`) -- the primitive `aivyx-vision-mold` uses
//! to keep its GPU-heavy generation calls from contending with local LLM
//! inference sharing the same machine. See `aivyx-broker/src/gpu_lock.rs`
//! for the lock's own implementation and
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5 for why this exists here
//! rather than reusing `aivyx-broker`'s llama-server-specific scheduling.
//!
//! Two defensive limits on top of the happy path above (Audit note:
//! `~/aivyx-audit-2026-10-04/C-report.md`, item V7): a cap on how much
//! response-body memory a single call will ever buffer
//! (`MAX_JSON_RESPONSE_BYTES`, enforced via `crate::body_limits`), and an
//! overall per-request deadline (`overall_timeout`, defaulting to
//! `DEFAULT_GPU_LOCK_OVERALL_TIMEOUT`) -- see that constant's doc comment
//! for why.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::body_limits::{self, CappedBodyError};

/// `aivyx-broker`'s GPU-lock responses are tiny, fixed-shape JSON
/// (`{"lease_id"}` on acquire, a short `{"error"}` on failure) -- nowhere
/// near large enough to need anything like `aivyx-vision-mold`'s own
/// `MAX_IMAGE_RESPONSE_BYTES`.
const MAX_JSON_RESPONSE_BYTES: usize = 64 * 1024;

/// How long `acquire()`/`release()` are allowed to take *overall* before
/// giving up, as a mechanism distinct from a per-read timeout -- see
/// `mold_client.rs`'s `DEFAULT_MOLD_OVERALL_TIMEOUT` for the general
/// problem this closes (a server trickling bytes so slowly that no
/// single read ever times out, while the request never completes
/// either).
///
/// Deliberately kept at least as long as `aivyx-broker`'s own
/// `gpu_lock_queue_timeout_secs` default (280s, see
/// `aivyx-broker/src/config.rs`) *plus* that config's own documented 20s
/// margin -- i.e. 300s, matching the read timeout this crate's HTTP
/// client is already configured with in `provider.rs`. Going any lower
/// would let this client give up on `acquire()` before the broker's own
/// queue-timeout response could arrive, defeating the margin
/// `aivyx-broker` already built in specifically so that doesn't happen.
const DEFAULT_GPU_LOCK_OVERALL_TIMEOUT: Duration = Duration::from_secs(300);

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
    #[error("broker response exceeded the {0}-byte limit")]
    ResponseTooLarge(usize),
}

fn map_capped_body_error(e: CappedBodyError) -> GpuLockClientError {
    match e {
        CappedBodyError::TooLarge(cap) => GpuLockClientError::ResponseTooLarge(cap),
        CappedBodyError::Transport(e) => classify_transport_error(e),
    }
}

#[derive(Clone)]
pub struct GpuLockClient {
    http: reqwest::Client,
    broker_url: String,
    overall_timeout: Duration,
}

/// Classifies a `reqwest::Error` from a broker request. A *connect*
/// timeout means the broker genuinely isn't reachable, so it stays
/// `Transport`; a timeout during the request/response itself maps to the
/// existing `Timeout` variant -- the same distinction `mold_client.rs`'s
/// `classify_transport_error` makes for `mold serve` requests.
fn classify_transport_error(e: reqwest::Error) -> GpuLockClientError {
    if e.is_timeout() && !e.is_connect() {
        GpuLockClientError::Timeout
    } else {
        GpuLockClientError::Transport(e.to_string())
    }
}

impl GpuLockClient {
    pub fn new(http: reqwest::Client, broker_url: String) -> Self {
        Self {
            http,
            broker_url,
            overall_timeout: DEFAULT_GPU_LOCK_OVERALL_TIMEOUT,
        }
    }

    /// Overrides the default overall per-request deadline (see
    /// `DEFAULT_GPU_LOCK_OVERALL_TIMEOUT`). A seam for tests that need a
    /// short deadline to exercise that path deterministically and
    /// quickly, without waiting on the real multi-minute default;
    /// production callers should rarely need this.
    pub fn with_overall_timeout(mut self, overall_timeout: Duration) -> Self {
        self.overall_timeout = overall_timeout;
        self
    }

    pub async fn acquire(&self) -> Result<LeaseId, GpuLockClientError> {
        match tokio::time::timeout(self.overall_timeout, self.acquire_inner()).await {
            Ok(result) => result,
            Err(_) => Err(GpuLockClientError::Timeout),
        }
    }

    async fn acquire_inner(&self) -> Result<LeaseId, GpuLockClientError> {
        let response = self
            .http
            .post(format!("{}/gpu-lock/acquire", self.broker_url))
            .send()
            .await
            .map_err(classify_transport_error)?;

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
        let bytes = body_limits::read_capped_body(response, MAX_JSON_RESPONSE_BYTES)
            .await
            .map_err(map_capped_body_error)?;
        let body: AcquireResponse = serde_json::from_slice(&bytes)
            .map_err(|e| GpuLockClientError::Transport(e.to_string()))?;
        Ok(LeaseId(body.lease_id))
    }

    pub async fn release(&self, lease: &LeaseId) -> Result<(), GpuLockClientError> {
        match tokio::time::timeout(self.overall_timeout, self.release_inner(lease)).await {
            Ok(result) => result,
            Err(_) => Err(GpuLockClientError::Timeout),
        }
    }

    async fn release_inner(&self, lease: &LeaseId) -> Result<(), GpuLockClientError> {
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
        let bytes = body_limits::read_capped_body(response, MAX_JSON_RESPONSE_BYTES)
            .await
            .map_err(map_capped_body_error)?;
        let body = String::from_utf8_lossy(&bytes).into_owned();
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

    #[tokio::test]
    async fn acquire_returns_timeout_when_the_broker_response_stalls_past_the_client_timeout() {
        let broker = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"lease_id": "l-slow"}))
                    .set_delay(std::time::Duration::from_millis(200)),
            )
            .mount(&broker)
            .await;

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();
        let client = GpuLockClient::new(http, broker.uri());
        let err = client.acquire().await.unwrap_err();
        assert!(matches!(err, GpuLockClientError::Timeout));
    }

    #[tokio::test]
    async fn acquire_fails_with_a_clear_error_when_the_broker_response_exceeds_the_cap() {
        let broker = MockServer::start().await;
        // A lease_id padded one byte over the cap, declared via
        // Content-Length -- the fix must reject this before ever
        // buffering the whole thing.
        let oversized_lease_id = "x".repeat(MAX_JSON_RESPONSE_BYTES + 1);
        Mock::given(method("POST"))
            .and(path("/gpu-lock/acquire"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": oversized_lease_id
            })))
            .mount(&broker)
            .await;

        let client = GpuLockClient::new(reqwest::Client::new(), broker.uri());
        let err = client.acquire().await.unwrap_err();
        assert!(matches!(
            err,
            GpuLockClientError::ResponseTooLarge(cap) if cap == MAX_JSON_RESPONSE_BYTES
        ));
    }

    #[tokio::test]
    async fn acquire_returns_timeout_when_the_overall_deadline_elapses_despite_no_single_stall() {
        // Same rationale as mold_client.rs's own trickle test: a raw
        // socket, because wiremock has no primitive for streaming a
        // response body slowly one write at a time, which is exactly the
        // shape of bug this is regression-testing (no single read ever
        // stalls long enough to trip a per-read timeout, but the request
        // as a whole never completes either).
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut discard = [0u8; 1024];
            let _ = socket.read(&mut discard).await;

            let body = br#"{"lease_id":"trickled-lease"}"#;
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            for chunk in body.chunks(4) {
                let _ = socket.write_all(chunk).await;
                let _ = socket.flush().await;
                // Each individual gap (30ms) is far below any reasonable
                // per-read timeout; only their sum, bounded by the
                // overall deadline below, catches this.
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
        });

        let client = GpuLockClient::new(reqwest::Client::new(), format!("http://{addr}"))
            .with_overall_timeout(Duration::from_millis(80));

        let err = client.acquire().await.unwrap_err();
        assert!(matches!(err, GpuLockClientError::Timeout));
    }
}
