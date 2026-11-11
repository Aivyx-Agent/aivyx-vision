//! Replays responses recorded from a real `mold serve` 0.37.0 (CUDA, RTX
//! 4090, model `flux2-klein:q8`) on 2026-11-11, so CI checks this crate
//! against mold's real shapes without a GPU (design spec §7; audit
//! 2026-11-11 M1).
//!
//! What the live run found, and these tests pin:
//! - `/api/generate` requires `model`, `width`, `height` and `steps`;
//! - request-validation failures are plain text, other errors JSON;
//! - sizes off the model's grid are refused (422), not adjusted;
//! - `UNKNOWN_MODEL` (404) and `UNAUTHORIZED` (401) codes;
//! - a successful reply carries `x-mold-seed-used` and
//!   `x-mold-request-warning`;
//! - sizes over the model's `max_pixels` are refused (422), and
//!   image-to-image takes `strength` (second live run, 2026-11-11).
//!
//! Recorded bodies are verbatim except the image itself: the real reply
//! was a 512x512 PNG. `fixtures/mold-0.37-no-pixels.png` is that reply with
//! its pixel data (`IDAT`) cut out; the header and the text chunks mold
//! embedded (prompt, model, settings) are byte for byte.

use aivyx_vision_core::{GenerationProvider, ImageRequest, VisionError};
use aivyx_vision_mold::{MoldConfig, MoldProvider};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `GET /api/models`, reduced to the downloaded model's fields this crate
/// reads (the real list has about 180 entries with many more fields).
const MODELS: &str = r#"[{"name":"flux2-klein:q8","family":"flux2","downloaded":true,"default_steps":4,"default_width":1024,"default_height":1024,"dimension_alignment":16,"max_pixels":1800000,"default_guidance":1.0},{"name":"flux-schnell:q8","family":"flux","downloaded":false,"default_steps":4,"default_width":1024,"default_height":1024,"dimension_alignment":16}]"#;
const WARNING: &str = "512x512 is not a recommended resolution for flux2 models. Suggested: 1024x1024, 1024x768, 768x1024, 1024x576, ... (6 total)";
const E401: &str = r#"{"error":"invalid API key","code":"UNAUTHORIZED"}"#;
const E404: &str = r#"{"error":"deferred generation preparation failed: unknown model 'no-such-model:q8'. Run 'mold list' to see available models.","code":"UNKNOWN_MODEL","recovery":{"job_id":"5e9aa8e1-9deb-4b44-9e95-c55e1d9db8a5","batch_id":"ef74a7cc-e640-4551-8c89-180b468a71fe","retryable":true}}"#;
const E422_MISSING: &str = "Failed to deserialize the JSON body into the target type: missing field `steps` at line 1 column 64";
const E422_TOO_BIG: &str = r#"{"error":"2048x2048 = 4.19 megapixels exceeds the 1.8MP limit (VAE VRAM constraint)","code":"VALIDATION_ERROR"}"#;
const PNG: &[u8] = include_bytes!("fixtures/mold-0.37-no-pixels.png");

async fn broker() -> MockServer {
    let broker = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/gpu-lock/acquire"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "replay"})),
        )
        .mount(&broker)
        .await;
    Mock::given(method("POST"))
        .and(path("/gpu-lock/release"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&broker)
        .await;
    broker
}

async fn mold_with_models() -> MockServer {
    let mold = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(MODELS, "application/json"))
        .mount(&mold)
        .await;
    mold
}

fn provider(broker: &MockServer, mold: &MockServer, dir: &std::path::Path) -> MoldProvider {
    MoldProvider::new(MoldConfig {
        broker_url: broker.uri(),
        mold_url: mold.uri(),
        api_key: Some("audit-test-key".to_string()),
        output_dir: dir.to_path_buf(),
        model: None,
    })
    .unwrap()
}

fn request(width: u32, height: u32) -> ImageRequest {
    ImageRequest {
        prompt: "a brass compass".to_string(),
        reference_image: None,
        width: Some(width),
        height: Some(height),
        seed: Some(3),
        strength: None,
        style_hint: None,
    }
}

/// Without every required field mold answers this plain-text 422, so the
/// mock only answers 200 when they are all present.
#[tokio::test]
async fn a_generation_sends_what_mold_requires_and_reads_its_headers() {
    let broker = broker().await;
    let mold = mold_with_models().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(serde_json::json!({
            "model": "flux2-klein:q8", "width": 512, "height": 512, "steps": 4, "seed": 3
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(PNG, "image/png")
                .insert_header("x-mold-seed-used", "3")
                .insert_header("x-mold-gpu", "0")
                .insert_header("x-mold-request-warning", WARNING)
                .insert_header("x-mold-dimension-warning", WARNING),
        )
        .mount(&mold)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(
            ResponseTemplate::new(422).set_body_raw(E422_MISSING, "text/plain; charset=utf-8"),
        )
        .mount(&mold)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let asset = provider(&broker, &mold, dir.path())
        .generate_image(request(512, 512))
        .await
        .unwrap();
    assert_eq!(asset.seed_used, Some(3));
    assert_eq!(asset.warnings, vec![WARNING.to_string()]);
    assert_eq!(asset.path.extension().unwrap(), "png");

    // mold's embedded prompt and settings don't reach the file; the
    // header does.
    let written = std::fs::read(&asset.path).unwrap();
    assert!(written.starts_with(&PNG[..33]), "signature and IHDR kept");
    assert!(PNG.windows(5).any(|w| w == b"mold:"));
    assert!(!written.windows(5).any(|w| w == b"mold:"), "{written:?}");
    assert!(!written.windows(13).any(|w| w == b"brass compass"));
}

/// mold refuses 500x500 on flux2 ("must be multiples of 16"); the size is
/// rounded first, and the agent is told.
#[tokio::test]
async fn an_off_grid_size_is_rounded_before_mold_sees_it() {
    let broker = broker().await;
    let mold = mold_with_models().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(
            serde_json::json!({"width": 496, "height": 496}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(PNG, "image/png"))
        .expect(1)
        .mount(&mold)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let asset = provider(&broker, &mold, dir.path())
        .generate_image(request(500, 500))
        .await
        .unwrap();
    assert!(
        asset.warnings[0].contains("496x496 instead of 500x500"),
        "{:?}",
        asset.warnings
    );
    mold.verify().await;
}

/// mold refuses 2048x2048 on flux2 ("exceeds the 1.8MP limit"); the size
/// is scaled down to fit first, keeping its shape, and the agent is told.
#[tokio::test]
async fn a_size_over_the_pixel_limit_is_scaled_before_mold_sees_it() {
    let broker = broker().await;
    let mold = mold_with_models().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(
            serde_json::json!({"width": 1328, "height": 1328}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(PNG, "image/png"))
        .expect(1)
        .mount(&mold)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(422).set_body_raw(E422_TOO_BIG, "application/json"))
        .mount(&mold)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let asset = provider(&broker, &mold, dir.path())
        .generate_image(request(2048, 2048))
        .await
        .unwrap();
    assert!(
        asset.warnings[0].contains("at most 1800000 pixels: generated 1328x1328"),
        "{:?}",
        asset.warnings
    );
    mold.verify().await;
}

#[tokio::test]
async fn molds_recorded_errors_read_clearly() {
    for (status, body, content_type, expect) in [
        (401, E401, "application/json", "rejected the API key"),
        (
            404,
            E404,
            "application/json",
            "unknown model 'no-such-model:q8'",
        ),
        (
            422,
            E422_MISSING,
            "text/plain; charset=utf-8",
            "missing field `steps`",
        ),
        (
            422,
            E422_TOO_BIG,
            "application/json",
            "exceeds the 1.8MP limit",
        ),
    ] {
        let broker = broker().await;
        let mold = mold_with_models().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(status).set_body_raw(body, content_type))
            .mount(&mold)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let err = provider(&broker, &mold, dir.path())
            .generate_image(request(512, 512))
            .await
            .unwrap_err();
        if status == 404 {
            assert!(
                matches!(err, VisionError::BackendError { status: 404, .. }),
                "{err:?}"
            );
        }
        assert!(err.to_string().contains(expect), "{status}: {err}");
    }
}
