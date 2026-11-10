//! Audit 2026-11-11 X1: `MoldProvider` talks only to the operator's own
//! `mold serve` and `aivyx-broker`, so it must never route through
//! `HTTP_PROXY`/`ALL_PROXY` -- reqwest would otherwise send even loopback
//! requests (the prompt, the API key, a reference image) to the proxy.
//!
//! Its own test binary: it sets process-wide proxy variables, which would
//! leak into any other test sharing the process.

use std::time::Duration;

use aivyx_vision_core::{GenerationProvider, ImageRequest, VisionError};
use aivyx_vision_mold::{MoldConfig, MoldProvider};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn generate_image_ignores_proxy_environment_variables() {
    let proxy = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_url = format!("http://{}", proxy.local_addr().unwrap());
    // SAFETY: this binary runs this one test; nothing else reads the
    // environment concurrently.
    unsafe {
        for var in [
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ] {
            std::env::set_var(var, &proxy_url);
        }
        std::env::remove_var("NO_PROXY");
        std::env::remove_var("no_proxy");
    }
    let proxied = tokio::spawn(async move {
        let (mut socket, _) = proxy.accept().await.unwrap();
        let mut buf = [0u8; 512];
        let n = socket.read(&mut buf).await.unwrap_or(0);
        let _ = socket
            .write_all(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n")
            .await;
        String::from_utf8_lossy(&buf[..n]).into_owned()
    });

    // Nothing listens on port 9 ("discard"), so a direct request fails to
    // connect; a proxied one would reach the listener above instead.
    let provider = MoldProvider::new(MoldConfig {
        broker_url: "http://127.0.0.1:9".to_string(),
        mold_url: "http://127.0.0.1:9".to_string(),
        api_key: Some("secret-key".to_string()),
        output_dir: std::env::temp_dir(),
    })
    .unwrap();
    let err = provider
        .generate_image(ImageRequest {
            prompt: "a red circle".to_string(),
            reference_image: None,
            width: None,
            height: None,
            seed: None,
            style_hint: None,
        })
        .await
        .unwrap_err();

    assert!(
        matches!(err, VisionError::BackendUnreachable(_)),
        "a direct connection to the closed port must fail as unreachable, got: {err:?}"
    );
    let seen = tokio::time::timeout(Duration::from_millis(300), proxied).await;
    assert!(
        seen.is_err(),
        "the request went through the proxy: {:?}",
        seen.ok().and_then(Result::ok)
    );
}
