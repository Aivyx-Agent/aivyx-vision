//! The per-request local-only check. `LocalOnlyDns` guards names at connect
//! time, but reqwest never calls a resolver for an IP-literal URL, so every
//! request is also checked here before it is sent.

/// `Ok` when `url` names a local or private-network host. The error is the
/// guard's own "... is not local (...)" text.
pub(crate) async fn ensure_local(url: &str) -> Result<(), String> {
    let url = url.to_string();
    // Resolving a name is a blocking lookup.
    tokio::task::spawn_blocking(move || aivyx_route::require_local(&url).map_err(|e| e.to_string()))
        .await
        .map_err(|e| format!("locality check failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_public_ip_literal_is_refused_and_loopback_is_allowed() {
        let err = ensure_local("http://203.0.113.7:7680").await.unwrap_err();
        assert!(err.contains("is not local"), "{err}");
        assert!(ensure_local("http://127.0.0.1:7680").await.is_ok());
    }
}
