//! A small helper shared by `mold_client.rs` and `gpu_lock_client.rs`:
//! reads an HTTP response body in bounded chunks via
//! `reqwest::Response::chunk()`, rather than `.bytes()`/`.json()`/`.text()`,
//! which all buffer the *entire* body regardless of its size. Each caller
//! passes its own cap -- large for `mold serve`'s generated image bytes,
//! much smaller for `aivyx-broker`'s tiny JSON control-endpoint responses
//! -- so a misbehaving or malicious server sending an unbounded body fails
//! with a clear, typed error instead of growing this process's memory
//! without limit. (Audit note: `~/aivyx-audit-2026-10-04/C-report.md`,
//! item V7.)

/// A response body either exceeded its caller-supplied cap, or the
/// underlying read itself failed partway through (a dropped connection,
/// a timeout, ...). Each client module maps this onto its own error type.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CappedBodyError {
    #[error("response body exceeded the {0}-byte limit")]
    TooLarge(usize),
    #[error("reading response body: {0}")]
    Transport(reqwest::Error),
}

/// Reads `response`'s body into memory, failing fast with
/// `CappedBodyError::TooLarge` the moment the total read so far (or a
/// declared `Content-Length`) would exceed `max_bytes`, instead of ever
/// buffering more than that much data.
pub(crate) async fn read_capped_body(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, CappedBodyError> {
    // A declared Content-Length over the cap is rejected before reading
    // any body bytes at all -- no reason to read even one chunk of a
    // response that already announced it's too big.
    if let Some(len) = response.content_length()
        && len > max_bytes as u64
    {
        return Err(CappedBodyError::TooLarge(max_bytes));
    }

    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(CappedBodyError::Transport)? {
        if buf.len() + chunk.len() > max_bytes {
            return Err(CappedBodyError::TooLarge(max_bytes));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}
