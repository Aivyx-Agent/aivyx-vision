# Aivyx-Vision Milestone 2, Pass A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `aivyx-vision` into a Cargo workspace and ship the
backend-agnostic `GenerationProvider` contract (`aivyx-vision-core`) plus
its first real implementation, `aivyx-vision-mold`'s synchronous
`generate_image` path (Pass A) — an HTTP client against an operator-run
`mold serve` instance, coordinated with local LLM inference on the same
machine via `aivyx-broker`'s GPU lock.

**Architecture:** `aivyx-vision-core` is pure types + one `async_trait`
(no I/O, no backend deps) so both consuming products can depend on it
without pulling in HTTP clients. `aivyx-vision-mold` has two small,
independent HTTP-client modules (`gpu_lock_client` against
`aivyx-broker`, `mold_client` against `mold serve`) composed by a
`provider` module that implements `GenerationProvider`: acquire the GPU
lock, call `mold serve`, always release the lock, write the returned
image bytes to disk, return a `GeneratedAsset`. `generate_3d` returns
`VisionError::Unsupported` for now — mold's own 3D surface
(`/api/mesh-workflows`) is a materially different async job-lifecycle
integration and is explicitly deferred to its own future build pass
("Pass B"), per
`aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
§7.

**Tech Stack:** Rust 2024 edition, Cargo workspace, `async-trait`,
`reqwest` (rustls, no native-tls), `serde`/`serde_json`, `thiserror`,
`wiremock` for HTTP-mocked tests, `tempfile` for filesystem tests.

## Global Constraints

- `cargo build --workspace`, `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo fmt --check` must stay clean — the same gate every sibling repo
  in this workspace already enforces.
- `aivyx-vision-core` has **no I/O and no backend dependencies** — only
  `async-trait` and `thiserror`. This is a hard constraint from the spec's
  own crate-layout comment (§2), not a style preference: other backends
  (`aivyx-vision-comfyui`, later) must be able to depend on this crate
  without pulling in `reqwest` or anything HTTP-specific.
- `ImageRequest`/`ThreeDRequest` stay backend-agnostic — no mold-specific
  field names (`negative_prompt`, `output_format`, etc.) leak into these
  types. Each backend crate translates internally (spec §4).
- `GeneratedAsset` carries a file path plus metadata, never raw bytes in
  memory (spec §4) — outputs can be large.
- `VisionError` must distinguish backend-unreachable, `Unsupported`,
  generation-timeout, and GPU-lock-acquisition-timeout as separate
  variants (spec §6) — a single opaque "generation failed" is explicitly
  rejected by the spec because the operator-facing fix differs per case.
- `aivyx-vision-mold` acquires the GPU lock before calling `mold serve`
  and releases it after, unconditionally (success or failure) — this is
  the entire reason the lock exists (spec §5); a leaked lease is only
  supposed to be reclaimed by `aivyx-broker`'s own `reap_expired` safety
  valve, not relied upon as the normal path.
- Dependency versions match what sibling repos in this workspace already
  pin, confirmed by reading `aivyx-broker/Cargo.toml` directly: `reqwest`
  `0.13.4` (`default-features = false`, features `json`/`rustls`, no
  `native-tls`), `tokio` `1.52.3`, `thiserror` `2.0.18`, `serde`/`serde_json`
  `1`, `wiremock` `0.6.5` (dev), `tempfile` `3.27.0` (dev).
- **CI never requires a GPU or a live backend.** Every test in this plan
  runs against `wiremock`-mocked HTTP servers or pure in-memory logic —
  nothing requires a real `mold serve` or `aivyx-broker` instance running
  (spec §7).
- Pass B (`generate_3d` for the mold backend) is explicitly out of scope
  for this plan. `MoldProvider::generate_3d` returns
  `VisionError::Unsupported` and does not touch the network at all.
- Real, researched grounding for `mold serve`'s HTTP contract (from
  `https://utensils.io/mold/api/`, confirmed 2026-09-18, default port
  `7680`): `POST /api/generate` is synchronous — JSON request
  (`prompt` required; `model`, `width`, `height`, `steps`, `seed`,
  `guidance`, `negative_prompt`, `source_image` [base64],
  `output_format` all optional; `batch_size` must stay at its default of
  1 for this endpoint) in, raw image bytes out on `200` with a real
  `Content-Type` header and an `x-mold-seed-used` header carrying the
  effective seed. Errors are JSON `{"error": "...", "code": "..."}` with
  HTTP status indicating category — `404`/`MODEL_NOT_FOUND` and
  `503`/`QUEUE_FULL` are the two documented typed codes this plan maps by
  name; anything else (`500` inference failures, etc.) falls through to a
  generic typed error carrying the real status/code/message. Optional
  `X-Api-Key` header when the operator has set `MOLD_API_KEY`.
- Real, verified contract for `aivyx-broker`'s GPU lock (built and merged
  in that repo on this same date — not researched, directly known):
  `POST /gpu-lock/acquire` → `200 {"lease_id": "<uuid>"}` or `503` on
  queue timeout; `POST /gpu-lock/release` with `{"lease_id": "<uuid>"}` →
  `200` on success, `404` if the lease is unknown/already
  released/expired, `400` if `lease_id` isn't a valid UUID.

---

## Task 1: Convert `aivyx-vision` into a Cargo workspace

**Files:**
- Move: `Cargo.toml` → `crates/aivyx-vision-svg/Cargo.toml`
- Move: `src/lib.rs` → `crates/aivyx-vision-svg/src/lib.rs`
- Create: new root `Cargo.toml` (workspace manifest)
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:** none — pure restructuring, zero behavior change. The
deliverable is: the exact same `aivyx-vision-svg` crate, now living under
`crates/`, building and testing identically to before.

This is not a TDD task (no new behavior), so its steps replace
write-test/verify-fail with record-baseline/verify-no-regression.

- [ ] **Step 1: Record the baseline**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test 2>&1 | tail -20
```

Expected: all existing `aivyx-vision-svg` tests pass (confirm the count
so Step 5 can be compared against it).

- [ ] **Step 2: Move the crate into `crates/aivyx-vision-svg/`**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
mkdir -p crates/aivyx-vision-svg
git mv src crates/aivyx-vision-svg/src
git mv Cargo.toml crates/aivyx-vision-svg/Cargo.toml
git rm Cargo.lock
```

(`Cargo.toml`'s own content needs no edits — every path in it is already
relative and path-independent. `Cargo.lock` is removed here because the
workspace gets a new, single lock file at the repo root once Step 3's
workspace manifest exists; `cargo build` regenerates it in Step 4.)

- [ ] **Step 3: Write the new root workspace manifest**

```toml
[workspace]
resolver = "2"
members = ["crates/*"]
```

Save as `/home/julian/Projects/Rust/aivyx-vision/Cargo.toml`.

- [ ] **Step 4: Build the workspace**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
```

Expected: succeeds, regenerates `Cargo.lock` at the repo root as the
workspace-level lock file.

- [ ] **Step 5: Verify no regression**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test --workspace 2>&1 | tail -20
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Expected: identical test count/pass rate to Step 1's baseline, clippy and
fmt clean.

- [ ] **Step 6: Update `CLAUDE.md`**

Replace the `## What this is` section's second sentence and the whole
`## Build, test, lint` section:

```markdown
`aivyx-vision` is a shared, local-first generation toolset for
`aivyx-pa` and `aivyx-coder` — image, 3D model, and vector/graphic-design
output as agent tool calls. This is now a Cargo workspace: `crates/aivyx-vision-svg`
(the vector/graphic-design milestone), `crates/aivyx-vision-core` (the
`GenerationProvider` trait + shared types, no I/O), and
`crates/aivyx-vision-mold` (mold-backed image generation — the image
milestone, 3D generation still pending its own build pass). See
`README.md` and
`aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
for the full rationale — this file only covers what's specific to working
in this repo's code.

No consumer depends on any of these crates yet — `aivyx-pa`'s and
`aivyx-coder`'s own adoption of `vision.*`-shaped tools is separate, later
work (each product's own `docs/superpowers/plans/`).

## Build, test, lint

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Cargo workspace, three members under `crates/` — use `-p <crate-name>` to
target one crate. Single test: `cargo test <test_name>`.
```

Update the `## Architecture` section's opening line from "Single file,
`src/lib.rs`:" to "`aivyx-vision-svg`'s own architecture (single file,
`crates/aivyx-vision-svg/src/lib.rs`):" — leave the bullet list under it
unchanged, it's still accurate.

- [ ] **Step 7: Update `README.md`**

Add one sentence to the top, after the existing first paragraph:

```markdown
This repo is now a Cargo workspace — `crates/aivyx-vision-svg` (this
crate), `crates/aivyx-vision-core`, and `crates/aivyx-vision-mold` are
its three members.
```

- [ ] **Step 8: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add -A
git commit -m "chore: convert aivyx-vision into a Cargo workspace

Relocates the existing aivyx-vision-svg crate to crates/aivyx-vision-svg
with zero behavior change, ahead of adding aivyx-vision-core and
aivyx-vision-mold as sibling workspace members (Milestone 2, Pass A)."
```

---

## Task 2: `aivyx-vision-core` — `GenerationProvider` trait, request/response types, `VisionError`

**Files:**
- Create: `crates/aivyx-vision-core/Cargo.toml`
- Create: `crates/aivyx-vision-core/src/lib.rs`

**Interfaces:**
- Produces: `GenerationProvider` trait (`async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError>`,
  `async fn generate_3d(&self, req: ThreeDRequest) -> Result<GeneratedAsset, VisionError>`),
  `ImageRequest { prompt: String, reference_image: Option<PathBuf>, width: Option<u32>, height: Option<u32>, seed: Option<u64>, style_hint: Option<String> }`,
  `ThreeDRequest { prompt: String, reference_image: Option<PathBuf>, style_hint: Option<String> }`,
  `GeneratedAsset { path: PathBuf, backend: String, seed_used: Option<u64>, generated_at: SystemTime }`,
  `VisionError` (`BackendUnreachable(String)`, `Unsupported(&'static str)`, `GenerationTimeout`, `GpuLockTimeout`, `BackendError { status: u16, code: Option<String>, message: String }`, `Io(String)`),
  `FakeGenerationProvider` (behind `#[cfg(any(test, feature = "testing"))]`, with `with_image_result`, `with_3d_result`, `captured_image_request`, `captured_3d_request`).

- [ ] **Step 1: Write the failing tests**

```rust
use std::sync::Arc;

use super::*;

#[tokio::test]
async fn generation_provider_is_usable_as_a_trait_object() {
    let provider: Arc<dyn GenerationProvider> =
        Arc::new(FakeGenerationProvider::with_image_result(Ok(GeneratedAsset {
            path: "/tmp/fake.png".into(),
            backend: "fake".to_string(),
            seed_used: Some(42),
            generated_at: std::time::SystemTime::now(),
        })));
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-core -- --nocapture
```

Expected: compile error — `aivyx-vision-core` doesn't exist as a crate
yet.

- [ ] **Step 3: Create the crate manifest**

```toml
[package]
name = "aivyx-vision-core"
description = "GenerationProvider trait, request/response types, and VisionError -- the shared, backend-agnostic contract for the Aivyx-Vision toolset. No I/O, no backend dependencies."
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[features]
testing = []

[dependencies]
async-trait = "0.1.89"
thiserror = "2.0.18"

[dev-dependencies]
tokio = { version = "1.52.3", features = ["macros", "rt"] }
```

Save as `crates/aivyx-vision-core/Cargo.toml`.

- [ ] **Step 4: Implement the crate**

```rust
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
    // (Step 1's tests land here.)
}
```

Save as `crates/aivyx-vision-core/src/lib.rs`.

- [ ] **Step 5: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-core -- --nocapture
```

Expected: all 5 new tests pass.

- [ ] **Step 6: Verify the `testing` feature compiles standalone**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build -p aivyx-vision-core --features testing
```

Expected: succeeds — this is the exact way a *consuming* crate (a future
`aivyx-pa`/`aivyx-coder` adapter) will pull in `FakeGenerationProvider`,
so it must compile outside of this crate's own `cfg(test)`.

- [ ] **Step 7: Run full workspace check**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

- [ ] **Step 8: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add -A
git commit -m "feat: add aivyx-vision-core -- GenerationProvider trait, types, VisionError

Pure types + one async_trait, no I/O, no backend dependencies -- the
shared contract aivyx-vision-mold (next) and later aivyx-vision-comfyui
implement. Includes FakeGenerationProvider (behind a 'testing' feature)
so both consuming products can test their own tool-adapter logic without
a real backend running, per the spec's own §7 testing strategy."
```

---

## Task 3: `aivyx-vision-mold` — `GpuLockClient`

**Files:**
- Create: `crates/aivyx-vision-mold/Cargo.toml`
- Create: `crates/aivyx-vision-mold/src/lib.rs`
- Create: `crates/aivyx-vision-mold/src/gpu_lock_client.rs`

**Interfaces:**
- Produces: `LeaseId` (opaque, `Display`-able, wraps the raw UUID string
  the broker returns — no local UUID parsing needed since this crate only
  ever round-trips the value verbatim), `GpuLockClientError` (`Transport(String)`,
  `Timeout`, `ReleaseRejected(String)`), `GpuLockClient::new(http: reqwest::Client, broker_url: String) -> Self`,
  `async fn acquire(&self) -> Result<LeaseId, GpuLockClientError>`,
  `async fn release(&self, lease: &LeaseId) -> Result<(), GpuLockClientError>`.

- [ ] **Step 1: Write the failing tests**

```rust
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold -- --nocapture
```

Expected: compile error — `aivyx-vision-mold` doesn't exist as a crate
yet.

- [ ] **Step 3: Create the crate manifest**

```toml
[package]
name = "aivyx-vision-mold"
description = "HTTP-client backend for Aivyx-Vision's GenerationProvider trait, talking to an operator-run `mold serve` instance for image generation, coordinated via aivyx-broker's GPU lock"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[dependencies]
aivyx-vision-core = { path = "../aivyx-vision-core" }
async-trait = "0.1.89"
reqwest = { version = "0.13.4", default-features = false, features = ["json", "rustls"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2.0.18"
tracing = "0.1.44"

[dev-dependencies]
tempfile = "3.27.0"
tokio = { version = "1.52.3", features = ["macros", "rt"] }
wiremock = "0.6.5"
```

Save as `crates/aivyx-vision-mold/Cargo.toml`.

- [ ] **Step 4: Implement `GpuLockClient`**

```rust
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
    // (Step 1's tests land here.)
}
```

Save as `crates/aivyx-vision-mold/src/gpu_lock_client.rs`.

- [ ] **Step 5: Wire the module into `lib.rs`**

```rust
//! HTTP-client backend for Aivyx-Vision's `GenerationProvider` trait,
//! talking to an operator-run `mold serve` instance for image
//! generation, coordinated via `aivyx-broker`'s GPU lock so it doesn't
//! contend with local LLM inference sharing the same GPU. See
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5/§7 (Milestone 2) for the full
//! design.
//!
//! 3D generation (`generate_3d`) is not yet implemented -- mold's own
//! `/api/mesh-workflows` job-lifecycle surface is materially more complex
//! than the synchronous `/api/generate` this crate currently wraps, and
//! is deliberately deferred to its own build pass ("Pass B").

pub mod gpu_lock_client;

pub use gpu_lock_client::{GpuLockClient, GpuLockClientError, LeaseId};
```

Save as `crates/aivyx-vision-mold/src/lib.rs`.

- [ ] **Step 6: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold -- --nocapture
```

Expected: all 5 new tests pass.

- [ ] **Step 7: Run full workspace check**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

- [ ] **Step 8: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add -A
git commit -m "feat: add aivyx-vision-mold's GpuLockClient

HTTP client against aivyx-broker's POST /gpu-lock/acquire and
POST /gpu-lock/release, matching that repo's real, verified contract
(200 {lease_id} or 503 on acquire; 200/404/400 on release). Round-trips
the lease id as an opaque string -- no UUID parsing or dependency needed
on this side."
```

---

## Task 4: `aivyx-vision-mold` — `MoldClient`

**Files:**
- Create: `crates/aivyx-vision-mold/src/mold_client.rs`
- Modify: `crates/aivyx-vision-mold/src/lib.rs`

**Interfaces:**
- Produces: `GenerateImageRequest { prompt: String, model: Option<String>, width: Option<u32>, height: Option<u32>, steps: Option<u32>, seed: Option<u64>, guidance: Option<f32>, negative_prompt: Option<String>, source_image: Option<String>, output_format: Option<String> }`
  (derives `Default` so callers only set the fields they use via
  `..Default::default()`), `GenerateImageResponse { bytes: Vec<u8>, content_type: String, seed_used: Option<u64> }`,
  `MoldClientError` (`Transport(String)`, `ModelNotFound(String)`, `QueueFull`, `BackendError { status: u16, code: Option<String>, message: String }`),
  `MoldClient::new(http: reqwest::Client, base_url: String, api_key: Option<String>) -> Self`,
  `async fn generate(&self, req: GenerateImageRequest) -> Result<GenerateImageResponse, MoldClientError>`.

- [ ] **Step 1: Write the failing tests**

```rust
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold mold_client -- --nocapture
```

Expected: compile error — `mold_client` module doesn't exist yet.

- [ ] **Step 3: Implement `MoldClient`**

```rust
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
        let response = builder
            .send()
            .await
            .map_err(|e| MoldClientError::Transport(e.to_string()))?;

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
                .map_err(|e| MoldClientError::Transport(e.to_string()))?
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
    // (Step 1's tests land here.)
}
```

Save as `crates/aivyx-vision-mold/src/mold_client.rs`.

- [ ] **Step 4: Wire the module into `lib.rs`**

```rust
//! HTTP-client backend for Aivyx-Vision's `GenerationProvider` trait,
//! talking to an operator-run `mold serve` instance for image
//! generation, coordinated via `aivyx-broker`'s GPU lock so it doesn't
//! contend with local LLM inference sharing the same GPU. See
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5/§7 (Milestone 2) for the full
//! design.
//!
//! 3D generation (`generate_3d`) is not yet implemented -- mold's own
//! `/api/mesh-workflows` job-lifecycle surface is materially more complex
//! than the synchronous `/api/generate` this crate currently wraps, and
//! is deliberately deferred to its own build pass ("Pass B").

pub mod gpu_lock_client;
pub mod mold_client;

pub use gpu_lock_client::{GpuLockClient, GpuLockClientError, LeaseId};
pub use mold_client::{GenerateImageRequest, GenerateImageResponse, MoldClient, MoldClientError};
```

Save as `crates/aivyx-vision-mold/src/lib.rs`.

- [ ] **Step 5: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold -- --nocapture
```

Expected: all 6 new tests pass (plus Task 3's 5, still green).

- [ ] **Step 6: Run full workspace check**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

- [ ] **Step 7: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add -A
git commit -m "feat: add aivyx-vision-mold's MoldClient

HTTP client against mold serve's POST /api/generate, grounded directly
against mold's own published API reference. Maps the two documented
typed error codes (MODEL_NOT_FOUND, QUEUE_FULL) to named error variants;
everything else (500s, unrecognized codes) falls through to a generic
typed BackendError carrying the real status/code/message."
```

---

## Task 5: `aivyx-vision-mold` — `MoldProvider`, config, docs, final verification

**Files:**
- Create: `crates/aivyx-vision-mold/src/provider.rs`
- Modify: `crates/aivyx-vision-mold/src/lib.rs`
- Modify: `crates/aivyx-vision-mold/Cargo.toml`
- Create: `crates/aivyx-vision-mold/README.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `GpuLockClient`/`GpuLockClientError` (Task 3), `MoldClient`/`GenerateImageRequest`/`GenerateImageResponse`/`MoldClientError` (Task 4), `aivyx_vision_core::{GenerationProvider, ImageRequest, ThreeDRequest, GeneratedAsset, VisionError}` (Task 2).
- Produces: `MoldConfig { broker_url: String, mold_url: String, api_key: Option<String>, output_dir: PathBuf }`,
  `MoldProvider::new(config: MoldConfig) -> Result<Self, VisionError>`,
  `impl GenerationProvider for MoldProvider`.

- [ ] **Step 1: Write the failing tests**

```rust
use std::path::PathBuf;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;

async fn test_provider(broker: &MockServer, mold: &MockServer, output_dir: PathBuf) -> MoldProvider {
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
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-1"})))
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
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-2"})))
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

#[test]
fn extension_for_content_type_maps_known_types_and_falls_back_to_bin() {
    assert_eq!(extension_for_content_type("image/png"), "png");
    assert_eq!(extension_for_content_type("image/jpeg"), "jpg");
    assert_eq!(extension_for_content_type("application/octet-stream"), "bin");
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold provider -- --nocapture
```

Expected: compile error — `provider` module, `MoldProvider`, `MoldConfig`
don't exist yet.

- [ ] **Step 3: Add `base64` and `uuid` dependencies**

```toml
[package]
name = "aivyx-vision-mold"
description = "HTTP-client backend for Aivyx-Vision's GenerationProvider trait, talking to an operator-run `mold serve` instance for image generation, coordinated via aivyx-broker's GPU lock"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[dependencies]
aivyx-vision-core = { path = "../aivyx-vision-core" }
async-trait = "0.1.89"
base64 = "0.22"
reqwest = { version = "0.13.4", default-features = false, features = ["json", "rustls"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2.0.18"
tracing = "0.1.44"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tempfile = "3.27.0"
tokio = { version = "1.52.3", features = ["macros", "rt"] }
wiremock = "0.6.5"
```

Save as `crates/aivyx-vision-mold/Cargo.toml`. (`uuid` here is only for
generating unique output filenames when writing a `GeneratedAsset` to
disk — unrelated to `GpuLockClient`'s `LeaseId`, which deliberately stays
`uuid`-free since it only ever round-trips the broker's own string.)

- [ ] **Step 4: Implement `MoldProvider`**

```rust
//! `GenerationProvider` implementation backed by `mold serve`: acquires
//! the GPU lock, calls mold's synchronous `/api/generate`, always
//! releases the lock (success or failure), writes the returned image
//! bytes to disk, and returns a `GeneratedAsset`. `generate_3d` returns
//! `VisionError::Unsupported` -- see this crate's own top-level doc
//! comment for why.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use aivyx_vision_core::{
    GeneratedAsset, GenerationProvider, ImageRequest, ThreeDRequest, VisionError,
};
use async_trait::async_trait;
use base64::Engine;

use crate::gpu_lock_client::{GpuLockClient, GpuLockClientError};
use crate::mold_client::{GenerateImageRequest, GenerateImageResponse, MoldClient, MoldClientError};

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

        Ok(Self {
            gpu_lock: GpuLockClient::new(http.clone(), config.broker_url),
            mold: MoldClient::new(http, config.mold_url, config.api_key),
            output_dir: config.output_dir,
        })
    }
}

#[async_trait]
impl GenerationProvider for MoldProvider {
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
        let lease = self
            .gpu_lock
            .acquire()
            .await
            .map_err(map_gpu_lock_error)?;

        let mold_req = build_mold_request(&req);
        let result = match mold_req {
            Ok(mold_req) => self.mold.generate(mold_req).await.map_err(map_mold_error),
            Err(e) => Err(e),
        };

        if let Err(e) = self.gpu_lock.release(&lease).await {
            tracing::warn!(
                error = %e,
                "gpu-lock release failed after generation; aivyx-broker's reap_expired will eventually reclaim it"
            );
        }

        let response = result?;
        write_generated_asset(&self.output_dir, response)
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
    // (Step 1's tests land here.)
}
```

Save as `crates/aivyx-vision-mold/src/provider.rs`.

- [ ] **Step 5: Wire the module into `lib.rs`**

```rust
//! HTTP-client backend for Aivyx-Vision's `GenerationProvider` trait,
//! talking to an operator-run `mold serve` instance for image
//! generation, coordinated via `aivyx-broker`'s GPU lock so it doesn't
//! contend with local LLM inference sharing the same GPU. See
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5/§7 (Milestone 2) for the full
//! design.
//!
//! 3D generation (`generate_3d`) is not yet implemented -- mold's own
//! `/api/mesh-workflows` job-lifecycle surface is materially more complex
//! than the synchronous `/api/generate` this crate currently wraps, and
//! is deliberately deferred to its own build pass ("Pass B").

pub mod gpu_lock_client;
pub mod mold_client;
mod provider;

pub use gpu_lock_client::{GpuLockClient, GpuLockClientError, LeaseId};
pub use mold_client::{GenerateImageRequest, GenerateImageResponse, MoldClient, MoldClientError};
pub use provider::{MoldConfig, MoldProvider};
```

Save as `crates/aivyx-vision-mold/src/lib.rs`.

- [ ] **Step 6: Run to verify they pass**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold -- --nocapture
```

Expected: all 8 new tests pass (plus Tasks 3+4's 11, still green — 19
total in this crate).

- [ ] **Step 7: Write `crates/aivyx-vision-mold/README.md`**

```markdown
# aivyx-vision-mold

`GenerationProvider` backend for the [Aivyx-Vision](../../README.md)
toolset, backed by [`mold serve`](https://github.com/utensils/mold) — a
pure-Rust, Candle-based local image generator. This crate is an HTTP
client against an operator-run `mold serve` instance; it does not bundle,
install, or manage `mold` itself, the same arm's-length relationship this
ecosystem already has with Ollama and ComfyUI.

## Status

- `generate_image` — implemented (Pass A). Calls `mold serve`'s
  synchronous `POST /api/generate`.
- `generate_3d` — **not yet implemented**, always returns
  `VisionError::Unsupported`. `mold`'s 3D surface
  (`POST /api/mesh-workflows`) is an async, durable job-lifecycle API
  (create → poll/SSE → resume/cancel), a materially different integration
  shape from the single synchronous call `generate_image` wraps —
  deliberately deferred to its own build pass ("Pass B").

## GPU coordination

Every `generate_image` call acquires a lease from
[`aivyx-broker`](https://github.com/Aivyx-Agent/aivyx-broker)'s
`POST /gpu-lock/acquire` before calling `mold serve`, and releases it via
`POST /gpu-lock/release` afterward — unconditionally, whether generation
succeeded or failed. This keeps `mold serve`'s GPU-heavy work from
contending with local LLM inference sharing the same machine through the
same broker. `aivyx-broker` must be running and reachable at
`MoldConfig::broker_url`; if the lock can't be acquired within the
broker's own configured queue timeout, `generate_image` returns
`VisionError::GpuLockTimeout`.

## Configuration

```rust
use aivyx_vision_core::GenerationProvider;
use aivyx_vision_mold::{MoldConfig, MoldProvider};

let provider = MoldProvider::new(MoldConfig {
    broker_url: "http://127.0.0.1:8899".to_string(),
    mold_url: "http://127.0.0.1:7680".to_string(),
    api_key: None, // Some("...") if the mold instance has MOLD_API_KEY set
    output_dir: "/path/to/generated/assets".into(),
})?;
```

`output_dir` is created if it doesn't already exist. Each generated file
is named `<uuid>.<ext>`, where `<ext>` is derived from `mold`'s own
`Content-Type` response header (`png`, `jpg`, `webp`; unrecognized types
fall back to `bin`).

## Honest tradeoffs

- **`mold serve` and `aivyx-broker` are both required-up dependencies**
  for `generate_image` to work at all — if either is down, the call fails
  with `VisionError::BackendUnreachable`, same trust model as this whole
  ecosystem already has with Ollama/`llama-server`.
- **No retry logic.** A transient `mold serve` failure or a GPU-lock
  timeout is returned as-is; retrying is the caller's decision.
- **`MOLD_READ_TIMEOUT` (300s) is a fixed constant, not yet
  configurable** via `MoldConfig` — generous enough for realistic local
  FLUX/SDXL generation time, but a v1 simplification.
```

- [ ] **Step 8: Update root `CLAUDE.md`**

Update the `crates/aivyx-vision-mold` mention in the `## What this is`
section (written in Task 1's Step 6) — replace "mold-backed image
generation — the image milestone, 3D generation still pending its own
build pass" with:

```markdown
`crates/aivyx-vision-mold` (mold-backed image generation — Milestone 2
Pass A shipped: `generate_image` via `mold serve`'s synchronous
`/api/generate`, coordinated through `aivyx-broker`'s GPU lock;
`generate_3d` returns `VisionError::Unsupported` until Pass B implements
mold's async `/api/mesh-workflows` job-lifecycle surface)
```

- [ ] **Step 9: Update root `README.md`**

Add a new section after the existing content:

```markdown
## Milestone 2 (Pass A): image generation

`crates/aivyx-vision-mold` implements `GenerationProvider::generate_image`
against an operator-run [`mold serve`](https://github.com/utensils/mold)
instance, coordinated with local LLM inference sharing the same GPU via
[`aivyx-broker`](https://github.com/Aivyx-Agent/aivyx-broker)'s GPU lock.
See `crates/aivyx-vision-mold/README.md` for configuration and status —
`generate_3d` is not yet implemented (Pass B, a separate future build
pass covering mold's async 3D job-lifecycle API).
```

- [ ] **Step 10: Final full-workspace verification**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
cargo test --workspace 2>&1 | tail -40
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green — 3 workspace members, every test across all of them
passing, clippy/fmt clean.

- [ ] **Step 11: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add -A
git commit -m "feat: add aivyx-vision-mold's MoldProvider -- GenerationProvider impl

Wires GpuLockClient and MoldClient together: acquire the GPU lock, call
mold serve, always release the lock (success or failure), write the
returned bytes to disk, return a GeneratedAsset. generate_3d returns
VisionError::Unsupported -- Pass B (mold's async mesh-workflows surface)
is a separate future build pass. Completes Milestone 2 Pass A per
aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md."
```

---

## Final verification (after all 5 tasks land)

- [ ] Run the complete workspace test suite once, not per-task:

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test --workspace 2>&1 | tail -40
```

- [ ] Run the documented clippy + fmt commands once more:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

- [ ] Confirm `aivyx-vision-core` still has zero I/O dependencies:

```bash
grep -E '^(reqwest|tokio|hyper|std::net|std::fs)' crates/aivyx-vision-core/Cargo.toml crates/aivyx-vision-core/src/lib.rs
```

Expected: no matches (only `async-trait` and `thiserror` in its
`Cargo.toml`, and `use std::path::PathBuf`/`use std::time::SystemTime`
in its `lib.rs` — no filesystem or network I/O).

- [ ] This plan does not decide whether to push a branch / open a PR —
  follow `superpowers:finishing-a-development-branch` once all tasks are
  individually reviewed and a final whole-branch review has passed, same
  as the `aivyx-broker` GPU-lock-extension plan executed just before
  this one.

## Explicitly out of scope for this plan

(Copied forward from the spec's own "What this spec does not decide" and
"Downstream" sections, so a future reader doesn't mistake this plan's
silence on these for an oversight.)

- Pass B: `generate_3d` for the mold backend (async
  `/api/mesh-workflows` job lifecycle).
- `aivyx-vision-comfyui` (deferred entirely per the spec's 2026-09-18
  amendment).
- Both products' own adapters (`aivyx-pa`'s `vision.generate_image` tool,
  `aivyx-coder`'s equivalent `Tool` impl) — separate, later work in each
  product's own repo.
- `vision.generate` capability-tier placement in `aivyx-pa`'s
  `CEILING_*` tables, and the A3 addendum count-drift update.
- Retention/cleanup policy for accumulated generated files.
- Rate-limiting / cost-budget integration with `aivyx-pa`'s Chapter K
  budget mechanism.
- `docs/TOOLS.md` (`aivyx-pa`) and the equivalent tool-catalog entry in
  `aivyx-coder`.
