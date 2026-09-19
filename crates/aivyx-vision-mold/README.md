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
- **Cancelling the caller's own future doesn't cancel the underlying GPU
  work or skip releasing the lock.** `generate_image` runs its
  `acquire -> generate -> release -> write` sequence on an internally
  spawned task; if the calling future is dropped (a timeout, a
  `tokio::select!`, a shutdown signal), that task keeps running to real
  completion regardless -- the lock is only ever released after
  generation genuinely finishes, never early. A cancelled call still
  costs real GPU time server-side; the caller just never sees the result.
  This closes *future-drop* cancellation specifically -- a panic inside
  the spawned task before `release()` runs (unrelated to cancellation,
  and no worse than before this task existed) still leaks the lease until
  `aivyx-broker`'s `reap_expired` reclaims it; there's no `Drop` guard for
  that narrower case either, by the same reasoning above.
