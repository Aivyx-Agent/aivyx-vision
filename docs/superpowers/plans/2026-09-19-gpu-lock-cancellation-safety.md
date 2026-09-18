# GPU-Lock Cancellation Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the GPU-lease leak `aivyx-vision-mold`'s own final
whole-branch review found and deliberately deferred: if the future
returned by `MoldProvider::generate_image` is dropped mid-flight (the
caller cancels — a timeout wrapper, `tokio::select!`, a shutdown signal),
the GPU lock is never released, and only `aivyx-broker`'s `reap_expired`
safety valve (up to `max_hold`, default 900s) eventually reclaims it.

**Architecture:** Move the entire `acquire → generate → release → write`
sequence inside a `tokio::spawn`'d task, and have `generate_image` itself
just spawn it and `.await` the resulting `JoinHandle`. If the caller's own
future is dropped, only that `.await` is abandoned — the spawned task
keeps running independently to real completion (mold's response fully
awaited, lock genuinely released only after generation actually finished)
regardless. This is deliberately **not** an RAII/`Drop`-based release
guard: a `Drop` guard would release the lock the instant the caller's
future is dropped, which could be *before* `mold serve` has actually
finished (or even noticed the disconnect) — letting a second caller
acquire the lock and start a second GPU job while the first is still
running server-side, exactly the double-GPU-usage scenario this lock
exists to prevent. Detaching the whole operation avoids that: the
trade-off is that a cancelled call still burns real GPU time to
completion server-side (the caller just never sees the result), which is
the correct trade-off given the lock's actual purpose.

**Tech Stack:** Rust, `tokio::spawn`/`JoinHandle` (promoting `tokio` from
a dev-only to a real dependency of `aivyx-vision-mold` — both consuming
products already run inside a Tokio runtime, so this isn't a new
transitive burden in practice), `wiremock`'s `set_delay` for the
timing-sensitive regression test.

## Global Constraints

- `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo fmt --check` must stay clean.
- `generate_3d` is untouched — it never blocks or holds a lock (always
  returns `VisionError::Unsupported` immediately), so there's nothing to
  detach there.
- The spawned task must do the **entire** remaining sequence — including
  `write_generated_asset`, not just up through `release` — so a
  cancelled-but-still-running call either fully succeeds (file written,
  lock released) or fully fails, never "GPU cost paid, no file written."
- `GpuLockClient` and `MoldClient` are already `Clone` (confirmed by
  reading `gpu_lock_client.rs`/`mold_client.rs` directly) — no new
  plumbing needed to move owned clones into the spawned task.
- Re-read every file cited by line number below before editing — accurate
  as of this plan's own research (2026-09-19) but the repo moves.

---

## Task 1: Detach `generate_image`'s GPU-lock sequence into a spawned task

**Files:**
- Modify: `/home/julian/Projects/Rust/aivyx-vision/crates/aivyx-vision-mold/Cargo.toml`
- Modify: `/home/julian/Projects/Rust/aivyx-vision/crates/aivyx-vision-mold/src/provider.rs`
- Modify: `/home/julian/Projects/Rust/aivyx-vision/crates/aivyx-vision-mold/README.md`

**Interfaces:** none new — `MoldProvider::generate_image`'s public
signature (`async fn generate_image(&self, req: ImageRequest) ->
Result<GeneratedAsset, VisionError>`, from the `GenerationProvider`
trait) is unchanged; only its internal implementation changes.

- [ ] **Step 1: Write the failing test**

Add to `provider.rs`'s existing `#[cfg(test)] mod tests` block. No new
import needed: the test module already has `use super::*;`, and the
parent module (`provider.rs`, line 9) already has `use std::time::{Duration,
SystemTime};` — private `use` bindings in a parent module are visible to
its descendant modules in Rust, so `Duration` is already reachable here
through that glob import.

```rust
#[tokio::test]
async fn generate_image_releases_the_lease_even_when_the_caller_cancels_mid_generation() {
    let broker = MockServer::start().await;
    let mold = MockServer::start().await;
    let output_dir = tempfile::tempdir().unwrap();

    Mock::given(method("POST"))
        .and(path("/gpu-lock/acquire"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"lease_id": "l-cancel"})),
        )
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
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(150))
                .set_body_bytes(vec![9, 9, 9])
                .insert_header("content-type", "image/png"),
        )
        .mount(&mold)
        .await;

    let provider = test_provider(&broker, &mold, output_dir.path().to_path_buf()).await;

    // Simulates a real caller losing interest mid-generation (a timeout
    // wrapper, tokio::select!, a shutdown signal, ...): spawn the call on
    // its own task, then abort *that outer task* -- not anything inside
    // generate_image itself -- while mold's response is still delayed.
    let outer = tokio::spawn(async move { provider.generate_image(sample_image_request()).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    outer.abort();

    // Give generate_image's own internally-spawned task (independent of
    // the outer task we just aborted) time to run past mold's artificial
    // delay and release the lock for real.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // The release mock's `.expect(1)` above is checked here, explicitly --
    // this is the whole point of the test: release must still happen even
    // though nothing is left awaiting generate_image's own result.
    broker.verify().await;
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold generate_image_releases_the_lease_even_when_the_caller_cancels -- --nocapture
```

Expected: the test fails (the release mock's `.expect(1)` is never
satisfied — with today's implementation, aborting the outer task also
tears down `generate_image`'s own in-flight `await` chain, since there is
no independent spawned task yet, so `gpu_lock.release()` is never
reached).

- [ ] **Step 3: Promote `tokio` to a real dependency**

In `Cargo.toml`, move `tokio` out of `[dev-dependencies]` into
`[dependencies]` with just the `rt` feature (this library doesn't choose
a runtime flavor — that's the consuming binary's decision; `rt` alone is
enough for `tokio::spawn` to compile), and drop the now-redundant `rt`
from `[dev-dependencies]` (Cargo unifies features across
`[dependencies]`/`[dev-dependencies]` for the same package within test
builds, so `macros` there still pulls in `rt` too):

```toml
[dependencies]
aivyx-vision-core = { path = "../aivyx-vision-core" }
async-trait = "0.1.89"
base64 = "0.22"
reqwest = { version = "0.13.4", default-features = false, features = ["json", "rustls"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2.0.18"
tokio = { version = "1.52.3", features = ["rt"] }
tracing = "0.1.44"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tempfile = "3.27.0"
tokio = { version = "1.52.3", features = ["macros"] }
wiremock = "0.6.5"
```

- [ ] **Step 4: Implement the spawn-detach**

Replace `generate_image`'s current body in `provider.rs`:

```rust
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
        let lease = self.gpu_lock.acquire().await.map_err(map_gpu_lock_error)?;

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
```

with:

```rust
    /// Detached from the caller's own future on purpose -- see this
    /// module's top-level doc comment (or the design rationale in
    /// `docs/superpowers/plans/2026-09-19-gpu-lock-cancellation-safety.md`
    /// if that's been pruned) for why a `Drop`-based release guard would
    /// have been the *wrong* fix: releasing the lock the instant the
    /// caller's future is dropped could free it before `mold serve` has
    /// actually finished the in-flight generation, letting a second
    /// caller start a second GPU job while the first is still running.
    /// Spawning the whole `acquire -> generate -> release -> write`
    /// sequence and only `.await`ing the `JoinHandle` here means a
    /// cancelled caller abandons *waiting on the result*, never the
    /// operation itself -- the lock is only ever released after
    /// generation has genuinely finished, cancelled or not.
    async fn generate_image(&self, req: ImageRequest) -> Result<GeneratedAsset, VisionError> {
        let gpu_lock = self.gpu_lock.clone();
        let mold = self.mold.clone();
        let output_dir = self.output_dir.clone();

        let handle = tokio::spawn(async move {
            let lease = gpu_lock.acquire().await.map_err(map_gpu_lock_error)?;

            let mold_req = build_mold_request(&req);
            let result = match mold_req {
                Ok(mold_req) => mold.generate(mold_req).await.map_err(map_mold_error),
                Err(e) => Err(e),
            };

            if let Err(e) = gpu_lock.release(&lease).await {
                tracing::warn!(
                    error = %e,
                    "gpu-lock release failed after generation; aivyx-broker's reap_expired will eventually reclaim it"
                );
            }

            let response = result?;
            write_generated_asset(&output_dir, response)
        });

        handle
            .await
            .map_err(|e| VisionError::BackendUnreachable(format!("generation task panicked: {e}")))?
    }
```

- [ ] **Step 5: Run to verify it passes**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test -p aivyx-vision-mold -- --nocapture
```

Expected: the new test passes, plus all pre-existing `aivyx-vision-mold`
tests still pass (record the count — should be 20: the 19 already there
plus this one new test).

- [ ] **Step 6: Update `README.md`**

In `crates/aivyx-vision-mold/README.md`'s `## Honest tradeoffs` section
(currently 3 bullets, lines 53-63), add a new bullet documenting the
guarantee this task adds — callers wrapping `generate_image` in their own
timeout/cancellation logic should know this:

```markdown
- **Cancelling the caller's own future doesn't cancel the underlying GPU
  work or skip releasing the lock.** `generate_image` runs its
  `acquire -> generate -> release -> write` sequence on an internally
  spawned task; if the calling future is dropped (a timeout, a
  `tokio::select!`, a shutdown signal), that task keeps running to real
  completion regardless -- the lock is only ever released after
  generation genuinely finishes, never early. A cancelled call still
  costs real GPU time server-side; the caller just never sees the result.
```

- [ ] **Step 7: Run full workspace verification**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo build --workspace
cargo test --workspace 2>&1 | tail -30
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
cd /home/julian/Projects/Rust/aivyx-vision
git add crates/aivyx-vision-mold/Cargo.toml crates/aivyx-vision-mold/src/provider.rs crates/aivyx-vision-mold/README.md
git commit -m "fix: detach generate_image's GPU-lock sequence so caller cancellation can't leak or double-use the lease

Closes the gap the engine's own final whole-branch review found and
deliberately deferred: if the caller's future was dropped mid-generation
(a timeout wrapper, tokio::select!, a shutdown signal), the GPU lock was
never released -- only aivyx-broker's reap_expired safety valve (up to
900s) eventually reclaimed it.

Deliberately not a Drop-based release guard: releasing on Drop would free
the lock the instant the caller's future is dropped, which could be
before mold serve has actually finished the in-flight generation --
letting a second caller start a second GPU job while the first is still
running server-side, the exact double-GPU-usage scenario this lock
exists to prevent. Instead, the whole acquire/generate/release/write
sequence now runs on an internally spawned task; generate_image itself
just awaits the JoinHandle, so a cancelled caller abandons waiting on the
result, never the operation itself -- the lock is only released after
generation has genuinely finished, cancelled or not. Promotes tokio from
a dev-only to a real dependency (rt feature only; the runtime flavor
stays the consuming binary's choice, both of which already run inside a
Tokio runtime)."
```

---

## Final verification

- [ ] Run the complete workspace test suite once, not per-task (this is a
  single-task plan, but this step still matters as the record of the
  final, merged state):

```bash
cd /home/julian/Projects/Rust/aivyx-vision
cargo test --workspace 2>&1 | tail -30
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

- [ ] This plan does not decide whether to push a branch / open a PR —
  follow `superpowers:finishing-a-development-branch` once the task is
  reviewed and a final whole-branch review has passed, same as every
  other plan executed this cycle.

## Explicitly out of scope for this plan

- `generate_3d` / Pass B — untouched, unaffected by this change.
- Any change to `aivyx-broker`'s own `reap_expired`/`max_hold` mechanism
  — it remains the real backstop for a genuinely crashed process (as
  opposed to a merely-cancelled future within a still-running process),
  which this plan does not and cannot address.
- Making `MOLD_READ_TIMEOUT` configurable, or adding retry logic — both
  already-documented, separate "Honest tradeoffs" items this plan doesn't
  touch.
