# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this is

`aivyx-vision` is a shared, local-first generation toolset for
`aivyx-pa` and `aivyx-coder` — image, 3D model, and vector/graphic-design
output as agent tool calls. This repo currently holds one crate,
`aivyx-vision-svg` (the vector/graphic-design milestone); image and 3D
generation land in later milestones as sibling crates
(`aivyx-vision-core`, `aivyx-vision-mold`, `aivyx-vision-comfyui`), at
which point this repo becomes a Cargo workspace. See `README.md` and
`aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
for the full rationale — this file only covers what's specific to working
in this repo's code.

No consumer depends on this crate yet — `aivyx-pa`'s and `aivyx-coder`'s
own adoption of `vision.generate_svg`-shaped tools is separate, later
work (each product's own `docs/superpowers/plans/`).

## Build, test, lint

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Single crate, no workspace yet — no `-p` flag needed. Single test:
`cargo test <test_name>`.

## Architecture

Single file, `src/lib.rs`:

- `TextCompleter` — this crate's only LLM seam: one method,
  `async fn complete(&self, prompt: &str) -> Result<String,
  TextCompleterError>`. Deliberately not `aivyx-pa`'s `LlmProvider` or
  `aivyx-coder`'s `LlmBackend` — this crate must never depend on either
  product. A consuming product's adapter implements this trait by
  delegating to its own real provider.
- `extract_svg_markup` — pure function, no I/O. Finds the first `<svg`
  and last `</svg>` in a raw completion response and slices between them,
  tolerating a markdown code fence or surrounding prose without needing
  to parse the fence syntax explicitly.
- `sanitize_svg` — wraps the `svg-hush` crate's allowlist-based filter.
  **Load-bearing, not optional**: SVG is executable-ish content, and this
  is the only thing standing between a model's raw output and whatever
  re-renders the result. `generate_svg_strips_a_script_tag`'s test is
  mutation-verified (see its git history) — don't weaken or bypass this
  path without re-verifying that test still fails when it should.
- `generate_svg` — the one public entry point, composing the three
  pieces above.

## Where to look next

- `README.md` — quick orientation and the design-doc pointer.
- `aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
  — the full design: why this exists, the two other planned milestones,
  and the explicit list of what that spec does not decide.
