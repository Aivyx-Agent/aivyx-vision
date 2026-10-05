# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this is

`aivyx-vision` is a shared, local-first generation toolset for
`aivyx-pa` and `aivyx-coder` — image, 3D model, and vector/graphic-design
output as agent tool calls. This is now a Cargo workspace: `crates/aivyx-vision-svg`
(the vector/graphic-design milestone), `crates/aivyx-vision-core` (the
`GenerationProvider` trait + shared types, no I/O), and
`crates/aivyx-vision-mold` (mold-backed image generation — Milestone 2
Pass A shipped: `generate_image` via `mold serve`'s synchronous
`/api/generate`, coordinated through `aivyx-broker`'s GPU lock;
`generate_3d` returns `VisionError::Unsupported` until Pass B implements
mold's async `/api/mesh-workflows` job-lifecycle surface). See
`README.md` and
`aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
for the full rationale — this file only covers what's specific to working
in this repo's code.

Both `aivyx-pa` and `aivyx-coder` depend on all three crates
(`aivyx-vision-svg`, `aivyx-vision-core`, `aivyx-vision-mold`) — `aivyx-pa`
via its `crates/aivyx-vision` tool process, `aivyx-coder` via
`crates/aivyx-tools`/`crates/aivyx-mcp-server`/`crates/aivyx`. Each
product's own `vision.*`-shaped tool wiring (the tool definitions
themselves, config, prompt surface) still lives in that product's own
`docs/superpowers/plans/`, not here.

## Build, test, lint

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Cargo workspace, three members under `crates/` — use `-p <crate-name>` to
target one crate. Single test: `cargo test <test_name>`.

## Architecture

`aivyx-vision-svg`'s own architecture (single file,
`crates/aivyx-vision-svg/src/lib.rs`):

- `TextCompleter` — this crate's only LLM seam: one method,
  `async fn complete(&self, prompt: &str) -> Result<String,
  TextCompleterError>`. Deliberately not `aivyx-pa`'s `LlmProvider` or
  `aivyx-coder`'s `LlmBackend` — this crate must never depend on either
  product. A consuming product's adapter implements this trait by
  delegating to its own real provider.
- `extract_svg_markup` — pure function, no I/O. Finds the first `<svg`
  and last `</svg>` in a raw completion response and slices between them.
  This tolerates a markdown code fence around the block, and prose that
  doesn't itself mention `<svg>` before the real block; a response that
  mentions `<svg` in prose ahead of the actual fenced block will fail
  sanitization with an XML-parsing error rather than being correctly
  extracted — a known, accepted limitation, not silently wrong (it fails
  closed, never emits a truncated or wrong block).
- `sanitize_svg` — input limits (`MAX_SVG_INPUT_BYTES`, `MAX_SVG_DEPTH`;
  `svg-hush`'s pretty-printing is quadratic in depth), then the `svg-hush`
  crate's allowlist-based filter, then `restrict_to_in_document_content`,
  a post-filter over `svg-hush`'s output that drops all CSS and every
  non-`#fragment`/non-`data:` reference (`svg-hush` misses CSS
  `image-set()` and rewrites off-document URLs to same-origin paths
  instead of removing them — audit 2026-10-04 V1/V3). Known, accepted
  losses: CSS-styled SVGs render with default fills (`SYSTEM_PROMPT` asks
  for presentation attributes instead), any attribute containing `\` is
  dropped, and `svg-hush` strips `xml:space="preserve"`, so runs of spaces
  in `<text>` collapse (audit V5, not fixed).
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
