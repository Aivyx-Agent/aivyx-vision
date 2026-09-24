# aivyx-vision

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)

LLM-prompted SVG generation, sanitized before return — the first shipped
slice of the Aivyx-Vision toolset (image and 3D model generation follow
in later milestones; see the design doc below).

This repo is now a Cargo workspace — `crates/aivyx-vision-svg` (this
crate), `crates/aivyx-vision-core`, and `crates/aivyx-vision-mold` are
its three members.

`generate_svg(completer, user_prompt)` prompts a caller-supplied
`TextCompleter` — a minimal, product-agnostic seam this crate defines
itself, not tied to any particular LLM provider — for SVG markup matching
`user_prompt`, extracts the `<svg>...</svg>` block from the response
(tolerating a markdown code fence around the block, and prose that
doesn't itself mention `<svg>` before the real block — a response that
mentions `<svg` in prose ahead of the actual fenced block will fail
sanitization with an XML-parsing error rather than being correctly
extracted, a known, accepted limitation, not silently wrong), and runs it
through [`svg-hush`](https://docs.rs/svg-hush)'s allowlist-based sanitizer
before returning the clean SVG source. Scripting and event-handler
attributes are stripped outright; cross-origin resource references are
neutralized by being rewritten to same-origin absolute paths, not
removed, so no cross-origin fetch survives, but the reference itself is
still present in the output — this crate never returns unsanitized model
output.

The returned string is always a full XML document — it begins with an
`<?xml version="1.0" encoding="utf-8"?>` processing instruction and is
pretty-printed with indentation, not a bare `<svg>...</svg>` fragment.
`svg-hush` always emits this shape and it can't be configured off, so a
caller splicing this directly into an existing HTML document should
strip the leading declaration first.

Each consuming product implements `TextCompleter` with a few lines
delegating to its own real, already-configured LLM provider — this crate
has no dependency on `aivyx-pa` or `aivyx-coder` and never talks to a
model itself.

See `aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
for the full design and the two other planned milestones (image and 3D
model generation).

## Milestone 2 (Pass A): image generation

`crates/aivyx-vision-mold` implements `GenerationProvider::generate_image`
against an operator-run [`mold serve`](https://github.com/utensils/mold)
instance, coordinated with local LLM inference sharing the same GPU via
[`aivyx-broker`](https://github.com/Aivyx-Agent/aivyx-broker)'s GPU lock.
See `crates/aivyx-vision-mold/README.md` for configuration and status —
`generate_3d` is not yet implemented (Pass B, a separate future build
pass covering mold's async 3D job-lifecycle API).
