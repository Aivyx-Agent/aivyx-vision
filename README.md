# aivyx-vision

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)

Local-first generation as agent tool calls, shared by
[Aivyx PA](https://github.com/Aivyx-Agent/aivyx-pa) and
[aivyx-coder](https://github.com/Aivyx-Agent/aivyx-coder). A Cargo
workspace with three crates:

| Crate | What it does | Status |
|---|---|---|
| `aivyx-vision-svg` | Asks the caller's own model for an SVG and sanitizes it | Shipped |
| `aivyx-vision-core` | The `GenerationProvider` trait, request and result types, `VisionError`; no I/O | Shipped |
| `aivyx-vision-mold` | Image generation through an operator-run [`mold serve`](https://github.com/utensils/mold), sharing the GPU through [`aivyx-broker`](https://github.com/Aivyx-Agent/aivyx-broker)'s GPU lock | Images shipped, not yet run against a real `mold serve`; 3D not built |

## SVG generation

`generate_svg(completer, user_prompt)` prompts a caller-supplied
`TextCompleter` (a one-method seam this crate defines, so it depends on
neither product) for SVG markup, picks the drawing out of the reply, and
sanitizes it before returning it. This crate never returns unsanitized
model output.

- **Finding the drawing:** the contents of a ```` ```svg ```` fence are
  tried first, then the reply's `<svg>…</svg>` blocks in order. The first
  one that sanitizes is returned, so prose that mentions `<svg>` or a
  second drawing doesn't break the call.
- **Sanitizing:** [`svg-hush`](https://docs.rs/svg-hush)'s allowlist
  removes scripting, event handlers and non-SVG elements. A second pass
  then removes all CSS (`<style>` and `style` attributes), because
  `svg-hush` doesn't filter CSS fetch syntax such as `image-set()`. It
  also removes every reference that doesn't point inside the document;
  only a `#fragment` or an inline PNG/JPEG/GIF `data:` URL is kept.
- **Limits:** input over 256 KiB or nested deeper than 64 elements is
  rejected.
- **Output:** a full XML document starting with
  `<?xml version="1.0" encoding="utf-8"?>`, not indented. Strip the
  declaration before splicing it into HTML.

## Image generation

`MoldProvider` implements `GenerationProvider::generate_image` against
`mold serve`'s `POST /api/generate`. It takes a GPU lease from the broker
first and always releases it afterwards. It talks to both services
directly, never through a proxy. Generated files are written owner-only.
`generate_3d` returns `VisionError::Unsupported` for now. See
[`crates/aivyx-vision-mold/README.md`](crates/aivyx-vision-mold/README.md)
for configuration and trade-offs.

## Licence

Source-available under BUSL-1.1: free for personal and other
non-commercial use. See [`LICENSE`](LICENSE) and
[`COMMERCIAL.md`](COMMERCIAL.md).
