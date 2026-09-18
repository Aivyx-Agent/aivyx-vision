# aivyx-vision

LLM-prompted SVG generation, sanitized before return — the first shipped
slice of the Aivyx-Vision toolset (image and 3D model generation follow
in later milestones; see the design doc below).

`generate_svg(completer, user_prompt)` prompts a caller-supplied
`TextCompleter` — a minimal, product-agnostic seam this crate defines
itself, not tied to any particular LLM provider — for SVG markup matching
`user_prompt`, extracts the `<svg>...</svg>` block from the response
(tolerating a markdown code fence or explanatory prose around it), and
runs it through [`svg-hush`](https://docs.rs/svg-hush)'s allowlist-based
sanitizer before returning the clean SVG source. Scripting, event-handler
attributes, and cross-origin resource references are stripped
unconditionally — this crate never returns unsanitized model output.

Each consuming product implements `TextCompleter` with a few lines
delegating to its own real, already-configured LLM provider — this crate
has no dependency on `aivyx-pa` or `aivyx-coder` and never talks to a
model itself.

See `aivyx-ecosystem/docs/superpowers/specs/2026-09-18-aivyx-vision-v1-design.md`
for the full design and the two other planned milestones (image and 3D
model generation).
