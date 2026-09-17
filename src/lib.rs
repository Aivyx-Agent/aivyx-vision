//! LLM-prompted SVG generation, sanitized before return. Part of the
//! Aivyx-Vision toolset (see `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md`) -- the vector/graphic-design
//! milestone, which deliberately needs no image/3D generation engine at
//! all: it prompts the *caller's own already-configured* text-completion
//! backend (see `TextCompleter`, added in a later commit) and sanitizes
//! whatever SVG markup comes back.
