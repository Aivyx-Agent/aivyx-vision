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
