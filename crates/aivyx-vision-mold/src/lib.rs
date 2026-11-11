//! HTTP-client backend for Aivyx-Vision's `GenerationProvider` trait,
//! talking to an operator-run `mold serve` instance for image
//! generation, coordinated via `aivyx-broker`'s GPU lock so it doesn't
//! contend with local LLM inference sharing the same GPU. The Aivyx-Vision
//! v1 design (§5, §7: Milestone 2) has the full reasoning.
//!
//! 3D generation (`generate_3d`) uses the same synchronous `/api/generate`
//! with mold's Hunyuan3D model: one image in, a binary glTF mesh out. The
//! 2026-11-11 live run found that enough; mold's async
//! `/api/mesh-workflows` job surface isn't needed.

mod body_limits;
pub mod gpu_lock_client;
mod metadata;
pub mod mold_client;
mod provider;

pub use gpu_lock_client::{GpuLockClient, GpuLockClientError, LeaseId};
pub use mold_client::{
    GenerateImageRequest, GenerateImageResponse, MeshOptions, MoldClient, MoldClientError,
    MoldModel, UpscaleImageRequest, UpscaleImageResponse,
};
pub use provider::{MoldConfig, MoldProvider};
