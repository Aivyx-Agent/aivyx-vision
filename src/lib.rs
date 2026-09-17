//! LLM-prompted SVG generation, sanitized before return. Part of the
//! Aivyx-Vision toolset (see `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md`) -- the vector/graphic-design
//! milestone, which deliberately needs no image/3D generation engine at
//! all: it prompts the *caller's own already-configured* text-completion
//! backend (see `TextCompleter`, added in a later commit) and sanitizes
//! whatever SVG markup comes back.

use async_trait::async_trait;

/// A minimal, product-agnostic "turn a prompt into text" seam. Each
/// product's own adapter (out of scope for this crate) implements this
/// by delegating to its real, already-configured LLM provider --
/// `aivyx-pa`'s `LlmProvider` or `aivyx-coder`'s `LlmBackend`. This crate
/// never talks to a model directly and has no opinion on which provider
/// backs it, no streaming, no cancellation -- just one prompt in, one
/// completion out.
#[async_trait]
pub trait TextCompleter: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String, TextCompleterError>;
}

/// An error from the caller-supplied [`TextCompleter`]. Opaque by
/// design -- this crate doesn't know or care whether the real backend is
/// Ollama, Anthropic, a local llama-server, or something else; the
/// message is whatever the adapter chose to surface.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("text completion failed: {0}")]
pub struct TextCompleterError(pub String);

/// Everything that can go wrong generating an SVG.
#[derive(Debug, thiserror::Error)]
pub enum SvgGenerationError {
    #[error(transparent)]
    Completion(#[from] TextCompleterError),
    #[error("no <svg>...</svg> markup found in the completion response")]
    NoSvgFound,
    #[error("SVG sanitization failed: {0}")]
    Sanitization(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A `TextCompleter` test double: returns a canned response (or
    /// error) exactly once, and records the prompt it was called with so
    /// tests can assert what was actually sent.
    struct FakeCompleter {
        response: Mutex<Option<Result<String, TextCompleterError>>>,
        captured_prompt: Mutex<Option<String>>,
    }

    impl FakeCompleter {
        fn returning(response: Result<String, TextCompleterError>) -> Self {
            Self {
                response: Mutex::new(Some(response)),
                captured_prompt: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl TextCompleter for FakeCompleter {
        async fn complete(&self, prompt: &str) -> Result<String, TextCompleterError> {
            *self.captured_prompt.lock().unwrap() = Some(prompt.to_string());
            self.response
                .lock()
                .unwrap()
                .take()
                .expect("FakeCompleter.complete called more than once")
        }
    }

    #[tokio::test]
    async fn text_completer_trait_object_is_usable_and_captures_its_prompt() {
        let fake = FakeCompleter::returning(Ok("hello".to_string()));
        let completer: &dyn TextCompleter = &fake;
        let result = completer.complete("a test prompt").await;
        assert_eq!(result.unwrap(), "hello");
        assert_eq!(
            fake.captured_prompt.lock().unwrap().as_deref(),
            Some("a test prompt")
        );
    }

    #[tokio::test]
    async fn text_completer_propagates_its_error() {
        let fake = FakeCompleter::returning(Err(TextCompleterError("backend down".into())));
        let completer: &dyn TextCompleter = &fake;
        let err = completer.complete("prompt").await.unwrap_err();
        assert_eq!(err.0, "backend down");
    }
}
