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

/// Pull the `<svg>...</svg>` block out of an arbitrary completion
/// response. Finds the first `<svg` and the last `</svg>` in the whole
/// response and slices between them (inclusive) -- this tolerates a
/// markdown code fence or explanatory prose around the block without
/// needing to parse or strip the fence syntax explicitly. Returns `None`
/// if no `<svg`/`</svg>` pair is present, or if the `</svg>` found is
/// before the `<svg` found (malformed / truncated response).
#[allow(dead_code)]
// Called by Task 4's generate_svg; until then, only test code uses it.
// Suppressed here so this commit doesn't leave the crate failing clippy.
fn extract_svg_markup(response: &str) -> Option<&str> {
    let start = response.find("<svg")?;
    let end = response.rfind("</svg>")? + "</svg>".len();
    if end <= start {
        return None;
    }
    Some(&response[start..end])
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

    #[test]
    fn extract_svg_markup_finds_a_bare_svg_block() {
        let response = "<svg xmlns=\"http://www.w3.org/2000/svg\"><circle r=\"5\"/></svg>";
        assert_eq!(extract_svg_markup(response), Some(response));
    }

    #[test]
    fn extract_svg_markup_strips_a_surrounding_markdown_fence() {
        let response = "Here you go:\n```svg\n<svg><rect/></svg>\n```\nEnjoy!";
        assert_eq!(extract_svg_markup(response), Some("<svg><rect/></svg>"));
    }

    #[test]
    fn extract_svg_markup_returns_none_when_no_svg_tag_present() {
        let response = "I can't generate images.";
        assert_eq!(extract_svg_markup(response), None);
    }

    #[test]
    fn extract_svg_markup_returns_none_for_an_unclosed_svg_tag() {
        let response = "<svg><circle r=\"5\"/>";
        assert_eq!(extract_svg_markup(response), None);
    }
}
