use std::path::Path;
use std::time::Duration;

use serde_json::Value;

/// Context window requested from Ollama on every call.
///
/// Not GPU-dependent: the request size is bounded by the image budget below
/// (image tokens + prompt + `num_predict`), so 4096 fits every call on any GPU.
/// Keeping a single constant also avoids Ollama reloading the model because
/// the requested context changed between calls.
pub const NUM_CTX: u32 = 4096;

/// Target number of visual tokens per image sent to the vision model.
/// Qwen2.5-VL maps each 28x28 pixel block to one token, so 1000 tokens is
/// roughly 0.78 MP (e.g. 1001x750 for a 4:3 photo). Benchmarked against a
/// ~3100-token variant: same media_type on 10/10 photos, ~2.6x faster per call.
pub const TARGET_IMAGE_TOKENS: u32 = 1000;

/// Pixels per visual token side for Qwen2.5-VL (14px patches merged 2x2).
const PX_PER_TOKEN: u32 = 28;
/// Lower bound so tiny budgets never collapse the image to nothing.
const MIN_IMAGE_TOKENS: u32 = 64;
/// Chat template / special tokens added by Ollama around the prompt.
const TEMPLATE_TOKENS: u32 = 64;
/// Slack for tokenizer/estimation differences (observed up to ~250 tokens).
const CTX_SAFETY_MARGIN: u32 = 256;

pub struct OllamaResponse {
    pub ok: bool,
    pub raw: String,
    #[allow(dead_code)]
    pub total_duration_s: f64,
    /// HTTP status when Ollama answered with an error status (4xx/5xx).
    pub http_status: Option<u16>,
}

impl OllamaResponse {
    pub fn error(msg: String) -> Self {
        Self {
            ok: false,
            raw: msg,
            total_duration_s: 0.0,
            http_status: None,
        }
    }

    /// True when retrying the same request cannot succeed (HTTP 4xx, e.g.
    /// "request exceeds the available context size" or model not found).
    pub fn is_client_error(&self) -> bool {
        matches!(self.http_status, Some(400..=499))
    }
}

/// Estimated visual tokens for an image of `w` x `h` pixels.
pub fn estimate_image_tokens(w: u32, h: u32) -> u32 {
    w.div_ceil(PX_PER_TOKEN) * h.div_ceil(PX_PER_TOKEN)
}

/// Visual-token budget for one call: the target, capped so that
/// image + prompt + generated tokens always fit in `NUM_CTX`.
pub fn image_token_budget(prompt: &str, num_predict: u32) -> u32 {
    // ~3 chars/token is conservative for English prompts and OCR text.
    let prompt_tokens = (prompt.len() as u32).div_ceil(3) + TEMPLATE_TOKENS;
    let available = NUM_CTX.saturating_sub(num_predict + prompt_tokens + CTX_SAFETY_MARGIN);
    TARGET_IMAGE_TOKENS.min(available).max(MIN_IMAGE_TOKENS)
}

/// Largest dimensions with the same aspect ratio whose estimated token
/// count is <= `max_tokens`. Returns the input unchanged if it already fits.
pub fn fit_dimensions(w: u32, h: u32, max_tokens: u32) -> (u32, u32) {
    if w == 0 || h == 0 || estimate_image_tokens(w, h) <= max_tokens {
        return (w, h);
    }
    let area = (max_tokens as f64) * (PX_PER_TOKEN as f64).powi(2);
    let scale = (area / (w as f64 * h as f64)).sqrt();
    let mut nw = ((w as f64 * scale).floor() as u32).max(1);
    let mut nh = ((h as f64 * scale).floor() as u32).max(1);
    while estimate_image_tokens(nw, nh) > max_tokens && nw > 1 && nh > 1 {
        nw = ((nw as f64 * 0.98).floor() as u32).max(1);
        nh = ((nh as f64 * 0.98).floor() as u32).max(1);
    }
    (nw, nh)
}

/// Bytes to send to the vision model for `path`.
///
/// Images above `max_tokens` (or with a non-default EXIF orientation) are
/// decoded, rotated upright, downscaled to fit and re-encoded as JPEG in
/// memory; nothing is written to disk. Anything that can't be decoded is
/// sent as-is (previous behavior).
pub fn prepare_image_bytes(path: &Path, max_tokens: u32) -> std::io::Result<Vec<u8>> {
    let original = std::fs::read(path)?;

    let dims = image::ImageReader::new(std::io::Cursor::new(&original))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.into_dimensions().ok());
    let Some((w, h)) = dims else {
        return Ok(original);
    };

    let orientation = crate::exif::extract_orientation(path);
    let (tw, th) = fit_dimensions(w, h, max_tokens);
    if (tw, th) == (w, h) && orientation <= 1 {
        return Ok(original);
    }

    let img = match image::load_from_memory(&original) {
        Ok(img) => img,
        Err(e) => {
            log::warn!("LLM image prep: cannot decode {}: {e}", path.display());
            return Ok(original);
        }
    };
    let img = crate::exif::apply_orientation(img, orientation);
    // Orientation 5-8 swaps axes; recompute the fit on the upright image.
    let (w, h) = (img.width(), img.height());
    let (tw, th) = fit_dimensions(w, h, max_tokens);
    let img = if (tw, th) != (w, h) {
        img.resize_exact(tw, th, image::imageops::FilterType::CatmullRom)
    } else {
        img
    };

    let rgb = img.to_rgb8();
    let mut buf = std::io::Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
    if let Err(e) = rgb.write_with_encoder(encoder) {
        log::warn!("LLM image prep: cannot encode {}: {e}", path.display());
        return Ok(original);
    }
    Ok(buf.into_inner())
}

pub fn ollama_generate(
    model: &str,
    prompt: &str,
    image_path: Option<&Path>,
    num_predict: u32,
    timeout_secs: u64,
    json_mode: bool,
) -> OllamaResponse {
    let mut payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "keep_alive": "5m",
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX,
        }
    });

    if json_mode {
        payload["format"] = serde_json::json!("json");
    }

    if let Some(img_path) = image_path {
        let budget = image_token_budget(prompt, num_predict);
        match prepare_image_bytes(img_path, budget) {
            Ok(bytes) => {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                payload["images"] = serde_json::json!([b64]);
            }
            Err(e) => {
                return OllamaResponse::error(format!("image_read_error: {e}"));
            }
        }
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_recv_body(Some(Duration::from_secs(timeout_secs)))
        .timeout_send_body(Some(Duration::from_secs(30)))
        // Read error bodies too, so Ollama's message ends up in the logs.
        .http_status_as_error(false)
        .build()
        .into();

    let result = agent
        .post(&format!("{}/api/generate", super::OLLAMA_BASE))
        .send_json(&payload);

    match result {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let body: String = match resp.body_mut().read_to_string() {
                Ok(s) => s,
                Err(e) => {
                    return OllamaResponse::error(format!("read_error: {e}"));
                }
            };
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            if status >= 400 {
                let msg = parsed
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or(body.as_str())
                    .to_string();
                log::warn!("Ollama /api/generate HTTP {status}: {msg}");
                return OllamaResponse {
                    ok: false,
                    raw: format!("http_error: {status}: {msg}"),
                    total_duration_s: 0.0,
                    http_status: Some(status),
                };
            }
            let raw = parsed
                .get("response")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let dur = parsed
                .get("total_duration")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                / 1_000_000_000.0;
            OllamaResponse {
                ok: true,
                raw,
                total_duration_s: dur,
                http_status: None,
            }
        }
        Err(e) => OllamaResponse::error(format!("http_error: {e}")),
    }
}

/// Extract the first balanced JSON object from free-form text.
pub fn extract_first_json_object(text: &str) -> Option<Value> {
    if text.is_empty() {
        return None;
    }
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            if v.is_object() {
                return Some(v);
            }
        }
    }
    for (start, _) in text.char_indices().filter(|(_, c)| *c == '{') {
        let mut depth = 0i32;
        for (i, ch) in text[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let candidate = &text[start..start + i + 1];
                        if let Ok(v) = serde_json::from_str::<Value>(candidate) {
                            if v.is_object() {
                                return Some(v);
                            }
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

pub fn parse_json_response(raw: &str) -> Option<Value> {
    if raw.is_empty() {
        return None;
    }
    extract_first_json_object(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_first_json_object_valid() {
        let input = r#"{"media_type":"anime","confidence":0.9}"#;
        let result = extract_first_json_object(input).unwrap();
        assert_eq!(result["media_type"], "anime");
    }

    #[test]
    fn extract_first_json_object_garbage_prefix() {
        let input = r#"Here is my answer: {"media_type":"photo","confidence":0.5} done"#;
        let result = extract_first_json_object(input).unwrap();
        assert_eq!(result["media_type"], "photo");
    }

    #[test]
    fn extract_first_json_object_nested_braces() {
        let input = r#"{"a":{"b":1},"c":2}"#;
        let result = extract_first_json_object(input).unwrap();
        assert_eq!(result["c"], 2);
    }

    #[test]
    fn extract_first_json_object_empty() {
        assert!(extract_first_json_object("").is_none());
    }

    #[test]
    fn extract_first_json_object_no_json() {
        assert!(extract_first_json_object("just some text").is_none());
    }

    #[test]
    fn estimate_image_tokens_rounds_up() {
        assert_eq!(estimate_image_tokens(28, 28), 1);
        assert_eq!(estimate_image_tokens(29, 28), 2);
        assert_eq!(estimate_image_tokens(1001, 750), 36 * 27);
    }

    #[test]
    fn fit_dimensions_keeps_small_images() {
        assert_eq!(fit_dimensions(800, 600, 1000), (800, 600));
        assert_eq!(fit_dimensions(0, 0, 1000), (0, 0));
    }

    #[test]
    fn fit_dimensions_downscales_large_photos_within_budget() {
        for (w, h) in [
            (4000, 3000),
            (3000, 4000),
            (3392, 2544),
            (4032, 1908),
            (1654, 2339),
        ] {
            let (nw, nh) = fit_dimensions(w, h, 1000);
            assert!(
                estimate_image_tokens(nw, nh) <= 1000,
                "{w}x{h} -> {nw}x{nh}"
            );
            // Aspect ratio preserved within 1%
            let r0 = w as f64 / h as f64;
            let r1 = nw as f64 / nh as f64;
            assert!((r0 - r1).abs() / r0 < 0.01, "{w}x{h} -> {nw}x{nh}");
            // Not shrunk far below the budget
            assert!(estimate_image_tokens(nw, nh) >= 900, "{w}x{h} -> {nw}x{nh}");
        }
    }

    #[test]
    fn image_token_budget_fits_every_call_in_context() {
        // (prompt length, num_predict) pairs used by classify.rs
        for (prompt_len, num_predict) in [(1100, 500), (1200, 600), (90, 2000), (900, 260)] {
            let prompt = "x".repeat(prompt_len);
            let budget = image_token_budget(&prompt, num_predict);
            assert!(budget <= TARGET_IMAGE_TOKENS);
            let total = budget + (prompt_len as u32).div_ceil(3) + TEMPLATE_TOKENS + num_predict;
            assert!(
                total <= NUM_CTX,
                "prompt={prompt_len} predict={num_predict}"
            );
        }
    }

    #[test]
    fn image_token_budget_is_target_for_primary_prompt() {
        let prompt = "x".repeat(1100);
        assert_eq!(image_token_budget(&prompt, 500), TARGET_IMAGE_TOKENS);
    }

    #[test]
    fn image_token_budget_never_below_minimum() {
        let prompt = "x".repeat(20_000);
        assert_eq!(image_token_budget(&prompt, 2000), MIN_IMAGE_TOKENS);
    }

    #[test]
    fn prepare_image_bytes_downscales_large_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.png");
        image::RgbImage::from_pixel(4000, 3000, image::Rgb([120, 60, 30]))
            .save(&path)
            .unwrap();
        let bytes = prepare_image_bytes(&path, 1000).unwrap();
        let out = image::load_from_memory(&bytes).unwrap();
        assert!(estimate_image_tokens(out.width(), out.height()) <= 1000);
        assert!(out.width() > out.height());
    }

    #[test]
    fn prepare_image_bytes_keeps_small_image_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.png");
        image::RgbImage::from_pixel(640, 480, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let bytes = prepare_image_bytes(&path, 1000).unwrap();
        assert_eq!(bytes, std::fs::read(&path).unwrap());
    }

    #[test]
    fn prepare_image_bytes_passes_through_undecodable_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not_an_image.jpg");
        std::fs::write(&path, b"definitely not an image").unwrap();
        let bytes = prepare_image_bytes(&path, 1000).unwrap();
        assert_eq!(bytes, b"definitely not an image");
    }

    #[test]
    fn prepare_image_bytes_missing_file_is_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(prepare_image_bytes(&dir.path().join("missing.jpg"), 1000).is_err());
    }

    #[test]
    fn client_error_detection() {
        let mut r = OllamaResponse::error("x".into());
        assert!(!r.is_client_error());
        r.http_status = Some(400);
        assert!(r.is_client_error());
        r.http_status = Some(500);
        assert!(!r.is_client_error());
    }
}
