//! Streaming text decoding and multi-token stop sequence matching for LLM inference.
//!
//! Handles incremental UTF-8 decoding across multi-byte character boundaries and
//! exact prefix/suffix stop sequence interception without leaking partial stop patterns.

/// Incremental UTF-8 decoder that buffers incomplete trailing multi-byte sequences.
#[derive(Debug, Default, Clone)]
pub struct StreamingUtf8Decoder {
    buffer: Vec<u8>,
}

impl StreamingUtf8Decoder {
    /// Creates a new streaming UTF-8 decoder.
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Appends incoming byte chunk and returns the longest valid, complete UTF-8 string prefix.
    /// Any incomplete trailing byte sequence is retained in the internal buffer for subsequent calls.
    pub fn add_bytes(&mut self, chunk: &[u8]) -> String {
        self.buffer.extend_from_slice(chunk);

        // Find the boundary of valid UTF-8
        match std::str::from_utf8(&self.buffer) {
            Ok(valid_str) => {
                let out = valid_str.to_string();
                self.buffer.clear();
                out
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if let Some(_error_len) = e.error_len() {
                    // There is an actual invalid sequence in the middle; decode lossily
                    let out = String::from_utf8_lossy(&self.buffer).to_string();
                    self.buffer.clear();
                    out
                } else {
                    // Incomplete multi-byte code point at the end
                    let valid_bytes = self.buffer[..valid_up_to].to_vec();
                    let remaining = self.buffer[valid_up_to..].to_vec();
                    self.buffer = remaining;
                    // SAFETY: valid_bytes was verified valid by e.valid_up_to()
                    String::from_utf8(valid_bytes).unwrap_or_default()
                }
            }
        }
    }

    /// Flushes any remaining bytes, converting lossily to UTF-8.
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            String::new()
        } else {
            let out = String::from_utf8_lossy(&self.buffer).to_string();
            self.buffer.clear();
            out
        }
    }

    /// Resets the internal byte buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Matches arbitrary stop strings during token streaming without leaking partial prefixes.
#[derive(Debug, Clone)]
pub struct StopSequenceMatcher {
    stop_sequences: Vec<String>,
    buffer: String,
    stopped: bool,
}

impl StopSequenceMatcher {
    /// Creates a new stop sequence matcher with target stop strings.
    pub fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            stop_sequences,
            buffer: String::new(),
            stopped: false,
        }
    }

    /// Returns whether any stop sequence was fully encountered.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Steps with newly generated text and returns `(text_to_yield, should_stop)`.
    pub fn step(&mut self, text: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }

        self.buffer.push_str(text);

        // Check if any stop sequence is fully contained
        for stop in &self.stop_sequences {
            if stop.is_empty() {
                continue;
            }
            if let Some(pos) = self.buffer.find(stop) {
                let text_to_emit = self.buffer[..pos].to_string();
                self.buffer.clear();
                self.stopped = true;
                return (text_to_emit, true);
            }
        }

        // Check if any suffix of self.buffer matches a prefix of any stop sequence.
        // Byte counts can land mid-UTF-8-char, so only cut at char boundaries.
        let mut max_prefix_len = 0;
        for stop in &self.stop_sequences {
            if stop.is_empty() {
                continue;
            }
            for k in (1..=stop.len().min(self.buffer.len())).rev() {
                let cut = self.buffer.len() - k;
                if !self.buffer.is_char_boundary(cut) {
                    continue;
                }
                let suffix = &self.buffer[cut..];
                if stop.starts_with(suffix) {
                    if k > max_prefix_len {
                        max_prefix_len = k;
                    }
                    break;
                }
            }
        }

        if max_prefix_len > 0 {
            // Hold back the candidate prefix suffix from emission
            let emit_len = self.buffer.len() - max_prefix_len;
            let text_to_emit = self.buffer[..emit_len].to_string();
            self.buffer = self.buffer[emit_len..].to_string();
            (text_to_emit, false)
        } else {
            let text_to_emit = std::mem::take(&mut self.buffer);
            (text_to_emit, false)
        }
    }

    /// Flushes any held-back buffer text when generation finishes without hitting a stop sequence.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    /// Resets the matcher state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.stopped = false;
    }
}
