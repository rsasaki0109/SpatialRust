//! Minimal dependency-free JSON value and streaming encoder/decoder.
//!
//! This is intentionally a small, well-tested JSON subset sufficient for the
//! `tiles3d` feature: objects, arrays, strings, numbers, booleans, and null.
//! Numbers are stored losslessly as `f64` text, and the encoder emits
//! deterministic output (no whitespace between tokens).

use crate::{InterchangeError, InterchangeResult};

/// A minimal JSON value used by the tileset and `pnts` codecs.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// JSON `null`.
    Null,
    /// JSON `true`/`false`.
    Bool(bool),
    /// JSON number, preserved as text for round-trip fidelity.
    Number(String),
    /// JSON string.
    String(String),
    /// JSON array.
    Array(Vec<Json>),
    /// JSON object with deterministic insertion order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Looks up a top-level object member.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(name, _)| name == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Returns the object member as a `&str` when present.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the object member as an `f64` when present.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => value.parse().ok(),
            _ => None,
        }
    }

    /// Returns the object member as a `u64` when present.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(value) => value.parse().ok(),
            _ => None,
        }
    }

    /// Returns the object member as a slice when present.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Builds a JSON object from key/value pairs.
    pub fn object(members: Vec<(&str, Json)>) -> Json {
        Json::Object(members.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
    }
}

/// Parses a JSON document into a [`Json`] value.
pub fn parse_json(input: &str) -> InterchangeResult<Json> {
    let mut parser = Parser { bytes: input.as_bytes(), offset: 0 };
    let value = parser.value()?;
    parser.whitespace();
    if parser.offset != parser.bytes.len() {
        return Err(json_error("trailing characters after JSON value"));
    }
    Ok(value)
}

/// Serializes a [`Json`] value deterministically (no whitespace).
pub fn serialize_json(value: &Json) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

struct Parser<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Parser<'a> {
    fn whitespace(&mut self) {
        while let Some(byte) = self.bytes.get(self.offset) {
            if matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
                self.offset += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.offset).copied()
    }

    fn value(&mut self) -> InterchangeResult<Json> {
        self.whitespace();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(byte) if byte == b'-' || byte.is_ascii_digit() => self.number(),
            _ => Err(json_error(&format!("unexpected JSON token at {}", self.offset))),
        }
    }

    fn object(&mut self) -> InterchangeResult<Json> {
        self.offset += 1; // consume '{'
        self.whitespace();
        let mut members = Vec::new();
        if self.peek() == Some(b'}') {
            self.offset += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.whitespace();
            if self.peek() != Some(b'"') {
                return Err(json_error("expected string key in object"));
            }
            let key = self.string()?;
            self.whitespace();
            if self.peek() != Some(b':') {
                return Err(json_error("expected ':' in object"));
            }
            self.offset += 1;
            let value = self.value()?;
            members.push((key, value));
            self.whitespace();
            match self.peek() {
                Some(b',') => self.offset += 1,
                Some(b'}') => {
                    self.offset += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(json_error("expected ',' or '}' in object")),
            }
        }
    }

    fn array(&mut self) -> InterchangeResult<Json> {
        self.offset += 1; // consume '['
        self.whitespace();
        let mut values = Vec::new();
        if self.peek() == Some(b']') {
            self.offset += 1;
            return Ok(Json::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.whitespace();
            match self.peek() {
                Some(b',') => self.offset += 1,
                Some(b']') => {
                    self.offset += 1;
                    return Ok(Json::Array(values));
                }
                _ => return Err(json_error("expected ',' or ']' in array")),
            }
        }
    }

    fn string(&mut self) -> InterchangeResult<String> {
        self.offset += 1; // consume '"'
        let mut out = String::new();
        loop {
            let byte =
                *self.bytes.get(self.offset).ok_or_else(|| json_error("unterminated string"))?;
            self.offset += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = *self
                        .bytes
                        .get(self.offset)
                        .ok_or_else(|| json_error("unterminated escape"))?;
                    self.offset += 1;
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.hex4()?;
                            let ch = char::from_u32(u32::from(code))
                                .ok_or_else(|| json_error("invalid unicode escape"))?;
                            out.push(ch);
                        }
                        _ => return Err(json_error("invalid escape sequence")),
                    }
                }
                byte if byte < 0x20 => {
                    return Err(json_error("unescaped control character in string"));
                }
                _ => {
                    // Single-byte path; multi-byte UTF-8 is copied verbatim below.
                    let start = self.offset - 1;
                    let mut len = 1;
                    while let Some(next) = self.bytes.get(start + len) {
                        if *next >= 0x20 && *next != b'"' && *next != b'\\' {
                            len += 1;
                        } else {
                            break;
                        }
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..start + len])
                            .map_err(|_| json_error("invalid UTF-8 in string"))?,
                    );
                    self.offset = start + len;
                }
            }
        }
    }

    fn hex4(&mut self) -> InterchangeResult<u16> {
        let mut value = 0u16;
        for _ in 0..4 {
            let byte = *self
                .bytes
                .get(self.offset)
                .ok_or_else(|| json_error("unterminated unicode escape"))?;
            self.offset += 1;
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a') + 10,
                b'A'..=b'F' => u16::from(byte - b'A') + 10,
                _ => return Err(json_error("invalid hex digit in unicode escape")),
            };
            value = (value << 4) | digit;
        }
        Ok(value)
    }

    fn number(&mut self) -> InterchangeResult<Json> {
        let start = self.offset;
        if self.peek() == Some(b'-') {
            self.offset += 1;
        }
        let mut digits = 0;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_digit() {
                digits += 1;
                self.offset += 1;
            } else {
                break;
            }
        }
        if digits == 0 {
            return Err(json_error("invalid number"));
        }
        if self.peek() == Some(b'.') {
            self.offset += 1;
            let mut fraction = 0;
            while let Some(byte) = self.peek() {
                if byte.is_ascii_digit() {
                    fraction += 1;
                    self.offset += 1;
                } else {
                    break;
                }
            }
            if fraction == 0 {
                return Err(json_error("invalid number fraction"));
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.offset += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.offset += 1;
            }
            let mut exponent = 0;
            while let Some(byte) = self.peek() {
                if byte.is_ascii_digit() {
                    exponent += 1;
                    self.offset += 1;
                } else {
                    break;
                }
            }
            if exponent == 0 {
                return Err(json_error("invalid number exponent"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.offset])
            .map_err(|_| json_error("invalid number bytes"))?
            .to_owned();
        text.parse::<f64>().map_err(|_| json_error("number out of range"))?;
        Ok(Json::Number(text))
    }

    fn literal(&mut self, expected: &str, value: Json) -> InterchangeResult<Json> {
        let end = self.offset + expected.len();
        let window = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| json_error("unexpected end of input"))?;
        if window != expected.as_bytes() {
            return Err(json_error(&format!("expected literal {expected} at {}", self.offset)));
        }
        self.offset = end;
        Ok(value)
    }
}

fn write_value(value: &Json, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(text) => out.push_str(text),
        Json::String(text) => write_string(text, out),
        Json::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(value, out);
            }
            out.push(']');
        }
        Json::Object(members) => {
            out.push('{');
            for (index, (key, value)) in members.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(value, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn json_error(message: &str) -> InterchangeError {
    InterchangeError::InvalidConfiguration(format!("JSON parse error: {message}"))
}

#[cfg(test)]
mod tests {
    use super::{parse_json, serialize_json, Json};

    #[test]
    fn round_trips_nested_document() {
        let document = r#"{"asset":{"version":"1.1"},"geometricError":500.5,"root":{"refine":"ADD","content":{"uri":"0.pnts"},"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]}}}"#;
        let parsed = parse_json(document).unwrap();
        let serialized = serialize_json(&parsed);
        assert_eq!(parse_json(&serialized).unwrap(), parsed);
    }

    #[test]
    fn preserves_number_text() {
        let parsed = parse_json(r#"{"a":1.250e0,"b":0.5}"#).unwrap();
        let value = parsed.get("a").unwrap();
        assert_eq!(value.as_f64(), Some(1.25));
        assert_eq!(value, &Json::Number("1.250e0".to_owned()));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(parse_json(r#"{"a":1} extra"#).is_err());
        assert!(parse_json(r#""unterminated"#).is_err());
        assert!(parse_json(r#"[1,]"#).is_err());
    }

    #[test]
    fn handles_escapes() {
        let parsed = parse_json(r#""line1\nline2\ttab""#).unwrap();
        assert_eq!(parsed, Json::String("line1\nline2\ttab".to_owned()));
        let backslash = parse_json(r#""a\\b""#).unwrap();
        assert_eq!(backslash, Json::String("a\\b".to_owned()));
        let quotes = parse_json(r#""say \"hi\"""#).unwrap();
        assert_eq!(quotes, Json::String("say \"hi\"".to_owned()));
    }

    #[test]
    fn serialize_is_whitespace_free() {
        let value = Json::object(vec![("b", Json::Number("1".to_owned())), ("a", Json::Null)]);
        assert_eq!(serialize_json(&value), r#"{"b":1,"a":null}"#);
    }
}
