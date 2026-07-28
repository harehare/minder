use serde::{Deserialize, Deserializer};

/// Deserializes an optional `timeout_secs`-style argument, accepting any
/// JSON number (not just an integer) and clamping it into a sane range --
/// LLM-generated tool calls occasionally send `30.0` or a negative number
/// for a duration, which a plain `Option<u64>` field rejects outright with
/// a cryptic `invalid type: floating point ..., expected u64` error instead
/// of just running the command. Missing/null still deserializes to `None`
/// via the field's own `#[serde(default)]`, since this is only called when
/// the key is present.
pub(crate) fn deserialize_timeout_secs<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = f64::deserialize(deserializer)?;
    Ok(Some(clamp_to_secs(raw)))
}

/// NaN/infinite/non-positive all fall back to the 1s floor rather than
/// erroring -- a timeout of "whatever's smallest" is a more useful recovery
/// than failing the whole tool call over a malformed duration. A huge value
/// saturates to `u64::MAX` (Rust's float-to-int cast is saturating), which
/// is effectively "no timeout" in practice.
fn clamp_to_secs(v: f64) -> u64 {
    if !v.is_finite() || v < 1.0 { 1 } else { v.round() as u64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Wrapper {
        #[serde(default, deserialize_with = "deserialize_timeout_secs")]
        timeout_secs: Option<u64>,
    }

    fn parse(json: serde_json::Value) -> Option<u64> {
        serde_json::from_value::<Wrapper>(json).unwrap().timeout_secs
    }

    #[test]
    fn missing_field_is_none() {
        assert_eq!(parse(serde_json::json!({})), None);
    }

    #[test]
    fn plain_integer_passes_through() {
        assert_eq!(parse(serde_json::json!({"timeout_secs": 30})), Some(30));
    }

    #[test]
    fn a_float_is_rounded_instead_of_rejected() {
        assert_eq!(parse(serde_json::json!({"timeout_secs": 30.4})), Some(30));
        assert_eq!(parse(serde_json::json!({"timeout_secs": 30.6})), Some(31));
    }

    #[test]
    fn negative_or_zero_clamps_to_the_1s_floor() {
        assert_eq!(parse(serde_json::json!({"timeout_secs": -5})), Some(1));
        assert_eq!(parse(serde_json::json!({"timeout_secs": 0})), Some(1));
    }

    #[test]
    fn a_huge_value_saturates_instead_of_panicking() {
        assert_eq!(parse(serde_json::json!({"timeout_secs": 1e30})), Some(u64::MAX));
    }
}
