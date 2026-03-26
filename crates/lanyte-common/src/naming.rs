use thiserror::Error;

const MAX_SEGMENT_LEN: usize = 63;
const MAX_NAME_LEN: usize = 253;

/// Validates a slug segment (single component).
pub fn validate_slug(s: &str) -> Result<(), NamingError> {
    if s.is_empty() {
        return Err(NamingError::Empty);
    }
    validate_slug_segment(s)
}

/// Validates an instance name (dot-separated slugs).
pub fn validate_instance_name(name: &str) -> Result<(), NamingError> {
    if name.is_empty() {
        return Err(NamingError::Empty);
    }
    if name.len() > MAX_NAME_LEN {
        return Err(NamingError::TooLong { len: name.len() });
    }

    for segment in name.split('.') {
        if segment.is_empty() {
            return Err(NamingError::EmptySegment {
                segment: segment.to_string(),
            });
        }
        validate_slug_segment(segment)?;
    }

    Ok(())
}

/// Validates a scope path (slash-separated slugs).
pub fn validate_scope_path(path: &str) -> Result<(), NamingError> {
    if path.is_empty() {
        return Err(NamingError::Empty);
    }
    if path.len() > MAX_NAME_LEN {
        return Err(NamingError::TooLong { len: path.len() });
    }

    for segment in path.split('/') {
        if segment.is_empty() {
            return Err(NamingError::EmptySegment {
                segment: segment.to_string(),
            });
        }
        validate_slug_segment(segment)?;
    }

    Ok(())
}

/// Validates a role slug (starts with letter).
pub fn validate_role_slug(slug: &str) -> Result<(), NamingError> {
    if slug.is_empty() {
        return Err(NamingError::Empty);
    }
    if slug.len() > MAX_SEGMENT_LEN {
        return Err(NamingError::SegmentTooLong {
            segment: slug.to_string(),
            len: slug.len(),
        });
    }

    let mut chars = slug.char_indices();
    let (_, first) = chars.next().expect("slug is non-empty");

    if first.is_ascii_digit() {
        return Err(NamingError::RoleStartsWithDigit { ch: first });
    }
    if !first.is_ascii_lowercase() {
        return Err(NamingError::InvalidChar {
            segment: slug.to_string(),
            ch: first,
            pos: 0,
        });
    }
    if slug.ends_with('-') {
        return Err(NamingError::TrailingHyphen {
            segment: slug.to_string(),
        });
    }

    for (pos, ch) in chars {
        if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
            return Err(NamingError::InvalidChar {
                segment: slug.to_string(),
                ch,
                pos,
            });
        }
    }

    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NamingError {
    #[error("empty name")]
    Empty,

    #[error("segment '{segment}' is empty (consecutive or trailing separator)")]
    EmptySegment { segment: String },

    #[error("segment '{segment}' exceeds 63 characters ({len})")]
    SegmentTooLong { segment: String, len: usize },

    #[error("segment '{segment}' contains invalid character '{ch}' at position {pos}")]
    InvalidChar {
        segment: String,
        ch: char,
        pos: usize,
    },

    #[error("segment '{segment}' starts with a hyphen")]
    LeadingHyphen { segment: String },

    #[error("segment '{segment}' ends with a hyphen")]
    TrailingHyphen { segment: String },

    #[error("name exceeds 253 characters ({len})")]
    TooLong { len: usize },

    #[error("role slug must start with a letter, got '{ch}'")]
    RoleStartsWithDigit { ch: char },
}

fn validate_slug_segment(segment: &str) -> Result<(), NamingError> {
    if segment.len() > MAX_SEGMENT_LEN {
        return Err(NamingError::SegmentTooLong {
            segment: segment.to_string(),
            len: segment.len(),
        });
    }

    if segment.starts_with('-') {
        return Err(NamingError::LeadingHyphen {
            segment: segment.to_string(),
        });
    }
    if segment.ends_with('-') {
        return Err(NamingError::TrailingHyphen {
            segment: segment.to_string(),
        });
    }

    for (pos, ch) in segment.char_indices() {
        if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
            return Err(NamingError::InvalidChar {
                segment: segment.to_string(),
                ch,
                pos,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_boundaries() {
        let max = "a".repeat(63);
        let too_long = "a".repeat(64);

        assert_eq!(validate_slug(&max), Ok(()));
        assert!(matches!(
            validate_slug(&too_long),
            Err(NamingError::SegmentTooLong { len: 64, .. })
        ));
    }

    #[test]
    fn instance_name_valid_examples() {
        assert_eq!(validate_instance_name("dev-local"), Ok(()));
        assert_eq!(validate_instance_name("a"), Ok(()));
        assert_eq!(validate_instance_name("lanytehq.prod.us-east-1"), Ok(()));
    }

    #[test]
    fn instance_name_invalid_examples() {
        assert!(matches!(
            validate_instance_name("Dev-Local"),
            Err(NamingError::InvalidChar {
                ch: 'D',
                pos: 0,
                ..
            })
        ));
        assert!(matches!(
            validate_instance_name("-leading"),
            Err(NamingError::LeadingHyphen { .. })
        ));
        assert!(matches!(
            validate_instance_name("trailing-"),
            Err(NamingError::TrailingHyphen { .. })
        ));
        assert!(matches!(
            validate_instance_name("a..b"),
            Err(NamingError::EmptySegment { .. })
        ));

        let too_long_segment = format!("{}.x", "x".repeat(64));
        assert!(matches!(
            validate_instance_name(&too_long_segment),
            Err(NamingError::SegmentTooLong { len: 64, .. })
        ));

        assert!(matches!(
            validate_instance_name(""),
            Err(NamingError::Empty)
        ));
    }

    #[test]
    fn scope_path_valid_examples() {
        assert_eq!(validate_scope_path("lanytehq"), Ok(()));
        assert_eq!(validate_scope_path("3leaps/ipcprims"), Ok(()));
        assert_eq!(
            validate_scope_path("lanytehq/lanyte/crates/gateway"),
            Ok(())
        );
    }

    #[test]
    fn scope_path_invalid_examples() {
        assert!(matches!(
            validate_scope_path("lanytehq/"),
            Err(NamingError::EmptySegment { .. })
        ));
        assert!(matches!(
            validate_scope_path("/lanytehq"),
            Err(NamingError::EmptySegment { .. })
        ));
    }

    #[test]
    fn role_slug_valid_examples() {
        assert_eq!(validate_role_slug("cxotech"), Ok(()));
        assert_eq!(validate_role_slug("devlead"), Ok(()));
        assert_eq!(validate_role_slug("a1"), Ok(()));
    }

    #[test]
    fn role_slug_invalid_examples() {
        assert!(matches!(
            validate_role_slug("1cxotech"),
            Err(NamingError::RoleStartsWithDigit { ch: '1' })
        ));
        assert!(matches!(
            validate_role_slug("CxoTech"),
            Err(NamingError::InvalidChar {
                ch: 'C',
                pos: 0,
                ..
            })
        ));
        assert!(matches!(
            validate_role_slug("devlead-"),
            Err(NamingError::TrailingHyphen { .. })
        ));
    }

    #[test]
    fn total_length_boundaries() {
        let valid_253 = format!(
            "{}.{}.{}.{}",
            "a".repeat(63),
            "b".repeat(63),
            "c".repeat(63),
            "d".repeat(61)
        );
        assert_eq!(valid_253.len(), 253);
        assert_eq!(validate_instance_name(&valid_253), Ok(()));

        let invalid_254 = format!(
            "{}.{}.{}.{}",
            "a".repeat(63),
            "b".repeat(63),
            "c".repeat(63),
            "d".repeat(62)
        );
        assert_eq!(invalid_254.len(), 254);
        assert!(matches!(
            validate_instance_name(&invalid_254),
            Err(NamingError::TooLong { len: 254 })
        ));
    }

    #[test]
    fn covers_invalid_char_variant() {
        assert!(matches!(
            validate_scope_path("lanytehq/my_scope"),
            Err(NamingError::InvalidChar { ch: '_', .. })
        ));
    }
}
