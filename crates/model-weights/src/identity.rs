//! Strong content and lifecycle identities.

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

use crate::{Error, ErrorCategory, Result};

/// A SHA-256 content digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentDigest([u8; 32]);

impl ContentDigest {
    /// Creates a digest from its 32 canonical bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Hashes ordered byte slices under a domain separator.
    #[must_use]
    pub fn hash(domain: &str, parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"model-weights-rs\0");
        hasher.update((domain.len() as u128).to_le_bytes());
        hasher.update(domain.as_bytes());
        for part in parts {
            let bytes = part.as_ref();
            hasher.update((bytes.len() as u128).to_le_bytes());
            hasher.update(bytes);
        }
        Self(hasher.finalize().into())
    }

    /// Returns the canonical digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Display for ContentDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for ContentDigest {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        if value.len() != 64 {
            return Err(Error::new(
                ErrorCategory::InvalidFormat,
                "SHA-256 digest must contain exactly 64 hexadecimal characters",
            ));
        }

        let mut bytes = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0]).ok_or_else(|| {
                Error::invalid("SHA-256 digest contains a non-hexadecimal character")
            })?;
            let low = decode_hex(pair[1]).ok_or_else(|| {
                Error::invalid("SHA-256 digest contains a non-hexadecimal character")
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

impl Serialize for ContentDigest {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ContentDigest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = <&str>::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

const fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

macro_rules! digest_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(ContentDigest);

        impl $name {
            /// Creates the identity from a domain-separated digest.
            #[must_use]
            pub const fn from_digest(digest: ContentDigest) -> Self {
                Self(digest)
            }

            /// Returns the underlying content digest.
            #[must_use]
            pub const fn digest(self) -> ContentDigest {
                self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
                Display::fmt(&self.0, formatter)
            }
        }
    };
}

digest_id!(
    /// Identifies one immutable repository snapshot.
    SnapshotId
);
digest_id!(
    /// Identifies normalized configuration facts used for planning.
    ManifestId
);
digest_id!(
    /// Identifies normalized component, variant, and overlay selection facts.
    SelectionId
);
digest_id!(
    /// Identifies a consumer-supplied target constant contract.
    ContractId
);
digest_id!(
    /// Identifies one canonical, validated binding plan.
    PlanId
);
digest_id!(
    /// Identifies prepared bytes for one plan and backend ABI.
    PreparedId
);
digest_id!(
    /// Identifies a runtime-owned allocation outside this crate.
    AllocationId
);
digest_id!(
    /// Identifies a consumer backend and layout ABI.
    BackendId
);
digest_id!(
    /// Identifies one independently invalidatable overlay layer.
    OverlayId
);

/// A validated identifier used in serialized provider and ABI contracts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct StableName(Box<str>);

impl StableName {
    /// Parses a stable ASCII identifier.
    ///
    /// Identifiers may contain ASCII alphanumerics plus `.`, `_`, `-`, `/`,
    /// and `:`. They must be non-empty and at most 128 bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for an empty, oversized, or unsupported
    /// identifier.
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let value = value.as_ref();
        let valid = !value.is_empty()
            && value.len() <= 128
            && value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
            });
        if !valid {
            return Err(Error::invalid(
                "stable name must contain 1 to 128 portable ASCII identifier bytes",
            ));
        }
        Ok(Self(value.into()))
    }

    /// Returns the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for StableName {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for StableName {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}

impl TryFrom<Box<str>> for StableName {
    type Error = Error;

    fn try_from(value: Box<str>) -> Result<Self> {
        Self::parse(value)
    }
}

impl<'de> Deserialize<'de> for StableName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Box::<str>::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

/// Identifies a byte-affecting provider implementation and version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ImplementationId {
    provider: StableName,
    operation: StableName,
    version: u32,
}

impl ImplementationId {
    /// Creates a versioned implementation identity.
    #[must_use]
    pub const fn new(provider: StableName, operation: StableName, version: u32) -> Self {
        Self {
            provider,
            operation,
            version,
        }
    }

    /// Returns the provider identifier.
    #[must_use]
    pub const fn provider(&self) -> &StableName {
        &self.provider
    }

    /// Returns the provider-defined operation identifier.
    #[must_use]
    pub const fn operation(&self) -> &StableName {
        &self.operation
    }

    /// Returns the byte-affecting implementation version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_name_deserialization_replays_validation() -> Result<()> {
        let valid = StableName::parse("provider/op:v1")?;
        let encoded = serde_json::to_vec(&valid).map_err(|error| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "serialize stable name test value",
                error,
            )
        })?;
        let decoded: StableName = serde_json::from_slice(&encoded).map_err(|error| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "deserialize stable name test value",
                error,
            )
        })?;

        assert_eq!(decoded, valid);
        for invalid in [r#""""#, r#""contains space""#, r#""🔥""#] {
            let Err(_) = serde_json::from_str::<StableName>(invalid) else {
                return Err(Error::invalid(
                    "invalid stable name unexpectedly deserialized",
                ));
            };
        }
        Ok(())
    }

    #[test]
    fn stable_name_try_from_rejects_oversized_input() -> Result<()> {
        let oversized = "a".repeat(129).into_boxed_str();

        let Err(_) = StableName::try_from(oversized) else {
            return Err(Error::invalid(
                "oversized stable name unexpectedly passed TryFrom",
            ));
        };
        Ok(())
    }
}
