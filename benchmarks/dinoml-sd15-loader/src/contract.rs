use std::error::Error as StdError;
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

use dinoml_autoencoder_kl::{ArtifactSet as VaeArtifactSet, AutoencoderKlCheckpoint, WorkflowKind};
use dinoml_checkpoint::{BindingOperation, ConstantBinding, ConstantStorage, ConstantTarget};
use dinoml_clip::{ArtifactSet as ClipArtifactSet, ClipCheckpoint, TowerKind};
use dinoml_runtime::TensorMetadata;
use dinoml_unet2d_condition::{ArtifactSet as UnetArtifactSet, Unet2dConditionCheckpoint};
use model_weights::identity::ContentDigest;
use model_weights::source::SourceDescriptor;
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::cli::Paths;

pub type AppResult<T> = Result<T, Box<dyn StdError>>;

pub const REFERENCE_TARGETS: usize = 939;
pub const REFERENCE_TARGET_BYTES: u64 = 2_065_322_934;
pub const REFERENCE_DIRECT_TARGETS: usize = 897;
pub const REFERENCE_DIRECT_BYTES: u64 = 1_946_881_974;
pub const REFERENCE_GROUPED_TARGETS: usize = 42;
pub const REFERENCE_GROUPED_BYTES: u64 = 118_440_960;
pub const REFERENCE_UNBOUND_TARGETS: usize = 1;
pub const REFERENCE_UNBOUND_BYTES: u64 = 616;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Component {
    ClipText,
    Unet,
    VaeDecode,
}

impl Component {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ClipText => "clip-text",
            Self::Unet => "unet",
            Self::VaeDecode => "vae-decode",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetStorage {
    Logical,
    CkKyxc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentitySource {
    TrustedOverride,
    ComputedSnapshot,
}

#[derive(Debug, Clone)]
pub struct ContractTarget {
    pub metadata: TensorMetadata,
    pub storage: TargetStorage,
    pub output_bytes: u64,
    pub logical_strides: Box<[usize]>,
    binding: ConstantBinding,
    source_names: Box<[String]>,
    grouped: bool,
}

impl ContractTarget {
    /// Returns the complete versioned checkpoint binding.
    #[must_use]
    pub const fn binding(&self) -> &ConstantBinding {
        &self.binding
    }

    /// Returns checkpoint tensor names in the binding's semantic input order.
    #[must_use]
    pub fn ordered_source_names(&self) -> &[String] {
        &self.source_names
    }

    /// Returns whether this target uses a grouped assembly operation.
    #[must_use]
    pub const fn is_grouped(&self) -> bool {
        self.grouped
    }

    /// Returns the sole source name for a non-grouped binding.
    #[must_use]
    pub fn single_source_name(&self) -> Option<&str> {
        if self.grouped {
            return None;
        }
        self.source_names.first().map(String::as_str)
    }
}

pub struct ComponentContract {
    pub component: Component,
    pub weights_path: PathBuf,
    pub digest: ContentDigest,
    pub identity_source: IdentitySource,
    pub identity_bytes_hashed: u64,
    pub source: SourceDescriptor,
    pub targets: Vec<ContractTarget>,
    pub unbound_targets: usize,
    pub unbound_bytes: u64,
}

pub struct Discovery {
    pub components: Vec<ComponentContract>,
    pub summary: ContractSummary,
}

#[derive(Debug, Serialize)]
pub struct ContractSummary {
    pub digest_sha256: String,
    pub target_count: usize,
    pub target_bytes: u64,
    pub direct_targets: usize,
    pub direct_bytes: u64,
    pub grouped_targets: usize,
    pub grouped_bytes: u64,
    pub unbound_targets: usize,
    pub unbound_bytes: u64,
    pub identity_bytes_hashed: u64,
    pub matches_reference: bool,
    pub components: Vec<ComponentSummary>,
}

#[derive(Debug, Serialize)]
pub struct ComponentSummary {
    pub component: Component,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub identity_source: IdentitySource,
    pub identity_bytes_hashed: u64,
    pub target_count: usize,
    pub target_bytes: u64,
    pub logical_targets: usize,
    pub ck_kyxc_targets: usize,
    pub direct_targets: usize,
    pub direct_bytes: u64,
    pub grouped_targets: usize,
    pub grouped_bytes: u64,
    pub unbound_targets: usize,
    pub unbound_bytes: u64,
}

pub fn discover(paths: &Paths) -> AppResult<Discovery> {
    let components = vec![
        discover_clip(paths)?,
        discover_unet(paths)?,
        discover_vae(paths)?,
    ];
    let summary = summarize(&components)?;
    Ok(Discovery {
        components,
        summary,
    })
}

fn discover_clip(paths: &Paths) -> AppResult<ComponentContract> {
    let checkpoint = ClipCheckpoint::open(&paths.clip_checkpoint)?;
    let artifacts = ClipArtifactSet::open(&paths.clip_artifacts, &checkpoint)?;
    let constants = uniform_constants(
        "CLIP text",
        artifacts
            .profiles()
            .filter(|profile| profile.tower() == TowerKind::Text)
            .map(|profile| profile.artifact().constants()),
    )?;
    drop(artifacts);
    let weights_path = checkpoint.weights().path().to_owned();
    let size = checked_snapshot_len(checkpoint.weights().snapshot_bytes())?;
    let (digest, identity_source, identity_bytes_hashed) =
        resolve_digest(paths.clip_sha256, size, || {
            *checkpoint.weights_fingerprint().as_bytes()
        });
    let source = retained_source(
        "clip/model.safetensors",
        &weights_path,
        size,
        digest,
        checkpoint,
    )?;
    build_component(
        Component::ClipText,
        weights_path,
        digest,
        identity_source,
        identity_bytes_hashed,
        source,
        constants,
    )
}

fn discover_unet(paths: &Paths) -> AppResult<ComponentContract> {
    let checkpoint =
        Unet2dConditionCheckpoint::open_with_weights(&paths.unet_checkpoint, &paths.unet_weights)?;
    let artifacts = UnetArtifactSet::open(&paths.unet_artifacts, &checkpoint)?;
    let constants = uniform_constants(
        "UNet",
        artifacts
            .profiles()
            .map(|profile| profile.artifact().constants()),
    )?;
    drop(artifacts);
    let weights_path = checkpoint.weights().path().to_owned();
    let size = checked_snapshot_len(checkpoint.weights().snapshot_bytes())?;
    let (digest, identity_source, identity_bytes_hashed) =
        resolve_digest(paths.unet_sha256, size, || {
            *checkpoint.weights_fingerprint().as_bytes()
        });
    let source = retained_source(
        "unet/weights.safetensors",
        &weights_path,
        size,
        digest,
        checkpoint,
    )?;
    build_component(
        Component::Unet,
        weights_path,
        digest,
        identity_source,
        identity_bytes_hashed,
        source,
        constants,
    )
}

fn discover_vae(paths: &Paths) -> AppResult<ComponentContract> {
    let checkpoint =
        AutoencoderKlCheckpoint::open_with_weights(&paths.vae_checkpoint, &paths.vae_weights)?;
    let artifacts = VaeArtifactSet::open(&paths.vae_artifacts, &checkpoint)?;
    let constants = uniform_constants(
        "VAE decoder",
        artifacts
            .profiles()
            .filter(|profile| profile.key().workflow() == WorkflowKind::Decode)
            .map(|profile| profile.artifact().constants()),
    )?;
    drop(artifacts);
    let weights_path = checkpoint.weights().path().to_owned();
    let size = checked_snapshot_len(checkpoint.weights().snapshot_bytes())?;
    let (digest, identity_source, identity_bytes_hashed) =
        resolve_digest(paths.vae_sha256, size, || {
            *checkpoint.weights_fingerprint().as_bytes()
        });
    let source = retained_source(
        "vae/weights.safetensors",
        &weights_path,
        size,
        digest,
        checkpoint,
    )?;
    build_component(
        Component::VaeDecode,
        weights_path,
        digest,
        identity_source,
        identity_bytes_hashed,
        source,
        constants,
    )
}

fn uniform_constants<'a>(
    label: &str,
    profiles: impl IntoIterator<Item = &'a [TensorMetadata]>,
) -> AppResult<Vec<TensorMetadata>> {
    let mut profiles = profiles.into_iter();
    let first = profiles
        .next()
        .ok_or_else(|| invalid(format!("{label} has no matching artifact profile")))?;
    for candidate in profiles {
        if candidate != first {
            return Err(invalid(format!(
                "{label} artifact profiles disagree on their constant contract"
            ))
            .into());
        }
    }
    Ok(first.to_vec())
}

fn build_component(
    component: Component,
    weights_path: PathBuf,
    digest: ContentDigest,
    identity_source: IdentitySource,
    identity_bytes_hashed: u64,
    source: SourceDescriptor,
    constants: Vec<TensorMetadata>,
) -> AppResult<ComponentContract> {
    let mut targets = Vec::new();
    let mut unbound_targets = 0_usize;
    let mut unbound_bytes = 0_u64;
    for metadata in constants {
        if metadata.binding.is_none() {
            unbound_targets = unbound_targets.saturating_add(1);
            unbound_bytes = unbound_bytes.saturating_add(metadata_bytes(&metadata)?);
            continue;
        }
        let binding = ConstantBinding::from_metadata(&metadata)?;
        match binding.operation() {
            BindingOperation::Tensor { .. }
            | BindingOperation::Concat { .. }
            | BindingOperation::Transpose { .. }
            | BindingOperation::Reshape { .. } => {
                targets.push(contract_target(metadata, binding)?);
            }
            _ => {
                return Err(invalid(format!(
                    "{} target {:?} uses a newer binding operation",
                    component.label(),
                    metadata.name
                ))
                .into());
            }
        }
    }
    Ok(ComponentContract {
        component,
        weights_path,
        digest,
        identity_source,
        identity_bytes_hashed,
        source,
        targets,
        unbound_targets,
        unbound_bytes,
    })
}

fn resolve_digest(
    trusted: Option<ContentDigest>,
    snapshot_bytes: u64,
    compute: impl FnOnce() -> [u8; 32],
) -> (ContentDigest, IdentitySource, u64) {
    match trusted {
        Some(digest) => (digest, IdentitySource::TrustedOverride, 0),
        None => (
            ContentDigest::from_bytes(compute()),
            IdentitySource::ComputedSnapshot,
            snapshot_bytes,
        ),
    }
}

fn contract_target(
    metadata: TensorMetadata,
    binding: ConstantBinding,
) -> AppResult<ContractTarget> {
    let target = ConstantTarget::from_metadata(&metadata)?;
    let storage = match target.storage() {
        ConstantStorage::Logical => TargetStorage::Logical,
        ConstantStorage::RocmCkKyxc => TargetStorage::CkKyxc,
        ConstantStorage::RocmCkKxc => {
            return Err(invalid(format!(
                "target {:?} uses CK KXC storage, which is outside the SD1.5 benchmark contract",
                metadata.name
            ))
            .into());
        }
        _ => {
            return Err(invalid(format!(
                "target {:?} uses a newer storage contract",
                metadata.name
            ))
            .into());
        }
    };
    let (source_names, grouped) = match binding.operation() {
        BindingOperation::Tensor { source }
        | BindingOperation::Transpose { source, .. }
        | BindingOperation::Reshape { source } => (vec![source.clone()].into_boxed_slice(), false),
        BindingOperation::Concat { sources, .. } => (sources.clone(), true),
        _ => {
            return Err(invalid(format!(
                "target {:?} uses a newer binding operation",
                metadata.name
            ))
            .into());
        }
    };
    Ok(ContractTarget {
        binding,
        storage,
        output_bytes: u64::try_from(target.storage_nbytes()?)?,
        logical_strides: target.logical_strides().into(),
        source_names,
        grouped,
        metadata,
    })
}

fn metadata_bytes(metadata: &TensorMetadata) -> AppResult<u64> {
    match metadata.nbytes {
        Some(bytes) => Ok(u64::try_from(bytes)?),
        None => Ok(u64::try_from(
            ConstantTarget::from_metadata(metadata)?.storage_nbytes()?,
        )?),
    }
}

fn checked_snapshot_len(bytes: &[u8]) -> AppResult<u64> {
    Ok(u64::try_from(bytes.len())?)
}

#[expect(
    clippy::too_many_lines,
    reason = "the summary validates all correlated contract totals in one pass"
)]
fn summarize(components: &[ComponentContract]) -> AppResult<ContractSummary> {
    let component_summaries = components
        .iter()
        .map(|component| {
            let target_bytes = component.targets.iter().try_fold(0_u64, |sum, target| {
                sum.checked_add(target.output_bytes)
                    .ok_or_else(|| invalid("target byte total overflow"))
            })?;
            let direct_targets = component
                .targets
                .iter()
                .filter(|target| !target.is_grouped())
                .count();
            let direct_bytes = sum_field(
                component
                    .targets
                    .iter()
                    .filter(|target| !target.is_grouped())
                    .map(|target| target.output_bytes),
            )?;
            let grouped_targets = component
                .targets
                .iter()
                .filter(|target| target.is_grouped())
                .count();
            let grouped_bytes = sum_field(
                component
                    .targets
                    .iter()
                    .filter(|target| target.is_grouped())
                    .map(|target| target.output_bytes),
            )?;
            Ok(ComponentSummary {
                component: component.component,
                checkpoint: component.weights_path.clone(),
                checkpoint_sha256: component.digest.to_string(),
                identity_source: component.identity_source,
                identity_bytes_hashed: component.identity_bytes_hashed,
                target_count: component.targets.len(),
                target_bytes,
                logical_targets: component
                    .targets
                    .iter()
                    .filter(|target| target.storage == TargetStorage::Logical)
                    .count(),
                ck_kyxc_targets: component
                    .targets
                    .iter()
                    .filter(|target| target.storage == TargetStorage::CkKyxc)
                    .count(),
                direct_targets,
                direct_bytes,
                grouped_targets,
                grouped_bytes,
                unbound_targets: component.unbound_targets,
                unbound_bytes: component.unbound_bytes,
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()?;
    let target_count = component_summaries
        .iter()
        .map(|summary| summary.target_count)
        .sum();
    let target_bytes = sum_field(
        component_summaries
            .iter()
            .map(|summary| summary.target_bytes),
    )?;
    let direct_targets = component_summaries
        .iter()
        .map(|summary| summary.direct_targets)
        .sum();
    let direct_bytes = sum_field(
        component_summaries
            .iter()
            .map(|summary| summary.direct_bytes),
    )?;
    let grouped_targets = component_summaries
        .iter()
        .map(|summary| summary.grouped_targets)
        .sum();
    let grouped_bytes = sum_field(
        component_summaries
            .iter()
            .map(|summary| summary.grouped_bytes),
    )?;
    let unbound_targets = component_summaries
        .iter()
        .map(|summary| summary.unbound_targets)
        .sum();
    let unbound_bytes = sum_field(
        component_summaries
            .iter()
            .map(|summary| summary.unbound_bytes),
    )?;
    let identity_bytes_hashed = sum_field(
        component_summaries
            .iter()
            .map(|summary| summary.identity_bytes_hashed),
    )?;
    let digest_sha256 = contract_digest(components)?;
    Ok(ContractSummary {
        digest_sha256,
        target_count,
        target_bytes,
        direct_targets,
        direct_bytes,
        grouped_targets,
        grouped_bytes,
        unbound_targets,
        unbound_bytes,
        identity_bytes_hashed,
        matches_reference: target_count == REFERENCE_TARGETS
            && target_bytes == REFERENCE_TARGET_BYTES
            && direct_targets == REFERENCE_DIRECT_TARGETS
            && direct_bytes == REFERENCE_DIRECT_BYTES
            && grouped_targets == REFERENCE_GROUPED_TARGETS
            && grouped_bytes == REFERENCE_GROUPED_BYTES
            && unbound_targets == REFERENCE_UNBOUND_TARGETS
            && unbound_bytes == REFERENCE_UNBOUND_BYTES,
        components: component_summaries,
    })
}

fn sum_field(values: impl IntoIterator<Item = u64>) -> io::Result<u64> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| invalid("contract byte total overflow"))
    })
}

#[derive(Serialize)]
struct CanonicalTarget<'a> {
    component: Component,
    name: &'a str,
    binding: &'a ConstantBinding,
    dtype: String,
    shape: &'a [usize],
    logical_strides: &'a [usize],
    storage: TargetStorage,
    output_bytes: u64,
}

fn contract_digest(components: &[ComponentContract]) -> AppResult<String> {
    let canonical = components
        .iter()
        .flat_map(|component| {
            component.targets.iter().map(|target| CanonicalTarget {
                component: component.component,
                name: &target.metadata.name,
                binding: target.binding(),
                dtype: target.metadata.dtype.to_string(),
                shape: &target.metadata.shape,
                logical_strides: &target.logical_strides,
                storage: target.storage,
                output_bytes: target.output_bytes,
            })
        })
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&canonical)?;
    let mut digest = Sha256::new();
    digest.update(b"dinoml-sd15-loader-contract-v2\0");
    digest.update(bytes);
    let digest = digest.finalize();
    Ok(digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    ))
}

#[cfg(windows)]
#[expect(
    unsafe_code,
    reason = "DinoML's retained mapping denies writes and deletes for the descriptor lifetime"
)]
fn retained_source<T>(
    logical_path: &str,
    local_path: &Path,
    size: u64,
    digest: ContentDigest,
    guard: T,
) -> model_weights::Result<SourceDescriptor>
where
    T: Send + Sync + 'static,
{
    // SAFETY: each DinoML checkpoint owns a read-only `MappedSafetensors`.
    // Its Windows handle permits only FILE_SHARE_READ, so retaining the
    // checkpoint as `guard` prevents mutation, replacement, deletion, and
    // truncation until every descriptor-derived mapping has been dropped.
    unsafe { SourceDescriptor::retained(logical_path, local_path, size, digest, guard) }
}

#[cfg(not(windows))]
fn retained_source<T>(
    _logical_path: &str,
    local_path: &Path,
    size: u64,
    digest: ContentDigest,
    guard: T,
) -> model_weights::Result<SourceDescriptor>
where
    T: Send + Sync + 'static,
{
    drop(guard);
    SourceDescriptor::local_with_digest(local_path, size, digest)
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use dinoml_checkpoint::ConstantBinding;
    use dinoml_runtime::{DType, TensorMetadata};

    use super::contract_target;

    fn metadata(binding: &ConstantBinding) -> TensorMetadata {
        TensorMetadata {
            name: "qkv".to_owned(),
            tensor: "qkv".to_owned(),
            shape: [6, 2].into(),
            shape_spec: None,
            dtype: DType::Float16,
            layout: None,
            semantic: None,
            binding: Some(serde_json::to_value(binding).expect("binding is serializable")),
            storage: None,
            nbytes: Some(24),
            offset: None,
        }
    }

    #[test]
    fn grouped_target_retains_complete_binding_and_semantic_source_order() {
        let binding = ConstantBinding::concat(
            0,
            vec!["query".to_owned(), "key".to_owned(), "value".to_owned()].into_boxed_slice(),
        );
        let target = contract_target(metadata(&binding), binding).expect("valid target");

        assert!(target.is_grouped());
        assert_eq!(target.ordered_source_names(), ["query", "key", "value"]);
        assert_eq!(target.single_source_name(), None);
        assert_eq!(target.output_bytes, 24);

        let serialized = serde_json::to_value(target.binding()).expect("binding is serializable");
        assert_eq!(serialized["schema_version"], 1);
        assert_eq!(serialized["kind"], "concat");
        assert_eq!(serialized["axis"], 0);
        assert_eq!(
            serialized["sources"],
            serde_json::json!(["query", "key", "value"])
        );
    }

    #[test]
    fn single_source_helper_covers_structural_bindings() {
        let binding = ConstantBinding::transpose("weight", [1, 0]);
        let target = contract_target(metadata(&binding), binding).expect("valid target");

        assert!(!target.is_grouped());
        assert_eq!(target.ordered_source_names(), ["weight"]);
        assert_eq!(target.single_source_name(), Some("weight"));
        let serialized = serde_json::to_value(target.binding()).expect("binding is serializable");
        assert_eq!(serialized["kind"], "transpose");
        assert_eq!(serialized["axes"], serde_json::json!([1, 0]));
    }
}
