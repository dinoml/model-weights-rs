use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use dinoml_checkpoint::{ConstantStorage, MappedSafetensors};

use crate::contract::{AppResult, Component, ComponentContract, ContractTarget, TargetStorage};
use crate::report::{LaneReport, OriginCounts, PeakBytes, TargetDigest, set_digest};

pub struct LegacySetup {
    components: Vec<LegacyComponent>,
}

struct LegacyComponent {
    component: Component,
    weights: MappedSafetensors,
    targets: Vec<ContractTarget>,
}

pub struct LegacyOutcome {
    pub binding_setup: Duration,
    pub report: LaneReport,
    pub digests: Vec<TargetDigest>,
}

impl LegacySetup {
    pub fn new(components: &[ComponentContract]) -> AppResult<Self> {
        let components = components
            .iter()
            .map(|component| {
                Ok(LegacyComponent {
                    component: component.component,
                    weights: MappedSafetensors::open(&component.weights_path)?,
                    targets: component.targets.clone(),
                })
            })
            .collect::<AppResult<Vec<_>>>()?;
        Ok(Self { components })
    }

    pub fn run(&self, validate: bool) -> AppResult<LegacyOutcome> {
        let binding_setup_started = Instant::now();
        let bindings = self
            .components
            .iter()
            .map(|component| component.weights.bindings())
            .collect::<Result<Vec<_>, _>>()?;
        let binding_setup = binding_setup_started.elapsed();
        let mut target_count = 0_u64;
        let mut delivered_bytes = 0_u64;
        let mut digests = Vec::new();
        let started = Instant::now();
        for (component, bindings) in self.components.iter().zip(&bindings) {
            for target in &component.targets {
                let transformed = bindings.materialize_metadata(&target.metadata)?;
                validate_metadata(target, &transformed)?;
                let bytes = transformed.bytes();
                target_count = target_count
                    .checked_add(1)
                    .ok_or("legacy target count overflow")?;
                delivered_bytes = delivered_bytes
                    .checked_add(u64::try_from(bytes.len())?)
                    .ok_or("legacy delivered byte count overflow")?;
                if validate {
                    digests.push(TargetDigest::from_bytes(
                        component.component,
                        &target.metadata.name,
                        bytes,
                    ));
                } else {
                    std::hint::black_box(bytes);
                }
            }
        }
        let elapsed = started.elapsed();
        let output_set_sha256 = validate.then(|| set_digest(&digests));
        Ok(LegacyOutcome {
            binding_setup,
            report: LaneReport {
                lane: "legacy",
                setup_ms: 0.0,
                materialization_ms: milliseconds(elapsed),
                pipeline_ms: None,
                target_count,
                delivered_bytes,
                throughput_mib_per_second: throughput(delivered_bytes, elapsed),
                workers: 1,
                execution_limits: None,
                cache_directory: None,
                prepared_entries_reset: 0,
                output_set_sha256,
                origins: OriginCounts {
                    other: target_count,
                    other_bytes: delivered_bytes,
                    ..OriginCounts::default()
                },
                peak_bytes: PeakBytes::default(),
                phases: BTreeMap::new(),
                operations: BTreeMap::new(),
                operation_nodes: Vec::new(),
            },
            digests,
        })
    }
}

fn validate_metadata(
    expected: &ContractTarget,
    actual: &dinoml_checkpoint::TransformedConstant,
) -> AppResult<()> {
    let expected_storage = match expected.storage {
        TargetStorage::Logical => ConstantStorage::Logical,
        TargetStorage::CkKyxc => ConstantStorage::RocmCkKyxc,
    };
    let actual_bytes = u64::try_from(actual.bytes().len())?;
    if actual.name() != expected.metadata.name
        || actual.dtype() != expected.metadata.dtype
        || actual.logical_shape() != expected.metadata.shape.as_ref()
        || actual.logical_strides() != expected.logical_strides.as_ref()
        || actual.storage() != expected_storage
        || actual_bytes != expected.output_bytes
    {
        return Err(format!(
            "legacy output for {:?} differs from the canonical target metadata",
            expected.metadata.name
        )
        .into());
    }
    Ok(())
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark byte counts are far below f64's exact integer range"
)]
fn throughput(bytes: u64, duration: Duration) -> f64 {
    if duration.is_zero() {
        return 0.0;
    }
    bytes as f64 / duration.as_secs_f64() / (1024.0 * 1024.0)
}
