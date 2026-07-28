use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::cli::ExecutionConfig;
use crate::contract::{Component, ContractSummary};

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub command: &'static str,
    pub consumption: &'static str,
    pub contract_setup_ms: f64,
    pub contract: ContractSummary,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lanes: Vec<LaneReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationReport>,
}

#[derive(Debug, Serialize)]
pub struct LaneReport {
    pub lane: &'static str,
    pub setup_ms: f64,
    pub materialization_ms: f64,
    pub pipeline_ms: Option<f64>,
    pub target_count: u64,
    pub delivered_bytes: u64,
    pub throughput_mib_per_second: f64,
    pub workers: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_limits: Option<ExecutionLimitReport>,
    pub cache_directory: Option<String>,
    pub prepared_entries_reset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_set_sha256: Option<String>,
    pub origins: OriginCounts,
    pub peak_bytes: PeakBytes,
    pub phases: BTreeMap<String, PhaseMetrics>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub operations: BTreeMap<String, OperationMetrics>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub operation_nodes: Vec<OperationNodeMetrics>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExecutionLimitReport {
    pub workers: usize,
    pub delivery_queue_depth: usize,
    pub dispatch_lookahead: usize,
    pub source_bytes: u64,
    pub scratch_bytes: u64,
    pub prepared_bytes: u64,
}

impl From<ExecutionConfig> for ExecutionLimitReport {
    fn from(config: ExecutionConfig) -> Self {
        Self {
            workers: config.workers,
            delivery_queue_depth: config.delivery_queue_depth,
            dispatch_lookahead: config.dispatch_lookahead,
            source_bytes: config.source_bytes,
            scratch_bytes: config.scratch_bytes,
            prepared_bytes: config.prepared_bytes,
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct OriginCounts {
    pub source: u64,
    pub source_bytes: u64,
    pub transform: u64,
    pub transform_bytes: u64,
    pub operation_graph: u64,
    pub operation_graph_bytes: u64,
    pub cache: u64,
    pub cache_bytes: u64,
    pub other: u64,
    pub other_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct PeakBytes {
    pub source: Option<u64>,
    pub scratch: Option<u64>,
    pub prepared: Option<u64>,
}

#[derive(Debug, Default, Serialize)]
pub struct PhaseMetrics {
    pub milliseconds: f64,
    pub bytes: u64,
    pub invocations: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct OperationMetrics {
    pub elapsed_sum_ms: f64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub materialized_output_bytes: u64,
    pub invocations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OperationLocationMetrics {
    Binding,
    PlannedTransform { index: usize },
    GraphNode { node_id: u32 },
    Other,
}

#[derive(Debug, Serialize)]
pub struct OperationNodeMetrics {
    pub component: Component,
    pub target: Option<String>,
    pub work_ordinal: Option<u64>,
    pub location: OperationLocationMetrics,
    pub kind: &'static str,
    pub elapsed_sum_ms: f64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub materialized_output_bytes: u64,
    pub invocations: u64,
}

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub matched: bool,
    pub mismatch_count: usize,
    pub target_count: usize,
    pub target_bytes: u64,
    pub legacy_set_sha256: String,
    pub model_weights_set_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetDigest {
    pub component: Component,
    pub name: String,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

impl TargetDigest {
    pub fn from_bytes(component: Component, name: &str, bytes: &[u8]) -> Self {
        Self {
            component,
            name: name.to_owned(),
            bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            sha256: Sha256::digest(bytes).into(),
        }
    }

    pub fn key(&self) -> (Component, &str) {
        (self.component, &self.name)
    }
}

pub fn set_digest(records: &[TargetDigest]) -> String {
    let mut records = records.iter().collect::<Vec<_>>();
    records.sort_by(|left, right| left.key().cmp(&right.key()));
    let mut digest = Sha256::new();
    digest.update(b"dinoml-sd15-loader-validation-v1\0");
    for record in records {
        digest.update(record.component.label().as_bytes());
        digest.update([0]);
        digest.update(record.name.as_bytes());
        digest.update([0]);
        digest.update(record.bytes.to_le_bytes());
        digest.update(record.sha256);
    }
    let digest = digest.finalize();
    digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    )
}
