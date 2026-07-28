use std::time::{Duration, Instant};

use crate::cli::{Arguments, Command, Consumption, Lane};
use crate::contract::{self, AppResult, ContractSummary, Discovery};
use crate::legacy_lane::LegacySetup;
use crate::model_lane::ModelSetup;
use crate::report::{LaneReport, Report, TargetDigest, ValidationReport, set_digest};

pub fn execute(arguments: &Arguments) -> AppResult<Report> {
    let contract_started = Instant::now();
    let discovery = contract::discover(&arguments.paths)?;
    let contract_setup = contract_started.elapsed();
    if !discovery.summary.matches_reference {
        return Err(format!(
            "discovered contract differs from the SD1.5 reference: {} targets / {} bytes, \
             {} direct targets / {} bytes, {} grouped targets / {} bytes, \
             {} unbound targets / {} bytes",
            discovery.summary.target_count,
            discovery.summary.target_bytes,
            discovery.summary.direct_targets,
            discovery.summary.direct_bytes,
            discovery.summary.grouped_targets,
            discovery.summary.grouped_bytes,
            discovery.summary.unbound_targets,
            discovery.summary.unbound_bytes,
        )
        .into());
    }
    match arguments.command {
        Command::Validate => validate(arguments, discovery, contract_setup),
        Command::Sample => sample(arguments, discovery, contract_setup),
        Command::Prime => prime(arguments, discovery, contract_setup),
    }
}

fn validate(
    arguments: &Arguments,
    discovery: Discovery,
    contract_setup: Duration,
) -> AppResult<Report> {
    let legacy_setup_started = Instant::now();
    let legacy = LegacySetup::new(&discovery.components)?;
    let legacy_setup = legacy_setup_started.elapsed();
    let mut legacy_outcome = legacy.run(true)?;
    legacy_outcome.report.setup_ms =
        milliseconds(legacy_setup.saturating_add(legacy_outcome.binding_setup));
    validate_lane_completeness(&legacy_outcome.report, &discovery.summary)?;
    validate_digest_completeness("legacy", &legacy_outcome.digests, &discovery.summary)?;

    let model_setup_started = Instant::now();
    let model = ModelSetup::new(&discovery.components, None)?;
    let model_setup = model_setup_started.elapsed();
    let mut model_outcome = model.run(arguments.execution, true, 0)?;
    model_outcome.report.setup_ms = milliseconds(model_setup);
    validate_lane_completeness(&model_outcome.report, &discovery.summary)?;
    validate_digest_completeness("model-weights", &model_outcome.digests, &discovery.summary)?;

    let validation = compare(&legacy_outcome.digests, &model_outcome.digests);
    Ok(Report {
        schema_version: 5,
        command: "validate",
        consumption: Consumption::Sha256.label(),
        contract_setup_ms: milliseconds(contract_setup),
        contract: discovery.summary,
        lanes: vec![legacy_outcome.report, model_outcome.report],
        validation: Some(validation),
    })
}

fn sample(
    arguments: &Arguments,
    discovery: Discovery,
    contract_setup: Duration,
) -> AppResult<Report> {
    let hash_outputs = arguments.consumption == Consumption::Sha256;
    let lane = match arguments.lane {
        Lane::Legacy => {
            let setup_started = Instant::now();
            let legacy = LegacySetup::new(&discovery.components)?;
            let setup = setup_started.elapsed();
            let mut outcome = legacy.run(hash_outputs)?;
            outcome.report.setup_ms = milliseconds(setup.saturating_add(outcome.binding_setup));
            outcome.report
        }
        Lane::ModelWeights => {
            let setup_started = Instant::now();
            let model = ModelSetup::new(&discovery.components, arguments.cache.as_deref())?;
            let reset = if arguments.reset_prepared {
                model.reset_prepared()?
            } else {
                0
            };
            let setup = setup_started.elapsed();
            let mut outcome = model.run(arguments.execution, hash_outputs, reset)?;
            outcome.report.setup_ms = milliseconds(setup);
            outcome.report
        }
    };
    validate_lane_completeness(&lane, &discovery.summary)?;
    Ok(Report {
        schema_version: 5,
        command: "sample",
        consumption: arguments.consumption.label(),
        contract_setup_ms: milliseconds(contract_setup),
        contract: discovery.summary,
        lanes: vec![lane],
        validation: None,
    })
}

fn prime(
    arguments: &Arguments,
    discovery: Discovery,
    contract_setup: Duration,
) -> AppResult<Report> {
    let setup_started = Instant::now();
    let model = ModelSetup::new(&discovery.components, arguments.cache.as_deref())?;
    let reset = if arguments.reset_prepared {
        model.reset_prepared()?
    } else {
        0
    };
    let setup = setup_started.elapsed();
    let mut outcome = model.run(
        arguments.execution,
        arguments.consumption == Consumption::Sha256,
        reset,
    )?;
    outcome.report.setup_ms = milliseconds(setup);
    validate_lane_completeness(&outcome.report, &discovery.summary)?;
    Ok(Report {
        schema_version: 5,
        command: "prime",
        consumption: arguments.consumption.label(),
        contract_setup_ms: milliseconds(contract_setup),
        contract: discovery.summary,
        lanes: vec![outcome.report],
        validation: None,
    })
}

fn validate_lane_completeness(lane: &LaneReport, contract: &ContractSummary) -> AppResult<()> {
    let expected_targets = u64::try_from(contract.target_count)?;
    if lane.target_count != expected_targets || lane.delivered_bytes != contract.target_bytes {
        return Err(format!(
            "{} lane delivered {} targets / {} bytes; contract requires {} targets / {} bytes",
            lane.lane,
            lane.target_count,
            lane.delivered_bytes,
            contract.target_count,
            contract.target_bytes,
        )
        .into());
    }
    Ok(())
}

fn validate_digest_completeness(
    lane: &str,
    digests: &[TargetDigest],
    contract: &ContractSummary,
) -> AppResult<()> {
    let digest_bytes = digests.iter().try_fold(0_u64, |sum, digest| {
        sum.checked_add(digest.bytes)
            .ok_or("validation digest byte count overflow")
    })?;
    if digests.len() != contract.target_count || digest_bytes != contract.target_bytes {
        return Err(format!(
            "{lane} validation produced {} digests / {digest_bytes} bytes; contract requires {} \
             digests / {} bytes",
            digests.len(),
            contract.target_count,
            contract.target_bytes,
        )
        .into());
    }
    Ok(())
}

fn compare(legacy: &[TargetDigest], model: &[TargetDigest]) -> ValidationReport {
    let mut legacy_sorted = legacy.to_vec();
    let mut model_sorted = model.to_vec();
    legacy_sorted.sort_by(|left, right| left.key().cmp(&right.key()));
    model_sorted.sort_by(|left, right| left.key().cmp(&right.key()));
    let pair_mismatches = legacy_sorted
        .iter()
        .zip(&model_sorted)
        .filter(|(left, right)| left != right)
        .count();
    let length_mismatches = legacy_sorted.len().abs_diff(model_sorted.len());
    let mismatch_count = pair_mismatches.saturating_add(length_mismatches);
    ValidationReport {
        matched: mismatch_count == 0,
        mismatch_count,
        target_count: legacy_sorted.len(),
        target_bytes: legacy_sorted.iter().map(|record| record.bytes).sum(),
        legacy_set_sha256: set_digest(&legacy_sorted),
        model_weights_set_sha256: set_digest(&model_sorted),
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::contract::{Component, ContractSummary};
    use crate::report::{LaneReport, OriginCounts, PeakBytes, TargetDigest};

    use super::{compare, validate_lane_completeness};

    fn digest(component: Component, name: &str, byte: u8) -> TargetDigest {
        TargetDigest {
            component,
            name: name.to_owned(),
            bytes: 1,
            sha256: [byte; 32],
        }
    }

    #[test]
    fn validation_is_order_independent_but_detects_changed_bytes() {
        let legacy = [
            digest(Component::ClipText, "a", 1),
            digest(Component::Unet, "b", 2),
        ];
        let reordered = [legacy[1].clone(), legacy[0].clone()];
        assert!(compare(&legacy, &reordered).matched);

        let changed = [digest(Component::ClipText, "a", 9), legacy[1].clone()];
        let report = compare(&legacy, &changed);
        assert!(!report.matched);
        assert_eq!(report.mismatch_count, 1);
    }

    #[test]
    fn lane_completeness_rejects_missing_targets_or_bytes() {
        let contract = ContractSummary {
            digest_sha256: "digest".to_owned(),
            target_count: 2,
            target_bytes: 8,
            direct_targets: 1,
            direct_bytes: 4,
            grouped_targets: 1,
            grouped_bytes: 4,
            unbound_targets: 1,
            unbound_bytes: 616,
            identity_bytes_hashed: 0,
            matches_reference: false,
            components: Vec::new(),
        };
        let lane = |target_count, delivered_bytes| LaneReport {
            lane: "test",
            setup_ms: 0.0,
            materialization_ms: 0.0,
            pipeline_ms: None,
            target_count,
            delivered_bytes,
            throughput_mib_per_second: 0.0,
            workers: 1,
            execution_limits: None,
            cache_directory: None,
            prepared_entries_reset: 0,
            output_set_sha256: None,
            origins: OriginCounts::default(),
            peak_bytes: PeakBytes::default(),
            phases: BTreeMap::new(),
            operations: BTreeMap::new(),
            operation_nodes: Vec::new(),
        };

        assert!(validate_lane_completeness(&lane(2, 8), &contract).is_ok());
        assert!(validate_lane_completeness(&lane(1, 8), &contract).is_err());
        assert!(validate_lane_completeness(&lane(2, 4), &contract).is_err());
    }
}
