use std::ffi::{OsStr, OsString};
use std::io;
use std::path::PathBuf;

use model_weights::identity::ContentDigest;
use model_weights::limits::ExecutionLimits;

const DEFAULT_CLIP_CHECKPOINT: &str = r"G:\checkpoints\openai\clip-vit-large-patch14";
const DEFAULT_CLIP_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\clip-l14-abi10";
const DEFAULT_UNET_CHECKPOINT: &str =
    r"G:\checkpoints\stable-diffusion-v1-5\stable-diffusion-v1-5\unet";
const DEFAULT_UNET_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\unet-sd15-abi10-profiled";
const DEFAULT_UNET_WEIGHTS: &str = "diffusion_pytorch_model.fp16.safetensors";
const DEFAULT_VAE_CHECKPOINT: &str =
    r"G:\checkpoints\stable-diffusion-v1-5\stable-diffusion-v1-5\vae";
const DEFAULT_VAE_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\vae-sd15-abi10";
const DEFAULT_VAE_WEIGHTS: &str = "diffusion_pytorch_model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Validate,
    Sample,
    Prime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Legacy,
    ModelWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consumption {
    Delivery,
    Sha256,
}

impl Consumption {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Delivery => "delivery",
            Self::Sha256 => "sha256",
        }
    }
}

#[derive(Debug)]
pub struct Arguments {
    pub command: Command,
    pub lane: Lane,
    pub consumption: Consumption,
    pub paths: Paths,
    pub cache: Option<PathBuf>,
    pub reset_prepared: bool,
    pub execution: ExecutionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionConfig {
    pub workers: usize,
    pub delivery_queue_depth: usize,
    pub dispatch_lookahead: usize,
    pub source_bytes: u64,
    pub scratch_bytes: u64,
    pub prepared_bytes: u64,
}

impl ExecutionConfig {
    pub const fn with_max_work_items(self, max_work_items: usize) -> ExecutionLimits {
        ExecutionLimits {
            workers: self.workers,
            max_work_items,
            delivery_queue_depth: self.delivery_queue_depth,
            dispatch_lookahead: self.dispatch_lookahead,
            source_bytes: self.source_bytes,
            scratch_bytes: self.scratch_bytes,
            prepared_bytes: self.prepared_bytes,
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        let defaults = ExecutionLimits::default();
        Self {
            workers: defaults.workers,
            delivery_queue_depth: defaults.delivery_queue_depth,
            dispatch_lookahead: defaults.dispatch_lookahead,
            source_bytes: defaults.source_bytes,
            scratch_bytes: defaults.scratch_bytes,
            prepared_bytes: defaults.prepared_bytes,
        }
    }
}

#[derive(Debug)]
pub struct Paths {
    pub clip_checkpoint: PathBuf,
    pub clip_artifacts: PathBuf,
    pub clip_sha256: Option<ContentDigest>,
    pub unet_checkpoint: PathBuf,
    pub unet_artifacts: PathBuf,
    pub unet_weights: String,
    pub unet_sha256: Option<ContentDigest>,
    pub vae_checkpoint: PathBuf,
    pub vae_artifacts: PathBuf,
    pub vae_weights: String,
    pub vae_sha256: Option<ContentDigest>,
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            clip_checkpoint: DEFAULT_CLIP_CHECKPOINT.into(),
            clip_artifacts: DEFAULT_CLIP_ARTIFACTS.into(),
            clip_sha256: None,
            unet_checkpoint: DEFAULT_UNET_CHECKPOINT.into(),
            unet_artifacts: DEFAULT_UNET_ARTIFACTS.into(),
            unet_weights: DEFAULT_UNET_WEIGHTS.to_owned(),
            unet_sha256: None,
            vae_checkpoint: DEFAULT_VAE_CHECKPOINT.into(),
            vae_artifacts: DEFAULT_VAE_ARTIFACTS.into(),
            vae_weights: DEFAULT_VAE_WEIGHTS.to_owned(),
            vae_sha256: None,
        }
    }
}

impl Arguments {
    pub fn parse() -> io::Result<Self> {
        parse_from(std::env::args_os().skip(1))
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "keeping the flat CLI option grammar in one match makes accepted benchmark inputs auditable"
)]
fn parse_from(arguments: impl IntoIterator<Item = OsString>) -> io::Result<Arguments> {
    let mut arguments = arguments.into_iter();
    let command = parse_command(arguments.next().as_deref().and_then(OsStr::to_str))?;
    let mut parsed = Arguments {
        command,
        lane: Lane::Legacy,
        consumption: Consumption::Sha256,
        paths: Paths::default(),
        cache: None,
        reset_prepared: false,
        execution: ExecutionConfig::default(),
    };
    while let Some(option) = arguments.next() {
        match option.to_str() {
            Some("--lane") => {
                parsed.lane = match required_value(&mut arguments, "--lane")?.to_str() {
                    Some("legacy") => Lane::Legacy,
                    Some("model-weights") => Lane::ModelWeights,
                    _ => return Err(invalid("--lane must be legacy or model-weights")),
                };
            }
            Some("--cache") => {
                parsed.cache = Some(required_value(&mut arguments, "--cache")?.into());
            }
            Some("--reset-prepared") => parsed.reset_prepared = true,
            Some("--workers") => {
                parsed.execution.workers = required_positive_usize(&mut arguments, "--workers")?;
            }
            Some("--delivery-queue-depth") => {
                parsed.execution.delivery_queue_depth =
                    required_positive_usize(&mut arguments, "--delivery-queue-depth")?;
            }
            Some("--dispatch-lookahead") => {
                parsed.execution.dispatch_lookahead =
                    required_positive_usize(&mut arguments, "--dispatch-lookahead")?;
            }
            Some("--source-bytes") => {
                parsed.execution.source_bytes =
                    required_positive_u64(&mut arguments, "--source-bytes")?;
            }
            Some("--scratch-bytes") => {
                parsed.execution.scratch_bytes =
                    required_positive_u64(&mut arguments, "--scratch-bytes")?;
            }
            Some("--prepared-bytes") => {
                parsed.execution.prepared_bytes =
                    required_positive_u64(&mut arguments, "--prepared-bytes")?;
            }
            Some("--consume") => {
                parsed.consumption = match required_value(&mut arguments, "--consume")?.to_str() {
                    Some("delivery") => Consumption::Delivery,
                    Some("sha256") => Consumption::Sha256,
                    _ => return Err(invalid("--consume must be delivery or sha256")),
                };
            }
            Some("--clip-checkpoint") => {
                parsed.paths.clip_checkpoint =
                    required_value(&mut arguments, "--clip-checkpoint")?.into();
            }
            Some("--clip-artifacts") => {
                parsed.paths.clip_artifacts =
                    required_value(&mut arguments, "--clip-artifacts")?.into();
            }
            Some("--clip-sha256") => {
                parsed.paths.clip_sha256 = Some(required_digest(&mut arguments, "--clip-sha256")?);
            }
            Some("--unet-checkpoint") => {
                parsed.paths.unet_checkpoint =
                    required_value(&mut arguments, "--unet-checkpoint")?.into();
            }
            Some("--unet-artifacts") => {
                parsed.paths.unet_artifacts =
                    required_value(&mut arguments, "--unet-artifacts")?.into();
            }
            Some("--unet-weights") => {
                parsed.paths.unet_weights = required_utf8(&mut arguments, "--unet-weights")?;
            }
            Some("--unet-sha256") => {
                parsed.paths.unet_sha256 = Some(required_digest(&mut arguments, "--unet-sha256")?);
            }
            Some("--vae-checkpoint") => {
                parsed.paths.vae_checkpoint =
                    required_value(&mut arguments, "--vae-checkpoint")?.into();
            }
            Some("--vae-artifacts") => {
                parsed.paths.vae_artifacts =
                    required_value(&mut arguments, "--vae-artifacts")?.into();
            }
            Some("--vae-weights") => {
                parsed.paths.vae_weights = required_utf8(&mut arguments, "--vae-weights")?;
            }
            Some("--vae-sha256") => {
                parsed.paths.vae_sha256 = Some(required_digest(&mut arguments, "--vae-sha256")?);
            }
            Some(value) => return Err(invalid(format!("unknown option {value:?}; {}", usage()))),
            None => return Err(invalid("options must be valid UTF-8")),
        }
    }
    if parsed.command == Command::Prime {
        parsed.lane = Lane::ModelWeights;
        if parsed.cache.is_none() {
            return Err(invalid("prime requires --cache DIR"));
        }
    }
    if parsed.command == Command::Validate {
        parsed.cache = None;
        parsed.reset_prepared = false;
    }
    if parsed.lane == Lane::Legacy && (parsed.cache.is_some() || parsed.reset_prepared) {
        return Err(invalid(
            "cache options apply only to the model-weights lane",
        ));
    }
    Ok(parsed)
}

fn parse_command(argument: Option<&str>) -> io::Result<Command> {
    Ok(match argument {
        Some("validate") => Command::Validate,
        Some("sample") => Command::Sample,
        Some("prime") => Command::Prime,
        _ => return Err(usage_error()),
    })
}

fn required_value(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> io::Result<OsString> {
    arguments
        .next()
        .ok_or_else(|| invalid(format!("{option} requires a value")))
}

fn required_utf8(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> io::Result<String> {
    required_value(arguments, option)?
        .into_string()
        .map_err(|_| invalid(format!("{option} requires UTF-8")))
}

fn required_digest(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> io::Result<ContentDigest> {
    required_utf8(arguments, option)?
        .parse()
        .map_err(|error| invalid(format!("{option} is invalid: {error}")))
}

fn required_positive_usize(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> io::Result<usize> {
    let value = required_utf8(arguments, option)?;
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{option} must be a positive integer")))
}

fn required_positive_u64(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> io::Result<u64> {
    let value = required_utf8(arguments, option)?;
    value
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{option} must be a positive integer")))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn usage_error() -> io::Error {
    invalid(usage())
}

fn usage() -> &'static str {
    "usage: dinoml-sd15-loader-benchmark <validate|sample|prime> \
     [--lane legacy|model-weights] [--cache DIR] [--reset-prepared] \
     [--workers N] [--delivery-queue-depth N] [--dispatch-lookahead N] \
     [--source-bytes N] [--scratch-bytes N] [--prepared-bytes N] \
     [--consume delivery|sha256] \
     [--clip-checkpoint DIR] [--clip-artifacts DIR] \
     [--clip-sha256 HEX] \
     [--unet-checkpoint DIR] [--unet-artifacts DIR] [--unet-weights FILE] \
     [--unet-sha256 HEX] [--vae-checkpoint DIR] [--vae-artifacts DIR] \
     [--vae-weights FILE] [--vae-sha256 HEX]"
}

#[cfg(test)]
mod tests {
    use super::{Command, Consumption, ExecutionConfig, Lane, parse_from};
    use std::ffi::OsString;

    #[test]
    fn parses_model_weights_sample() {
        let arguments = parse_from(
            [
                "sample",
                "--lane",
                "model-weights",
                "--cache",
                "cache",
                "--workers",
                "2",
                "--delivery-queue-depth",
                "3",
                "--dispatch-lookahead",
                "5",
                "--source-bytes",
                "7",
                "--scratch-bytes",
                "11",
                "--prepared-bytes",
                "13",
                "--consume",
                "delivery",
            ]
            .map(OsString::from),
        )
        .expect("valid arguments");

        assert_eq!(arguments.command, Command::Sample);
        assert_eq!(arguments.lane, Lane::ModelWeights);
        assert_eq!(arguments.consumption, Consumption::Delivery);
        assert_eq!(
            arguments.execution,
            ExecutionConfig {
                workers: 2,
                delivery_queue_depth: 3,
                dispatch_lookahead: 5,
                source_bytes: 7,
                scratch_bytes: 11,
                prepared_bytes: 13,
            }
        );
        assert!(arguments.cache.is_some());
    }

    #[test]
    fn omitted_execution_options_use_core_defaults() {
        let arguments = parse_from([OsString::from("sample")]).expect("valid arguments");

        assert_eq!(arguments.execution, ExecutionConfig::default());
    }

    #[test]
    fn execution_options_reject_zero() {
        for option in [
            "--workers",
            "--delivery-queue-depth",
            "--dispatch-lookahead",
            "--source-bytes",
            "--scratch-bytes",
            "--prepared-bytes",
        ] {
            assert!(
                parse_from(["sample", option, "0"].map(OsString::from)).is_err(),
                "{option} accepted zero"
            );
        }
    }

    #[test]
    fn prime_requires_cache() {
        assert!(parse_from([OsString::from("prime")]).is_err());
    }
}
