//! Loads the canonical `DinoML` SD1.5 native component set without inference.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use dinoml_autoencoder_kl::{
    ArtifactSet as VaeArtifactSet, AutoencoderKl, AutoencoderKlCheckpoint,
};
use dinoml_clip::{ArtifactSet as ClipArtifactSet, Clip, ClipCheckpoint, Device, TowerSelection};
use dinoml_stable_diffusion::{
    ComponentIdentity, ComponentSetIdentity, NativeComponents, SD15_CLIP_CHECKPOINT_ID,
    SD15_UNET_CHECKPOINT_ID, SD15_VAE_CHECKPOINT_ID, StableDiffusion15,
};
use dinoml_unet2d_condition::{
    ArtifactSet as UnetArtifactSet, Unet2dCondition, Unet2dConditionCheckpoint,
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};

type AppError = Box<dyn Error>;
type AppResult<T> = Result<T, AppError>;

const DEFAULT_CLIP_CHECKPOINT: &str = r"G:\checkpoints\openai\clip-vit-large-patch14";
const DEFAULT_CLIP_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\clip-l14-abi10";
const DEFAULT_UNET_CHECKPOINT: &str =
    r"G:\checkpoints\stable-diffusion-v1-5\stable-diffusion-v1-5\unet";
const DEFAULT_UNET_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\unet-sd15-abi10-profiled";
const DEFAULT_UNET_WEIGHTS: &str = "diffusion_pytorch_model.fp16.safetensors";
const DEFAULT_VAE_CHECKPOINT: &str =
    r"G:\checkpoints\stable-diffusion-v1-5\stable-diffusion-v1-5\vae";
const DEFAULT_VAE_ARTIFACTS: &str = r"H:\dinoml_v2\build\iter007\vae-sd15-abi10";
const DEFAULT_ARCHITECTURE: &str = "gfx1201";
const DEFAULT_DEVICE_INDEX: u32 = 0;
const DEFAULT_QUEUE_CAPACITY: usize = 2;

const HELP: &str = "\
DinoML SD1.5 full native-load benchmark

Usage:
  full_native --trust-native-artifacts [OPTIONS]

Options:
  --trust-native-artifacts       Acknowledge that the selected native libraries are trusted
  --clip-checkpoint PATH         CLIP checkpoint directory
  --clip-artifacts PATH          CLIP artifact-set directory
  --unet-checkpoint PATH         UNet checkpoint directory
  --unet-artifacts PATH          UNet artifact-set directory
  --unet-weights FILE            Relative UNet Safetensors filename
  --vae-checkpoint PATH          VAE checkpoint directory
  --vae-artifacts PATH           VAE artifact-set directory
  --architecture NAME            ROCm architecture (default: gfx1201)
  --device-index INDEX           ROCm device index (default: 0)
  --queue-capacity COUNT         Per-service bounded queue capacity (default: 2)
  -h, --help                     Print this help

The program loads and composes CLIP, UNet, and VAE services, then tears them
down without calling encode, forward, decode, or generation APIs. It emits one
JSON report on stdout.
";

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
    clip_checkpoint: PathBuf,
    clip_artifacts: PathBuf,
    unet_checkpoint: PathBuf,
    unet_artifacts: PathBuf,
    unet_weights: String,
    vae_checkpoint: PathBuf,
    vae_artifacts: PathBuf,
    architecture: String,
    device_index: u32,
    queue_capacity: usize,
    trusted_native_artifacts: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            clip_checkpoint: PathBuf::from(DEFAULT_CLIP_CHECKPOINT),
            clip_artifacts: PathBuf::from(DEFAULT_CLIP_ARTIFACTS),
            unet_checkpoint: PathBuf::from(DEFAULT_UNET_CHECKPOINT),
            unet_artifacts: PathBuf::from(DEFAULT_UNET_ARTIFACTS),
            unet_weights: DEFAULT_UNET_WEIGHTS.to_owned(),
            vae_checkpoint: PathBuf::from(DEFAULT_VAE_CHECKPOINT),
            vae_artifacts: PathBuf::from(DEFAULT_VAE_ARTIFACTS),
            architecture: DEFAULT_ARCHITECTURE.to_owned(),
            device_index: DEFAULT_DEVICE_INDEX,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            trusted_native_artifacts: false,
        }
    }
}

#[derive(Debug)]
enum Command {
    Help,
    Run(Box<BenchmarkConfig>),
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Elapsed {
    nanoseconds: u64,
    milliseconds: f64,
}

impl From<Duration> for Elapsed {
    fn from(duration: Duration) -> Self {
        Self {
            nanoseconds: u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            milliseconds: duration.as_secs_f64() * 1_000.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct PhaseTimings {
    exact_component_identity: Elapsed,
    clip_native_build: Elapsed,
    unet_native_build: Elapsed,
    vae_native_build: Elapsed,
    stable_diffusion_compose: Elapsed,
    ready_total: Elapsed,
    teardown: Elapsed,
    run_through_teardown: Elapsed,
}

#[derive(Debug, Serialize)]
struct IdentityEntry {
    role: Box<str>,
    source: Box<str>,
    digest: Box<str>,
}

impl From<&ComponentIdentity> for IdentityEntry {
    fn from(identity: &ComponentIdentity) -> Self {
        Self {
            role: identity.role.clone(),
            source: identity.source.clone(),
            digest: identity.digest.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct IdentityReport {
    text_encoder: IdentityEntry,
    denoiser: IdentityEntry,
    decoder: IdentityEntry,
}

impl From<&ComponentSetIdentity> for IdentityReport {
    fn from(identities: &ComponentSetIdentity) -> Self {
        Self {
            text_encoder: IdentityEntry::from(&identities.text_encoder),
            denoiser: IdentityEntry::from(&identities.denoiser),
            decoder: IdentityEntry::from(&identities.decoder),
        }
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    benchmark: &'static str,
    identity_mode: &'static str,
    backend: &'static str,
    inference_executed: bool,
    graph_replay_enabled: bool,
    artifact_content_reverification_enabled: bool,
    clip_tower_selection: &'static str,
    vae_workflow_selection: &'static str,
    config: BenchmarkConfig,
    identities: IdentityReport,
    phases: PhaseTimings,
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("full native-load benchmark failed: {error}");
            let mut cause = error.source();
            while let Some(source) = cause {
                eprintln!("caused by: {source}");
                cause = source.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn try_main() -> AppResult<()> {
    match parse_command()? {
        Command::Help => {
            print!("{HELP}");
            Ok(())
        }
        Command::Run(config) => run_benchmark(*config),
    }
}

fn parse_command() -> AppResult<Command> {
    let mut config = BenchmarkConfig::default();
    let mut args = env::args_os().skip(1);

    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("-h" | "--help") => return Ok(Command::Help),
            Some("--trust-native-artifacts") => config.trusted_native_artifacts = true,
            Some("--clip-checkpoint") => {
                config.clip_checkpoint =
                    PathBuf::from(required_value(&mut args, "--clip-checkpoint")?);
            }
            Some("--clip-artifacts") => {
                config.clip_artifacts =
                    PathBuf::from(required_value(&mut args, "--clip-artifacts")?);
            }
            Some("--unet-checkpoint") => {
                config.unet_checkpoint =
                    PathBuf::from(required_value(&mut args, "--unet-checkpoint")?);
            }
            Some("--unet-artifacts") => {
                config.unet_artifacts =
                    PathBuf::from(required_value(&mut args, "--unet-artifacts")?);
            }
            Some("--unet-weights") => {
                config.unet_weights = required_utf8_value(&mut args, "--unet-weights")?;
            }
            Some("--vae-checkpoint") => {
                config.vae_checkpoint =
                    PathBuf::from(required_value(&mut args, "--vae-checkpoint")?);
            }
            Some("--vae-artifacts") => {
                config.vae_artifacts = PathBuf::from(required_value(&mut args, "--vae-artifacts")?);
            }
            Some("--architecture") => {
                config.architecture = required_utf8_value(&mut args, "--architecture")?;
            }
            Some("--device-index") => {
                config.device_index = required_utf8_value(&mut args, "--device-index")?
                    .parse()
                    .map_err(|source| {
                        invalid_input(format!("--device-index must be a u32: {source}"))
                    })?;
            }
            Some("--queue-capacity") => {
                config.queue_capacity = required_utf8_value(&mut args, "--queue-capacity")?
                    .parse()
                    .map_err(|source| {
                        invalid_input(format!("--queue-capacity must be a usize: {source}"))
                    })?;
            }
            Some(option) => {
                return Err(invalid_input(format!("unrecognized option {option:?}")));
            }
            None => {
                return Err(invalid_input(format!(
                    "option is not valid UTF-8: {}",
                    Path::new(&argument).display()
                )));
            }
        }
    }

    validate_config(&config)?;
    Ok(Command::Run(Box::new(config)))
}

fn required_value(args: &mut impl Iterator<Item = OsString>, option: &str) -> AppResult<OsString> {
    args.next()
        .ok_or_else(|| invalid_input(format!("{option} requires a value")))
}

fn required_utf8_value(
    args: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> AppResult<String> {
    required_value(args, option)?
        .into_string()
        .map_err(|value| {
            invalid_input(format!(
                "{option} value is not valid UTF-8: {}",
                Path::new(&value).display()
            ))
        })
}

fn validate_config(config: &BenchmarkConfig) -> AppResult<()> {
    if !config.trusted_native_artifacts {
        return Err(invalid_input(
            "--trust-native-artifacts is required because native artifact loading executes code",
        ));
    }
    if config.architecture.trim().is_empty() {
        return Err(invalid_input("--architecture must not be empty"));
    }
    if config.queue_capacity == 0 {
        return Err(invalid_input("--queue-capacity must be positive"));
    }

    require_directory(&config.clip_checkpoint, "CLIP checkpoint")?;
    require_directory(&config.clip_artifacts, "CLIP artifact set")?;
    require_directory(&config.unet_checkpoint, "UNet checkpoint")?;
    require_directory(&config.unet_artifacts, "UNet artifact set")?;
    require_file(
        &config.unet_checkpoint.join(&config.unet_weights),
        "UNet weights",
    )?;
    require_directory(&config.vae_checkpoint, "VAE checkpoint")?;
    require_directory(&config.vae_artifacts, "VAE artifact set")
}

fn require_directory(path: &Path, label: &str) -> AppResult<()> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{label} directory does not exist: {}",
            path.display()
        )))
    }
}

fn require_file(path: &Path, label: &str) -> AppResult<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{label} file does not exist: {}",
            path.display()
        )))
    }
}

fn invalid_input(message: impl Into<String>) -> AppError {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn measure<T>(operation: impl FnOnce() -> AppResult<T>) -> AppResult<(T, Elapsed)> {
    let started = Instant::now();
    let value = operation()?;
    Ok((value, started.elapsed().into()))
}

fn run_benchmark(config: BenchmarkConfig) -> AppResult<()> {
    let run_started = Instant::now();

    let (identities, identity_elapsed) = measure(|| exact_component_identities(&config))?;
    let identity_report = IdentityReport::from(&identities);
    let device = Device::rocm(config.device_index);

    let (clip, clip_elapsed) = measure(|| load_clip(&config, device))?;
    let (unet, unet_elapsed) = measure(|| load_unet(&config, device))?;
    let (vae, vae_elapsed) = measure(|| load_vae(&config, device))?;

    let compose_started = Instant::now();
    let pipeline = StableDiffusion15::new(NativeComponents::new(clip, unet, vae, identities)?);
    let compose_elapsed = compose_started.elapsed().into();
    let _ = black_box(&pipeline);
    let ready_elapsed = run_started.elapsed().into();

    let teardown_started = Instant::now();
    drop(pipeline);
    let teardown_elapsed = teardown_started.elapsed().into();
    let run_elapsed = run_started.elapsed().into();

    let report = BenchmarkReport {
        schema_version: 1,
        benchmark: "dinoml-sd15-full-native-load",
        identity_mode: "exact-local-sd15-component-identities",
        backend: "rocm",
        inference_executed: false,
        graph_replay_enabled: true,
        artifact_content_reverification_enabled: false,
        clip_tower_selection: "text",
        vae_workflow_selection: "encode_and_decode",
        config,
        identities: identity_report,
        phases: PhaseTimings {
            exact_component_identity: identity_elapsed,
            clip_native_build: clip_elapsed,
            unet_native_build: unet_elapsed,
            vae_native_build: vae_elapsed,
            stable_diffusion_compose: compose_elapsed,
            ready_total: ready_elapsed,
            teardown: teardown_elapsed,
            run_through_teardown: run_elapsed,
        },
    };

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}

fn exact_component_identities(config: &BenchmarkConfig) -> AppResult<ComponentSetIdentity> {
    let clip_checkpoint = ClipCheckpoint::open(&config.clip_checkpoint)?;
    let clip_artifacts = ClipArtifactSet::open(&config.clip_artifacts, &clip_checkpoint)?;
    let clip_digest = digest(
        b"dinoml-sd15-clip-component-v1\0",
        &[
            clip_checkpoint.fingerprint().as_bytes(),
            clip_checkpoint.tokenizer_fingerprint().as_bytes(),
            clip_checkpoint.weights_fingerprint().as_bytes(),
            clip_artifacts.content_fingerprint()?.as_bytes(),
        ],
    )?;

    let unet_checkpoint = Unet2dConditionCheckpoint::open_with_weights(
        &config.unet_checkpoint,
        &config.unet_weights,
    )?;
    let unet_artifacts = UnetArtifactSet::open(&config.unet_artifacts, &unet_checkpoint)?;
    let unet_digest = digest(
        b"dinoml-sd15-unet-component-v1\0",
        &[
            unet_checkpoint.config_fingerprint().as_bytes(),
            unet_checkpoint.weights_fingerprint().as_bytes(),
            unet_artifacts.content_fingerprint()?.as_bytes(),
        ],
    )?;

    let vae_checkpoint = AutoencoderKlCheckpoint::open(&config.vae_checkpoint)?;
    let vae_artifacts = VaeArtifactSet::open(&config.vae_artifacts, &vae_checkpoint)?;
    let vae_digest = digest(
        b"dinoml-sd15-vae-component-v1\0",
        &[
            vae_checkpoint.config_fingerprint().as_bytes(),
            vae_checkpoint.weights_fingerprint().as_bytes(),
            vae_artifacts.content_fingerprint()?.as_bytes(),
        ],
    )?;

    Ok(ComponentSetIdentity {
        text_encoder: ComponentIdentity::new("text_encoder", SD15_CLIP_CHECKPOINT_ID, clip_digest)?,
        denoiser: ComponentIdentity::new("denoiser", SD15_UNET_CHECKPOINT_ID, unet_digest)?,
        decoder: ComponentIdentity::new("decoder", SD15_VAE_CHECKPOINT_ID, vae_digest)?,
    })
}

fn digest(domain: &[u8], parts: &[&[u8]]) -> AppResult<String> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        let length = u64::try_from(part.len())
            .map_err(|_| invalid_input("component identity part is too large"))?;
        hasher.update(length.to_le_bytes());
        hasher.update(part);
    }

    let mut output = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut output, "{byte:02x}")?;
    }
    Ok(output)
}

#[expect(
    unsafe_code,
    reason = "the caller explicitly attests that the selected CLIP native artifacts are trusted"
)]
fn load_clip(config: &BenchmarkConfig, device: Device) -> AppResult<Clip> {
    // SAFETY: `validate_config` requires the caller's explicit trust
    // attestation before this process loads any selected native library.
    let clip = unsafe {
        Clip::builder(&config.clip_checkpoint, &config.clip_artifacts)
            .execution_device(device)
            .execution_architecture(&config.architecture)
            .queue_capacity(config.queue_capacity)
            .towers(TowerSelection::Text)
            .build()
    }?;
    Ok(clip)
}

#[expect(
    unsafe_code,
    reason = "the caller explicitly attests that the selected UNet native artifacts are trusted"
)]
fn load_unet(config: &BenchmarkConfig, device: Device) -> AppResult<Unet2dCondition> {
    // SAFETY: `validate_config` requires the caller's explicit trust
    // attestation before this process loads any selected native library.
    let unet = unsafe {
        Unet2dCondition::builder(&config.unet_checkpoint, &config.unet_artifacts)
            .checkpoint_filename(&config.unet_weights)
            .execution_device(device)
            .execution_architecture(&config.architecture)
            .queue_capacity(config.queue_capacity)
            .build()
    }?;
    Ok(unet)
}

#[expect(
    unsafe_code,
    reason = "the caller explicitly attests that the selected VAE native artifacts are trusted"
)]
fn load_vae(config: &BenchmarkConfig, device: Device) -> AppResult<AutoencoderKl> {
    // SAFETY: `validate_config` requires the caller's explicit trust
    // attestation before this process loads any selected native library.
    let vae = unsafe {
        AutoencoderKl::builder(&config.vae_checkpoint, &config.vae_artifacts)
            .execution_device(device)
            .execution_architecture(&config.architecture)
            .queue_capacity(config.queue_capacity)
            .build()
    }?;
    Ok(vae)
}
