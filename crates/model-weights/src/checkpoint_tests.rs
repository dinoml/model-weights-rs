#[cfg(feature = "mmap")]
use std::fmt;
use std::fs::File;
use std::io::Write as _;
#[cfg(feature = "mmap")]
use std::sync::Arc;
#[cfg(feature = "mmap")]
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use sha2::{Digest as _, Sha256};
use tempfile::TempDir;

use super::{AccessMode, Checkpoint, CheckpointBuilder, TensorData};
use crate::identity::ContentDigest;
use crate::inventory::DigestState;
use crate::source::{DigestPolicy, SourceDescriptor, SourceKind};
use crate::{CancellationToken, ErrorCategory, Result};

fn write_safetensors(
    directory: &TempDir,
    name: &str,
    header_json: &str,
    payload: &[u8],
) -> Result<std::path::PathBuf> {
    let mut header = header_json.as_bytes().to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let path = directory.path().join(name);
    let mut file = File::create(&path)
        .map_err(|source| crate::Error::io("test could not create checkpoint", source))?;
    let length = u64::try_from(header.len())
        .map_err(|_error| crate::Error::limit("test header length does not fit u64"))?;
    file.write_all(&length.to_le_bytes())
        .and_then(|()| file.write_all(&header))
        .and_then(|()| file.write_all(payload))
        .map_err(|source| crate::Error::io("test could not write checkpoint", source))?;
    Ok(path)
}

#[test]
fn single_file_inventory_is_sorted_and_payload_is_lazy() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let payload = [0_u8, 0, 0x80, 0x3f, 0, 1, 2, 3];
    let path = write_safetensors(
        &directory,
        "weights.safetensors",
        r#"{"z":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
        &payload,
    )?;

    let checkpoint = Checkpoint::open(&path)?;
    let names = checkpoint
        .inventory()
        .iter()
        .map(crate::inventory::TensorRecord::name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["a", "z"]);
    let encoded_inventory = serde_json::to_vec(checkpoint.inventory()).map_err(|source| {
        crate::Error::with_source(
            ErrorCategory::InvalidFormat,
            "test could not serialize inventory",
            source,
        )
    })?;
    let decoded_inventory: crate::inventory::Inventory = serde_json::from_slice(&encoded_inventory)
        .map_err(|source| {
            crate::Error::with_source(
                ErrorCategory::InvalidFormat,
                "test could not deserialize inventory",
                source,
            )
        })?;
    assert_eq!(&decoded_inventory, checkpoint.inventory());

    let TensorData::Plain(tensor) = checkpoint.tensor("a")? else {
        panic!("plain safetensors dtype must remain plain");
    };
    assert_eq!(tensor.bytes().as_slice(), &payload[..4]);

    let cancellation = CancellationToken::new();
    assert_eq!(
        checkpoint.snapshot_id(&cancellation)?,
        checkpoint.snapshot_id(&cancellation)?
    );
    Ok(())
}

#[test]
fn pre_cancelled_open_and_span_read_stop_before_work() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "cancel.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[7],
    )?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        Checkpoint::open_with_cancellation(&path, &cancellation)
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Cancelled)
    );

    let checkpoint = Checkpoint::open(path)?;
    let span = checkpoint
        .inventory()
        .tensor("x")
        .ok_or_else(|| crate::Error::binding("test tensor is absent"))?
        .storage()
        .span();
    assert_eq!(
        checkpoint
            .read_span_with_cancellation(span, &cancellation)
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Cancelled)
    );
    Ok(())
}

#[test]
fn duplicate_names_and_noncontiguous_payloads_are_rejected() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let duplicate = write_safetensors(
        &directory,
        "duplicate.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        &[1, 2],
    )?;
    assert_eq!(
        Checkpoint::open(duplicate)
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::InvalidFormat)
    );

    let gap = write_safetensors(
        &directory,
        "gap.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        &[0, 1],
    )?;
    assert_eq!(
        Checkpoint::open(gap).err().map(|error| error.category()),
        Some(ErrorCategory::InvalidFormat)
    );
    Ok(())
}

#[test]
fn sub_byte_safetensors_dtypes_remain_explicitly_quantized() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "packed.safetensors",
        r#"{"f4":{"dtype":"F4","shape":[4],"data_offsets":[0,2]},"f6":{"dtype":"F6_E3M2","shape":[4],"data_offsets":[2,5]}}"#,
        &[0x12, 0x34, 0x56, 0x78, 0x9a],
    )?;
    let checkpoint = Checkpoint::open(path)?;

    let TensorData::Quantized(f4) = checkpoint.tensor("f4")? else {
        panic!("F4 safetensors storage must remain explicitly quantized");
    };
    let TensorData::Quantized(f6) = checkpoint.tensor("f6")? else {
        panic!("F6 safetensors storage must remain explicitly quantized");
    };
    assert_eq!(f4.bytes().as_slice(), &[0x12, 0x34]);
    assert_eq!(f6.bytes().as_slice(), &[0x56, 0x78, 0x9a]);
    Ok(())
}

#[test]
fn shard_index_is_membership_and_location_truth() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    write_safetensors(
        &directory,
        "one.safetensors",
        r#"{"one":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[1],
    )?;
    write_safetensors(
        &directory,
        "two.safetensors",
        r#"{"two":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[2],
    )?;
    let index_path = directory.path().join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        r#"{"weight_map":{"one":"two.safetensors","two":"one.safetensors"}}"#,
    )
    .map_err(|source| crate::Error::io("test could not write shard index", source))?;

    assert_eq!(
        Checkpoint::open_index(index_path)
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Integrity)
    );
    Ok(())
}

#[test]
fn empty_tensors_and_multiple_tensors_in_one_shard_are_supported() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    write_safetensors(
        &directory,
        "shared.safetensors",
        r#"{"empty":{"dtype":"BF16","shape":[0,7],"data_offsets":[0,0]},"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0x5a],
    )?;
    let index_path = directory.path().join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        r#"{"weight_map":{"empty":"shared.safetensors","value":"shared.safetensors"}}"#,
    )
    .map_err(|source| crate::Error::io("test could not write shard index", source))?;

    let checkpoint = Checkpoint::open_index(index_path)?;

    assert_eq!(checkpoint.inventory().files().len(), 1);
    assert_eq!(checkpoint.tensor("empty")?.shape(), [0, 7]);
    assert!(checkpoint.tensor("empty")?.bytes().as_slice().is_empty());
    assert_eq!(checkpoint.tensor("value")?.bytes().as_slice(), [0x5a]);
    Ok(())
}

#[test]
fn unsafe_shard_paths_and_truncated_payloads_are_rejected() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let unsafe_index = directory.path().join("unsafe.safetensors.index.json");
    std::fs::write(
        &unsafe_index,
        r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .map_err(|source| crate::Error::io("test could not write shard index", source))?;
    let unsafe_error = Checkpoint::open_index(unsafe_index)
        .expect_err("parent traversal in a shard index must fail");
    assert_eq!(unsafe_error.category(), ErrorCategory::InvalidPath);
    assert_eq!(
        unsafe_error.message(),
        "repository path must be a safe slash-separated relative path"
    );

    let truncated = write_safetensors(
        &directory,
        "truncated.safetensors",
        r#"{"weight":{"dtype":"U32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0_u8; 4],
    )?;
    let truncated_error =
        Checkpoint::open(truncated).expect_err("truncated tensor payload must fail");
    assert_eq!(truncated_error.category(), ErrorCategory::InvalidFormat);
    assert_eq!(
        truncated_error.message(),
        "safetensors tensor offset lies outside its data section"
    );
    Ok(())
}

#[test]
fn explicit_mapping_rejects_an_unretained_local_file() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "weights.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[7],
    )?;
    let source = SourceDescriptor::local(path)?;
    assert_eq!(
        CheckpointBuilder::new(source)
            .access_mode(AccessMode::Mmap)
            .open()
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Unsupported)
    );
    Ok(())
}

#[test]
fn trusted_local_digest_skips_rehash_without_enabling_mapping() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "trusted.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[13],
    )?;
    let size = std::fs::metadata(&path)
        .map_err(|source| crate::Error::io("test could not inspect checkpoint", source))?
        .len();
    let upstream_digest = ContentDigest::hash("upstream-test-digest-v1", [b"trusted"]);
    let source = SourceDescriptor::local_with_trusted_digest(&path, size, upstream_digest)?;

    assert_eq!(
        source.digest_policy(),
        DigestPolicy::TrustExternal(upstream_digest)
    );
    assert_eq!(source.kind(), SourceKind::Local);
    let checkpoint = Checkpoint::open_source(source.clone())?;
    assert_eq!(
        checkpoint.inventory().files()[0].digest_state(),
        DigestState::Trusted(upstream_digest)
    );
    assert_eq!(
        checkpoint
            .source_digests(&CancellationToken::new())?
            .as_ref(),
        &[upstream_digest]
    );
    assert_eq!(checkpoint.tensor("x")?.bytes().as_slice(), &[13]);
    assert_eq!(
        CheckpointBuilder::new(source)
            .access_mode(AccessMode::Mmap)
            .open()
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Unsupported)
    );
    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
#[expect(
    unsafe_code,
    reason = "the temporary checkpoint remains immutable while the retained test view exists"
)]
fn mapped_tensor_view_retains_the_snapshot_guard() -> Result<()> {
    #[derive(Clone)]
    struct Guard(Arc<AtomicUsize>);

    impl fmt::Debug for Guard {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("Guard")
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "weights.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[9],
    )?;
    let file_bytes = std::fs::read(&path)
        .map_err(|source| crate::Error::io("test could not read checkpoint", source))?;
    let digest = ContentDigest::from_bytes(Sha256::digest(file_bytes).into());
    let size = std::fs::metadata(&path)
        .map_err(|source| crate::Error::io("test could not inspect checkpoint", source))?
        .len();
    let drops = Arc::new(AtomicUsize::new(0));
    // SAFETY: the temporary file remains alive and unmodified through every
    // checkpoint and mapped-view lifetime in this test.
    let source = unsafe {
        SourceDescriptor::retained(
            "weights.safetensors",
            &path,
            size,
            digest,
            Guard(Arc::clone(&drops)),
        )?
    };
    let checkpoint = Checkpoint::open_source(source)?;
    let view = checkpoint.tensor("x")?.bytes().clone();
    drop(checkpoint);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(view.as_slice(), &[9]);
    drop(view);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    Ok(())
}

#[cfg(not(feature = "mmap"))]
#[test]
#[expect(
    unsafe_code,
    reason = "the temporary checkpoint remains immutable for the descriptor lifetime"
)]
fn automatic_access_reads_retained_sources_when_mapping_is_disabled() -> Result<()> {
    let directory = TempDir::new()
        .map_err(|source| crate::Error::io("test could not create directory", source))?;
    let path = write_safetensors(
        &directory,
        "weights.safetensors",
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[11],
    )?;
    let file_bytes = std::fs::read(&path)
        .map_err(|source| crate::Error::io("test could not read checkpoint", source))?;
    let digest = ContentDigest::from_bytes(Sha256::digest(file_bytes).into());
    let size = std::fs::metadata(&path)
        .map_err(|source| crate::Error::io("test could not inspect checkpoint", source))?
        .len();
    // SAFETY: the temporary file remains alive and unmodified until the
    // checkpoint is dropped.
    let source =
        unsafe { SourceDescriptor::retained("weights.safetensors", &path, size, digest, ())? };
    let checkpoint = Checkpoint::open_source(source)?;

    assert_eq!(checkpoint.tensor("x")?.bytes().as_slice(), &[11]);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn bounded_parser_round_trips_small_plain_tensor_headers(
        shape in prop::collection::vec(0_u64..8, 0..4),
        dtype in prop_oneof![
            Just(("U8", 1_u64)),
            Just(("I16", 2_u64)),
            Just(("F32", 4_u64)),
        ],
    ) {
        let elements = shape.iter().copied().product::<u64>();
        let byte_len = elements * dtype.1;
        let payload_len = usize::try_from(byte_len)
            .map_err(|_error| TestCaseError::fail("generated payload does not fit usize"))?;
        let header = serde_json::json!({
            "tensor": {
                "dtype": dtype.0,
                "shape": &shape,
                "data_offsets": [0, byte_len],
            }
        })
        .to_string();
        let directory = TempDir::new()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let payload = vec![0x5a; payload_len];
        let path = write_safetensors(
            &directory,
            "property.safetensors",
            &header,
            &payload,
        )
        .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let checkpoint = Checkpoint::open(path)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let tensor = checkpoint
            .tensor("tensor")
            .map_err(|error| TestCaseError::fail(error.to_string()))?;

        prop_assert_eq!(tensor.shape(), shape.as_slice());
        prop_assert_eq!(tensor.bytes().as_slice(), payload.as_slice());
    }
}
