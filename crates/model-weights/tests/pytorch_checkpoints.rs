//! The ZIP64 and legacy stream fixtures were made with `PyTorch` using one
//! 4 × 6 arange storage. The ZIP32 fixture was repacked with big-endian
//! payload bytes. The historical tar models `PyTorch`'s legacy tar records
//! and was cross-checked using `PyTorch`'s own tar loader.

use std::fs;
use std::path::{Path, PathBuf};

use model_weights::limits::ResourceLimits;
use model_weights::plan::{SourceTensor, TensorName};
use model_weights::source::SourceDescriptor;
use model_weights::tensor::DType;
use model_weights::{
    CancellationToken, Checkpoint, CheckpointFormat, ErrorCategory, Result, TensorData,
};

const ZIP64: &[u8] = include_bytes!("fixtures/pytorch/zip64.pt");
const ZIP32: &[u8] = include_bytes!("fixtures/pytorch/zip32.ckpt");
const LEGACY: &[u8] = include_bytes!("fixtures/pytorch/legacy.bin");
const TAR: &[u8] = include_bytes!("fixtures/pytorch/legacy.tar");

fn write(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
    let path = dir.join(name);
    fs::write(&path, bytes)?;
    Ok(path)
}

fn bytes(values: &[u8]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32::from(value).to_le_bytes())
        .collect()
}

fn check_tensor(checkpoint: &Checkpoint, name: &str, shape: &[u64], values: &[u8]) -> Result<()> {
    let data = checkpoint.tensor(name)?;
    let TensorData::Plain(tensor) = data else {
        panic!("expected plain tensor {name}");
    };
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.bytes().as_slice(), bytes(values));
    Ok(())
}

#[test]
fn zip64_zip32_and_legacy_stream_deliver_contiguous_and_strided_tensors()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    for (filename, fixture) in [
        ("model.pt", ZIP64),
        ("model.ckpt", ZIP32),
        ("pytorch_model.bin", LEGACY),
    ] {
        let checkpoint = Checkpoint::open(write(dir.path(), filename, fixture)?)?;
        assert_eq!(checkpoint.format(), CheckpointFormat::PyTorch);
        assert_eq!(checkpoint.inventory().len(), 5);
        check_tensor(&checkpoint, "full", &[4, 6], &(0..24).collect::<Vec<_>>())?;
        check_tensor(&checkpoint, "offset", &[2, 6], &(6..18).collect::<Vec<_>>())?;
        check_tensor(
            &checkpoint,
            "stride",
            &[4, 3],
            &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        )?;
        check_tensor(&checkpoint, "broadcast", &[100, 100], &[0; 10000])?;
        let tensor = checkpoint
            .inventory()
            .tensor("transpose")
            .expect("transpose inventory");
        let source = SourceTensor::new(
            TensorName::parse("transpose")?,
            tensor.shape().to_vec(),
            tensor.storage().clone(),
        )?;
        let slice = source.slice_rows(TensorName::parse("transposed_rows")?, 1, 3)?;
        let serialized = serde_json::to_vec(slice.storage())?;
        let storage: model_weights::quantization::Storage = serde_json::from_slice(&serialized)?;
        assert_eq!(
            checkpoint.read_span(storage.span())?.as_slice(),
            bytes(&[1, 7, 13, 19, 2, 8, 14, 20])
        );
    }
    Ok(())
}

#[test]
fn historical_tar_resolves_persistent_tensor_ids_and_storage_views()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let checkpoint = Checkpoint::open(write(dir.path(), "checkpoint.pt", TAR)?)?;
    assert_eq!(checkpoint.format(), CheckpointFormat::PyTorch);
    assert_eq!(checkpoint.inventory().len(), 2);
    check_tensor(
        &checkpoint,
        "full",
        &[6, 4],
        &[
            0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23,
        ],
    )?;
    check_tensor(&checkpoint, "offset", &[2, 6], &(6..18).collect::<Vec<_>>())?;
    Ok(())
}

#[test]
fn pytorch_metadata_limits_distinguish_resource_errors_and_cancellation()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    for (name, fixture) in [("modern", ZIP64), ("old", LEGACY), ("oldest", TAR)] {
        let path = write(dir.path(), name, fixture)?;
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert_eq!(
            Checkpoint::open_with_cancellation(&path, &cancellation)
                .expect_err("cancelled")
                .category(),
            ErrorCategory::Cancelled
        );
        assert_eq!(
            Checkpoint::builder(SourceDescriptor::local(path)?)
                .limits(ResourceLimits::builder().max_header_bytes(32).build()?)
                .open()
                .expect_err("resource limit")
                .category(),
            ErrorCategory::ResourceLimit
        );
    }
    Ok(())
}

#[test]
fn pytorch_rejects_truncated_archives() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    for (name, fixture, length) in [
        ("zip", ZIP64, ZIP64.len() - 15),
        ("legacy", LEGACY, LEGACY.len() - 1),
        ("tar", TAR, 512),
    ] {
        Checkpoint::open(write(dir.path(), name, &fixture[..length])?)
            .expect_err("truncated container");
    }
    Ok(())
}

#[test]
fn weight_map_loads_a_zip_shard_with_a_legacy_stream_shard()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    write(dir.path(), "pytorch_model-00001-of-00002.bin", ZIP64)?;
    write(
        dir.path(),
        "pytorch_model-00002-of-00002.bin",
        include_bytes!("fixtures/pytorch/standalone.pt"),
    )?;
    let index = write(
        dir.path(),
        "pytorch_model.bin.index.json",
        br#"{
        "weight_map": {
            "broadcast": "pytorch_model-00001-of-00002.bin",
            "full": "pytorch_model-00001-of-00002.bin",
            "offset": "pytorch_model-00001-of-00002.bin",
            "stride": "pytorch_model-00001-of-00002.bin",
            "transpose": "pytorch_model-00001-of-00002.bin",
            "tensor": "pytorch_model-00002-of-00002.bin"
        }
    }"#,
    )?;
    let checkpoint = Checkpoint::open(&index)?;
    assert_eq!(checkpoint.format(), CheckpointFormat::PyTorch);
    assert_eq!(checkpoint.inventory().files().len(), 2);
    check_tensor(&checkpoint, "tensor", &[2], &[4, 9])?;
    check_tensor(&checkpoint, "offset", &[2, 6], &(6..18).collect::<Vec<_>>())?;
    Ok(())
}

#[derive(serde::Deserialize)]
struct TorchReference {
    dtype: String,
    shape: Vec<u64>,
    bytes: String,
}

const SPECIAL_DEFLATE: &[u8] = include_bytes!("fixtures/pytorch/special-deflated.pt");
const SPECIAL_LEGACY: &[u8] = include_bytes!("fixtures/pytorch/special-legacy.bin");
const SPECIAL_REFERENCE: &str = include_str!("fixtures/pytorch/special-reference.json");

#[test]
fn sparse_quantized_and_deflated_weights_match_pytorch_cpu()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let references: std::collections::BTreeMap<String, TorchReference> =
        serde_json::from_str(SPECIAL_REFERENCE)?;
    let dir = tempfile::tempdir()?;
    for (name, fixture) in [
        ("special.pt", SPECIAL_DEFLATE),
        ("special.bin", SPECIAL_LEGACY),
    ] {
        let checkpoint = Checkpoint::open(write(dir.path(), name, fixture)?)?;
        assert_eq!(checkpoint.inventory().len(), references.len());
        for (name, reference) in &references {
            let TensorData::Plain(tensor) = checkpoint.tensor(name)? else {
                panic!("expected normalized tensor {name}");
            };
            let expected_dtype = match reference.dtype.as_str() {
                "float32" => DType::F32,
                "float16" => DType::F16,
                "int32" => DType::I32,
                "complex64" => DType::C64,
                "bool" => DType::Bool,
                dtype => panic!("unexpected fixture dtype {dtype}"),
            };
            assert_eq!(tensor.dtype(), expected_dtype, "{name}");
            assert_eq!(tensor.shape(), reference.shape, "{name}");
            let expected = (0..reference.bytes.len())
                .step_by(2)
                .map(|offset| u8::from_str_radix(&reference.bytes[offset..offset + 2], 16))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert_eq!(tensor.bytes().as_slice(), expected, "{name}");
            if let [rows, _] = reference.shape.as_slice() {
                let source = checkpoint.inventory().tensor(name).expect("tensor record");
                let source = SourceTensor::new(
                    TensorName::parse(name)?,
                    source.shape().to_vec(),
                    source.storage().clone(),
                )?;
                let view =
                    source.slice_rows(TensorName::parse(format!("{name}_rows"))?, 1, *rows)?;
                let row_bytes = expected.len() / usize::try_from(*rows)?;
                assert_eq!(
                    checkpoint.read_span(view.storage().span())?.as_slice(),
                    &expected[row_bytes..],
                    "{name} rows"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn deflated_tensor_declared_expansion_is_bounded()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = write(dir.path(), "compressed.pt", SPECIAL_DEFLATE)?;
    let error = Checkpoint::builder(SourceDescriptor::local(path)?)
        .limits(ResourceLimits::builder().max_zip_member_bytes(16).build()?)
        .open()
        .expect_err("expanded tensor member should exceed configured limit");
    assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    Ok(())
}

#[test]
fn deflated_storage_crc_is_checked_on_read() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let mut corrupted = SPECIAL_DEFLATE.to_vec();
    let mut modified = 0;
    for at in 0..corrupted.len() - 46 {
        if &corrupted[at..at + 4] == b"PK\x01\x02"
            && corrupted[at + 46..].starts_with(b"archive/data/")
        {
            corrupted[at + 16] ^= 1;
            modified += 1;
        }
    }
    assert!(modified > 0);
    let dir = tempfile::tempdir()?;
    let checkpoint = Checkpoint::open(write(dir.path(), "corrupt.pt", &corrupted)?)?;
    assert_eq!(
        checkpoint
            .tensor("strided")
            .expect_err("bad storage CRC")
            .category(),
        ErrorCategory::Integrity
    );
    Ok(())
}

#[test]
fn sparse_dense_reads_have_an_independent_output_limit()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = write(dir.path(), "sparse.bin", SPECIAL_LEGACY)?;
    let checkpoint = Checkpoint::builder(SourceDescriptor::local(path)?)
        .limits(
            ResourceLimits::builder()
                .max_pytorch_read_bytes(24)
                .build()?,
        )
        .open()?;
    assert_eq!(
        checkpoint
            .tensor("csr")
            .expect_err("bounded dense allocation")
            .category(),
        ErrorCategory::ResourceLimit
    );
    let source = checkpoint.inventory().tensor("csr").expect("CSR tensor");
    let source = SourceTensor::new(
        TensorName::parse("csr")?,
        source.shape().to_vec(),
        source.storage().clone(),
    )?;
    let row = source.slice_rows(TensorName::parse("csr_row")?, 0, 1)?;
    assert_eq!(checkpoint.read_span(row.storage().span())?.len(), 16);
    Ok(())
}

#[test]
fn sparse_coordinates_are_validated_before_scattering()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = write(
        dir.path(),
        "out-of-bounds.pt",
        include_bytes!("fixtures/pytorch/invalid-coo.pt"),
    )?;
    let checkpoint = Checkpoint::open(path)?;
    assert_eq!(
        checkpoint
            .tensor("out_of_bounds")
            .expect_err("COO coordinate outside its shape")
            .category(),
        ErrorCategory::InvalidFormat
    );
    Ok(())
}
