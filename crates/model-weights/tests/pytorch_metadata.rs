//! Metadata reads through local files and seekable range sources.

use std::io::{self, Cursor, Read, Seek, SeekFrom};

use model_weights::limits::ResourceLimits;
use model_weights::metadata::{read_pytorch_field, read_pytorch_field_from_reader};
use model_weights::{CancellationToken, ErrorCategory};

const DEFLATED: &[u8] = include_bytes!("fixtures/pytorch/metadata-deflated.pt");
const LEGACY: &[u8] = include_bytes!("fixtures/pytorch/metadata-legacy.bin");
const TAR: &[u8] = include_bytes!("fixtures/pytorch/legacy.tar");

#[test]
fn metadata_round_trips_data_types_and_distinguishes_missing_from_null()
-> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::new();
    let expected = serde_json::json!({
        "enabled":true, "disabled":false, "steps":5, "large":1_u64 << 63,
        "ratio":0.25, "label":"config", "optional":null, "nested":[[1,2],{"x":-8}],
    });
    for archive in [DEFLATED, LEGACY] {
        let read =
            |field| read_pytorch_field_from_reader(Cursor::new(archive), field, &limits, &cancel);
        assert_eq!(read("example_config")?, Some(expected.clone()));
        assert_eq!(read("nullable")?, Some(serde_json::Value::Null));
        assert_eq!(read("absent")?, None);
        assert_eq!(
            read("model").expect_err("tensor reference").category(),
            ErrorCategory::Unsupported
        );
        assert_eq!(
            read("set")
                .expect_err("set is outside JSON subset")
                .category(),
            ErrorCategory::Unsupported
        );
        assert_eq!(
            read("cycle").expect_err("cyclic metadata").category(),
            ErrorCategory::InvalidFormat
        );
        assert_eq!(
            read("deep").expect_err("nesting limit").category(),
            ErrorCategory::ResourceLimit
        );
    }
    let folder = tempfile::tempdir()?;
    let file = folder.path().join("metadata.pt");
    std::fs::write(&file, DEFLATED)?;
    assert_eq!(
        read_pytorch_field(file, "example_config", &limits, &cancel)?,
        Some(expected)
    );
    Ok(())
}

#[test]
fn metadata_amplification_is_bounded_and_reads_are_cancellable()
-> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::builder().max_metadata_bytes(512).build()?;
    let cancel = CancellationToken::new();
    let read =
        || read_pytorch_field_from_reader(Cursor::new(DEFLATED), "repeated", &limits, &cancel);
    assert_eq!(
        read()
            .expect_err("repeated reference must charge each occurrence")
            .category(),
        ErrorCategory::ResourceLimit
    );
    cancel.cancel();
    assert_eq!(
        read().expect_err("pre-cancelled").category(),
        ErrorCategory::Cancelled
    );
    Ok(())
}

// The fixture was repacked with Python zipfile and provides its compressed
// lengths in the local headers. Forbid accesses to every tensor member,
// including its local header, to exercise the remote-friendly read path.
struct MetadataOnly {
    reader: Cursor<&'static [u8]>,
    forbidden: std::ops::Range<u64>,
}

impl Read for MetadataOnly {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let start = self.reader.position();
        let end = start + output.len() as u64;
        if start < self.forbidden.end && end > self.forbidden.start {
            return Err(io::Error::other(
                "tensor member read during metadata access",
            ));
        }
        self.reader.read(output)
    }
}

impl Seek for MetadataOnly {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.reader.seek(position)
    }
}

#[test]
fn zip_metadata_does_not_read_tensor_members() -> Result<(), Box<dyn std::error::Error>> {
    let name = b"archive/data/0";
    let name_start = DEFLATED
        .windows(name.len())
        .position(|window| window == name)
        .expect("fixture tensor storage name");
    let header = name_start - 30;
    assert_eq!(&DEFLATED[header..header + 4], b"PK\x03\x04");
    let compressed = u32::from_le_bytes(DEFLATED[header + 18..header + 22].try_into()?);
    let extra = u16::from_le_bytes(DEFLATED[header + 28..header + 30].try_into()?);
    let end = (name_start + name.len()) as u64 + u64::from(extra) + u64::from(compressed);
    let value = read_pytorch_field_from_reader(
        MetadataOnly {
            reader: Cursor::new(DEFLATED),
            forbidden: header as u64..end,
        },
        "example_config",
        &ResourceLimits::default(),
        &CancellationToken::new(),
    )?
    .expect("config present");
    assert_eq!(value["enabled"], true);
    Ok(())
}

#[test]
fn historical_tar_accessor_handles_missing_fields_without_reading_storage()
-> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::new();
    let read = |field| read_pytorch_field_from_reader(Cursor::new(TAR), field, &limits, &cancel);
    assert_eq!(read("absent")?, None);
    assert_eq!(
        read("full")
            .expect_err("historical tensor reference")
            .category(),
        ErrorCategory::Unsupported
    );
    Ok(())
}
