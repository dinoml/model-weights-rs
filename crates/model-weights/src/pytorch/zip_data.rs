use std::fs::File;
use std::io::{BufRead, Write};

use flate2::{Decompress, FlushDecompress, Status};

use crate::{CancellationToken, Error, Result};

use super::container::{Range, Source, ZipEntry};

#[derive(Debug, Clone, Copy)]
pub(super) enum Backing {
    Stored(Range),
    Deflate {
        compressed: Range,
        size: u64,
        crc: u32,
    },
}

impl Backing {
    pub(super) fn zip(entry: &ZipEntry, limit: u64) -> Result<Self> {
        match entry.method {
            0 => Ok(Self::Stored(entry.range)),
            8 if entry.size <= limit => Ok(Self::Deflate {
                compressed: entry.range,
                size: entry.size,
                crc: entry.crc,
            }),
            8 => Err(Error::limit(
                "expanded PyTorch ZIP member exceeds its limit",
            )),
            method => Err(Error::unsupported(format!(
                "unsupported PyTorch ZIP compression method {method}"
            ))),
        }
    }

    pub(super) fn len(self) -> u64 {
        match self {
            Self::Stored(range) => range.len,
            Self::Deflate { size, .. } => size,
        }
    }

    pub(super) fn open<'a>(
        self,
        file: &'a File,
        cancellation: &CancellationToken,
    ) -> Result<Opened<'a>> {
        if let Self::Stored(range) = self {
            return Ok(Opened::Stored(file, range));
        }
        let mut temp = tempfile::tempfile()
            .map_err(|error| Error::io("creating PyTorch ZIP scratch file", error))?;
        let source = Source::new(
            file,
            match self {
                Self::Stored(range) => range.end(),
                Self::Deflate { compressed, .. } => compressed.end(),
            },
        );
        self.inflate(&source, &mut temp, cancellation)?;
        Ok(Opened::Expanded(temp, self.len()))
    }

    pub(super) fn inflate(
        self,
        source: &Source<'_>,
        output: &mut impl Write,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let Self::Deflate {
            compressed,
            size,
            crc,
        } = self
        else {
            return Err(Error::invalid("expected a DEFLATE member"));
        };
        let mut input = source.reader(compressed);
        let mut decoder = Decompress::new(false);
        let mut hash = crc32fast::Hasher::new();
        let mut block = [0; 16 * 1024];
        let mut remaining = size;
        loop {
            cancellation.check()?;
            let available = input
                .fill_buf()
                .map_err(|error| Error::io("reading compressed PyTorch ZIP member", error))?;
            let max = block
                .len()
                .min(usize::try_from(remaining.saturating_add(1)).unwrap_or(usize::MAX));
            let before = (decoder.total_in(), decoder.total_out());
            let status = decoder
                .decompress(available, &mut block[..max], FlushDecompress::None)
                .map_err(|error| {
                    Error::with_source(
                        crate::ErrorCategory::InvalidFormat,
                        "invalid PyTorch ZIP DEFLATE stream",
                        error,
                    )
                })?;
            let used = usize::try_from(decoder.total_in() - before.0)
                .map_err(|_error| Error::invalid("PyTorch ZIP input count overflow"))?;
            let count = usize::try_from(decoder.total_out() - before.1)
                .map_err(|_error| Error::invalid("PyTorch ZIP output count overflow"))?;
            input.consume(used);
            remaining = remaining.checked_sub(count as u64).ok_or_else(|| {
                Error::invalid("PyTorch ZIP member expands beyond its declared length")
            })?;
            hash.update(&block[..count]);
            output
                .write_all(&block[..count])
                .map_err(|error| Error::io("writing expanded PyTorch ZIP member", error))?;
            if status == Status::StreamEnd {
                break;
            }
            if used == 0 && count == 0 {
                return Err(Error::invalid("truncated PyTorch ZIP DEFLATE stream"));
            }
        }
        if remaining != 0 || decoder.total_in() != compressed.len || hash.finalize() != crc {
            return Err(Error::integrity(
                "PyTorch ZIP member length or CRC mismatch",
            ));
        }
        Ok(())
    }
}

pub(super) enum Opened<'a> {
    Stored(&'a File, Range),
    Expanded(File, u64),
}

impl Opened<'_> {
    pub(super) fn source(&self) -> (&File, Range) {
        match self {
            Self::Stored(file, range) => (file, *range),
            Self::Expanded(file, len) => (
                file,
                Range {
                    start: 0,
                    len: *len,
                },
            ),
        }
    }
}
