use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

use crate::checkpoint::read_exact_at;
use crate::limits::ResourceLimits;
use crate::{CancellationToken, Error, Result};

use super::pickle::Budget;

#[derive(Debug, Clone, Copy)]
pub(super) struct Range {
    pub start: u64,
    pub len: u64,
}

impl Range {
    pub(super) fn sub(self, offset: u64, len: u64) -> Result<Self> {
        if offset > self.len || len > self.len - offset {
            return Err(Error::invalid("PyTorch range extends past its container"));
        }
        Ok(Self {
            start: self.start + offset,
            len,
        })
    }

    pub(super) fn end(self) -> u64 {
        self.start + self.len
    }
}

pub(super) trait SeekRead: Read + Seek {}
impl<T: Read + Seek + ?Sized> SeekRead for T {}

#[derive(Clone, Copy)]
enum Input<'a> {
    File(&'a File),
    Reader(&'a RefCell<dyn SeekRead + 'a>),
}

impl Input<'_> {
    fn read(self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        match self {
            Self::File(file) => read_exact_at(file, offset, bytes),
            Self::Reader(reader) => {
                let mut reader = reader.borrow_mut();
                reader
                    .seek(SeekFrom::Start(offset))
                    .and_then(|_| reader.read_exact(bytes))
                    .map_err(|error| Error::io("reading PyTorch checkpoint metadata", error))
            }
        }
    }
}

pub(super) struct Source<'a> {
    input: Input<'a>,
    pub range: Range,
}

impl<'a> Source<'a> {
    pub(super) fn new(file: &'a File, size: u64) -> Self {
        Self {
            input: Input::File(file),
            range: Range {
                start: 0,
                len: size,
            },
        }
    }

    pub(super) fn from_reader(reader: &'a RefCell<dyn SeekRead + 'a>, size: u64) -> Self {
        Self {
            input: Input::Reader(reader),
            range: Range {
                start: 0,
                len: size,
            },
        }
    }

    pub(super) fn array<const N: usize>(&self, offset: u64) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read(self.range.sub(offset, N as u64)?, &mut bytes)?;
        Ok(bytes)
    }

    pub(super) fn bytes(&self, range: Range) -> Result<Vec<u8>> {
        let size = usize::try_from(range.len)
            .map_err(|_error| Error::limit("PyTorch metadata is too large"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_error| Error::limit("could not allocate PyTorch metadata"))?;
        bytes.resize(size, 0);
        self.read(range, &mut bytes)?;
        Ok(bytes)
    }

    fn read(&self, range: Range, buffer: &mut [u8]) -> Result<()> {
        self.range.sub(range.start, range.len)?;
        self.input.read(range.start, buffer)
    }

    pub(super) fn reader(&self, range: Range) -> BufReader<Positional<'a>> {
        BufReader::with_capacity(
            4096,
            Positional {
                input: self.input,
                remaining: range,
            },
        )
    }
}

pub(super) struct Positional<'a> {
    input: Input<'a>,
    remaining: Range,
}

impl Positional<'_> {
    pub(super) fn position(&self) -> u64 {
        self.remaining.start
    }
}

impl Read for Positional<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = self.remaining.len.min(bytes.len() as u64);
        let size = usize::try_from(count).map_err(io::Error::other)?;
        if size != 0 {
            self.input
                .read(self.remaining.start, &mut bytes[..size])
                .map_err(io::Error::other)?;
            self.remaining.start += count;
            self.remaining.len -= count;
        }
        Ok(size)
    }
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from(u16_at(bytes, at)) | (u32::from(u16_at(bytes, at + 2)) << 16)
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from(u32_at(bytes, at)) | (u64::from(u32_at(bytes, at + 4)) << 32)
}

fn bad_zip() -> Error {
    Error::invalid("malformed PyTorch ZIP archive")
}

fn zip_directory(source: &Source<'_>) -> Result<(u64, Range)> {
    if source.range.len < 22 {
        return Err(bad_zip());
    }
    let last = source.range.len - 22;
    let candidate = source.array::<22>(last)?;
    let tail;
    let (end_at, footer) = if candidate.starts_with(b"PK\x05\x06") && u16_at(&candidate, 20) == 0 {
        (last, candidate.as_slice())
    } else {
        let tail_size = source.range.len.min(65_557);
        let tail_start = source.range.len - tail_size;
        tail = source.bytes(source.range.sub(tail_start, tail_size)?)?;
        let end = (0..=tail.len() - 22)
            .rev()
            .find(|&at| {
                tail[at..].starts_with(b"PK\x05\x06")
                    && usize::from(u16_at(&tail, at + 20)) == tail.len() - at - 22
            })
            .ok_or_else(bad_zip)?;
        (tail_start + end as u64, &tail[end..])
    };
    if u16_at(footer, 4) != 0 || u16_at(footer, 6) != 0 || u16_at(footer, 8) != u16_at(footer, 10) {
        return Err(Error::unsupported("multi-disk PyTorch ZIP archive"));
    }
    let locator = end_at
        .checked_sub(20)
        .map(|at| source.array::<20>(at))
        .transpose()?;
    if let Some(locator) = locator.filter(|bytes| bytes.starts_with(b"PK\x06\x07")) {
        let offset = u64_at(&locator, 8);
        let record = source.array::<56>(offset)?;
        let size = u64_at(&record, 4);
        if !record.starts_with(b"PK\x06\x06")
            || size < 44
            || offset
                .checked_add(12)
                .and_then(|start| start.checked_add(size))
                != Some(end_at - 20)
            || u32_at(&locator, 4) != 0
            || u32_at(&locator, 16) != 1
            || u32_at(&record, 16) != 0
            || u32_at(&record, 20) != 0
            || u64_at(&record, 24) != u64_at(&record, 32)
        {
            return Err(bad_zip());
        }
        let directory = source.range.sub(u64_at(&record, 48), u64_at(&record, 40))?;
        if directory.end() > offset {
            return Err(bad_zip());
        }
        return Ok((u64_at(&record, 32), directory));
    }
    let count = u16_at(footer, 10);
    let size = u32_at(footer, 12);
    let offset = u32_at(footer, 16);
    if count == u16::MAX || size == u32::MAX || offset == u32::MAX {
        return Err(bad_zip());
    }
    let directory = source.range.sub(u64::from(offset), u64::from(size))?;
    if directory.end() > end_at {
        return Err(bad_zip());
    }
    Ok((u64::from(count), directory))
}

fn zip64(extra: &[u8]) -> Result<&[u8]> {
    let mut fields = extra;
    while fields.len() >= 4 {
        let len = usize::from(u16_at(fields, 2));
        let value = fields.get(4..4 + len).ok_or_else(bad_zip)?;
        if u16_at(fields, 0) == 1 {
            return Ok(value);
        }
        fields = &fields[4 + len..];
    }
    Err(bad_zip())
}

fn zip_number(value: u32, extended: &mut &[u8]) -> Result<u64> {
    if value != u32::MAX {
        return Ok(u64::from(value));
    }
    let bytes = extended.get(..8).ok_or_else(bad_zip)?;
    *extended = &extended[8..];
    Ok(u64_at(bytes, 0))
}

pub(super) struct ZipEntry {
    pub range: Range,
    pub method: u16,
    pub size: u64,
    pub crc: u32,
}

#[derive(Debug)]
pub(super) struct CentralEntry {
    local_at: u64,
    directory_start: u64,
    size: u64,
    compressed: u64,
    method: u16,
    flags: u16,
    crc: u32,
}

impl CentralEntry {
    pub(super) fn read(
        &self,
        source: &Source<'_>,
        name: &str,
        budget: &mut Budget<'_>,
    ) -> Result<ZipEntry> {
        let local = source.array::<30>(self.local_at)?;
        if !local.starts_with(b"PK\x03\x04")
            || u16_at(&local, 6) != self.flags
            || u16_at(&local, 8) != self.method
            || usize::from(u16_at(&local, 26)) != name.len()
        {
            return Err(bad_zip());
        }
        let local_name = source.range.sub(self.local_at + 30, name.len() as u64)?;
        budget.charge(local_name.len)?;
        if source.bytes(local_name)? != name.as_bytes() {
            return Err(bad_zip());
        }
        let start = local_name
            .end()
            .checked_add(u64::from(u16_at(&local, 28)))
            .ok_or_else(bad_zip)?;
        let range = source.range.sub(start, self.compressed)?;
        if range.end() > self.directory_start {
            return Err(bad_zip());
        }
        Ok(ZipEntry {
            range,
            method: self.method,
            size: self.size,
            crc: self.crc,
        })
    }
}

pub(super) fn zip_catalog(
    source: &Source<'_>,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
) -> Result<BTreeMap<Box<str>, CentralEntry>> {
    let (count, directory) = zip_directory(source)?;
    budget.charge(directory.len)?;
    if count > directory.len / 46 {
        return Err(bad_zip());
    }
    let directory_start = directory.start;
    let bytes = source.bytes(directory)?;
    let mut remaining = bytes.as_slice();
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        cancellation.check()?;
        let header = remaining.get(..46).ok_or_else(bad_zip)?;
        if !header.starts_with(b"PK\x01\x02") || u16_at(header, 34) != 0 {
            return Err(bad_zip());
        }
        let name_len = usize::from(u16_at(header, 28));
        let extra_len = usize::from(u16_at(header, 30));
        let comment_len = usize::from(u16_at(header, 32));
        if name_len == 0 || name_len > limits.max_name_bytes() {
            return Err(Error::limit("PyTorch ZIP entry name exceeds limits"));
        }
        let name = remaining.get(46..46 + name_len).ok_or_else(bad_zip)?;
        let extra = remaining
            .get(46 + name_len..46 + name_len + extra_len)
            .ok_or_else(bad_zip)?;
        remaining = remaining
            .get(46 + name_len + extra_len + comment_len..)
            .ok_or_else(bad_zip)?;
        let method = u16_at(header, 10);
        let flags = u16_at(header, 8);
        if flags & 1 != 0 {
            return Err(Error::unsupported("encrypted PyTorch ZIP entry"));
        }
        let mut extended = if [20, 24, 42]
            .into_iter()
            .any(|at| u32_at(header, at) == u32::MAX)
        {
            zip64(extra)?
        } else {
            &[]
        };
        let size = zip_number(u32_at(header, 24), &mut extended)?;
        let compressed = zip_number(u32_at(header, 20), &mut extended)?;
        let local_at = zip_number(u32_at(header, 42), &mut extended)?;
        if local_at >= directory_start || (method == 0 && size != compressed) {
            return Err(bad_zip());
        }
        let name = std::str::from_utf8(name).map_err(|_error| bad_zip())?;
        let entry = CentralEntry {
            local_at,
            directory_start,
            size,
            compressed,
            method,
            flags,
            crc: u32_at(header, 16),
        };
        if entries.insert(name.into(), entry).is_some() {
            return Err(bad_zip());
        }
    }
    if !remaining.is_empty() {
        return Err(bad_zip());
    }
    Ok(entries)
}

pub(super) fn zip(
    source: &Source<'_>,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
) -> Result<BTreeMap<Box<str>, ZipEntry>> {
    let catalog = zip_catalog(source, limits, cancellation, budget)?;
    let mut entries = BTreeMap::new();
    let mut ranges = Vec::with_capacity(catalog.len());
    for (name, metadata) in catalog {
        cancellation.check()?;
        let entry = metadata.read(source, &name, budget)?;
        ranges.push((metadata.local_at, entry.range.end()));
        entries.insert(name, entry);
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(bad_zip());
    }
    Ok(entries)
}

fn tar_num(bytes: &[u8]) -> Option<u64> {
    if bytes.first()? & 0x80 != 0 {
        let (&first, rest) = bytes.split_first()?;
        if first != 0x80 {
            return None;
        }
        return rest.iter().try_fold(0_u64, |n, &byte| {
            n.checked_mul(256)?.checked_add(u64::from(byte))
        });
    }
    let text = std::str::from_utf8(bytes).ok()?.trim_matches([' ', '\0']);
    if text.is_empty() {
        Some(0)
    } else {
        u64::from_str_radix(text, 8).ok()
    }
}

pub(super) fn tar_header_valid(header: &[u8; 512]) -> bool {
    let sum = header[..148]
        .iter()
        .chain(&header[156..])
        .map(|&byte| u64::from(byte))
        .sum::<u64>();
    tar_num(&header[148..156]) == Some(sum + 8 * 32)
}

fn tar_text(bytes: &[u8]) -> Result<&str> {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end])
        .map_err(|_error| Error::invalid("invalid PyTorch tar member name"))
}

fn pax_overrides(mut bytes: &[u8]) -> Result<(Option<Box<str>>, Option<u64>)> {
    let mut path = None;
    let mut size = None;
    while !bytes.is_empty() {
        let space = bytes
            .iter()
            .position(|&byte| byte == b' ')
            .ok_or_else(bad_tar)?;
        let length: usize = std::str::from_utf8(&bytes[..space])
            .map_err(|_error| bad_tar())?
            .parse()
            .map_err(|_error| bad_tar())?;
        let record = bytes
            .get(space + 1..length)
            .and_then(|line| line.strip_suffix(b"\n"))
            .ok_or_else(bad_tar)?;
        if let Some(value) = record.strip_prefix(b"path=") {
            path = Some(tar_text(value)?.into());
        } else if let Some(value) = record.strip_prefix(b"size=") {
            size = Some(tar_text(value)?.parse().map_err(|_error| bad_tar())?);
        } else if record.starts_with(b"GNU.sparse.") {
            return Err(Error::unsupported("sparse PyTorch tar member"));
        }
        bytes = &bytes[length..];
    }
    Ok((path, size))
}

fn bad_tar() -> Error {
    Error::invalid("malformed PyTorch tar archive")
}

pub(super) fn tar(
    source: &Source<'_>,
    budget: &mut Budget<'_>,
) -> Result<BTreeMap<Box<str>, Range>> {
    let mut entries = BTreeMap::new();
    let mut at = 0_u64;
    let mut override_path = None;
    let mut override_size = None;
    loop {
        budget.charge(512)?;
        let header = source.array::<512>(at)?;
        if header.iter().all(|&byte| byte == 0) {
            if override_path.is_some() || override_size.is_some() {
                return Err(bad_tar());
            }
            break;
        }
        if !tar_header_valid(&header) {
            return Err(bad_tar());
        }
        let kind = header[156];
        let raw_size = tar_num(&header[124..136]).ok_or_else(bad_tar)?;
        let size = if matches!(kind, b'x' | b'g' | b'L' | b'K') {
            raw_size
        } else {
            override_size.take().unwrap_or(raw_size)
        };
        let data = source.range.sub(at + 512, size)?;
        let padded = size.checked_add(511).ok_or_else(bad_tar)? / 512 * 512;
        at = source.range.sub(data.start, padded)?.end();
        match kind {
            b'x' | b'g' => {
                budget.charge(data.len)?;
                let (path, size) = pax_overrides(&source.bytes(data)?)?;
                if kind == b'g' && (path.is_some() || size.is_some()) {
                    return Err(Error::unsupported("global path or size in PyTorch tar"));
                }
                if path.is_some() {
                    override_path = path;
                }
                if size.is_some() {
                    override_size = size;
                }
            }
            b'L' => {
                budget.charge(data.len)?;
                override_path = Some(
                    tar_text(&source.bytes(data)?)?
                        .trim_end_matches('\n')
                        .into(),
                );
            }
            _ => {
                let prefix = if &header[257..263] == b"ustar\0" {
                    tar_text(&header[345..500])?
                } else {
                    ""
                };
                let name = tar_text(&header[..100])?;
                let name = override_path.take().unwrap_or_else(|| {
                    if prefix.is_empty() {
                        name.into()
                    } else {
                        format!("{prefix}/{name}").into()
                    }
                });
                if matches!(&*name, "storages" | "tensors" | "pickle")
                    && (!matches!(kind, 0 | b'0') || entries.insert(name, data).is_some())
                {
                    return Err(bad_tar());
                }
            }
        }
    }
    Ok(entries)
}
