use std::fs::File;

use crate::checkpoint::read_exact_at;
use crate::tensor::{ByteView, DType, SourceSpan};
use crate::{CancellationToken, Error, Result};

use super::container::Range;
use super::special::{Quantized, Sparse};
use super::zip_data::{Backing, Opened};
use super::{invalid, multiply, scalar_bytes};

const BLOCK_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) struct TensorView(Projection);

#[derive(Debug)]
enum Projection {
    Dense(DenseView),
    Sparse(Box<Sparse>),
    Quantized(Box<Quantized>),
}

impl TensorView {
    pub(super) fn dense(view: DenseView) -> Self {
        Self(Projection::Dense(view))
    }

    pub(super) fn sparse(view: Sparse) -> Self {
        Self(Projection::Sparse(Box::new(view)))
    }

    pub(super) fn quantized(view: Quantized) -> Self {
        Self(Projection::Quantized(Box::new(view)))
    }

    pub(crate) fn len(&self) -> u64 {
        match &self.0 {
            Projection::Dense(view) => view.length,
            Projection::Sparse(view) => view.length,
            Projection::Quantized(view) => view.length,
        }
    }

    pub(crate) fn read(
        &self,
        file: &File,
        span: SourceSpan,
        cancellation: &CancellationToken,
    ) -> Result<ByteView> {
        if span.end() > self.len() {
            return Err(invalid("tensor view span exceeds its logical length"));
        }
        if span.is_empty() {
            return Ok(ByteView::from_boxed(Box::new([])));
        }
        let width = match &self.0 {
            Projection::Dense(view) => view.width,
            Projection::Sparse(view) => scalar_bytes(view.dtype),
            Projection::Quantized(_) => 4,
        };
        let start = span.offset() / width;
        let end = (span.end() - 1) / width + 1;
        let length = usize::try_from(multiply(end - start, width)?)
            .map_err(|_error| Error::limit("normalized tensor length does not fit usize"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_error| Error::limit("could not allocate normalized tensor bytes"))?;
        bytes.resize(length, 0);
        match &self.0 {
            Projection::Dense(view) => view.read(file, start, &mut bytes, cancellation)?,
            Projection::Sparse(view) => view.read(file, start, &mut bytes, cancellation)?,
            Projection::Quantized(view) => view.read(file, start, &mut bytes, cancellation)?,
        }
        let first = usize::try_from(span.offset() % width)
            .map_err(|_error| invalid("invalid tensor offset"))?;
        let last = first
            + usize::try_from(span.len()).map_err(|_error| invalid("invalid tensor length"))?;
        ByteView::from_boxed(bytes.into_boxed_slice()).slice(first..last)
    }
}

#[derive(Debug)]
pub(super) struct DenseView {
    pub(super) storage: Backing,
    pub(super) dtype: DType,
    pub(super) shape: Box<[u64]>,
    first_byte: u64,
    length: u64,
    width: u64,
    endian_unit: usize,
    strides: Box<[u64]>,
}

impl DenseView {
    pub(super) fn new(
        storage: Backing,
        dtype: DType,
        big_endian: bool,
        offset: u64,
        shape: Vec<u64>,
        strides: &[u64],
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(invalid("PyTorch tensor shape and stride rank differ"));
        }
        let width = scalar_bytes(dtype);
        let length = dtype.byte_len(&shape)?;
        let first_byte = multiply(offset, width)?;
        let endian_unit = if big_endian && width > 1 {
            if dtype == DType::C64 {
                4
            } else {
                usize::try_from(width).map_err(|_error| invalid("invalid PyTorch scalar size"))?
            }
        } else {
            0
        };
        if length != 0 {
            let last = shape
                .iter()
                .zip(strides)
                .try_fold(0_u64, |total, (&dim, &stride)| {
                    total
                        .checked_add(multiply(dim - 1, stride)?)
                        .ok_or_else(|| Error::limit("PyTorch tensor stride extent overflows"))
                })?;
            let extent = last
                .checked_add(1)
                .ok_or_else(|| Error::limit("PyTorch tensor extent overflows"))?;
            if first_byte > storage.len() || multiply(width, extent)? > storage.len() - first_byte {
                return Err(invalid("PyTorch tensor extent exceeds its storage"));
            }
        }
        Ok(Self {
            storage,
            dtype,
            shape: shape.into_boxed_slice(),
            first_byte,
            length,
            width,
            endian_unit,
            strides: strides.into(),
        })
    }

    pub(super) fn direct(&self) -> Result<Option<Range>> {
        let Backing::Stored(storage) = self.storage else {
            return Ok(None);
        };
        if self.length == 0 {
            return storage.sub(self.first_byte.min(storage.len), 0).map(Some);
        }
        if self.endian_unit == 0 && super::contiguous(&self.shape, &self.strides)? {
            return storage.sub(self.first_byte, self.length).map(Some);
        }
        Ok(None)
    }

    pub(super) fn elements(&self) -> u64 {
        self.length / self.width
    }

    pub(super) fn open<'a>(
        &'a self,
        file: &'a File,
        cancellation: &CancellationToken,
    ) -> Result<DenseRead<'a>> {
        Ok(DenseRead {
            view: self,
            storage: self.storage.open(file, cancellation)?,
            page: Page::new(self.storage.len()),
        })
    }

    fn read(
        &self,
        file: &File,
        start: u64,
        output: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        if self.endian_unit == 0 && super::contiguous(&self.shape, &self.strides)? {
            let opened = self.storage.open(file, cancellation)?;
            let (file, storage) = opened.source();
            let range = storage.sub(self.first_byte + start * self.width, output.len() as u64)?;
            let mut offset = range.start;
            for block in output.chunks_mut(BLOCK_BYTES) {
                cancellation.check()?;
                read_exact_at(file, offset, block)?;
                offset += block.len() as u64;
            }
            return Ok(());
        }
        let mut reader = self.open(file, cancellation)?;
        let width =
            usize::try_from(self.width).map_err(|_error| invalid("invalid tensor scalar size"))?;
        for (index, bytes) in output.chunks_exact_mut(width).enumerate() {
            if index % 1024 == 0 {
                cancellation.check()?;
            }
            let value = reader.scalar(start + index as u64)?;
            bytes.copy_from_slice(&value[..width]);
        }
        Ok(())
    }

    fn physical_offset(&self, mut linear: u64) -> u64 {
        let mut offset = 0;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            offset += (linear % dimension) * stride;
            linear /= dimension;
        }
        self.first_byte + offset * self.width
    }
}

pub(super) struct DenseRead<'a> {
    view: &'a DenseView,
    storage: Opened<'a>,
    page: Page,
}

impl DenseRead<'_> {
    pub(super) fn scalar(&mut self, index: u64) -> Result<[u8; 8]> {
        if index >= self.view.elements() {
            return Err(invalid("PyTorch tensor element is out of bounds"));
        }
        let mut bytes = [0; 8];
        let width = usize::try_from(self.view.width)
            .map_err(|_error| invalid("invalid tensor scalar size"))?;
        let (file, storage) = self.storage.source();
        self.page.read(
            file,
            storage,
            self.view.physical_offset(index),
            &mut bytes[..width],
        )?;
        if self.view.endian_unit != 0 {
            for component in bytes[..width].chunks_exact_mut(self.view.endian_unit) {
                component.reverse();
            }
        }
        Ok(bytes)
    }

    pub(super) fn index(&mut self, index: u64) -> Result<u64> {
        let bytes = self.scalar(index)?;
        let value = if self.view.dtype == DType::I32 {
            i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            i64::from_le_bytes(bytes)
        };
        u64::try_from(value).map_err(|_error| invalid("negative PyTorch sparse index"))
    }
}

struct Page {
    start: u64,
    length: usize,
    bytes: Vec<u8>,
}

impl Page {
    fn new(storage_len: u64) -> Self {
        let size = usize::try_from(storage_len.min(BLOCK_BYTES as u64)).unwrap_or(BLOCK_BYTES);
        Self {
            start: 0,
            length: 0,
            bytes: vec![0; size],
        }
    }

    fn read(
        &mut self,
        file: &File,
        storage: Range,
        mut offset: u64,
        mut output: &mut [u8],
    ) -> Result<()> {
        while !output.is_empty() {
            if offset >= storage.len {
                return Err(invalid("tensor view element exceeds its storage"));
            }
            if offset < self.start || offset - self.start >= self.length as u64 {
                self.start = offset / BLOCK_BYTES as u64 * BLOCK_BYTES as u64;
                self.length =
                    usize::try_from((storage.len - self.start).min(self.bytes.len() as u64))
                        .map_err(|_error| {
                            Error::limit("tensor view block size does not fit usize")
                        })?;
                read_exact_at(
                    file,
                    storage.start + self.start,
                    &mut self.bytes[..self.length],
                )?;
            }
            let at = usize::try_from(offset - self.start)
                .map_err(|_error| invalid("invalid tensor view block offset"))?;
            let take = output.len().min(self.length - at);
            output[..take].copy_from_slice(&self.bytes[at..at + take]);
            output = &mut output[take..];
            offset += take as u64;
        }
        Ok(())
    }
}
