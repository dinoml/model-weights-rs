use std::fs::File;

use crate::inventory::TensorRecord;
use crate::tensor::DType;
use crate::{CancellationToken, Error, Result, limits::ResourceLimits};

use super::pickle::{Document, Id, Node};
use super::view::{DenseRead, DenseView, TensorView};
use super::zip_data::Backing;
use super::{
    StorageSpec, TensorInventory, contiguous, dimensions, invalid, multiply, scalar_bytes,
    storage_spec, tensor,
};

mod affine;
mod sparse;
pub(super) use affine::Quantized;
pub(super) use sparse::Sparse;

pub(super) fn record<'a>(
    doc: &'a Document,
    mut value: Id,
    name: &str,
    inventory: &mut TensorInventory,
    limits: &ResourceLimits,
    storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
) -> Result<Option<TensorRecord>> {
    for _ in 0..8 {
        let Node::Call(callable, arguments) = doc.nodes[value] else {
            return Ok(None);
        };
        match doc.global(callable) {
            Some(("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state")) => {
                value = *doc
                    .seq(arguments)?
                    .first()
                    .ok_or_else(|| invalid("missing PyTorch parameter tensor"))?;
            }
            Some(("torch._utils", "_rebuild_sparse_tensor")) => {
                let (shape, sparse) =
                    sparse::parse(doc, arguments, limits, inventory.big_endian, storage)?;
                return inventory
                    .project(
                        name,
                        sparse.dtype,
                        shape.into_boxed_slice(),
                        TensorView::sparse(sparse),
                    )
                    .map(Some);
            }
            Some(("torch._utils", "_rebuild_qtensor")) => {
                let (shape, quantized) = affine::parse(doc, arguments, limits, inventory, storage)?;
                return inventory
                    .project(
                        name,
                        DType::F32,
                        shape.into_boxed_slice(),
                        TensorView::quantized(quantized),
                    )
                    .map(Some);
            }
            _ => return Ok(None),
        }
    }
    Err(Error::limit(
        "PyTorch parameter nesting exceeds supported depth",
    ))
}

fn component<'a>(
    doc: &'a Document,
    id: Id,
    limits: &ResourceLimits,
    big_endian: bool,
    storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
) -> Result<DenseView> {
    let tensor = tensor(doc, id, limits)?
        .ok_or_else(|| invalid("expected a dense PyTorch tensor component"))?;
    DenseView::new(
        storage(tensor.storage)?,
        tensor.dtype,
        big_endian,
        tensor.offset,
        tensor.shape,
        &tensor.strides,
    )
}

fn elements(shape: &[u64]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1, |count, &dimension| multiply(count, dimension))
}

struct SparseOutput<'a> {
    bytes: &'a mut [u8],
    start: u64,
    width: usize,
    dtype: DType,
    cancel: &'a CancellationToken,
}

impl SparseOutput<'_> {
    fn add(
        &mut self,
        target: u64,
        source: u64,
        count: u64,
        values: &mut DenseRead<'_>,
    ) -> Result<()> {
        let end = self.start + (self.bytes.len() / self.width) as u64;
        let first = target.max(self.start);
        let last = (target + count).min(end);
        for index in first..last {
            if (index - first).is_multiple_of(1024) {
                self.cancel.check()?;
            }
            let output_index = usize::try_from(index - self.start)
                .map_err(|_error| invalid("invalid sparse output offset"))?
                * self.width;
            let scalar = values.scalar(source + index - target)?;
            sparse::add(
                self.dtype,
                &mut self.bytes[output_index..output_index + self.width],
                scalar,
            );
        }
        Ok(())
    }
}
