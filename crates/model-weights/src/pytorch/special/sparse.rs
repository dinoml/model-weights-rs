use super::{
    Backing, CancellationToken, DType, DenseView, Document, Error, File, Id, Node, ResourceLimits,
    Result, SparseOutput, StorageSpec, component, dimensions, elements, invalid, scalar_bytes,
};

#[derive(Debug)]
pub(in crate::pytorch) struct Sparse {
    pub(in crate::pytorch) dtype: DType,
    pub(in crate::pytorch) length: u64,
    values: DenseView,
    layout: Layout,
}

#[derive(Debug)]
enum Layout {
    Coo {
        coordinates: DenseView,
        dimensions: Box<[u64]>,
        nnz: u64,
        inner: u64,
        coalesced: bool,
    },
    Compressed {
        pointers: DenseView,
        indices: DenseView,
        rows: u64,
        cols: u64,
        row_major: bool,
        height: u64,
        width: u64,
        inner: u64,
        batches: u64,
        nnz: u64,
    },
}

pub(super) fn parse<'a>(
    doc: &'a Document,
    args: Id,
    limits: &ResourceLimits,
    big_endian: bool,
    storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
) -> Result<(Vec<u64>, Sparse)> {
    let &[layout, data] = doc.seq(args)? else {
        return Err(invalid("invalid PyTorch sparse tensor arguments"));
    };
    let Node::Call(function, names) = doc.nodes[layout] else {
        return Err(invalid("invalid PyTorch sparse layout"));
    };
    let &[layout_name] = doc.seq(names)? else {
        return Err(invalid("invalid PyTorch sparse layout arguments"));
    };
    if doc.global(function) != Some(("torch.serialization", "_get_layout")) {
        return Err(invalid("invalid PyTorch sparse layout constructor"));
    }
    let name = doc.text(layout_name)?;
    let data = doc.seq(data)?;
    let mut read = |id| component(doc, id, limits, big_endian, storage);
    let (shape, values, layout) = match (name, data) {
        ("torch.sparse_coo", &[coordinates, values, shape, ref tail @ ..]) if tail.len() <= 1 => {
            let shape = dimensions(doc, shape, limits)?;
            let coordinates = read(coordinates)?;
            let values = read(values)?;
            let &[rank, nnz] = coordinates.shape.as_ref() else {
                return Err(invalid("PyTorch COO indices must have rank two"));
            };
            let rank =
                usize::try_from(rank).map_err(|_error| invalid("invalid COO sparse rank"))?;
            if coordinates.dtype != DType::I64
                || rank > shape.len()
                || values.shape.first() != Some(&nnz)
                || values.shape[1..] != shape[rank..]
            {
                return Err(invalid(
                    "PyTorch COO indices or values disagree with the tensor shape",
                ));
            }
            let coalesced = match tail {
                [] => false,
                [id] if matches!(doc.nodes[*id], Node::Bool(true) | Node::Int(1)) => true,
                [id] if matches!(
                    doc.nodes[*id],
                    Node::None | Node::Bool(false) | Node::Int(0)
                ) =>
                {
                    false
                }
                _ => return Err(invalid("invalid PyTorch COO coalesced flag")),
            };

            let layout = Layout::Coo {
                coordinates,
                dimensions: shape[..rank].into(),
                nnz,
                inner: elements(&shape[rank..])?,
                coalesced,
            };
            (shape, values, layout)
        }
        (
            "torch.sparse_csr" | "torch.sparse_csc" | "torch.sparse_bsr" | "torch.sparse_bsc",
            &[pointers, indices, values, shape],
        ) => {
            let shape = dimensions(doc, shape, limits)?;
            let pointers = read(pointers)?;
            let indices = read(indices)?;
            let values = read(values)?;
            let layout = compressed(name, &shape, pointers, indices, &values)?;
            (shape, values, layout)
        }
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported PyTorch sparse layout {name}"
            )));
        }
    };
    let dtype = values.dtype;
    ensure_addable(dtype)?;
    let length = dtype.byte_len(&shape)?;
    Ok((
        shape,
        Sparse {
            dtype,
            length,
            values,
            layout,
        },
    ))
}

fn compressed(
    name: &str,
    shape: &[u64],
    pointers: DenseView,
    indices: DenseView,
    values: &DenseView,
) -> Result<Layout> {
    let Some((&pointer_count, batch)) = pointers.shape.split_last() else {
        return Err(invalid(
            "PyTorch sparse pointers must have at least one dimension",
        ));
    };
    let Some((&nnz, index_batch)) = indices.shape.split_last() else {
        return Err(invalid(
            "PyTorch sparse indices must have at least one dimension",
        ));
    };
    let brank = batch.len();
    let Some([rows, cols, tail @ ..]) = shape.get(brank..) else {
        return Err(invalid(
            "PyTorch compressed sparse tensor requires two sparse dimensions",
        ));
    };
    if batch != &shape[..brank]
        || index_batch != batch
        || !matches!(pointers.dtype, DType::I32 | DType::I64)
        || pointers.dtype != indices.dtype
        || values.shape.get(..brank) != Some(batch)
        || values.shape.get(brank) != Some(&nnz)
    {
        return Err(invalid("PyTorch compressed sparse component mismatch"));
    }
    let blocked = matches!(name, "torch.sparse_bsr" | "torch.sparse_bsc");
    let row_major = matches!(name, "torch.sparse_csr" | "torch.sparse_bsr");
    let rest = &values.shape[brank + 1..];
    let (height, width) = if blocked {
        let [height, width, value_tail @ ..] = rest else {
            return Err(invalid("invalid PyTorch sparse block dimensions"));
        };
        if *height == 0
            || *width == 0
            || value_tail != tail
            || rows % height != 0
            || cols % width != 0
        {
            return Err(invalid(
                "PyTorch sparse block shape does not tile its tensor",
            ));
        }
        (*height, *width)
    } else {
        if rest != tail {
            return Err(invalid(
                "PyTorch sparse values have the wrong dense dimensions",
            ));
        }
        (1, 1)
    };
    let major = if row_major {
        rows / height
    } else {
        cols / width
    };
    if pointer_count.checked_sub(1) != Some(major) {
        return Err(invalid(
            "PyTorch compressed sparse pointer length does not match its shape",
        ));
    }
    let batches = elements(batch)?;
    let inner = elements(tail)?;
    // Bound the intermediate strides even when a dense dimension is zero.
    elements(&[*rows, *cols, inner])?;
    elements(&[height, width, inner])?;
    Ok(Layout::Compressed {
        pointers,
        indices,
        rows: *rows,
        cols: *cols,
        row_major,
        height,
        width,
        inner,
        batches,
        nnz,
    })
}

impl Sparse {
    #[allow(
        clippy::too_many_lines,
        reason = "keeps COO and compressed-layout walks in one dispatch"
    )]
    pub(in crate::pytorch) fn read(
        &self,
        file: &File,
        start: u64,
        output: &mut [u8],
        cancel: &CancellationToken,
    ) -> Result<()> {
        let width = usize::try_from(scalar_bytes(self.dtype))
            .map_err(|_error| invalid("invalid sparse dtype width"))?;
        let mut values = self.values.open(file, cancel)?;
        let mut output = SparseOutput {
            bytes: output,
            start,
            width,
            dtype: self.dtype,
            cancel,
        };
        match &self.layout {
            Layout::Coo {
                coordinates,
                dimensions,
                nnz,
                inner,
                coalesced,
            } => {
                let mut coords = coordinates.open(file, cancel)?;
                let mut previous = None;
                for n in 0..*nnz {
                    if n % 1024 == 0 {
                        cancel.check()?;
                    }
                    let mut offset = 0;
                    for (axis, &dimension) in dimensions.iter().enumerate() {
                        let coord = coords.index(axis as u64 * nnz + n)?;
                        if coord >= dimension {
                            return Err(invalid("PyTorch COO index exceeds its tensor shape"));
                        }
                        offset = offset * dimension + coord;
                    }
                    if *coalesced && previous.is_some_and(|previous| previous >= offset) {
                        return Err(invalid(
                            "coalesced PyTorch COO indices are not strictly ordered",
                        ));
                    }
                    previous = Some(offset);
                    output.add(offset * inner, n * inner, *inner, &mut values)?;
                }
            }
            Layout::Compressed {
                pointers,
                indices,
                rows,
                cols,
                row_major,
                height,
                width: block_width,
                inner,
                batches,
                nnz,
            } => {
                let mut pointer = pointers.open(file, cancel)?;
                let mut index = indices.open(file, cancel)?;
                let (major, minor) = if *row_major {
                    (rows / height, cols / block_width)
                } else {
                    (cols / block_width, rows / height)
                };
                for batch in 0..*batches {
                    cancel.check()?;
                    let base = batch * (major + 1);
                    let mut from = pointer.index(base)?;
                    if from != 0 {
                        return Err(invalid("PyTorch sparse pointer must start at zero"));
                    }
                    for major_index in 0..major {
                        if major_index % 1024 == 0 {
                            cancel.check()?;
                        }
                        let to = pointer.index(base + major_index + 1)?;
                        if to < from || to > *nnz {
                            return Err(invalid("invalid PyTorch compressed sparse pointer"));
                        }
                        let mut previous = None;
                        for slot in from..to {
                            if slot % 1024 == 0 {
                                cancel.check()?;
                            }
                            let minor_index = index.index(batch * nnz + slot)?;
                            if minor_index >= minor || previous.is_some_and(|p| p >= minor_index) {
                                return Err(invalid(
                                    "invalid or unsorted PyTorch compressed sparse index",
                                ));
                            }
                            previous = Some(minor_index);
                            let (row, col) = if *row_major {
                                (major_index, minor_index)
                            } else {
                                (minor_index, major_index)
                            };
                            let block = (batch * nnz + slot) * height * block_width * inner;
                            for dr in 0..*height {
                                if dr.is_multiple_of(1024) {
                                    cancel.check()?;
                                }
                                let target = ((batch * rows + row * height + dr) * cols
                                    + col * block_width)
                                    * inner;
                                output.add(
                                    target,
                                    block + dr * block_width * inner,
                                    block_width * inner,
                                    &mut values,
                                )?;
                            }
                        }
                        from = to;
                    }
                    if from != *nnz {
                        return Err(invalid(
                            "PyTorch sparse pointer must end at its nonzero count",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn ensure_addable(dtype: DType) -> Result<()> {
    match dtype {
        DType::F8E5M2 | DType::F8E4M3 | DType::F8E8M0 | DType::F8E4M3Fnuz | DType::F8E5M2Fnuz => {
            Err(Error::unsupported(
                "unsupported PyTorch COO accumulation dtype",
            ))
        }
        _ => Ok(()),
    }
}

pub(super) fn add(dtype: DType, output: &mut [u8], source: [u8; 8]) {
    let mut current = [0; 8];
    current[..output.len()].copy_from_slice(output);
    macro_rules! combine {
        ($type:ty, $count:literal, $combine:expr) => {{
            let left =
                <$type>::from_le_bytes(current[..$count].try_into().expect("fixed scalar width"));
            let right =
                <$type>::from_le_bytes(source[..$count].try_into().expect("fixed scalar width"));
            output.copy_from_slice(&$combine(left, right).to_le_bytes());
        }};
    }
    match dtype {
        DType::Bool => output[0] = u8::from(output[0] != 0 || source[0] != 0),
        DType::F16 => combine!(u16, 2, |a, b| (half::f16::from_bits(a)
            + half::f16::from_bits(b))
        .to_bits()),
        DType::Bf16 => combine!(u16, 2, |a, b| (half::bf16::from_bits(a)
            + half::bf16::from_bits(b))
        .to_bits()),
        DType::F32 => combine!(f32, 4, |a, b| a + b),
        DType::F64 => combine!(f64, 8, |a, b| a + b),
        DType::C64 => {
            for (a, b) in output.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
                let x = f32::from_le_bytes(a.try_into().expect("fixed complex component"));
                let y = f32::from_le_bytes(b.try_into().expect("fixed complex component"));
                a.copy_from_slice(&(x + y).to_le_bytes());
            }
        }
        _ => {
            let sum = u64::from_le_bytes(current).wrapping_add(u64::from_le_bytes(source));
            output.copy_from_slice(&sum.to_le_bytes()[..output.len()]);
        }
    }
}
