use super::{
    Backing, CancellationToken, DType, DenseRead, DenseView, Document, Error, File, Id, Node,
    ResourceLimits, Result, StorageSpec, TensorInventory, component, contiguous, dimensions,
    elements, invalid, storage_spec,
};

#[derive(Debug)]
pub(in crate::pytorch) struct Quantized {
    pub(in crate::pytorch) length: u64,
    values: DenseView,
    params: Params,
    bits: u8,
}

#[derive(Debug)]
enum Params {
    Tensor { scale: f64, zero: i64 },
    Channel(Box<Channel>),
}

#[derive(Debug)]
struct Channel {
    scales: Numbers,
    zeros: Numbers,
    count: u64,
    step: u64,
    float: bool,
}

#[derive(Debug)]
enum Numbers {
    Tensor(DenseView),
    Inline(Box<[f64]>),
}

enum ReadNumbers<'a> {
    Tensor(DType, DenseRead<'a>),
    Inline(&'a [f64]),
}

#[allow(
    clippy::cast_precision_loss,
    reason = "quantized CPU kernels convert operands to floating point"
)]
fn number(doc: &Document, id: Id) -> Result<f64> {
    match doc.nodes[id] {
        Node::Float(value) => Ok(value),
        Node::Int(value) => Ok(value as f64),
        _ => Err(invalid("invalid PyTorch quantizer number")),
    }
}

fn integers(doc: &Document, id: Id) -> Result<i64> {
    match doc.nodes[id] {
        Node::Int(value) => {
            i64::try_from(value).map_err(|_error| invalid("invalid PyTorch zero point"))
        }
        _ => Err(invalid("invalid PyTorch integer zero point")),
    }
}

impl Numbers {
    fn parse<'a>(
        doc: &'a Document,
        id: Id,
        count: u64,
        limits: &ResourceLimits,
        big_endian: bool,
        storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
        inline_budget: &mut usize,
    ) -> Result<Self> {
        if let Node::Seq(values) = &doc.nodes[id] {
            if values.len() as u64 != count {
                return Err(invalid("PyTorch per-channel parameter length mismatch"));
            }
            *inline_budget = inline_budget
                .checked_sub(values.len().saturating_mul(8))
                .ok_or_else(|| {
                    Error::limit("inline PyTorch quantization parameters exceed metadata limit")
                })?;
            return Ok(Self::Inline(
                values
                    .iter()
                    .map(|&id| number(doc, id))
                    .collect::<Result<_>>()?,
            ));
        }
        let view = component(doc, id, limits, big_endian, storage)?;
        if view.elements() != count {
            return Err(Error::invalid(format!(
                "PyTorch per-channel parameter expected {count} values, got {}",
                view.elements()
            )));
        }
        Ok(Self::Tensor(view))
    }

    fn open<'a>(&'a self, file: &'a File, cancel: &CancellationToken) -> Result<ReadNumbers<'a>> {
        match self {
            Self::Tensor(view) => Ok(ReadNumbers::Tensor(view.dtype, view.open(file, cancel)?)),
            Self::Inline(values) => Ok(ReadNumbers::Inline(values)),
        }
    }
}

impl ReadNumbers<'_> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "quantized CPU kernels convert zero points to floating point"
    )]
    fn get(&mut self, index: u64) -> Result<f64> {
        match self {
            Self::Tensor(dtype, view) => {
                let bytes = view.scalar(index)?;
                Ok(match dtype {
                    DType::F32 => {
                        f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    DType::F64 => f64::from_le_bytes(bytes),
                    DType::I64 => i64::from_le_bytes(bytes) as f64,
                    _ => return Err(invalid("unsupported PyTorch per-channel parameter dtype")),
                })
            }
            Self::Inline(values) => {
                let index =
                    usize::try_from(index).map_err(|_error| invalid("invalid channel index"))?;
                values
                    .get(index)
                    .copied()
                    .ok_or_else(|| invalid("invalid channel index"))
            }
        }
    }
}

pub(super) fn parse<'a>(
    doc: &'a Document,
    args: Id,
    limits: &ResourceLimits,
    inventory: &mut TensorInventory,
    storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
) -> Result<(Vec<u64>, Quantized)> {
    let &[stored, offset, shape, strides, qparams, _, _] = doc.seq(args)? else {
        return Err(invalid("malformed PyTorch quantized tensor arguments"));
    };
    let Node::Persistent(id) = doc.nodes[stored] else {
        return Err(invalid("missing PyTorch quantized storage reference"));
    };
    let spec = storage_spec(doc, id)?;
    let dtype = spec
        .dtype
        .ok_or_else(|| invalid("invalid PyTorch quantized storage dtype"))?;
    let descriptor = doc.seq(id)?;
    let bits = match doc.global(descriptor[1]) {
        Some(("torch", "QUInt2x4Storage")) => 2,
        Some(("torch", "QUInt4x2Storage")) => 4,
        Some(("torch", "QUInt8Storage" | "QInt8Storage")) => 8,
        Some(("torch", "QInt32Storage")) => 32,
        _ => {
            return Err(Error::unsupported(
                "unsupported PyTorch quantized storage type",
            ));
        }
    };
    let offset = doc.int(offset)?;
    let shape = dimensions(doc, shape, limits)?;
    let strides = dimensions(doc, strides, limits)?;
    let length = DType::F32.byte_len(&shape)?;
    let stored = storage(spec)?;
    let big_endian = inventory.big_endian;
    let values = if bits < 8 {
        if offset != 0 || shape.len() != strides.len() || !contiguous(&shape, &strides)? {
            return Err(Error::unsupported(
                "noncontiguous sub-byte PyTorch quantized tensor",
            ));
        }
        DenseView::new(
            stored,
            DType::U8,
            false,
            0,
            vec![(length / 4).div_ceil(8 / u64::from(bits))],
            &[1],
        )?
    } else {
        DenseView::new(stored, dtype, big_endian, offset, shape.clone(), &strides)?
    };
    let params = parse_params(
        doc,
        qparams,
        limits,
        inventory,
        storage,
        &shape,
        (dtype, bits),
    )?;
    Ok((
        shape,
        Quantized {
            length,
            values,
            params,
            bits,
        },
    ))
}

fn parse_params<'a>(
    doc: &'a Document,
    qparams: Id,
    limits: &ResourceLimits,
    inventory: &mut TensorInventory,
    storage: &mut impl FnMut(StorageSpec<'a>) -> Result<Backing>,
    shape: &[u64],
    encoding: (DType, u8),
) -> Result<Params> {
    let (dtype, bits) = encoding;
    let big_endian = inventory.big_endian;
    let params = match *doc.seq(qparams)? {
        [scheme, scale, zero] if doc.global(scheme) == Some(("torch", "per_tensor_affine")) => {
            let zero = integers(doc, zero)?;
            if !in_domain(zero, dtype, bits) {
                return Err(invalid("PyTorch quantizer zero point is out of bounds"));
            }
            Params::Tensor {
                scale: number(doc, scale)?,
                zero,
            }
        }
        [scheme, scales, zeros, axis] => {
            let explicit_float = match doc.global(scheme) {
                Some(("torch", "per_channel_affine")) => false,
                Some(("torch", "per_channel_affine_float_qparams")) => true,
                _ => {
                    return Err(Error::unsupported(
                        "unsupported PyTorch quantization scheme",
                    ));
                }
            };
            let axis = usize::try_from(doc.int(axis)?)
                .map_err(|_error| invalid("invalid quantizer axis"))?;
            let count = *shape
                .get(axis)
                .ok_or_else(|| invalid("PyTorch quantization axis exceeds tensor rank"))?;
            let step = elements(&shape[axis + 1..])?;
            let scales = Numbers::parse(
                doc,
                scales,
                count,
                limits,
                big_endian,
                storage,
                &mut inventory.inline_budget,
            )?;
            let zeros = Numbers::parse(
                doc,
                zeros,
                count,
                limits,
                big_endian,
                storage,
                &mut inventory.inline_budget,
            )?;
            // torch.save uses the historical per_channel_affine label for both
            // schemes. The tensor parameter dtypes select the actual scheme.
            let float = match (&scales, &zeros) {
                (Numbers::Tensor(s), Numbers::Tensor(z))
                    if s.dtype == DType::F32 && z.dtype == DType::F32 =>
                {
                    true
                }
                (Numbers::Tensor(s), Numbers::Tensor(z))
                    if !explicit_float && s.dtype == DType::F64 && z.dtype == DType::I64 =>
                {
                    false
                }
                (Numbers::Inline(_), Numbers::Inline(_)) => explicit_float,
                _ => {
                    return Err(invalid(
                        "PyTorch per-channel scales and zero point dtypes do not match their scheme",
                    ));
                }
            };
            if !float && bits < 8 {
                return Err(Error::unsupported(
                    "PyTorch packed per-channel weights require floating zero points",
                ));
            }
            Params::Channel(Box::new(Channel {
                scales,
                zeros,
                count,
                step,
                float,
            }))
        }
        _ => {
            return Err(Error::unsupported(
                "unsupported PyTorch quantization scheme",
            ));
        }
    };
    Ok(params)
}

fn in_domain(zero: i64, dtype: DType, bits: u8) -> bool {
    match dtype {
        DType::I8 => i8::try_from(zero).is_ok(),
        DType::I32 => i32::try_from(zero).is_ok(),
        _ => (0..(1_i64 << bits)).contains(&zero),
    }
}

impl Quantized {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "mirrors the intermediate and result precision of the PyTorch CPU kernels"
    )]
    pub(in crate::pytorch) fn read(
        &self,
        file: &File,
        first: u64,
        output: &mut [u8],
        cancel: &CancellationToken,
    ) -> Result<()> {
        let mut values = self.values.open(file, cancel)?;
        let lanes = (8 / u64::from(self.bits)).max(1);
        let value = |at: u64, source: &mut DenseRead<'_>| -> Result<i64> {
            let bytes = source.scalar(at / lanes)?;
            Ok(match self.values.dtype {
                DType::I32 => {
                    i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                }
                DType::I8 => i64::from(i8::from_le_bytes([bytes[0]])),
                _ if lanes != 1 => i64::from(
                    (bytes[0] >> ((at % lanes) * u64::from(self.bits))) & ((1_u8 << self.bits) - 1),
                ),
                _ => i64::from(bytes[0]),
            })
        };
        match &self.params {
            Params::Tensor { scale, zero } => {
                for (i, bytes) in output.chunks_exact_mut(4).enumerate() {
                    if i % 1024 == 0 {
                        cancel.check()?;
                    }
                    let q = value(first + i as u64, &mut values)?;
                    // PyTorch CPU subtracts the zero point in a signed 32-bit lane.
                    let result = (q as i32).wrapping_sub(*zero as i32) as f32 * *scale as f32;
                    bytes.copy_from_slice(&result.to_le_bytes());
                }
            }
            Params::Channel(channel) => {
                let Channel {
                    scales,
                    zeros,
                    count,
                    step,
                    float,
                } = channel.as_ref();
                let mut scales = scales.open(file, cancel)?;
                let mut zeros = zeros.open(file, cancel)?;
                let mut current = None;
                let mut params = (0., 0.);
                for (i, bytes) in output.chunks_exact_mut(4).enumerate() {
                    if i % 1024 == 0 {
                        cancel.check()?;
                    }
                    let at = first + i as u64;
                    let channel = (at / step) % count;
                    if current != Some(channel) {
                        params = (scales.get(channel)?, zeros.get(channel)?);
                        if !float
                            && (!params.1.is_finite()
                                || params.1.fract() != 0.
                                || !in_domain(params.1 as i64, self.values.dtype, self.bits))
                        {
                            return Err(invalid("PyTorch per-channel zero point is out of bounds"));
                        }
                        current = Some(channel);
                    }
                    let q = value(at, &mut values)? as f32;
                    let diff = q - params.1 as f32;
                    let result = if *float {
                        diff * params.0 as f32
                    } else {
                        (f64::from(diff) * params.0) as f32
                    };
                    bytes.copy_from_slice(&result.to_le_bytes());
                }
            }
        }
        Ok(())
    }
}
