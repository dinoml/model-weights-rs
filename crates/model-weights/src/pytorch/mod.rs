use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Cursor};

use crate::inventory::TensorRecord;
use crate::limits::ResourceLimits;
use crate::quantization::Storage;
use crate::tensor::{DType, FileId, SourceSpan};
use crate::{CancellationToken, Error, Result};

mod container;
pub(crate) mod metadata;
mod pickle;
mod special;
mod view;
mod zip_data;

pub(crate) use view::TensorView;

use container::{Positional, Range, Source, ZipEntry};
use pickle::{Budget, Document, Id, Node};
use view::DenseView;
use zip_data::Backing;

const LEGACY_MAGIC: i128 = 0x1950_a86a_20f9_469c_fc6c;

#[derive(Clone, Copy)]
pub(crate) enum Container {
    Zip,
    Stream,
    Tar,
}

pub(crate) fn detect(file: &File, size: u64, magic: [u8; 4]) -> Result<Option<Container>> {
    detect_source(&Source::new(file, size), magic)
}

fn detect_source(source: &Source<'_>, magic: [u8; 4]) -> Result<Option<Container>> {
    if magic == *b"PK\x03\x04" {
        return Ok(Some(Container::Zip));
    }
    let size = source.range.len;
    if (magic[0] == 0x80 || magic[0] == b'L') && size >= 28 {
        let prefix = source.array::<28>(0)?;
        let supports_binary = prefix[0] == 0x80
            && prefix
                .windows(12)
                .any(|part| part == b"\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19");
        if supports_binary || prefix.starts_with(b"L119547037146038801333356L\n.") {
            return Ok(Some(Container::Stream));
        }
    }
    if size >= 512 {
        // Valid Safetensors starts with a length and a JSON object.
        let prefix = source.array::<9>(0)?;
        if prefix[8] != b'{' || prefix[4..8] != [0; 4] {
            let header = source.array::<512>(0)?;
            if header[257..].starts_with(b"ustar") || container::tar_header_valid(&header) {
                return Ok(Some(Container::Tar));
            }
        }
    }
    Ok(None)
}

pub(crate) fn parse(
    file: &File,
    size: u64,
    file_id: FileId,
    container: Container,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<(Vec<TensorRecord>, Vec<TensorView>)> {
    let source = Source::new(file, size);
    let mut budget = Budget::new(limits, cancellation);
    let mut inventory = TensorInventory {
        file_id,
        big_endian: false,
        inline_budget: limits.max_metadata_bytes(),
        views: Vec::new(),
    };
    let output = match container {
        Container::Zip => zip(&source, &mut inventory, limits, cancellation, &mut budget),
        Container::Stream => stream(&source, &mut inventory, limits, cancellation, &mut budget),
        Container::Tar => tar(&source, &mut inventory, limits, cancellation, &mut budget),
    }?;
    if output.is_empty() {
        return Err(Error::unsupported(
            "PyTorch checkpoint contains no supported named tensors",
        ));
    }
    Ok((output, inventory.views))
}

fn invalid(message: &'static str) -> Error {
    Error::invalid(message)
}

fn storage_dtype(module: &str, name: &str) -> Result<Option<DType>> {
    if !matches!(module, "torch" | "torch.storage" | "torch.cuda") {
        return Err(Error::unsupported("unknown PyTorch storage module"));
    }
    Ok(Some(match name {
        "UntypedStorage" => return Ok(None),
        "BoolStorage" => DType::Bool,
        "ByteStorage" | "QUInt8Storage" | "QUInt2x4Storage" | "QUInt4x2Storage" => DType::U8,
        "CharStorage" | "QInt8Storage" => DType::I8,
        "ShortStorage" => DType::I16,
        "IntStorage" | "QInt32Storage" => DType::I32,
        "LongStorage" => DType::I64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::Bf16,
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "ComplexFloatStorage" => DType::C64,
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported PyTorch storage type {name}"
            )));
        }
    }))
}

fn dtype_symbol(doc: &Document, id: Id) -> Result<DType> {
    Ok(match doc.global(id) {
        Some(("torch", "bool")) => DType::Bool,
        Some(("torch", "uint8")) => DType::U8,
        Some(("torch", "uint16")) => DType::U16,
        Some(("torch", "uint32")) => DType::U32,
        Some(("torch", "uint64")) => DType::U64,
        Some(("torch", "int8")) => DType::I8,
        Some(("torch", "int16")) => DType::I16,
        Some(("torch", "int32")) => DType::I32,
        Some(("torch", "int64")) => DType::I64,
        Some(("torch", "float16")) => DType::F16,
        Some(("torch", "bfloat16")) => DType::Bf16,
        Some(("torch", "float32")) => DType::F32,
        Some(("torch", "float64")) => DType::F64,
        Some(("torch", "complex64")) => DType::C64,
        Some(("torch", "float8_e5m2")) => DType::F8E5M2,
        Some(("torch", "float8_e4m3fn")) => DType::F8E4M3,
        Some(("torch", "float8_e5m2fnuz")) => DType::F8E5M2Fnuz,
        Some(("torch", "float8_e4m3fnuz")) => DType::F8E4M3Fnuz,
        Some(("torch", "float8_e8m0fnu")) => DType::F8E8M0,
        _ => return Err(Error::unsupported("unsupported PyTorch tensor dtype")),
    })
}

fn scalar_bytes(dtype: DType) -> u64 {
    u64::from(dtype.bits()) / 8
}

fn multiply(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::limit("PyTorch tensor byte range overflows"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StorageSpec<'a> {
    key: &'a str,
    dtype: Option<DType>,
    count: u64,
    view: Option<(u64, u64)>,
}

fn storage_spec(doc: &Document, id: Id) -> Result<StorageSpec<'_>> {
    if matches!(doc.nodes[id], Node::Text(_)) {
        return Err(Error::unsupported(
            "textual PyTorch storage references are unsupported",
        ));
    }
    let [tag, dtype, key, location, count, extra @ ..] = doc.seq(id)? else {
        return Err(invalid("malformed PyTorch storage reference"));
    };
    if doc.text(*tag)? != "storage" || extra.len() > 1 {
        return Err(Error::unsupported(
            "unsupported PyTorch persistent reference",
        ));
    }
    let (module, name) = doc
        .global(*dtype)
        .ok_or_else(|| invalid("invalid PyTorch storage type"))?;
    let dtype = storage_dtype(module, name)?;
    doc.text(*location)?;
    let count = doc.int(*count)?;
    let view = match extra {
        [view] if !matches!(doc.nodes[*view], Node::None) => {
            let [key, offset, size] = doc.seq(*view)? else {
                return Err(invalid("invalid PyTorch storage view"));
            };
            doc.text(*key)?;
            Some((doc.int(*offset)?, doc.int(*size)?))
        }
        _ => None,
    };
    Ok(StorageSpec {
        key: doc.text(*key)?,
        dtype,
        count,
        view,
    })
}

fn storage_width(spec: StorageSpec<'_>) -> u64 {
    spec.dtype.map_or(1, scalar_bytes)
}

struct Tensor<'a> {
    storage: StorageSpec<'a>,
    dtype: DType,
    offset: u64,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

fn dimensions(doc: &Document, id: Id, limits: &ResourceLimits) -> Result<Vec<u64>> {
    let values = doc.seq(id)?;
    if values.len() > limits.max_rank() {
        return Err(Error::limit("PyTorch tensor rank exceeds configured limit"));
    }
    values.iter().map(|&value| doc.int(value)).collect()
}

fn tensor<'a>(
    doc: &'a Document,
    mut id: Id,
    limits: &ResourceLimits,
) -> Result<Option<Tensor<'a>>> {
    for _ in 0..8 {
        let Node::Call(callable, arguments) = doc.nodes[id] else {
            if matches!(doc.nodes[id], Node::Persistent(_)) {
                return Err(Error::unsupported("unexpected PyTorch tensor reference"));
            }
            return Ok(None);
        };
        let Some(("torch._utils", rebuild)) = doc.global(callable) else {
            return Err(Error::unsupported(
                "state dictionary contains a non-tensor object",
            ));
        };
        let args = doc.seq(arguments)?;
        if matches!(
            rebuild,
            "_rebuild_parameter" | "_rebuild_parameter_with_state"
        ) {
            id = *args
                .first()
                .ok_or_else(|| invalid("missing PyTorch parameter tensor"))?;
            continue;
        }
        if !matches!(
            rebuild,
            "_rebuild_tensor" | "_rebuild_tensor_v2" | "_rebuild_tensor_v3"
        ) {
            return Err(Error::unsupported(format!(
                "unsupported PyTorch tensor constructor {rebuild}"
            )));
        }
        let [storage, offset, shape, strides, ..] = args else {
            return Err(invalid("malformed PyTorch tensor arguments"));
        };
        let metadata_at = if rebuild == "_rebuild_tensor_v3" {
            7
        } else {
            6
        };
        if let Some(&metadata) = args.get(metadata_at)
            && !matches!(doc.nodes[metadata], Node::None)
            && !matches!(&doc.nodes[metadata], Node::Dict(entries) if entries.is_empty())
        {
            return Err(Error::unsupported(
                "PyTorch tensor has conjugate, negative, or unknown metadata",
            ));
        }
        let Node::Persistent(storage_id) = doc.nodes[*storage] else {
            return Err(invalid("missing PyTorch tensor storage reference"));
        };
        let storage = storage_spec(doc, storage_id)?;
        let dtype = if rebuild == "_rebuild_tensor_v3" {
            dtype_symbol(
                doc,
                *args
                    .get(6)
                    .ok_or_else(|| invalid("missing PyTorch tensor dtype"))?,
            )?
        } else {
            storage
                .dtype
                .ok_or_else(|| invalid("untyped PyTorch storage lacks an explicit dtype"))?
        };
        return Ok(Some(Tensor {
            storage,
            dtype,
            offset: doc.int(*offset)?,
            shape: dimensions(doc, *shape, limits)?,
            strides: dimensions(doc, *strides, limits)?,
        }));
    }
    Err(Error::limit(
        "PyTorch parameter nesting exceeds supported depth",
    ))
}

struct TensorInventory {
    file_id: FileId,
    big_endian: bool,
    inline_budget: usize,
    views: Vec<TensorView>,
}

impl TensorInventory {
    fn record(
        &mut self,
        name: &str,
        dtype: DType,
        storage: Backing,
        offset: u64,
        shape: Vec<u64>,
        strides: &[u64],
    ) -> Result<TensorRecord> {
        let view = DenseView::new(storage, dtype, self.big_endian, offset, shape, strides)?;
        if let Some(range) = view.direct()? {
            let span = SourceSpan::new(self.file_id, range.start, range.len)?;
            return Ok(TensorRecord::new(
                name.into(),
                view.shape,
                Storage::Plain { dtype, span },
            ));
        }
        self.project(name, dtype, view.shape.clone(), TensorView::dense(view))
    }

    fn project(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Box<[u64]>,
        view: TensorView,
    ) -> Result<TensorRecord> {
        let id = u32::try_from(self.views.len())
            .map_err(|_error| Error::limit("too many PyTorch tensor views"))?;
        let span = SourceSpan::for_tensor_view(self.file_id, id, view.len())?;
        self.views.push(view);
        Ok(TensorRecord::new(
            name.into(),
            shape,
            Storage::Plain { dtype, span },
        ))
    }
}

fn contiguous(shape: &[u64], strides: &[u64]) -> Result<bool> {
    let mut stride = 1;
    for (&dim, &actual) in shape.iter().zip(strides).rev() {
        if dim > 1 && actual != stride {
            return Ok(false);
        }
        stride = multiply(dim, stride)?;
    }
    Ok(true)
}

fn dict(doc: &Document, id: Id) -> Result<&[(Id, Id)]> {
    if let Node::Dict(ref items) = doc.nodes[id] {
        Ok(items)
    } else {
        Err(Error::unsupported(
            "PyTorch checkpoint must contain a named state dictionary",
        ))
    }
}

fn tensor_constructor(doc: &Document, id: Id) -> bool {
    match doc.nodes[id] {
        Node::Call(callable, _) => matches!(doc.global(callable), Some(("torch._utils", name))
            if name.starts_with("_rebuild_tensor") || name.starts_with("_rebuild_parameter") || matches!(name, "_rebuild_qtensor" | "_rebuild_sparse_tensor")),
        Node::Persistent(_) => true,
        _ => false,
    }
}

fn packed_parameters(doc: &Document, name: &str, id: Id) -> Option<[Id; 2]> {
    if !name.ends_with("_packed_params") {
        return None;
    }
    let Node::Seq(entries) = &doc.nodes[id] else {
        return None;
    };
    let [weight, bias] = entries.as_slice() else {
        return None;
    };
    let Node::Call(callable, _) = doc.nodes[*weight] else {
        return None;
    };
    (doc.global(callable) == Some(("torch._utils", "_rebuild_qtensor"))).then_some([*weight, *bias])
}

fn state_dict<'a>(doc: &'a Document, cancellation: &CancellationToken) -> Result<&'a [(Id, Id)]> {
    let outer = dict(doc, doc.root)?;
    let mut explicit = None;
    let mut inferred = None;
    let mut ambiguous = false;
    let mut direct = false;
    let mut visited = BTreeSet::new();
    for &(key, value) in outer {
        cancellation.check()?;
        if tensor_constructor(doc, value)
            || doc
                .text(key)
                .ok()
                .and_then(|name| packed_parameters(doc, name, value))
                .is_some()
        {
            direct = true;
        }
        let Node::Dict(ref entries) = doc.nodes[value] else {
            continue;
        };
        if doc.text(key).ok() == Some("state_dict") && explicit.replace(value).is_some() {
            return Err(invalid("duplicate state_dict in PyTorch checkpoint"));
        }
        if visited.insert(value) {
            let mut has_tensors = false;
            for &(key, value) in entries {
                cancellation.check()?;
                if tensor_constructor(doc, value)
                    || doc
                        .text(key)
                        .ok()
                        .and_then(|name| packed_parameters(doc, name, value))
                        .is_some()
                {
                    has_tensors = true;
                    break;
                }
            }
            ambiguous |= has_tensors && inferred.replace(value).is_some();
        }
    }
    if let Some(id) = explicit {
        return dict(doc, id);
    }
    if direct {
        return Ok(outer);
    }
    if ambiguous {
        return Err(Error::unsupported(
            "checkpoint contains several weight dictionaries; expected state_dict",
        ));
    }
    inferred.map_or(Ok(outer), |id| dict(doc, id))
}

fn for_each_weight(
    doc: &Document,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    mut visit: impl FnMut(&str, Id) -> Result<()>,
) -> Result<()> {
    let count = |len| {
        if len > limits.max_tensors() {
            return Err(Error::limit(
                "PyTorch checkpoint exceeds configured tensor limit",
            ));
        }
        Ok(())
    };
    match &doc.nodes[doc.root] {
        Node::Dict(_) => {
            let entries = state_dict(doc, cancellation)?;
            count(entries.len())?;
            let mut names = BTreeSet::new();
            let mut count = entries.len();
            for &(name, value) in entries {
                cancellation.check()?;
                let name = doc.text(name)?;
                if !names.insert(name) {
                    return Err(invalid("duplicate PyTorch tensor name"));
                }
                if let Some([weight, bias]) = packed_parameters(doc, name, value) {
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| Error::limit("too many PyTorch weight entries"))?;
                    if count > limits.max_tensors()
                        || name.len().saturating_add(2) > limits.max_name_bytes()
                    {
                        return Err(Error::limit(
                            "PyTorch packed parameters exceed configured limits",
                        ));
                    }
                    visit(&format!("{name}.0"), weight)?;
                    visit(&format!("{name}.1"), bias)?;
                } else if name != "_extra_state" && !name.ends_with("._extra_state") {
                    visit(name, value)?;
                }
            }
        }
        Node::Seq(values) => {
            count(values.len())?;
            for (index, &value) in values.iter().enumerate() {
                cancellation.check()?;
                visit(&index.to_string(), value)?;
            }
        }
        _ if tensor_constructor(doc, doc.root) => visit("tensor", doc.root)?,
        _ => {
            return Err(Error::unsupported(
                "PyTorch checkpoint has no tensor collection",
            ));
        }
    }
    Ok(())
}

fn records<'a>(
    doc: &'a Document,
    inventory: &mut TensorInventory,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    mut storage: impl FnMut(StorageSpec<'a>) -> Result<Backing>,
) -> Result<Vec<TensorRecord>> {
    let mut output = Vec::new();
    for_each_weight(doc, limits, cancellation, |name, value| {
        if let Some(record) = special::record(doc, value, name, inventory, limits, &mut storage)? {
            output.push(record);
            return Ok(());
        }
        let descriptor = match doc.nodes[value] {
            Node::Persistent(id) => {
                let storage = storage_spec(doc, id)?;
                let dtype = storage.dtype.unwrap_or(DType::U8);
                let count = storage.view.map_or(storage.count, |(_, count)| count);
                Some(Tensor {
                    storage,
                    dtype,
                    offset: 0,
                    shape: vec![count],
                    strides: vec![1],
                })
            }
            _ => tensor(doc, value, limits)?,
        };
        if let Some(tensor) = descriptor {
            let range = storage(tensor.storage)?;
            output.push(inventory.record(
                name,
                tensor.dtype,
                range,
                tensor.offset,
                tensor.shape,
                &tensor.strides,
            )?);
        }
        Ok(())
    })?;
    Ok(output)
}

fn zip_metadata(
    source: &Source<'_>,
    entry: &ZipEntry,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>> {
    match Backing::zip(entry, entry.size)? {
        Backing::Stored(range) => source.bytes(range),
        backing @ Backing::Deflate { .. } => {
            let size = usize::try_from(entry.size)
                .map_err(|_error| Error::limit("PyTorch ZIP metadata is too large"))?;
            let mut data = Vec::new();
            data.try_reserve_exact(size)
                .map_err(|_error| Error::limit("could not allocate PyTorch ZIP metadata"))?;
            backing.inflate(source, &mut data, cancellation)?;
            Ok(data)
        }
    }
}

fn zip_root<T>(entries: &BTreeMap<Box<str>, T>) -> Result<(String, &str)> {
    let mut pickles = entries.keys().filter_map(|name| {
        name.strip_suffix("/data.pkl")
            .filter(|prefix| !prefix.is_empty())
            .or_else(|| (name.as_ref() == "data.pkl").then_some(""))
            .map(|prefix| (prefix, name.as_ref()))
    });
    let (prefix, name) = pickles
        .next()
        .ok_or_else(|| Error::unsupported("ZIP archive has no PyTorch data.pkl"))?;
    if pickles.next().is_some() {
        return Err(invalid("ambiguous PyTorch data.pkl archive root"));
    }
    let prefix = if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}/")
    };
    if entries.contains_key(format!("{prefix}constants.pkl").as_str()) {
        return Err(Error::unsupported(
            "TorchScript model archives are unsupported",
        ));
    }
    Ok((prefix, name))
}

fn zip_document(
    source: &Source<'_>,
    data: &ZipEntry,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
    buffer_stored: bool,
) -> Result<Document> {
    if data.size > limits.max_header_bytes() || data.range.len > limits.max_header_bytes() {
        return Err(Error::limit("PyTorch data.pkl exceeds the header limit"));
    }
    // Keep the uncompressed path streaming and avoid doubling the pickle buffer.
    let doc = if data.method == 0 && !buffer_stored {
        let mut input = source.reader(data.range);
        let doc = pickle::read(&mut input, budget)?;
        if position(&input) != data.range.end() {
            return Err(invalid("trailing bytes in PyTorch data.pkl"));
        }
        doc
    } else {
        let mut input = Cursor::new(zip_metadata(source, data, cancellation)?);
        let doc = pickle::read(&mut input, budget)?;
        if input.position() != data.size {
            return Err(invalid("trailing bytes in PyTorch data.pkl"));
        }
        doc
    };
    Ok(doc)
}

fn zip(
    source: &Source<'_>,
    inventory: &mut TensorInventory,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
) -> Result<Vec<TensorRecord>> {
    let mut entries = container::zip(source, limits, cancellation, budget)?;
    let (prefix, name) = zip_root(&entries)?;
    let data = &entries[name];
    if let Some(byteorder) = entries.get(format!("{prefix}byteorder").as_str()) {
        if byteorder.size > 6 {
            return Err(invalid("invalid PyTorch byteorder entry"));
        }
        match zip_metadata(source, byteorder, cancellation)?.as_slice() {
            b"little" => {}
            b"big" => inventory.big_endian = true,
            _ => return Err(invalid("invalid PyTorch byteorder entry")),
        }
    }
    let doc = zip_document(source, data, limits, cancellation, budget, false)?;
    let storage_prefix = format!("{prefix}data/");
    entries.retain(|name, _| name.starts_with(&storage_prefix));
    records(&doc, inventory, limits, cancellation, |spec| {
        if spec.key.is_empty() || spec.key.contains(['/', '\\', '\0']) || spec.view.is_some() {
            return Err(invalid("invalid PyTorch ZIP storage reference"));
        }
        let entry = entries
            .get(format!("{storage_prefix}{}", spec.key).as_str())
            .ok_or_else(|| invalid("missing PyTorch ZIP storage entry"))?;
        if entry.size != multiply(spec.count, storage_width(spec))? {
            return Err(invalid(
                "PyTorch ZIP storage size differs from its declaration",
            ));
        }
        Backing::zip(entry, limits.max_zip_member_bytes())
    })
}

fn position(input: &BufReader<Positional<'_>>) -> u64 {
    input.get_ref().position() - input.buffer().len() as u64
}

fn read_document(
    source: &Source<'_>,
    bounds: Range,
    at: &mut u64,
    budget: &mut Budget<'_>,
) -> Result<Document> {
    let offset = at
        .checked_sub(bounds.start)
        .ok_or_else(|| invalid("invalid PyTorch metadata offset"))?;
    let mut reader = source.reader(
        bounds.sub(
            offset,
            bounds
                .len
                .checked_sub(offset)
                .ok_or_else(|| invalid("truncated PyTorch metadata"))?,
        )?,
    );
    let doc = pickle::read(&mut reader, budget)?;
    *at = position(&reader);
    Ok(doc)
}

fn raw_u64(
    source: &Source<'_>,
    bounds: Range,
    at: &mut u64,
    budget: &mut Budget<'_>,
) -> Result<u64> {
    budget.charge(8)?;
    let offset = at
        .checked_sub(bounds.start)
        .ok_or_else(|| invalid("invalid PyTorch integer offset"))?;
    let range = bounds.sub(offset, 8)?;
    let value = u64::from_le_bytes(source.array::<8>(range.start)?);
    *at = range.end();
    Ok(value)
}

fn raw_storage(
    source: &Source<'_>,
    bounds: Range,
    at: &mut u64,
    budget: &mut Budget<'_>,
    width: u64,
    expected: Option<u64>,
) -> Result<Range> {
    // The legacy storage writer encodes both the count and payload in
    // little-endian order, independently of the host byte order in sys_info.
    let count = raw_u64(source, bounds, at, budget)?;
    if expected.is_some_and(|expected| expected != count) {
        return Err(invalid(
            "legacy PyTorch storage count differs from its declaration",
        ));
    }
    let range = bounds.sub(*at - bounds.start, multiply(count, width)?)?;
    *at = range.end();
    Ok(range)
}

fn stream_document(source: &Source<'_>, at: &mut u64, budget: &mut Budget<'_>) -> Result<Document> {
    let mut next = || read_document(source, source.range, at, budget);
    let magic = next()?;
    if !matches!(magic.nodes[magic.root], Node::Int(LEGACY_MAGIC)) {
        return Err(invalid("invalid legacy PyTorch magic number"));
    }
    let version = next()?;
    if version.int(version.root)? != 1001 {
        return Err(Error::unsupported(
            "unsupported legacy PyTorch serialization version",
        ));
    }
    let system = next()?;
    dict(&system, system.root)?;
    next()
}

fn stream(
    source: &Source<'_>,
    inventory: &mut TensorInventory,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
) -> Result<Vec<TensorRecord>> {
    let mut at = 0;
    let doc = stream_document(source, &mut at, budget)?;
    let keys = read_document(source, source.range, &mut at, budget)?;
    let mut specs = BTreeMap::new();
    for node in &doc.nodes {
        cancellation.check()?;
        let Node::Persistent(id) = *node else {
            continue;
        };
        let parts = doc.seq(id)?;
        if parts.first().and_then(|&key| doc.text(key).ok()) != Some("storage") {
            continue;
        }
        let mut spec = storage_spec(&doc, id)?;
        spec.view = None;
        // The first reference determines the count prefix's element width.
        // Later references can describe the same bytes as an untyped storage.
        let first = *specs.entry(spec.key).or_insert(spec);
        if multiply(first.count, storage_width(first))?
            != multiply(spec.count, storage_width(spec))?
        {
            return Err(invalid("conflicting legacy PyTorch storage declarations"));
        }
    }
    let mut ranges = BTreeMap::new();
    for &id in keys.seq(keys.root)? {
        cancellation.check()?;
        let key = keys.text(id)?;
        let spec = specs
            .remove(key)
            .ok_or_else(|| invalid("unexpected legacy PyTorch storage key"))?;
        let range = raw_storage(
            source,
            source.range,
            &mut at,
            budget,
            storage_width(spec),
            Some(spec.count),
        )?;
        ranges.insert(key, range);
    }
    if !specs.is_empty() || at != source.range.end() {
        return Err(invalid(
            "incomplete or trailing data in legacy PyTorch checkpoint",
        ));
    }
    records(&doc, inventory, limits, cancellation, |spec| {
        let range = *ranges
            .get(spec.key)
            .ok_or_else(|| invalid("missing legacy PyTorch storage"))?;
        match spec.view {
            Some((offset, count)) => range
                .sub(
                    multiply(offset, storage_width(spec))?,
                    multiply(count, storage_width(spec))?,
                )
                .map(Backing::Stored),
            None => Ok(Backing::Stored(range)),
        }
    })
}

struct TarTensor {
    dtype: DType,
    storage: Range,
    offset: u64,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

fn tar(
    source: &Source<'_>,
    inventory: &mut TensorInventory,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
    budget: &mut Budget<'_>,
) -> Result<Vec<TensorRecord>> {
    let members = container::tar(source, budget)?;
    let member = |name| {
        members
            .get(name)
            .copied()
            .ok_or_else(|| invalid("missing PyTorch tar member"))
    };
    let storages = tar_storages(source, member("storages")?, limits, budget)?;
    let tensors = tar_tensors(source, member("tensors")?, &storages, limits, budget)?;
    let bounds = member("pickle")?;
    let mut at = bounds.start;
    let doc = read_document(source, bounds, &mut at, budget)?;
    if at != bounds.end() {
        return Err(invalid("trailing bytes in PyTorch tar pickle"));
    }
    let mut output = Vec::new();
    for_each_weight(&doc, limits, cancellation, |name, value| {
        match doc.nodes[value] {
            Node::Persistent(id) => {
                let key = match &doc.nodes[id] {
                    Node::Text(key) => key
                        .parse()
                        .map_err(|_error| invalid("invalid PyTorch tar tensor ID"))?,
                    _ => doc.int(id)?,
                };
                let record = if let Some(tensor) = tensors.get(&key) {
                    inventory.record(
                        name,
                        tensor.dtype,
                        Backing::Stored(tensor.storage),
                        tensor.offset,
                        tensor.shape.clone(),
                        &tensor.strides,
                    )?
                } else if let Some(&(dtype, range)) = storages.get(&key) {
                    inventory.record(
                        name,
                        dtype,
                        Backing::Stored(range),
                        0,
                        vec![range.len / scalar_bytes(dtype)],
                        &[1],
                    )?
                } else {
                    return Err(invalid("missing PyTorch tar tensor or storage ID"));
                };
                output.push(record);
            }
            Node::Call(..) => {
                return Err(Error::unsupported(
                    "PyTorch tar checkpoint contains a non-tensor object",
                ));
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(output)
}

fn tar_storages(
    source: &Source<'_>,
    bounds: Range,
    limits: &ResourceLimits,
    budget: &mut Budget<'_>,
) -> Result<BTreeMap<u64, (DType, Range)>> {
    let mut at = bounds.start;
    let count = read_document(source, bounds, &mut at, budget)?;
    let count = count.int(count.root)?;
    if count > limits.max_tensors() as u64 {
        return Err(Error::limit(
            "PyTorch tar storage count exceeds configured limit",
        ));
    }
    let mut storages = BTreeMap::new();
    for _ in 0..count {
        let doc = read_document(source, bounds, &mut at, budget)?;
        let [key, location, dtype] = doc.seq(doc.root)? else {
            return Err(invalid("invalid PyTorch tar storage descriptor"));
        };
        doc.text(*location)?;
        let (module, name) = doc
            .global(*dtype)
            .ok_or_else(|| invalid("invalid PyTorch tar storage dtype"))?;
        let dtype =
            storage_dtype(module, name)?.ok_or_else(|| invalid("untyped PyTorch tar storage"))?;
        let range = raw_storage(source, bounds, &mut at, budget, scalar_bytes(dtype), None)?;
        if storages.insert(doc.int(*key)?, (dtype, range)).is_some() {
            return Err(invalid("duplicate PyTorch tar storage ID"));
        }
    }
    let views = read_document(source, bounds, &mut at, budget)?;
    for &view in views.seq(views.root)? {
        budget.charge(0)?;
        let [key, root, offset, count] = views.seq(view)? else {
            return Err(invalid("invalid PyTorch tar storage view"));
        };
        let (dtype, root) = *storages
            .get(&views.int(*root)?)
            .ok_or_else(|| invalid("missing PyTorch tar root storage ID"))?;
        let bytes = scalar_bytes(dtype);
        let range = root.sub(
            multiply(views.int(*offset)?, bytes)?,
            multiply(views.int(*count)?, bytes)?,
        )?;
        if storages.insert(views.int(*key)?, (dtype, range)).is_some() {
            return Err(invalid("duplicate PyTorch tar storage view ID"));
        }
    }
    if at != bounds.end() {
        return Err(invalid("trailing bytes in PyTorch tar storages"));
    }
    Ok(storages)
}

fn tar_tensors(
    source: &Source<'_>,
    bounds: Range,
    storages: &BTreeMap<u64, (DType, Range)>,
    limits: &ResourceLimits,
    budget: &mut Budget<'_>,
) -> Result<BTreeMap<u64, TarTensor>> {
    let mut at = bounds.start;
    let count = read_document(source, bounds, &mut at, budget)?;
    let count = count.int(count.root)?;
    if count > limits.max_tensors() as u64 {
        return Err(Error::limit(
            "PyTorch tar tensor count exceeds configured limit",
        ));
    }
    let mut tensors = BTreeMap::new();
    for _ in 0..count {
        let doc = read_document(source, bounds, &mut at, budget)?;
        let [key, storage, _kind] = doc.seq(doc.root)? else {
            return Err(invalid("invalid PyTorch tar tensor descriptor"));
        };
        let (dtype, storage) = *storages
            .get(&doc.int(*storage)?)
            .ok_or_else(|| invalid("missing PyTorch tar tensor storage ID"))?;
        // PyTorch uses a 32-bit rank followed by four bytes of legacy padding.
        let rank = raw_u64(source, bounds, &mut at, budget)? & u64::from(u32::MAX);
        if rank > limits.max_rank() as u64 {
            return Err(Error::limit(
                "PyTorch tar tensor rank exceeds configured limit",
            ));
        }
        let shape = (0..rank)
            .map(|_| raw_u64(source, bounds, &mut at, budget))
            .collect::<Result<_>>()?;
        let strides = (0..rank)
            .map(|_| raw_u64(source, bounds, &mut at, budget))
            .collect::<Result<_>>()?;
        let offset = raw_u64(source, bounds, &mut at, budget)?;
        if tensors
            .insert(
                doc.int(*key)?,
                TarTensor {
                    dtype,
                    storage,
                    offset,
                    shape,
                    strides,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate PyTorch tar tensor ID"));
        }
    }
    if at != bounds.end() {
        return Err(invalid("trailing bytes in PyTorch tar tensors"));
    }
    Ok(tensors)
}
