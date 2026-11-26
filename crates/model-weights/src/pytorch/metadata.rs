use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};

use serde_json::{Map, Number, Value};

use crate::limits::ResourceLimits;
use crate::{CancellationToken, Error, Result};

use super::container::{self, Source};
use super::pickle::{Budget, Document, Id, Node};
use super::{Container, detect_source, read_document, stream_document, zip_document, zip_root};

const MAX_DEPTH: usize = 64;

pub(crate) fn read<R: Read + Seek>(
    mut reader: R,
    field: &str,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Option<Value>> {
    cancellation.check()?;
    if field.len() > limits.max_name_bytes() {
        return Err(Error::limit(
            "PyTorch metadata field name exceeds its limit",
        ));
    }
    let size = reader
        .seek(SeekFrom::End(0))
        .map_err(|error| Error::io("seeking PyTorch checkpoint metadata", error))?;
    let reader = RefCell::new(reader);
    let source = Source::from_reader(&reader, size);
    let magic = source.array(0)?;
    let container = detect_source(&source, magic)?
        .ok_or_else(|| Error::unsupported("unsupported PyTorch metadata container"))?;
    let mut budget = Budget::new(limits, cancellation);
    let doc = match container {
        Container::Zip => {
            let entries = container::zip_catalog(&source, limits, cancellation, &mut budget)?;
            let (_, name) = zip_root(&entries)?;
            let entry = entries[name].read(&source, name, &mut budget)?;
            zip_document(&source, &entry, limits, cancellation, &mut budget, true)?
        }
        Container::Stream => {
            let mut position = 0;
            stream_document(&source, &mut position, &mut budget)?
        }
        Container::Tar => {
            let members = container::tar(&source, &mut budget)?;
            let range = *members
                .get("pickle")
                .ok_or_else(|| Error::invalid("missing PyTorch tar pickle"))?;
            let mut position = range.start;
            let doc = read_document(&source, range, &mut position, &mut budget)?;
            if position != range.end() {
                return Err(Error::invalid("trailing bytes in PyTorch tar pickle"));
            }
            doc
        }
    };
    let Node::Dict(entries) = &doc.nodes[doc.root] else {
        return Ok(None);
    };
    let mut selected = None;
    for &(key, value) in entries {
        cancellation.check()?;
        if doc.text(key).ok() == Some(field) && selected.replace(value).is_some() {
            return Err(Error::invalid("duplicate top-level PyTorch metadata field"));
        }
    }
    let Some(id) = selected else { return Ok(None) };
    let mut projection = Projection {
        doc: &doc,
        remaining: limits.max_metadata_bytes(),
        cancellation,
        active: BTreeSet::new(),
    };
    projection.value(id, 0).map(Some)
}

struct Projection<'a> {
    doc: &'a Document,
    remaining: usize,
    cancellation: &'a CancellationToken,
    active: BTreeSet<Id>,
}

impl Projection<'_> {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| Error::limit("expanded PyTorch metadata exceeds its byte budget"))?;
        Ok(())
    }

    fn value(&mut self, id: Id, depth: usize) -> Result<Value> {
        self.cancellation.check()?;
        if depth >= MAX_DEPTH {
            return Err(Error::limit("PyTorch metadata exceeds its nesting limit"));
        }
        self.charge(size_of::<Value>())?;
        match &self.doc.nodes[id] {
            Node::None => Ok(Value::Null),
            Node::Bool(value) => Ok(Value::Bool(*value)),
            Node::Int(value) => i64::try_from(*value)
                .map(Number::from)
                .or_else(|_error| u64::try_from(*value).map(Number::from))
                .map(Value::Number)
                .map_err(|_error| Error::unsupported("PyTorch metadata integer does not fit JSON")),
            Node::Float(value) => Number::from_f64(*value)
                .map(Value::Number)
                .ok_or_else(|| Error::unsupported("non-finite PyTorch metadata float")),
            Node::Text(value) => {
                self.charge(value.len())?;
                Ok(Value::String(value.to_string()))
            }
            Node::OversizedText => Err(Error::limit(
                "PyTorch metadata string exceeds its name limit",
            )),
            Node::Seq(values) => {
                self.enter(id)?;
                let mut output = Vec::new();
                for &child in values {
                    output.push(self.value(child, depth + 1)?);
                }
                self.active.remove(&id);
                Ok(Value::Array(output))
            }
            Node::Dict(entries) => {
                self.enter(id)?;
                let mut output = Map::new();
                for &(key, value) in entries {
                    let name = self.doc.text(key)?;
                    if output.contains_key(name) {
                        return Err(Error::invalid(
                            "duplicate key in PyTorch metadata dictionary",
                        ));
                    }
                    // Account for the key and ordered map bookkeeping in addition to the value.
                    self.charge(name.len() + size_of::<String>() + 4 * size_of::<usize>())?;
                    output.insert(name.to_owned(), self.value(value, depth + 1)?);
                }
                self.active.remove(&id);
                Ok(Value::Object(output))
            }
            _ => Err(Error::unsupported(
                "PyTorch metadata field contains a non-data value",
            )),
        }
    }

    fn enter(&mut self, id: Id) -> Result<()> {
        if self.active.insert(id) {
            Ok(())
        } else {
            Err(Error::invalid("cyclic PyTorch metadata"))
        }
    }
}
