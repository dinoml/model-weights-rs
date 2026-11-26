use std::collections::BTreeMap;
use std::io::{self, BufRead};

use crate::limits::ResourceLimits;
use crate::{CancellationToken, Error, Result};

pub(super) type Id = usize;

#[derive(Debug)]
pub(super) enum Node {
    None,
    Int(i128),
    Bool(bool),
    Float(f64),
    Text(Box<str>),
    Other,
    OversizedText,
    Global(Box<str>, Box<str>),
    Seq(Vec<Id>),
    Set(Vec<Id>),
    Dict(Vec<(Id, Id)>),
    Call(Id, Id),
    Persistent(Id),
}

pub(super) struct Document {
    pub nodes: Vec<Node>,
    pub root: Id,
}

impl Document {
    pub(super) fn seq(&self, id: Id) -> Result<&[Id]> {
        match &self.nodes[id] {
            Node::Seq(values) => Ok(values),
            _ => Err(Error::invalid("expected a PyTorch pickle sequence")),
        }
    }

    pub(super) fn text(&self, id: Id) -> Result<&str> {
        match &self.nodes[id] {
            Node::Text(value) => Ok(value),
            Node::OversizedText => Err(Error::limit(
                "PyTorch metadata string exceeds its name limit",
            )),
            _ => Err(Error::invalid("expected a PyTorch pickle string")),
        }
    }

    pub(super) fn int(&self, id: Id) -> Result<u64> {
        match &self.nodes[id] {
            Node::Bool(value) => Ok(u64::from(*value)),
            Node::Int(value) => u64::try_from(*value)
                .map_err(|_error| Error::invalid("PyTorch pickle integer is out of range")),
            _ => Err(Error::invalid("expected a PyTorch pickle integer")),
        }
    }

    pub(super) fn global(&self, id: Id) -> Option<(&str, &str)> {
        if let Node::Global(module, name) = &self.nodes[id] {
            Some((module, name))
        } else {
            None
        }
    }
}

pub(super) struct Budget<'a> {
    bytes: u64,
    ops: usize,
    name: usize,
    cancellation: &'a CancellationToken,
}

impl<'a> Budget<'a> {
    pub(super) fn new(limits: &ResourceLimits, cancellation: &'a CancellationToken) -> Self {
        Self {
            bytes: limits.max_header_bytes(),
            ops: limits
                .max_tensors()
                .saturating_mul(128)
                .clamp(4096, 2_000_000),
            name: limits.max_name_bytes(),
            cancellation,
        }
    }

    pub(super) fn charge(&mut self, bytes: u64) -> Result<()> {
        self.cancellation.check()?;
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or_else(|| Error::limit("PyTorch metadata exceeds the configured byte limit"))?;
        Ok(())
    }
}

struct Vm<'a, 'b, R> {
    input: &'a mut R,
    budget: &'a mut Budget<'b>,
    nodes: Vec<Node>,
    stack: Vec<Id>,
    marks: Vec<usize>,
    memo: BTreeMap<usize, Id>,
    frame_left: Option<u64>,
}

pub(super) fn read<R: BufRead>(input: &mut R, budget: &mut Budget<'_>) -> Result<Document> {
    Vm {
        input,
        budget,
        nodes: Vec::new(),
        stack: Vec::new(),
        marks: Vec::new(),
        memo: BTreeMap::new(),
        frame_left: None,
    }
    .run()
}

fn malformed() -> Error {
    Error::invalid("malformed PyTorch pickle")
}

fn read_error(error: io::Error) -> Error {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        malformed()
    } else {
        Error::io("reading PyTorch pickle", error)
    }
}

impl<R: BufRead> Vm<'_, '_, R> {
    fn take(&mut self, len: u64) -> Result<()> {
        self.budget.charge(len)?;
        if let Some(left) = &mut self.frame_left {
            *left = left.checked_sub(len).ok_or_else(malformed)?;
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N as u64)?;
        let mut bytes = [0; N];
        self.input.read_exact(&mut bytes).map_err(read_error)?;
        Ok(bytes)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn line(&mut self) -> Result<Vec<u8>> {
        let mut line = Vec::new();
        loop {
            let byte = self.byte()?;
            if byte == b'\n' {
                return Ok(line);
            }
            if line.len() >= self.budget.name {
                return Err(Error::limit(
                    "PyTorch pickle line exceeds the configured name limit",
                ));
            }
            line.push(byte);
        }
    }

    fn string(&mut self, length: u64, utf8: bool) -> Result<Node> {
        self.take(length)?;
        let length = usize::try_from(length).map_err(|_error| malformed())?;
        if !utf8 {
            self.skip(length)?;
            return Ok(Node::Other);
        }
        // Non-name strings in training metadata can be large. Skip them without
        // allocating and require a usable string when interpreting a name.
        if length > self.budget.name {
            self.skip(length)?;
            return Ok(Node::OversizedText);
        }
        let mut bytes = vec![0; length];
        self.input.read_exact(&mut bytes).map_err(read_error)?;
        Ok(match String::from_utf8(bytes) {
            Ok(value) => Node::Text(value.into_boxed_str()),
            Err(_) => Node::Other,
        })
    }

    fn skip(&mut self, mut length: usize) -> Result<()> {
        let mut scratch = [0; 4096];
        while length != 0 {
            self.budget.cancellation.check()?;
            let count = length.min(scratch.len());
            self.input
                .read_exact(&mut scratch[..count])
                .map_err(read_error)?;
            length -= count;
        }
        Ok(())
    }

    fn long(&mut self, length: u64) -> Result<Node> {
        if length > 16 {
            return self.string(length, false);
        }
        self.take(length)?;
        let length = usize::try_from(length).map_err(|_error| malformed())?;
        let mut bytes = [0; 16];
        self.input
            .read_exact(&mut bytes[..length])
            .map_err(read_error)?;
        if length != 0 && bytes[length - 1] & 128 != 0 {
            bytes[length..].fill(255);
        }
        Ok(Node::Int(i128::from_le_bytes(bytes)))
    }

    fn line_text(&mut self) -> Result<Box<str>> {
        String::from_utf8(self.line()?)
            .map(String::into_boxed_str)
            .map_err(|_error| malformed())
    }

    fn push(&mut self, node: Node) {
        self.stack.push(self.nodes.len());
        self.nodes.push(node);
    }

    fn top(&self) -> Result<Id> {
        let value = *self.stack.last().ok_or_else(malformed)?;
        if self.marks.last() == Some(&self.stack.len()) {
            return Err(malformed());
        }
        Ok(value)
    }

    fn pop(&mut self) -> Result<Id> {
        let value = self.top()?;
        self.stack.pop();
        Ok(value)
    }

    fn marked(&mut self) -> Result<Vec<Id>> {
        let mark = self.marks.pop().ok_or_else(malformed)?;
        Ok(self.stack.split_off(mark))
    }

    fn fixed(&mut self, count: usize) -> Result<Vec<Id>> {
        let start = self.stack.len().checked_sub(count).ok_or_else(malformed)?;
        if self.marks.last().is_some_and(|mark| start < *mark) {
            return Err(malformed());
        }
        Ok(self.stack.split_off(start))
    }

    fn pairs(values: &[Id]) -> Result<Vec<(Id, Id)>> {
        let chunks = values.chunks_exact(2);
        if !chunks.remainder().is_empty() {
            return Err(malformed());
        }
        Ok(chunks.map(|pair| (pair[0], pair[1])).collect())
    }

    fn apply(&mut self, callable: Id, arguments: Id) -> Result<()> {
        if matches!(&self.nodes[callable], Node::Global(module, name)
            if (module.as_ref(), name.as_ref()) == ("collections", "OrderedDict"))
        {
            let entries = match &self.nodes[arguments] {
                Node::Seq(args) if args.is_empty() => Some(Vec::new()),
                Node::Seq(args) if args.len() == 1 => match &self.nodes[args[0]] {
                    Node::Seq(pairs) => {
                        self.budget.charge(pairs.len() as u64 * 16)?;
                        pairs
                            .iter()
                            .map(|pair| match &self.nodes[*pair] {
                                Node::Seq(pair) if pair.len() == 2 => Some((pair[0], pair[1])),
                                _ => None,
                            })
                            .collect()
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(entries) = entries {
                self.push(Node::Dict(entries));
                return Ok(());
            }
        }
        if matches!(&self.nodes[callable], Node::Global(module, name)
            if (module.as_ref(), name.as_ref()) == ("torch", "Size"))
            && let Node::Seq(args) = &self.nodes[arguments]
            && let [arg] = args.as_slice()
            && matches!(self.nodes[*arg], Node::Seq(_))
        {
            self.stack.push(*arg);
            return Ok(());
        }
        // Symbolic only. A later tensor decoder recognizes specific calls.
        self.push(Node::Call(callable, arguments));
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "pickle opcode dispatch is a single table"
    )]
    fn run(mut self) -> Result<Document> {
        loop {
            self.budget.ops = self
                .budget
                .ops
                .checked_sub(1)
                .ok_or_else(|| Error::limit("PyTorch pickle operation limit exceeded"))?;
            if self.frame_left == Some(0) {
                self.frame_left = None;
            }
            let opcode = self.byte()?;
            let node = match opcode {
                b'.' => {
                    let root = self.pop()?;
                    if !self.stack.is_empty()
                        || !self.marks.is_empty()
                        || self.frame_left.is_some_and(|left| left != 0)
                    {
                        return Err(malformed());
                    }
                    return Ok(Document {
                        nodes: self.nodes,
                        root,
                    });
                }
                0x80 => {
                    if self.byte()? > 5 {
                        return Err(Error::unsupported("unsupported PyTorch pickle protocol"));
                    }
                    continue;
                }
                0x95 => {
                    if self.frame_left.is_some() {
                        return Err(malformed());
                    }
                    let size = u64::from_le_bytes(self.array()?);
                    if size > self.budget.bytes {
                        return Err(Error::limit("PyTorch pickle frame exceeds the byte limit"));
                    }
                    self.frame_left = Some(size);
                    continue;
                }
                b'(' => {
                    self.marks.push(self.stack.len());
                    continue;
                }
                b'0' if self.marks.last() == Some(&self.stack.len()) => {
                    self.marks.pop();
                    continue;
                }
                b'0' => {
                    self.pop()?;
                    continue;
                }
                b'1' => {
                    self.marked()?;
                    continue;
                }
                b'2' => {
                    let top = self.top()?;
                    self.stack.push(top);
                    continue;
                }
                b'N' => Node::None,
                0x88 => Node::Bool(true),
                0x89 => Node::Bool(false),
                b'J' => Node::Int(i128::from(i32::from_le_bytes(self.array()?))),
                b'K' => Node::Int(i128::from(self.byte()?)),
                b'M' => Node::Int(i128::from(u16::from_le_bytes(self.array()?))),
                b'I' | b'L' => {
                    let value = self.line_text()?;
                    match (opcode, value.as_ref()) {
                        (b'I', "00") => Node::Bool(false),
                        (b'I', "01") => Node::Bool(true),
                        _ => value
                            .trim_end_matches('L')
                            .parse::<i128>()
                            .map_or(Node::Other, Node::Int),
                    }
                }
                0x8a | 0x8b => {
                    let size = if opcode == 0x8a {
                        u64::from(self.byte()?)
                    } else {
                        u64::from(u32::from_le_bytes(self.array()?))
                    };
                    self.long(size)?
                }
                b'G' => Node::Float(f64::from_be_bytes(self.array()?)),
                b'F' => Node::Float(self.line_text()?.parse().map_err(|_error| malformed())?),
                b'X' | b'T' | b'B' | b'U' | b'C' | 0x8c..=0x8e | 0x96 => {
                    let size = match opcode {
                        b'U' | b'C' | 0x8c => u64::from(self.byte()?),
                        0x8d | 0x8e | 0x96 => u64::from_le_bytes(self.array()?),
                        _ => u64::from(u32::from_le_bytes(self.array()?)),
                    };
                    self.string(size, matches!(opcode, b'X' | b'T' | b'U' | 0x8c | 0x8d))?
                }
                b'V' => Node::Text(self.line_text()?),
                b'S' => Node::Text(decode_quoted(&self.line()?)?),
                b'c' => Node::Global(self.line_text()?, self.line_text()?),
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (&self.nodes[module], &self.nodes[name]) {
                        (Node::Text(module), Node::Text(name)) => {
                            self.budget.charge((module.len() + name.len()) as u64)?;
                            Node::Global(module.clone(), name.clone())
                        }
                        _ => return Err(malformed()),
                    }
                }
                b')' | b']' => Node::Seq(Vec::new()),
                0x8f => Node::Set(Vec::new()),
                b't' | b'l' => Node::Seq(self.marked()?),
                0x91 => Node::Set(self.marked()?),
                0x85..=0x87 => Node::Seq(self.fixed(usize::from(opcode - 0x84))?),
                b'}' => Node::Dict(Vec::new()),
                b'd' => Node::Dict(Self::pairs(&self.marked()?)?),
                b'a' | b'e' | 0x90 => {
                    let mut values = if opcode == b'a' {
                        vec![self.pop()?]
                    } else {
                        self.marked()?
                    };
                    let target = self.top()?;
                    match &mut self.nodes[target] {
                        Node::Seq(items) if opcode != 0x90 => items.append(&mut values),
                        Node::Set(items) if opcode == 0x90 => items.append(&mut values),
                        Node::Call(..) => {}
                        _ => return Err(malformed()),
                    }
                    continue;
                }
                b's' | b'u' => {
                    let values = if opcode == b's' {
                        self.fixed(2)?
                    } else {
                        self.marked()?
                    };
                    let target = self.top()?;
                    let mut pairs = Self::pairs(&values)?;
                    match &mut self.nodes[target] {
                        Node::Dict(items) => items.append(&mut pairs),
                        Node::Call(..) => {}
                        _ => return Err(malformed()),
                    }
                    continue;
                }
                b'R' | 0x81 => {
                    let arguments = self.pop()?;
                    let callable = self.pop()?;
                    self.apply(callable, arguments)?;
                    continue;
                }
                0x92 => {
                    self.pop()?;
                    let arguments = self.pop()?;
                    let callable = self.pop()?;
                    Node::Call(callable, arguments)
                }
                b'b' => {
                    self.pop()?;
                    self.top()?;
                    continue;
                }
                b'Q' => Node::Persistent(self.pop()?),
                b'P' => {
                    let id = self.line_text()?;
                    let value = self.nodes.len();
                    self.nodes.push(Node::Text(id));
                    Node::Persistent(value)
                }
                b'p' | b'q' | b'r' | b'g' | b'h' | b'j' | 0x94 => {
                    let key = match opcode {
                        b'p' | b'g' => self.line_text()?.parse().map_err(|_error| malformed())?,
                        b'q' | b'h' => usize::from(self.byte()?),
                        0x94 => self.memo.len(),
                        _ => usize::try_from(u32::from_le_bytes(self.array()?))
                            .map_err(|_error| malformed())?,
                    };
                    if matches!(opcode, b'g' | b'h' | b'j') {
                        self.stack.push(*self.memo.get(&key).ok_or_else(malformed)?);
                    } else {
                        self.memo.insert(key, self.top()?);
                    }
                    continue;
                }
                _ => {
                    return Err(Error::unsupported(format!(
                        "unsupported PyTorch pickle opcode {opcode:#04x}"
                    )));
                }
            };
            self.push(node);
        }
    }
}

// Protocol 0 Python byte strings occur in historical checkpoint archives.
fn decode_quoted(bytes: &[u8]) -> Result<Box<str>> {
    let [quote @ (b'\'' | b'"'), inner @ .., end] = bytes else {
        return Err(malformed());
    };
    if quote != end {
        return Err(malformed());
    }
    let mut chars = inner.iter().copied();
    let mut decoded = Vec::new();
    while let Some(byte) = chars.next() {
        let next = if byte == b'\\' {
            match chars.next().ok_or_else(malformed)? {
                b'x' => {
                    let high = chars
                        .next()
                        .and_then(|byte| char::from(byte).to_digit(16))
                        .ok_or_else(malformed)?;
                    let low = chars
                        .next()
                        .and_then(|byte| char::from(byte).to_digit(16))
                        .ok_or_else(malformed)?;
                    u8::try_from(high * 16 + low).map_err(|_error| malformed())?
                }
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                other => other,
            }
        } else {
            byte
        };
        decoded.push(next);
    }
    String::from_utf8(decoded)
        .map(String::into_boxed_str)
        .map_err(|_error| malformed())
}
