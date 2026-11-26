# model-weights

`model-weights` is a safe, deterministic Rust substrate for turning immutable
model checkpoint bytes into runtime-ready weight deliveries.

It owns:

- bounded Safetensors, GGUF, and shard-index inventory;
- deterministic tensor selection, aliases, ordered multi-source target binding,
  typed operation graphs, and overlay plans;
- versioned dtype/layout preparation and provider recipes;
- content-addressed prepared-weight caching;
- bounded parallel materialization with cancellation, progress, and telemetry;
- explicit packed-weight routes without choosing a runtime's quantization policy.

It deliberately does **not** acquire files from a network, interpret model
configuration, compile an inference graph, allocate device memory, or run
inference. Callers supply local paths or retained immutable snapshots, a target
contract, and any runtime-specific preparation or packed-weight providers.

The main consumer is [DinoML](https://github.com/hlky/dinoml_v2), while the
crate's contracts remain runtime-neutral. It is designed to compose with
[hf-store-rs](https://github.com/dinoml/hf-store-rs) for immutable snapshot
lifetimes and [model-configs-rs](https://github.com/dinoml/model-configs-rs)
for normalized configuration facts.

## Current scope

Safetensors and single- or multi-file GGUF inputs are parsed with
caller-configurable bounds and without copying tensor payloads during
inventory. GGUF v2 and v3 metadata, including nested arrays, is retained as
typed values. Every currently file-valid GGML scalar and packed storage type is
inventoried with its exact block geometry; scalar storage remains plain and
packed payloads retain their bytes and normalized row-block descriptors.
Removed, reserved, and unknown GGML type codes fail closed. Ordinary files use
positional reads and are hashed only when a content identity is needed.
Retained immutable snapshots can reuse a previously verified digest and
provide zero-copy mapped views. Authorizing that mapping is an explicit audited
`unsafe` boundary, normally contained in the snapshot-store adapter. An
upstream-verified digest can also be reused for a conservative copied source
without asserting mmap safety.

Binding-plan schema v2 can bind an ordered set of named source tensors to one
target. A small, versioned operation graph covers N-input concatenation,
logical or physical axis permutation, zero-padded physical storage, slicing,
ordered splitting, metadata-only reshape, and provider-backed preparation such
as dtype conversion. Graph
construction infers and validates every intermediate before allocation.

Tensor contracts distinguish the consumer-visible logical shape and strides
from physical storage shape and strides. This allows a target to retain logical
OIHW semantics while storing bytes as OHWI/KYXC, including channel-padded
storage, without presenting the physical byte order as a different logical
tensor.

The bounded core host executor can materialize a graph, or the materializer can
delegate the same validated graph and exact ordered input views to a consumer
for fused CPU or GPU execution. Cache identities include ordered sources and
spans, graph parameters and implementation versions, and the complete target
contract; changing Q/K/V order therefore cannot reuse another result.
On x86-64, F16/BF16 3x3 OIHW-to-OHWI materialization uses a runtime-detected
AVX2 transpose; other processors retain the portable scalar path.

Packed encodings are represented explicitly. Versioned routes describe whether
a consumer will keep packed bytes, dequantize on the host, dequantize into
device scratch, or repack. Kernel execution and scratch allocation stay with
the consumer. This lets DinoML use fused in-tile or GPU dequantization while
other runtimes can register different providers. Typed operation graphs do not
implicitly decode packed data or select a quantization policy; expanded
quantized encodings and decode paths remain separate work.

Conversion policy is similarly extensible. Declarative, versioned recipes can
normalize source namespaces and layouts without embedding one framework's
model catalog in this crate. A future Python binding can run reference
converters—for example Hugging Face Diffusers single-file conversion—as an
offline validation oracle while Rust remains the production path.

See [ADR 0001](docs/adr/0001-weight-substrate-boundaries.md) for the detailed
boundary and identity model and
[ADR 0002](docs/adr/0002-typed-operation-graphs.md) for the schema-v2 graph,
execution, and cache-identity decision. The
[ecosystem integration guide](docs/integrations.md) documents the concrete
`hf-store-rs`, `model-configs-rs`, DinoML, GGUF/provider, and Diffusers/Python
adapter seams. The [DinoML integration handoff](docs/dinoml-integration-handoff.md)
turns the checked SD1.5 adapter into a staged runtime migration. The
[benchmark protocol](docs/benchmarking.md) defines the Stable Diffusion 1.5
regression comparison.

## PyTorch weights

`Checkpoint` recognizes modern PyTorch ZIP/ZIP64 archives, legacy pickle
streams, and historical PyTorch tar checkpoints by content. Hugging Face
`pytorch_model.bin.index.json` can use the same `weight_map` entry point as
Safetensors. A checkpoint can contain a flat state dictionary, a
`state_dict` wrapper, a single unambiguous nested weight dictionary, a
standalone tensor (named `tensor`), or a sequence of tensors (named by
their zero-based positions). Bare storages are exposed as one-dimensional
tensors. Quantized linear weight/bias pairs stored under `_packed_params`
are exposed with `.0` and `.1` suffixes.

The first-party parser interprets pickle as data and observes resource limits
and cancellation. ZIP members may be stored or DEFLATE-compressed. Compressed
metadata is expanded within the header budget. Compressed storage members are
expanded on demand into temporary files, with length and CRC validation and a
configurable expanded-size limit per member.

Contiguous little-endian tensors in stored members use direct byte spans.
Strided and big-endian tensors produce row-major little-endian bytes on
demand. Sparse COO, CSR, CSC, BSR and BSC tensors are delivered as dense bytes
in their original value dtype. PyTorch affine quantized tensors are delivered
as dequantized FP32 bytes. A configurable per-read limit bounds normalized
output. These views use owned reads and are unavailable with forced
`AccessMode::Mmap`.

Use `metadata::read_pytorch_field` for a named top-level field such as
`ckpt_config`. It returns a bounded JSON-compatible value without loading
tensor storage. `metadata::read_pytorch_field_from_reader` accepts a seekable
source, including a caller-supplied range reader. Model-specific schema
validation belongs to the consumer.

TorchScript and Python model reconstruction are outside the weight reader's
supported subset.

## Safety and compatibility

- Rust 1.95 or newer, edition 2024.
- Apache-2.0 licensed.
- No payload hashing during header-only inventory.
- A path ending in `.gguf` must contain GGUF magic and never falls back to
  Safetensors parsing.
- Packed GGUF decoding and target dtype conversions use explicit preparation routes.
- Operation-graph input order is semantic and participates in plan and cache
  identity.
- Tensor dtype, rank, logical and storage geometry, layout, byte length, and
  checked arithmetic are validated before graph output allocation.
- Preparation providers declare exact output and scratch bytes; undeclared
  size-dependent workspace allocations are forbidden.
- No path traversal through shard indexes.
- Parser counts, pipeline work items, and cache maintenance scans are bounded.
- Serialized, byte-affecting contracts carry schema and implementation
  versions.
- Public APIs are additive-first and use stable error categories.

## Development

```text
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo test --workspace --no-default-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

The repository is early-stage. Semver compatibility begins with the first
published release; serialized schema versions are independent and are already
treated as compatibility boundaries.
