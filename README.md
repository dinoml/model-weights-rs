# model-weights

`model-weights` is a safe, deterministic Rust substrate for turning immutable
model checkpoint bytes into runtime-ready weight deliveries.

It owns:

- bounded safetensors and shard-index inventory;
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

Safetensors inputs are parsed with caller-configurable bounds and without
copying tensor payloads during inventory. Ordinary files use positional reads
and are hashed only when a content identity is needed. Retained immutable
snapshots can reuse a previously verified digest and provide zero-copy mapped
views. Authorizing that mapping is an explicit audited `unsafe` boundary,
normally contained in the snapshot-store adapter. An upstream-verified digest
can also be reused for a conservative copied source without asserting mmap
safety.

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

## Safety and compatibility

- Rust 1.85 or newer, edition 2024.
- Apache-2.0 licensed.
- No payload hashing during header-only inventory.
- No implicit quantized decode or dtype conversion.
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
