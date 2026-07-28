# ADR 0001: Weight substrate boundaries and stable identities

- Status: accepted
- Date: 2026-07-28

## Context

Model runtimes repeatedly implement the same error-prone path from repository
files to runtime constants: locate shards, parse checkpoint metadata, choose
variants, rename tensors, convert dtypes/layouts, cache results, and stream
bytes into backend-owned allocations. Those implementations often mix file
acquisition, model configuration, graph construction, and device execution.
That makes validation difficult and makes cache reuse unsafe across policy or
kernel changes.

DinoML is the first consumer, but this layer can be useful to other Rust and
Python projects. Diffusers also demonstrates the breadth of legacy/single-file
conversion logic; reproducing that catalog independently in every runtime is
not sustainable.

## Decision

`model-weights` is a model-agnostic substrate with five explicit stages:

1. **Inventory** validates local or retained files and records exact tensor
   storage spans. Inventory reads metadata only.
2. **Plan** deterministically selects source tensors, applies declared aliases
   and overlays, and binds them to a consumer-supplied target contract.
   Binding-plan schema v2 supports semantically ordered multi-source bindings
   through a typed operation graph.
3. **Prepare** executes declared, versioned in-process dtype/layout transforms
   and structural graph operations. Identity and metadata-only reshape preserve
   zero-copy views. Graphs may instead be delegated with exact ordered source
   views for consumer-owned fused execution. Recipe and quantized provider work
   remains an explicit, separate handoff; provider-finalized host bytes can be
   published to and reused from the prepared cache. Each planned transform pins
   exact output and caller-owned scratch lengths. Validation is cancellable and
   metadata-only; execution receives exact engine-owned slices and may not
   allocate undeclared size-dependent workspace.
4. **Cache** publishes complete prepared artifacts atomically under
   content-addressed identities and protects live entries with leases.
5. **Deliver** streams bounded prepared views to consumer callbacks with
   cancellation, progress, and structured telemetry.

Repository acquisition belongs in `hf-store-rs`. Model configuration and
architecture facts belong in `model-configs-rs`. Graph compilation, device
allocation, kernels, and inference belong in DinoML or another consumer.
Adapters and general-purpose import frontends may be separate crates built on
the core contracts.

All externally meaningful choices are explicit inputs to stable identities:

- immutable file content and logical repository paths;
- normalized model/configuration manifest facts;
- component, variant, and overlay selection;
- target constant names, shapes, dtypes, layouts, and requiredness;
- ordered source descriptors and spans, graph edges, operation parameters, and
  logical and physical tensor facts;
- provider name, operation name, and byte-affecting implementation version;
- preparation schema version and backend layout ABI.

Map iteration order, thread scheduling, absolute local paths, timestamps, and
temporary directory names never affect an identity.

## Source trust and lifetimes

An ordinary local source is opened once and uses positional reads. Its SHA-256
is deferred until a content-addressed operation needs it; a predeclared digest
is verified then. Metadata changes detected during hashing are an integrity
failure, but callers remain responsible for excluding malicious concurrent
rewrites. Retained immutable snapshots are the strong integrity path.

A retained immutable snapshot includes a lifetime guard, expected length, and
already verified digest. The guard is held by checkpoints and mapped byte views.
Only this source class may be mapped, avoiding untracked mutation and
use-after-cleanup risks. Constructing it is an explicit `unsafe` adapter
boundary because the operating system cannot prove another handle or process
will not mutate or truncate the mapped file. If mapping is disabled or
unavailable, positional reads remain available.

Shard paths are normalized repository-relative paths. Local index resolution
canonicalizes both the index directory and each shard and rejects any resolved
path outside the directory. As with all ordinary local sources, the caller must
exclude concurrent directory mutation between resolution and open.

## Quantization

Packed storage is data, not a promise to decode it. A descriptor records the
encoding identity, logical dtype/shape, byte span, block geometry, and bounded
opaque parameters. A preparation route then declares one of:

- preserve packed bytes for direct/fused kernel consumption;
- host dequantization to a requested scalar dtype;
- device dequantization to consumer-managed scratch;
- repacking to another encoding.

The core does not ship a GGUF decoder in the initial implementation. Libraries
such as `libgguf`, DinoML kernels, or ecosystem providers can implement routes.
Provider identity and version participate in plan/prepared cache keys, so a
kernel or algorithm change cannot silently reuse incompatible bytes.

Typed operation graphs are not an implicit quantization route. They operate on
validated dense scalar facts and leave packed-direct, fused in-tile,
host-dequant, device-scratch, and repack choices under this separate contract.
Expanded quantized encoding and decode support is deferred independently of
ordered graph assembly.

## Conversion recipes and Python validation

Generally useful dense mechanical operations are represented by the typed
operation graph described in
[ADR 0002](0002-typed-operation-graphs.md). Broader namespace/layout conversion
is represented as declarative, versioned provider recipes. Architecture
catalogs and framework-specific heuristics live in providers or importer
crates.

Python bindings may expose the same inventory and recipe contracts and invoke
reference Python converters offline. Their useful role is source-of-truth
differential validation and migration tooling, not a Python runtime dependency
in the Rust loading path. Exact input facts, recipe/provider version, and output
manifest make those comparisons reproducible.

## Error, cancellation, and recovery contract

Errors have stable categories: I/O, invalid path, invalid format, integrity,
resource limit, binding, unsupported capability, cache, cancellation, and
delivery. Diagnostics add context without changing category-based recovery.

Parsing, hashing, preparation, cache waits, and delivery observe a cloneable
cooperative cancellation token. Cancellation never publishes a partial cache
entry. Bounded queues and byte budgets provide backpressure; a slow consumer
cannot cause unbounded prepared-memory growth.

## Consequences

The core cannot automatically “do the right thing” for an unknown architecture
or encoding. The caller must select a recipe and packed-weight route, which
makes behavior auditable and cache-safe.

Consumers gain deterministic inventories and plans, reusable cache artifacts,
zero-copy retained paths, and a conformance surface independent of their graph
or kernel implementation. New ecosystems can integrate without depending on
DinoML types.
