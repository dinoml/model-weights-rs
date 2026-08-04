# Ecosystem integration guide

`model-weights` is the checkpoint-to-runtime boundary. It does not acquire Hub
files, interpret model classes, choose an inference graph, allocate device
memory, or own GPU kernels. The intended composition is:

```text
hf-store Snapshot
       |
       v
model-configs normalized facts + consumer artifact contract
       |
       v
model-weights Checkpoint -> BindingPlan -> WeightDelivery
       |
       v
DinoML or another runtime/provider
```

The public names in this guide are current APIs in
[`hf-store-rs`](https://github.com/dinoml/hf-store-rs),
[`model-configs-rs`](https://github.com/dinoml/model-configs-rs),
[`model-weights-rs`](https://github.com/dinoml/model-weights-rs), and
[`DinoML`](https://github.com/hlky/dinoml_v2). The repositories intentionally
have no mandatory dependencies on one another. Consequently, the wiring below
is clearly marked pseudocode and belongs in a consumer adapter crate.

## `hf-store-rs`: retained immutable sources

Online `HubStore::fetch` and transport-free `OfflineStore::open_request` both
return an `hf_store::Snapshot`. Its `files()` are in canonical repository-path
order, and each `SnapshotFile` exposes `path()`, `local_path()`, `sha256()`,
`size()`, and `form()`.

The production zero-copy adapter uses
`model_weights::source::SourceDescriptor::retained`. That constructor is
`unsafe`: a valid digest and a live cache lease are necessary, but the adapter
must also guarantee that no cooperating or out-of-band writer can mutate,
replace, or truncate the file while any derived checkpoint, mapping, or
`ByteView` remains alive.

Use the hf-store-owned cache for this path and retain the whole `Snapshot`, not
an individual `SnapshotFile`. `Snapshot::clone` retains the shared reader
lease; `SnapshotFile::clone` does not own that lease.

Illustrative adapter pseudocode (not compiled in this repository):

```rust
// PSEUDOCODE: error conversion and file selection are adapter policy.
fn retained_weight_sources(
    snapshot: &hf_store::Snapshot,
) -> model_weights::Result<Vec<model_weights::source::SourceDescriptor>> {
    use hf_store::{CacheMode, SnapshotFileForm};
    use model_weights::identity::ContentDigest;
    use model_weights::source::SourceDescriptor;
    use model_weights::{Error, ErrorCategory};

    if snapshot.cache_mode() != CacheMode::Owned {
        return Err(Error::from_category(
            ErrorCategory::Unsupported,
            "retained mmap requires an hf-store-owned immutable cache",
        ));
    }

    snapshot
        .files()
        .iter()
        .filter(|file| file.path().as_str().ends_with(".safetensors"))
        .map(|file| {
            if file.form() != SnapshotFileForm::Owned {
                return Err(Error::from_category(
                    ErrorCategory::Unsupported,
                    "retained mmap requires an owned snapshot file",
                ));
            }
            let digest = file.sha256().parse::<ContentDigest>()?;

            // SAFETY: this adapter has exclusive cache-root ownership; hf-store
            // published an immutable owned file, validated `size` and `sha256`,
            // and `snapshot.clone()` keeps its reader lease alive until every
            // descriptor-derived owner has been dropped.
            unsafe {
                SourceDescriptor::retained(
                    file.path().as_str(),
                    file.local_path(),
                    file.size(),
                    digest,
                    snapshot.clone(),
                )
            }
        })
        .collect()
}
```

The safety comment is part of the adapter contract, not boilerplate. If another
process can edit the cache root, or when using `CacheMode::Compatible`, do not
assert the retained-mapping invariant. When hf-store's digest validation is an
accepted integrity boundary, use
`SourceDescriptor::local_with_trusted_digest(...).with_logical_path(...)`; this
reuses the verified digest but remains an ordinary copied source under
`AccessMode::Auto`. Use `local_with_digest` instead when model-weights must
independently verify the declared digest. Both local modes require the consumer
to prevent concurrent mutation while the checkpoint is in use.

Pass only the selected component and variant shards to
`CheckpointBuilder::from_sources`. Prefer an available checkpoint variant whose
dtype already matches the target contract; a preparation cast is a fallback
for a genuine source/target mismatch, not a normal load step. The builder sorts
the selected shards by canonical logical path; that resulting order defines
`PlanInputs::source_digests()` and quantized companion file ordinals. Do not
introduce a second ordinal convention in the adapter. If a safetensors index is
selection truth, parse its bounded bytes with
`inventory::ShardIndex::from_json` and attach it with
`CheckpointBuilder::shard_index` before `open_with_cancellation`.

## `model-configs-rs`: views, manifest identity, and selection

`ModelRepository::read` parses supported data files and retains exact source
bytes. Two surfaces serve different adapter needs:

- `ModelRepository::normalized()` returns `ModelRepositoryConfig`, including
  `architecture`, `architecture_source`, `task`, `components`, and
  `applied_defaults`.
- `SourceDocument::typed_view()` preserves format-specific distinctions. For a
  Diffusers `model_index.json`, `TypedDocumentView::ModelIndex` exposes
  `ModelIndexView::class_name`, `diffusers_version`, `pipeline_tag`, `task`,
  and the `components()` iterator. Each `DiffusersComponent` exposes `name()`,
  `raw()`, and `value()`.
- `TypedDocumentView::SafetensorsIndex` exposes
  `SafetensorsIndexView::weight_map`; a present
  `SafetensorsWeightMapView::entries()` yields each tensor name and typed shard
  path without loading weight payloads.

Typed `SourceField` values deliberately distinguish missing, null, invalid, and
present data. Adapters should not collapse those cases before selection and
diagnostics are complete.

`ModelRepository::manifest()` produces a `CompatibilityManifest`, and
`CompatibilityManifest::to_json_pretty()` is documented as deterministic.
`model-configs` and `model-weights` do not share an ID type, so the bridge is an
adapter convention that must be domain-separated and versioned:

```rust
// PSEUDOCODE: the domain string is an adapter ABI and must be frozen.
let repository = model_configs::ModelRepository::read(snapshot.directory())?;
let normalized = repository.normalized()?;
let manifest = repository.manifest()?;
let manifest_bytes = manifest.to_json_pretty()?;

let manifest_id = model_weights::identity::ManifestId::from_digest(
    model_weights::identity::ContentDigest::hash(
        "model-configs-compatibility-manifest-v1",
        [manifest_bytes.as_bytes()],
    ),
);
```

Do not cast or equate `hf_store::SelectionId` with
`model_weights::identity::SelectionId`. The former identifies an exact path
set; the latter must commit to all component, variant, overlay, and other
consumer selection facts that affect planning. Serialize those facts with a
canonical, versioned adapter schema, include `snapshot.selection_id()` as one
lowercase hexadecimal string via `to_string()`, then domain-hash the resulting
bytes.

Likewise, derive:

- `ContractId` from the canonical consumer target-constant contract;
- `BackendId` from the backend, layout ABI, and other byte-affecting runtime
  compatibility facts.

Configuration selects the architecture and component dialect. It does not by
itself define the target constants. In DinoML, the validated
`Artifact::constants()` entries are the authoritative names, shapes, dtypes,
layouts, checkpoint bindings, and storage requirements.

Once the adapter has built `TargetTensor` values, the core plan assembly uses
the public API directly:

```rust
// PSEUDOCODE: target translation and canonical ID helpers are consumer code.
let sources = checkpoint
    .inventory()
    .tensors()
    .iter()
    .map(model_weights::plan::SourceTensor::try_from)
    .collect::<model_weights::Result<Vec<_>>>()?;

let inputs = model_weights::plan::PlanInputs::new(
    manifest_id,
    selection_id,
    contract_id,
    backend_id,
    checkpoint.source_digests(&cancellation)?,
);

let plan = model_weights::plan::BindingPlan::builder(inputs)
    .sources(sources)
    .targets(targets)
    .extra_source_policy(model_weights::plan::ExtraSourcePolicy::Allow)
    .build()?;
```

Use `ExtraSourcePolicy::Allow` only when extra inventory tensors are expected
and retained as explicit `unused_sources()` evidence. It must not substitute
for selecting the correct component or variant in hf-store.

## DinoML delivery

DinoML currently materializes constants through
`dinoml_checkpoint::MappedSafetensors` and
`CheckpointBindings::materialize_metadata`, then constructs
`dinoml_runtime::Tensor` values and calls
`ModuleBuilder::set_shared_constant`. A model-weights adapter can replace the
checkpoint/materialization part while retaining the runtime module boundary.

`Materializer::execute` accepts any
`PreparedSink<WeightDelivery>`. Delivery is in canonical target order and owns
its `PreparedItem`, so a sink may either consume it synchronously or move it
into a runtime queue. Once the callback returns, any queued-byte budget and
backpressure are the consumer's responsibility.

Enable reuse with `Materializer::with_cache` before execution. Provider
publication through `Materializer::publish_prepared_bytes` requires that cache
and a binding with a final host-output address.

Illustrative DinoML sink pseudocode (not compiled in this repository):

```rust
// PSEUDOCODE: checked dtype/layout/error translation is adapter code.
let mut sink = |ordinal,
                item: model_weights::pipeline::PreparedItem<
                    model_weights::materialize::WeightDelivery,
                >,
                cancellation: &model_weights::CancellationToken| {
    cancellation.check()?;

    match item.into_value() {
        WeightDelivery::Prepared(weight) => {
            let metadata = constant_by_name(weight.target_name())?;
            validate_representation(metadata, weight.representation())?;
            let shape = checked_usize_shape(weight.shape())?;
            let dtype = to_dinoml_dtype(weight.representation().dtype())?;

            let tensor = dinoml_runtime::Tensor::from_cpu_bytes(
                dtype,
                shape,
                weight.bytes().as_slice(),
            )
            .map_err(delivery_error)?;

            module_builder
                .set_constant(weight.target_name().as_str(), tensor.view())
                .map_err(delivery_error)?;
        }
        WeightDelivery::Operation(handoff) => {
            operation_provider.submit(ordinal, handoff, cancellation)?;
        }
        WeightDelivery::Conversion(handoff) => {
            conversion_provider.submit(ordinal, handoff, cancellation)?;
        }
        WeightDelivery::Quantized(handoff) => {
            quantized_provider.submit(ordinal, handoff, cancellation)?;
        }
        WeightDelivery::Overlay(handoff) => {
            overlay_provider.submit(ordinal, handoff, cancellation)?;
        }
        _ => return Err(unsupported_future_delivery()),
    }
    Ok(())
};

materializer.execute(&pipeline, &mut sink, &observer)?;
```

An observer that needs a transform breakdown can opt into
`ExecutionEvent::OperationFinished` by wrapping it with
`model_weights::telemetry::with_operation_events(observer)`. Events identify
the binding, `PlannedTransform`, or graph `NodeId`, classify the executed
operation, and report elapsed time plus input, output, and actually materialized
byte spans.
Pipeline events carry the stable work ordinal. A direct
`materialize_with_observer` call has no ordinal, so callers that need target
attribution should scope that observer to the named call.
`OperationKind::Cast` is emitted only for a dtype-changing transform that
actually ran. A matching source contract emits `Identity` with zero
materialized bytes. Ordinary observers, including closures, do not request
per-operation timers unless explicitly wrapped or their
`operation_events_enabled` implementation returns `true`.

That contiguous constructor is only the simple case. The real adapter must
validate the matching `TensorMetadata::layout` and use
`Tensor::from_cpu_bytes_with_strides` or
`Tensor::from_cpu_storage_bytes_with_strides` for the corresponding physical
contract. It must explicitly map the supported intersection of
`model_weights::tensor::DType` and `dinoml_runtime::DType`, rejecting the rest.

`ModuleBuilder::set_constant` copies or stages a CPU tensor. For multiple
compatible profiles, DinoML can allocate/upload once through the exact shared
`DeviceRuntime` and pass `Arc<Tensor>` to
`ModuleBuilder::set_shared_constant`. Package-authorized state paths can use
`ModuleBuilder::bind_constant_verified` after the adapter has computed and
validated the final content SHA-256.

### Binding-plan v2 and typed operation graphs

Binding-plan schema v2 supports an ordered set of external sources for one
target. `OperationGraphBuilder::add_input` preserves insertion order, and the
resolver binds sources in that exact order rather than sorting them by name.
Node order, edge order, split-output order, and the selected final output are
also stable serialized semantics.

Each graph input and output carries `TensorFacts` with separate logical shape,
storage shape, logical strides, storage strides, scalar dtype, physical layout,
and exact byte length:

- logical shape and strides describe the tensor DinoML exposes to model code;
- storage shape describes the physical axis order; and
- storage strides are indexed by logical axis and locate those coordinates in
  physical storage.

For example, a storage permutation with order `[0, 2, 3, 1]` retains logical
OIHW semantics while producing OHWI/KYXC storage. The adapter should construct
the DinoML view from both stride contracts instead of replacing the logical
shape with physical axis order.

The graph supports:

- `Concat`: two or more compatible inputs joined along one logical axis;
- `Permute`: either logical-axis transpose into contiguous storage or a
  physical-storage permutation that retains logical axes;
- `Slice`: one half-open range per logical axis, producing a materialized
  output;
- `Split`: ordered, non-empty, non-overlapping ranges along one logical axis,
  producing ordered outputs;
- `Reshape`: a metadata-only reinterpretation of compatible contiguous
  storage; and
- `Prepare`: one versioned `PlannedTransform`, such as a dtype cast.

A graph selects one final value for its target binding. A `Split` node can
produce several values for subsequent nodes, but schema v2 does not make
several target deliveries one atomic publication.

Graph construction validates dtype, rank, logical and storage shapes, both
stride mappings, layout, exact bytes, edge availability, operation parameters,
and checked arithmetic before output allocation. Deserialization replays the
same inference and rejects unsupported graph schema versions or altered
derived facts.

`Materializer` defaults to `OperationExecution::Host`. The bounded host
interpreter preserves zero-copy source, reshape, and fact-preserving identity
permutation paths, materializes the other structural operations, and uses the
preparation engine for `Prepare`.
On x86-64 processors with AVX2, F16/BF16 3x3 OIHW-to-OHWI storage
permutations use a runtime-gated 16-channel SIMD transpose. Other processors
retain the portable scalar builder, and both paths preserve bounded
cancellation.
`OperationExecution::Delegate` instead delivers `WeightDelivery::Operation`;
its `OperationHandoff::inputs()` are immutable source views in exact semantic
order, suitable for DinoML fused CPU or GPU execution. A delegated provider may
publish a validated final host result through
`Materializer::publish_prepared_bytes`.

A genuine concat or storage permutation cannot be represented as one
contiguous `ByteView` without changing bytes. For those routes, “zero-copy”
means retaining the ordered source views through delegation and letting the
consumer fuse assembly/layout conversion into its device upload, scratch
transform, or weight-consuming kernel. Host materialization remains the
portable reference and fallback path.

Pipeline admission accounts for unique source spans, the graph's checked peak
live intermediates and provider scratch, and final prepared bytes. Host kernels
and providers observe cooperative cancellation. Plan and prepared-cache
identity include all ordered source descriptors and spans, graph ordering,
operation parameters and implementation versions, target facts, and schema
versions. Reordering Q/K/V is therefore a different plan and cache address.

Operation graphs cover dense scalar assembly only. They neither interpret
packed GGUF data nor choose dequantization or kernel policy; those remain the
separate quantized handoffs described below.

### Preparation provider memory and cancellation contract

Binding-plan schema v2 requires every in-process `PlannedTransform`, whether in
the legacy unary path or an operation-graph `Prepare` node, to serialize both
its exact output length and its peak provider-workspace length. Both values
participate in binding-plan and prepared-cache identity.
`PlannedTransform::new` records zero workspace; providers that need workspace
must set it explicitly with `with_scratch_bytes`.

`PreparationProvider::validate` is cancellable and metadata-only. It may inspect
the transform, shape, representation, and source length, but it must not read or
convert payload bytes or allocate output/workspace. An allocating provider
declares both lengths through `OutputStrategy::Allocate`. The engine rejects
either mismatch before allocation. The default provider path creates exact
zero-initialized output and scratch slices and passes both to `prepare_into`.
A provider that can safely construct every output byte in order may override
`prepare_allocated` to avoid the redundant output initialization while
retaining the same declared sizes and cancellation contract. The built-in
contiguous float conversion uses this initialized-output path only when the
source and target dtypes differ; matching representations reuse the source.

`prepare_into` implementations must use the supplied scratch slice for all
size-dependent temporary storage. A `prepare_allocated` override must likewise
stay within its declared output and scratch lengths. Undeclared host
allocations, device allocations, thread-local arenas, and other hidden
workspace are forbidden; small fixed-size stack temporaries are permitted.
Long validation or execution loops must check the supplied cancellation token
at bounded intervals.

For the legacy unary path, pipeline scratch admission uses the maximum, across
transform steps, of provider workspace plus the previous and next intermediate
buffers that are simultaneously live. Operation graphs compute the equivalent
bound from node liveness and provider-declared transient scratch. Arithmetic is
checked, so an unrepresentable peak rejects the work instead of wrapping its
budget.

### Quantized handoffs

`QuantizedHandoff` exposes `source()`, `target()`, `storage()`,
`capability()`, `payload()`, `companions()`, and `resident_bytes()`. The
consumer must dispatch on `handoff.capability().route()`:

- `QuantizedRoute::HostDequant { target_dtype }`: a pinned provider such as
  [`libgguf`](https://github.com/hlky/libgguf) decodes to the declared final
  dtype and layout. After validating the exact target contract, the adapter may
  call `Materializer::publish_prepared_bytes`; the current delivery is still
  the provider's responsibility, while later runs may receive
  `PreparedOrigin::Cache`.
- `PackedDirect`: retain or copy the packed payload and companions into the
  allocation ABI expected by the selected kernel.
- `FusedInTile`: bind packed storage to a kernel that decodes while consuming a
  tile.
- `DeviceDequantToScratch { target_dtype }`: upload the packed inputs and let
  the runtime allocate, budget, and recycle transient scratch.
- `Repack { target_encoding }`: run the pinned provider and bind the resulting
  packed representation.

Only final host output from `HostDequant` has a prepared-cache address.
Device-scratch, direct, fused, and repack routes intentionally do not. Provider
operation/version, source encoding, backend, and target layout are carried by
`RouteCapability` and participate in plan/cache identity.

The core parses bounded single- or multi-file GGUF v2 and v3 containers and
retains typed model metadata, including nested arrays, in file-ordinal order.
Every currently file-valid GGML scalar and packed storage type is inventoried
with its exact block geometry. Scalar storage remains plain; packed storage is
represented by row-blocked `QuantizedStorage` descriptors and delivered
byte-for-byte. Removed, reserved, and unknown type codes are rejected
explicitly. DinoML selects `PackedDirect` or another route according to its
actual kernel and memory policy; this crate does not decode, quantize, or
upload the payload.

### Overlay handoffs

`OverlayHandoff::base()` returns the already prepared or delegated base;
`plan()`, `binding()`, and `target_digest()` identify the exact ordered
composition. DinoML may fuse the referenced operations into a kernel or send
them to an eager provider. A validated final host result can be published with
`Materializer::publish_prepared_bytes`; the per-target overlay digest prevents
unrelated layers from invalidating that entry.

## Diffusers recipes and Python differential validation

Diffusers' conversion catalog is broader than a single model loader. Relevant
reference surfaces include its
[`scripts/`](https://github.com/huggingface/diffusers/tree/main/scripts),
[`single_file_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py),
and
[`single_file_model.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_model.py).
A separate Python binding/provider can use those implementations as a pinned
differential oracle without making Python part of the Rust production load
path.

The language-neutral core types are `ConversionRecipe`, `RecipeStep`,
`RecipeInput`, `RecipeValue`, and `ImplementationId`. On a cache miss,
`WeightDelivery::Conversion` supplies a `ConversionHandoff` with the recipe,
exact source descriptor, exact source bytes, and complete output target.
Binding-plan schema v2 still treats that recipe as the whole external
source-to-target stage; it cannot be combined with a planned transform, typed
operation graph, or quantized route.

A reference-validation harness should:

1. Pin the Diffusers commit, Python package lock, recipe schema, and adapter
   `ImplementationId`.
2. Record exact input content digests and the recipe's canonical digest.
3. Run the Python reference converter in a separate process with bounded,
   explicit inputs; do not import arbitrary repository code.
4. Emit an output manifest containing tensor names, shapes, dtypes, layouts,
   byte lengths, and content digests.
5. Run the Rust/provider implementation from the same recipe and inputs.
6. Compare the complete output set and metadata, then exact bytes where the
   operation is specified as bit-exact. Any tolerance-based numeric comparison
   must be an explicit conformance rule, not an implicit fallback.
7. Publish bytes through `Materializer::publish_prepared_bytes` only after the
   provider has validated shape, dtype, layout, semantics, and exact output
   length. The core publication method enforces length and cache identity; it
   does not infer conversion semantics.

Binding-plan schema v2 removes the general single-source limitation for typed
operation graphs. Built-in graphs cover ordered joins, transposes, physical
layout permutations, zero-padded storage, slices, splits, reshapes, and
provider preparation. The
external `ConversionRecipe` contract remains single-input and each binding
still delivers one selected target. A recipe may name several outputs and be
reused by target bindings that explicitly opt into one `shared_source_group`,
but the pipeline does not execute or cache those targets as one atomic group.

Diffusers conversions beyond the built-in graph semantics still need a pinned
provider, an importer-owned intermediate checkpoint, or a future atomic
multi-target conversion contract. The Python differential harness should be
the acceptance oracle for those extensions.

## Requirements for a reproducible SD1.5 comparison

The generic `benchmark_checkpoint` example and public conformance tests cover
the model-weights stages, but they are not an end-to-end DinoML Stable
Diffusion 1.5 result. A publishable comparison still requires downstream work:

1. Add a DinoML adapter that turns the exact CLIP, UNet, and AutoencoderKL
   artifact constants into `TargetTensor` contracts and freezes the manifest,
   selection, contract, and backend identity schemas.
2. Replace or run alongside each current
   `CheckpointBindings::materialize_metadata` loop with a retained hf-store
   `Checkpoint`, `BindingPlan`, `Materializer`, and bounded DinoML sink.
3. Encode every existing `BindingOperation` semantic in schema v2. Combined
   Q/K/V targets must use ordered `Concat` inputs, and logical versus physical
   transposes must use the corresponding `Permute` mode. Any operation outside
   the built-in graph must use an explicitly versioned provider or importer.
4. Implement the production upload queue, allocator-domain sharing, custom
   layout mapping, cancellation propagation, and provider error translation.
   The queue must retain delivered owners and enforce its own byte budget after
   accepting an item.
5. Add Diffusers/Python differential fixtures for every SD1.5 namespace or
   layout conversion used by that adapter, then verify final constant names,
   shapes, dtypes, layouts, and content before timing.
6. Run the old `dinoml-checkpoint` path and new model-weights path against the
   same immutable snapshot, artifact set, component/variant selection, output
   dtype, backend, device, and release build.
7. Collect at least ten process-cold and ten process-warm samples after a
   discarded setup run. Report median and p95 wall time, peak RSS, bytes read,
   hashed, and copied, cache hits/misses, selected/unused tensors, pipeline
   peaks, and delivered bytes. A warm in-process repeat alone is not a
   process-warm measurement.

Acceptance requires equivalent target contracts; comparing all checkpoint
tensors on one path with only graph-selected constants on the other is useful
diagnostically but is not the final result. The measured reports, environment,
and artifact/snapshot identities should be retained with the benchmark record
so later kernel, recipe, or cache changes remain attributable.
