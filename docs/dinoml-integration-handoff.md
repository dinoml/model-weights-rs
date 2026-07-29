# DinoML integration handoff

## Outcome

Integrate `model-weights` as DinoML's checkpoint inventory, binding-plan,
preparation, and bounded-delivery path. Keep device allocation, host-to-device
transfer, allocator-domain reuse, and `ModuleBuilder::set_shared_constant` in
DinoML.

The checked SD1.5 harness is the executable reference adapter:

- `benchmarks/dinoml-sd15-loader/src/contract.rs` translates DinoML artifact
  constants into one target contract.
- `benchmarks/dinoml-sd15-loader/src/model_lane.rs` builds `TargetTensor`
  values, operation graphs, IDs, and a `BindingPlan`, then consumes
  `WeightDelivery`.
- `benchmarks/dinoml-sd15-loader/src/legacy_lane.rs` is parity evidence only
  and must not become part of the production adapter.

The 939-target, 2,065,322,934-byte CLIP text + UNet + VAE decode contract
validates with zero metadata or byte-digest mismatches.

As of the first DinoML integration slice, migration steps 1 and 2 below are
implemented in `H:\dinoml_v2`:

- `dinoml-checkpoint::PlannedWeights` translates and validates artifact
  constants, executes bounded preparation, and delivers owned CPU tensors.
- `ClipBuilder::planned_weight_loading(true)` selects the path for text-only
  services while DinoML retains synchronous device upload and module binding.
- A real CLIP B/32 ROCm test produced a byte-identical normalized embedding
  through legacy and planned loaders.
- A real CLIP L/14 ROCm test, matching the SD1.5 text encoder, produced a
  nonzero inference result through the planned loader.

## Boundary

```text
hf-store snapshot + model-configs facts + DinoML artifact constants
                                |
                                v
             model-weights inventory / plan / prepare
                                |
                                v
                  PreparedSink<WeightDelivery>
                                |
                                v
        DinoML host staging / GPU upload / shared constants
```

`model-weights` must not load a GPU runtime, allocate device memory, choose a
device, or own upload streams. Its terminal host object is
`WeightDelivery::Prepared`; other delivery variants are explicit provider or
runtime handoffs rather than implicit fallback behavior.

## Model-weights source contract

Use the source mode matching the guarantee available to the adapter:

- `SourceDescriptor::retained` enables mmap and reuses the upstream digest. It
  is an audited unsafe boundary: the retained guard must prevent mutation,
  replacement, deletion, and truncation for every derived mapped-view
  lifetime.
- `SourceDescriptor::local_with_trusted_digest` reuses an upstream-verified
  digest without enabling mmap. This is the conservative hf-store integration
  when the cache lease is accepted for identity but cannot prove the stronger
  mapping invariant.
- `SourceDescriptor::local_with_digest` independently rehashes and verifies an
  ordinary file.

Do not use hf-store's ID as a model-weights `SelectionId`. Hash a canonical,
versioned DinoML selection record containing hf-store's selection ID plus
component, variant, overlay, and other byte-affecting selection facts.

Production retained mmap from an hf-store-owned cache still needs an explicit
exclusive-cache-root policy. A snapshot reader lease protects against
cooperating cache maintenance; by itself it does not prove that an out-of-band
process cannot edit a weight file.

## Recommended DinoML shape

Add the adapter at the shared checkpoint/runtime boundary, not separately in
CLIP, UNet, and VAE services. `dinoml-checkpoint` is the smallest existing
dependency surface that already owns `ConstantBinding`, `ConstantTarget`, and
the legacy materialization path. A parallel planned-loader type can be
introduced there and migrated component by component before the legacy loader
is removed.

The shared adapter should:

1. Accept only the selected checkpoint shards and optional bounded
   safetensors index.
2. Open one `model_weights::Checkpoint` and retain it through materialization.
3. Translate the authoritative `Artifact::constants()` entries into
   `TargetTensor` values. Preserve complete binding operations, logical shape
   and strides, physical storage shape and strides, dtype, and exact output
   byte count.
4. Freeze versioned `ManifestId`, `SelectionId`, `ContractId`, and `BackendId`
   schemas. The backend ID must include every byte-affecting DinoML layout ABI
   fact.
5. Build and validate one canonical `BindingPlan` before allocating or
   uploading constants.
6. Execute one bounded pipeline per selected component contract and route
   deliveries by target name. Do not materialize the same constant separately
   for every artifact profile.
7. Convert finalized host bytes into DinoML tensors, upload through the exact
   `ModuleBuilder::device_runtime()` allocator domain, and share the resulting
   `Arc<Tensor>` only among compatible builders.

For the first integration, the existing synchronous runtime path is adequate:
construct a CPU tensor from the prepared bytes and call
`DeviceRuntime::copy_tensor`, then `set_shared_constant`. This preserves the
current GPU ownership boundary but introduces one host copy because DinoML's
byte constructors allocate their own CPU storage.

A later DinoML optimization may add a runtime-owned host-staging API that
accepts finalized bytes directly or fills a `PinnedHostBuffer`, then uses the
existing asynchronous ROCm stream/event APIs. That optimization belongs in
DinoML and must account for the model-weights pipeline's prepared-byte budget
until the upload queue has accepted ownership.

## Migration order

1. **Complete:** add the shared adapter and contract/ID tests in
   `dinoml-checkpoint`.
2. **Complete:** migrate CLIP text profiles behind an opt-in loader selection.
3. Migrate SD1.5 UNet and VAE decode using the same adapter.
4. Run the checked 939-target parity harness and full native module-build probe.
5. Make the planned loader the default after cold/warm and failure-path
   conformance is green.
6. Remove legacy materialization only after all remaining
   `MappedSafetensors` consumers have an explicit migration decision.

The repository contains additional direct users in GLM OCR, Qwen3 TTS, and
external-state generation. Replacing SD1.5 loading does not authorize silently
changing those consumers.

## Acceptance criteria

- Exact target name, dtype, logical shape, logical stride, physical layout,
  output length, and byte-digest parity for all 939 checked SD1.5 targets.
- Single-file and sharded-index checkpoints select the same exact resolved
  snapshot as configuration discovery.
- Float16, bfloat16, and float32 source paths are covered, including planned
  casts, ordered concat, reshape, transpose, and CK KXC/KYXC storage.
- Missing required targets, ambiguous aliases, unsupported bindings, corrupt
  payloads, and shard-index disagreement fail before module construction.
- Upstream-trusted source digests are not redundantly scanned by
  model-weights.
- Cancellation and source/scratch/prepared budgets remain effective through
  delivery. A downstream upload queue owns and reports any budget after sink
  acceptance.
- One device upload is reused only inside the same allocator/runtime domain.
- Prepared-cache cold and warm runs preserve exact output identity.
- GPU upload and module binding are measured separately from host preparation;
  neither timing is mislabeled as the other.

## Remaining model-weights work

Land the `DigestPolicy::TrustExternal` and
`SourceDescriptor::local_with_trusted_digest` changes before DinoML updates its
pinned `model-weights` revision. The current DinoML slice pins commit
`f9f7fd0e9132e42cb91985d0569a900e9b8a494f`, so its ordinary local source still
performs a complete identity hash. Debug-mode L/14 startup made that redundant
scan visible; do not describe current startup measurements as the final
materialization result.

`Operation::Pad` now provides deterministic high-end zero padding while
permuting logical axes into larger physical storage. DinoML should use it for
padded CK KYXC and validate logical shape, storage shape, storage strides, byte
length, and the zero-filled channel tail against the legacy materializer.

Broader replacement still needs ecosystem conformance for multi-shard
repositories, variant selection, corruption and cancellation, concurrent
loads, and non-SD1.5 consumers. Quantized runtime/provider handoffs must remain
explicit and should be added only with a concrete backend contract.
