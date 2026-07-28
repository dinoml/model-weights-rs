# Benchmarking checkpoint loading

Issue #10 uses Stable Diffusion 1.5 as the first end-to-end regression target:
load time should be no worse than the existing DinoML checkpoint path while
avoiding full payload scans during inventory, redundant hashes, redundant
copies, and preparation of unused variants.

Run the portable probe against either one safetensors file or its shard index,
followed by the exact tensor names in the target contract:

```text
cargo run --release --example benchmark_checkpoint -- \
  [--dtype source|f16|bf16|f32] [--cache DIR] \
  PATH TENSOR [TENSOR ...]
```

`--dtype source` is the zero-copy baseline and is the default. The other dtype
values add a pinned built-in contiguous float transform, so source, transform,
and prepared-cache behavior can be compared without changing the selected
tensor set. Built-in casts accept f16, bf16, and f32 sources. The generic probe
accepts plain safetensors tensors; packed tensors require a consumer adapter to
declare the intended quantized route.

`--cache DIR` enables the prepared cache at a persistent directory. Before the
cold pass, the probe removes only the exact prepared-cache addresses for the
selected bindings; plan-cache entries and unrelated prepared entries are left
untouched. The warm pass immediately repeats the same bounded execution. A
dtype transform should therefore report `cold.origin.transform` followed by
`warm.origin.cache`. Source-compatible zero-copy bindings are intentionally
not duplicated in the prepared cache and report `origin.source` in both
passes.

The probe builds a deterministic binding plan, permits unselected inventory
tensors as explicit extras, and executes only its bindings through
`Materializer` and the bounded `Pipeline`. The checkpoint handle used to
establish source identity is reused for both passes, so execution does not
perform a second full-file hash. It reports:

- setup inventory, identity, and planning durations, plus the bytes that
  required identity hashing;
- pending digest bytes immediately before execution (normally zero after
  successful setup identity);
- cold and warm end-to-end and pipeline durations;
- hashing, mapping, source-read, cache-lookup, transform, preparation,
  queue-wait, and delivery-callback duration, byte, and invocation totals;
- submitted, prepared, delivered, failed, and delivered-byte counters;
- peak delivery queue depth and peak source, scratch, and prepared reservations;
- delivered MiB/s and source/transform/cache/other origin counts;
- selected and explicitly unused inventory tensor counts;
- the exact work-item, worker, queue, and byte limits used by the run.

Phase totals can overlap: for example, preparation includes its nested source
read, cache lookup, or transform. The cold pass starts with the selected
prepared-cache entries removed, but deliberately reuses the already-identified
checkpoint handle; `cold.phase.hashing.bytes` should therefore be zero. It does
not flush the operating-system page cache. Use the comparison protocol below
for process-cold measurements.

Example transform/cache run:

```text
cargo run --release --example benchmark_checkpoint -- \
  --dtype f16 --cache .model-weights-bench-cache \
  model.safetensors model.diffusion_model.input_blocks.0.0.weight
```

Representative output keys (values are machine- and checkpoint-dependent):

```text
selected_tensors=1
unused_tensors=...
setup_identity_bytes=...
execution_pending_digest_bytes=0
cold.phase.transform.ms=...
cold.origin.transform=1
cold.origin.cache=0
warm.phase.cache_lookup.ms=...
warm.origin.transform=0
warm.origin.cache=1
warm.peak_prepared_bytes=...
warm.throughput_mib_per_second=...
```

The public-API conformance test exercises the same stage boundaries with cold
and warm prepared-cache behavior, bounded pipeline delivery, typed telemetry,
and cooperative cancellation:

```text
cargo test --test end_to_end_conformance --all-features
```

For retained `hf-store-rs` snapshots, benchmark through the consumer adapter so
the already verified digest and snapshot guard are passed to
`SourceDescriptor::retained`. This is the small audited `unsafe` integration
boundary: the adapter guarantees that the file cannot mutate or truncate for
the guard lifetime. It is the production zero-copy path and should not be
approximated by a mutable local path.

## Comparison protocol

Use the same local snapshot, storage device, release profile, target model
variant, and output dtype for both implementations. Record at least ten
process-cold and ten process-warm samples after one discarded setup run. Report
median and p95 wall time, peak resident host memory, bytes read, bytes hashed,
bytes copied, cache hits/misses, and prepared/delivered tensor counts.

The old and new paths must deliver the same target contract. Pass exactly that
contract's tensor names to the probe. Comparing “load every checkpoint tensor”
with “load only selected graph constants” is useful diagnostically but is not
an acceptance comparison.

Performance changes should include the raw reports and environment facts.
There is intentionally no hard-coded CI duration threshold: shared runners are
too noisy, and the SD1.5 fixture is too large for ordinary repository CI. This
repository does not currently publish an SD1.5 comparison result; record one
only after running the protocol above against both implementations.
