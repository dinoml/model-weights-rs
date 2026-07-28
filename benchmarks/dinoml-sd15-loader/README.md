# DinoML SD1.5 loader benchmark

This standalone workspace compares legacy DinoML host constant materialization
with `model-weights` against one canonical SD1.5 target contract. It intentionally
does not join either parent workspace, and uses local path dependencies because
the integration crates are not published.

The primary contract is every checkpoint-bound constant used by the local CLIP
text tower, profiled UNet, and VAE decoder artifacts. With the checked local
`iter007` artifacts this is 939 targets and 2,065,322,934 output bytes, including
42 ordered multi-source concat targets (118,440,960 bytes). One embedded CLIP
`position_ids` constant remains intentionally unbound (616 bytes).

## Commands

```powershell
cargo run --release --bin dinoml-sd15-loader-benchmark -- validate
cargo run --release --bin dinoml-sd15-loader-benchmark -- sample --lane legacy --consume sha256
cargo run --release --bin dinoml-sd15-loader-benchmark -- sample --lane model-weights --consume sha256

.\run_samples.ps1 .\results\local\sha256-host-defaults `
  -Samples 10 -Consumption sha256

cargo run --release --features full-native --bin full_native -- `
  --trust-native-artifacts
```

All output is one JSON document on stdout. Paths default to the local SD1.5
fixtures used by `dinoml_v2`; every checkpoint/artifact path can be overridden
with the options shown by `--help`.

When `--workers` is omitted, the model-weights lane uses
`std::thread::available_parallelism()`. This is a portable starting value, not a
claim about the throughput limit of any host. Result-channel capacity and
dispatch lookahead are independent controls:
`--delivery-queue-depth N` bounds the worker-to-coordinator channel, while
`--dispatch-lookahead N` bounds admitted work that has not completed ordered
delivery.

Source-read, transform-scratch, and prepared-output admission budgets are also
independent. They default to 268,435,456, 67,108,864, and 536,870,912 bytes,
respectively, and can be set with `--source-bytes`, `--scratch-bytes`, and
`--prepared-bytes`. The benchmark rejects a target whose individual reservation
cannot fit its corresponding budget. Tune each control for the CPU topology,
memory bandwidth, backend, and concurrent load being measured, and record the
effective values from each lane's `execution_limits` report object.

The model-weights lane reports `operations` totals keyed by `identity`, `cast`,
`concat`, `permute`, `slice`, `split`, `reshape`, or `prepare`. Each entry
contains an explicitly cumulative `elapsed_sum_ms`, invocation count, logical
input/output bytes, and bytes actually materialized. Concurrent operation
durations overlap and must not be added to pipeline wall time. Operation
durations stop before event aggregation; the instrumented pipeline wall time
still includes the observer's synchronous map update.

`operation_nodes` breaks the same measurements down by component, target,
component-local work ordinal, location, and operation kind. Locations distinguish
the source-satisfies-binding case, an indexed planned transform, and an indexed
operation-graph node. The target name is resolved from the exact
`BindingPlan::bindings()` ordering submitted to that component's pipeline. If a
future event has no ordinal, or an ordinal outside that plan, `target` is null
while the event's ordinal and location remain available.

The `origins` object records target counts and delivered byte spans for source,
planned-transform, operation-graph, prepared-cache, and other routes. These are
route origins, not allocation classifications. In particular, an operation
graph containing only a reshape or fact-preserving identity permutation can
retain a zero-copy source view while still reporting the `operation_graph`
route. Source-route bytes also quantify retained views, not physical bytes read
from storage. Use each operation entry's `materialized_output_bytes` to
distinguish retained views from materialized host outputs.
The operation map describes host operations that actually ran. A prepared-cache
hit or delegated consumer operation therefore has no host operation entry.
Cast entries appear only when the selected checkpoint dtype differs from the
target contract. Prefer a matching Hugging Face/hf-store variant when one is
available.

For process-cold collection, pass externally verified checkpoint identities with
`--clip-sha256`, `--unet-sha256`, and `--vae-sha256`. These values are trusted:
they must describe the exact files protected by the retained DinoML mappings.
The report records `identity_source` and `identity_bytes_hashed` per component.
When an override is omitted, setup computes SHA-256 from the retained snapshot
and reports the scanned bytes; that scan warms the filesystem cache and therefore
cannot support a filesystem-cold claim.

`validate` materializes both lanes and compares target metadata, byte lengths,
and SHA-256 digests. The executable rejects a discovered contract that differs
from the checked SD1.5 reference instead of silently timing a partial fixture.

`sample` defaults to `--consume sha256`, which reads every delivered byte and
emits an output-set digest. The sample driver requires that digest to remain
identical across both lanes and every child process, including prepared-cache
runs. Use `--consume delivery` only to isolate host delivery/transform latency:
that mode intentionally preserves `model-weights` zero-copy mmap delivery and
does not imply that all source pages were made resident.

GPU upload and module construction are deliberately outside the primary host
boundary. The feature-gated `full_native` binary separately measures the
current downstream CLIP + UNet + VAE ROCm builders and SD1.5 composition,
without inference.

For reportable runs, launch a fresh process per sample, discard one setup run,
then collect at least ten samples per lane/state in interleaved AB/BA order.
`run_samples.ps1` implements this protocol, polls each child's `WorkingSet64`,
retains raw stdout/stderr/process envelopes, and writes medians and nearest-rank
p95 values to `aggregate.json`.
For a prepared-cache-cold model-weights sample, remove only the exact cache
addresses for this contract before launching the child. For a warm sample, prime
those addresses in a separate untimed process and then launch a fresh child.
Legacy DinoML and source-compatible model-weights tensors have no prepared-cache
entry; their warm state is the filesystem page cache. On Windows, call a run
filesystem-cold only when the standby-list reset method is controlled and
recorded; otherwise label it `process-fresh, prepared-cache-cold,
filesystem-cache-uncontrolled`.

Record the two repository commits, clean-worktree status, Rust version, build
profile, CPU/storage details, checkpoint identities, sample order, and the
emitted contract digest beside the result set.

See [RESULTS.md](RESULTS.md) for the validated 939-target 2026-07-28 local run,
plus the earlier 897-target subset retained for historical comparison.
