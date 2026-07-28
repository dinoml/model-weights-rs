# Architecture

```text
local files / retained snapshot
              |
              v
        bounded inventory
              |
   normalized facts + target contract
              |
              v
      deterministic binding plan
              |
     +--------+--------------------+
     |        |                    |
 prepared   bounded host       delegated graph /
 cache hit  operation graph    recipe / packed route
     |        |                    |
     +--------+--------------------+
              |
       bounded delivery
              |
              v
 runtime-owned allocation / packed kernel route
```

Each arrow passes an immutable value with a stable schema. The checkpoint owns
open handles; mapped views additionally own the retained snapshot guard.
Runtime allocations are referenced by opaque consumer identifiers and are
never owned by this crate.

## Tensor contracts and operation graphs

Binding-plan schema v2 binds graph inputs in caller-defined semantic order.
Inputs are not sorted by tensor name. Graph nodes are stored in topological
order and use typed edges, typed operation parameters, inferred outputs, and
versioned implementation identities.

`TensorFacts` separates logical tensor semantics from physical storage:

- logical shape and logical strides are the consumer-visible view;
- storage shape describes physical axis dimensions;
- storage strides map logical coordinates to physical elements; and
- representation records scalar dtype and physical layout.

The two shapes contain the same number of elements, while both stride maps must
be dense, non-overlapping, and rank-compatible. This distinction lets a logical
OIHW constant keep its model-facing shape and strides when a storage
permutation writes OHWI/KYXC bytes. Rank, shape, strides, dtype, layout, exact
byte length, and arithmetic overflow are checked while the graph is built and
again when it is deserialized. The final inferred facts must exactly match the
target contract.

The structural operation set is intentionally small: N-input concat, logical
and physical permutation, materialized slice, ordered multi-output split, and
metadata-only contiguous reshape. `Prepare` embeds one existing versioned
provider transform, such as a dtype cast. A graph selects one final value for
one target binding.

## Execution and cache identity

Host execution is the default. It uses checked allocations, preserves
zero-copy input and reshape paths, accounts for live intermediates, uses only
provider-declared scratch for `Prepare`, and observes cooperative cancellation.
Delegated execution delivers the validated graph and immutable source views in
exact graph-input order so a consumer can fuse the same semantics on CPU or
GPU.

Plan and prepared-cache identity cover every ordered source descriptor and
span, graph input and edge order, operation parameters, implementation
versions, complete logical and storage facts, target contract, and schema
version. Source order is therefore byte-affecting identity, not presentation
metadata.

## Integration seams

`hf-store-rs` can build a retained `SourceDescriptor` from a snapshot file,
passing its logical repository path, expected length, SHA-256, and an owned
snapshot guard. Its adapter owns the narrow `unsafe` proof that those bytes
cannot mutate or truncate for the guard lifetime. `model-configs-rs` can
serialize normalized configuration facts into a manifest identity and generate
target contracts. Neither is a mandatory dependency.

DinoML supplies target constants, allocators, upload callbacks, and optional
preparation or delegated-operation providers. Packed routes separately let it
choose direct in-tile consumption, GPU scratch dequantization, host
dequantization, or repacking per backend. A typed dense operation graph never
implicitly chooses one of those packed routes.

Importer and Python packages can supply conversion recipes over the same public
inventory/plan schema. They should emit normalized facts and provider versions,
not runtime-specific closures, so conversion results remain reproducible.

In-process preparation providers declare exact output and scratch byte lengths
before allocation. Validation is metadata-only and cancellable. The engine owns
both buffers and passes exact mutable slices to execution; providers must use
the declared scratch slice for all size-dependent workspace and must not make
hidden host or device workspace allocations. Serialized scratch lengths affect
plan and prepared-cache identity, while pipeline admission accounts for
provider scratch and simultaneously live adjacent intermediates.
