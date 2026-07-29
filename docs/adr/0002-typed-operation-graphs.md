# ADR 0002: Ordered typed operation graphs

- Status: accepted
- Date: 2026-07-28

## Context

Binding-plan schema v1 selected one external source for each target. That was
enough for aliases and unary preparation, but it could not faithfully describe
common checkpoint assembly such as Q/K/V concatenation. It also left physical
layout conversion ambiguous when a consumer-visible logical tensor, such as
OIHW, had to be stored in a backend layout such as OHWI/KYXC.

Treating those cases as importer conventions or benchmark-local providers made
source order, layout geometry, validation, resource accounting, and cache
identity dependent on code outside the plan. The same conversion could then be
implemented differently by each consumer.

## Decision

Binding-plan schema v2 adds ordered multi-source bindings and a versioned,
typed `OperationGraph`. Graph external inputs retain caller insertion order;
nodes retain topological order; input edges, split outputs, and the selected
final value retain serialized order. Deserialization reconstructs the graph
through the same inference checks rather than trusting serialized derived
facts.

Every graph value has `TensorFacts` containing:

- consumer-visible logical shape;
- physical storage shape;
- consumer-visible element strides indexed by logical axis;
- physical element strides indexed by logical axis;
- scalar dtype and physical layout representation; and
- exact physical byte length.

Logical and storage shapes must have equal element counts. Both stride mappings
must be dense and non-overlapping, their ranks must agree, and all element,
stride, offset, and byte calculations use checked arithmetic. The selected
graph output facts must exactly equal the target contract before any output is
allocated.

Operation-graph schema v1 supports:

- N-input concatenation along one logical axis;
- complete logical-axis permutation into contiguous storage;
- physical-storage permutation that retains logical axes, including
  OIHW-to-OHWI/KYXC order `[0, 2, 3, 1]`;
- high-end zero padding while permuting logical axes into larger physical
  storage;
- materialized slicing with one half-open range per logical axis;
- ordered, non-overlapping split ranges with ordered outputs;
- metadata-only reshape of compatible contiguous storage; and
- one versioned `PlannedTransform` preparation node, such as a dtype cast.

Each structural node pins the built-in `model-weights` operation identity and
version. A preparation node pins its provider, operation, and byte-affecting
implementation version.

## Execution, resources, and identity

`OperationExecution::Host` is the default. It runs structural operations in the
bounded host interpreter and routes `Prepare` nodes through the existing
preparation engine. Reshape aliases compatible storage; concat, permutation,
padding, slice, and split materialize output. Execution checks cancellation at bounded
intervals, including while zero-initializing large output allocations.

Before execution, the graph computes a checked liveness bound for simultaneously
live intermediates and provider-declared scratch. Pipeline admission accounts
separately for unique source spans, graph scratch, and the final prepared
output. A request is rejected before work when any configured source, scratch,
prepared, or arithmetic bound is exceeded.

`OperationExecution::Delegate` instead delivers an `OperationHandoff` containing
the validated graph, complete target contract, and immutable source views in
exact graph-input order. A consumer can fuse the graph into CPU or GPU kernels
and may publish a fully validated final host result through the prepared cache.

Plan and prepared-cache identities include the binding-plan and graph schema
versions, every ordered source descriptor and byte span, graph input and edge
order, operation parameters and implementation identities, and complete output
facts. Ordered inputs are never canonicalized as a set: changing K/Q/V into
Q/K/V changes identity even when the same source names and bytes are present.
Graphs whose selected output is a direct external input or a reshape-only alias
have no prepared-cache address, so a warm lookup cannot replace their zero-copy
source view with a copied cache payload.

## Scope boundary

An operation graph is one complete source-to-one-target path. `Split` has
multiple ordered node outputs, but a graph selects one final value for its
binding; the schema does not promise atomic publication of several targets from
one execution.

Operation graphs describe dense scalar tensor assembly. They do not interpret
packed quantized encodings, choose GGUF decode policy, or move device scratch
ownership into the core. Packed-direct, fused in-tile, host-dequant,
device-dequant-to-scratch, and repack policies remain explicit quantized routes
and provider or consumer responsibilities. Expanded quantized encoding and
decode support is intentionally separate from this decision and remains tracked
in [issue #8](https://github.com/dinoml/model-weights-rs/issues/8).

## Consequences

Consumers can express grouped checkpoint assembly and physical layout
conversion without benchmark-local or model-specific core providers. Host and
delegated implementations share one validated semantic contract and one cache
identity.

The plan schema version changes from 1 to 2. Serialized schema versions remain
compatibility boundaries; callers must regenerate v1 plans rather than treating
them as v2.
