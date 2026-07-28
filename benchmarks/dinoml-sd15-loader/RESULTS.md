# Local SD1.5 loader results — 2026-07-28

These are workstation results, not portable performance claims. Each row is ten
fresh-process samples per lane in alternating AB/BA order after one discarded
pair. The filesystem cache was uncontrolled (and warmed by validation), the
prepared cache was disabled, checkpoint SHA-256 identities were supplied from
verified Hugging Face LFS metadata, and the `model-weights` lane used four
workers unless a section states otherwise.

## AVX2 permutation follow-up

The runtime-gated AVX2 path for F16/BF16 3x3 OIHW-to-OHWI permutations was
measured with the full 939-target contract. A production-graph microbenchmark
over one 58,982,400-byte tensor, including initialized output allocation and
cancellation checks, improved from 85.144/89.320 ms before SIMD to
32.386/32.165 ms after SIMD, approximately 2.7x.

The full validator used trusted download-time checkpoint digests, hashed zero
checkpoint bytes on load, and matched all 2,065,322,934 output bytes with zero
mismatches. The following delivery-only results use the same executable and
ten-sample interleaved protocol:

| Workers / lookahead | Lane | Materialization median | p95 | Process median | Peak working-set median |
|---|---|---:|---:|---:|---:|
| 4 / 4 | DinoML legacy | 6,268.8 ms | 6,523.6 ms | 7,064.2 ms | 2.260 GiB |
| 4 / 4 | `model-weights` | 1,877.4 ms | 2,145.4 ms | 2,722.6 ms | 1.734 GiB |
| Host default 36 / 36 | DinoML legacy | 5,926.0 ms | 6,747.6 ms | 6,806.1 ms | 2.258 GiB |
| Host default 36 / 36 | `model-weights` | 881.6 ms | 1,076.9 ms | 1,695.3 ms | 1.746 GiB |

At the fixed four-worker setting, the pre-SIMD telemetry median was 2,465.8 ms
for materialization and 2,233.9 ms for cumulative permutation CPU. The SIMD
run reduced those values to 1,877.4 ms (-23.86%) and 1,215.5 ms (-45.59%).
At the host-default setting, the corresponding medians changed from
1,143.35 ms to 881.62 ms (-22.89%) and from 2,597.3 ms to 1,618.1 ms
(-37.70%). Cumulative operation CPU overlaps across workers and is not added to
pipeline wall time. Worker count and lookahead remain system-specific controls;
the 36-worker result is a host default, not a portable optimum.

The benchmark executable SHA-256 was
`8794e2e6ef62beb30952c82092246c1ddbd1645bf33716e1ca3b1c2ffb9452f9`.
Raw aggregates:

- `results/local/simd-avx2-delivery-workers4/aggregate.json`
  (`ded0bbbd6fad95f1db616c196053fba5c0659cae999bcf194141664a74a1afc0`)
- `results/local/simd-avx2-delivery-host-defaults/aggregate.json`
  (`e1c8498dc7dca20b3045692a7ef60200ba7cc2e0d4cac3c96bf6633d75c8440d`)

## Issue #11 full contract

The current benchmark covers all 939 checkpoint-bound targets and
2,065,322,934 output bytes, including 42 ordered multi-source targets and all 85
CK/KYXC storage permutations. The full-byte validation found zero mismatches:

- Contract SHA-256:
  `0c6400faa63434d596226b4ad827bd6a811ac5f4f7f05e1316f5658b6a33ed55`
- Legacy and `model-weights` output-set SHA-256:
  `651db48d7c124f9641347ce64025109aa12ceb73665671efd48cd63beaa57842`
- Grouped coverage: 42 targets, 118,440,960 bytes
- Direct coverage: 897 targets, 1,946,881,974 bytes
- Validation mismatch count: 0

Both result sets used the same final executable, SHA-256
`c8b41e3023cf1c23d8a77ea53218b986576a2d94f8667bcab0214315245f5163`.
The delivery run skipped a duplicate validation invocation; the full-byte run
validated that exact executable and contract immediately before sampling.

| Mode | Lane | Materialization median | p95 | Process median | Peak working-set median |
|---|---:|---:|---:|---:|---:|
| Full-byte SHA-256 consumption | DinoML legacy | 18,457.8 ms | 18,762.1 ms | 19,244.1 ms | 2.259 GiB |
| Full-byte SHA-256 consumption | `model-weights` | 14,906.1 ms | 15,117.9 ms | 15,604.8 ms | 2.262 GiB |
| Delivery/transform only | DinoML legacy | 6,135.5 ms | 6,851.6 ms | 7,160.9 ms | 2.258 GiB |
| Delivery/transform only | `model-weights` | 3,028.7 ms | 3,505.1 ms | 3,753.4 ms | 1.736 GiB |

For full-byte consumption, `model-weights` reduced median materialization time
by 19.24% (1.238x throughput) and process wall time by 18.91%; median working
set was effectively tied (+0.14%). For delivery/transform only, it reduced
median materialization time by 50.64% (2.026x throughput), process wall time by
47.58%, and median working set by 23.12%. The delivery result preserves mmap
views for source-compatible weights and therefore does not claim that every
checkpoint byte became resident.

Raw aggregates:

- `results/local/issue11-final-sha256-workers4/aggregate.json`
  (`d7f8b9e6c8eece73d7134ed5739572f226652149ba9270bd8203e421bdbc708a`)
- `results/local/issue11-final-delivery-workers4/aggregate.json`
  (`d0553c5496f554abca9bfca711ebe72406d533addb6cf903dfece9f3eebb77b4`)

## Historical 897-target subset

The following measurements predate issue #11. They cover only 897 direct
targets and must not be presented as full-contract results.

The downstream checkout was dirty at DinoML commit
`e7d9cbd64711cf29c247aaaa7dca0aa8503a670f`; the benchmark executable SHA-256
was `a7c43d8f01406fcf786774014b44cbb0ec2a08cfe8cec9227e17eae3c8792e85`.
`model-weights-rs` had an unborn Git branch, so there is no source commit for
this snapshot.

### Historical validated subset

- Contract: 897 targets, 1,946,881,974 bytes
- Contract SHA-256:
  `7ba79464474d6637c8e573ec76069cb028b0afc55e5522e47b5c69844050f1cb`
- Output-set SHA-256:
  `02c8a3fa8145036b83d7a63d43d1c074497fb03a2d74de5e60dfc1143255f0ce`
- Legacy/model-weights mismatch count: 0
- Coverage: 94.265% of the CLIP-text + UNet + VAE-decoder target bytes
- Excluded by schema v1: 42 multi-source concat targets, 118,440,960 bytes

### Historical results

| Mode | Lane | Materialization median | p95 | Process median | Peak working-set median |
|---|---:|---:|---:|---:|---:|
| Full-byte SHA-256 consumption | DinoML legacy | 17,401.9 ms | 17,455.0 ms | 17,953.0 ms | 2.106 GiB |
| Full-byte SHA-256 consumption | `model-weights` | 13,304.0 ms | 13,587.9 ms | 13,853.3 ms | 2.109 GiB |
| Delivery/transform only | DinoML legacy | 5,681.8 ms | 5,843.7 ms | 6,196.9 ms | 2.105 GiB |
| Delivery/transform only | `model-weights` | 1,707.4 ms | 1,719.5 ms | 2,199.7 ms | 1.582 GiB |

For the primary full-byte comparison, `model-weights` reduced median
materialization time by 23.55% (1.308x throughput) and process wall time by
22.84%; working set was effectively tied (+0.12%). For delivery/transform only,
it reduced median materialization time by 69.95% (3.328x) and working set by
24.86%. The latter isolates the benefit of retaining source-compatible mmap
views and must not be interpreted as full payload residency.

Raw aggregates:

- `results/local/sha256-workers4/aggregate.json`
  (`11a237a3fce553e50db84577037baca4c6864525e5d303806351a8b01fc82fdf`)
- `results/local/delivery-workers4-final2/aggregate.json`
  (`33706ecb9873ebcf4c4f1da20bb36b1a6150ef3e6eae793d2fbce330c16605d8`)

## Native downstream composition probe

The separate ROCm builder probe could not produce a timing because the checked
`iter007` artifacts cannot currently coexist on Windows. VAE encode contains a
100,352-byte `dinoml_rocm_runtime.dll` with SHA-256
`3b66ba462ba91707d519bf11b8d4f75c88b2ff9a6f072e787aae746b111efa6b`,
while CLIP text, UNet, and VAE decode contain a 114,688-byte DLL with SHA-256
`2521d5f687ba9783dbafcdf8c060807e3b7af6152048037288ccf124751ce390`.
The runtime correctly rejects same-named support DLLs with different content.

The captured failure is
`results/local/full-native/sample-1.stderr.txt`
(`a5ea4403f63c033b9efdcaff52fdfc3c0bc9abe141e4b1833b80012e9d100c29`).
Rebuild the artifact sets from one runtime revision—or make the downstream VAE
builder decoder-selective—before collecting the full native composition timing.

## Host

- Windows 10 IoT Enterprise LTSC 10.0.19044
- Intel Xeon E5-2686 v4, 18 cores / 36 threads
- 128 GiB RAM
- Rust 1.95.0, optimized release build with debug information
- Checkpoints and workspace on separate Crucial CT2000P3SSD8 NVMe drives
