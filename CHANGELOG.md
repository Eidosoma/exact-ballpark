# Changelog

All notable changes are recorded here. Versions follow `0.y.z` until the interface settles. Each release names the
upstream commits it is checked against (destiny, carbonengine/math, DirectXMath) and the platform it is qualified on.

## [Unreleased]

- The first public tree: the package (`src/exact_ballpark`), its tests, the conformance kit, the benchmark harness's
  core and an example.
  - Checked against destiny `114e89fa`, carbonengine/math `82733220` and DirectXMath `f5d82574`; qualified on GCC
    14.3 with glibc 2.34 (aarch64 Linux) and on the NVIDIA GH200 (sm_90).
  - libstdc++'s hash-table order is reproduced by a clean-room implementation (`docs/provenance.md`, section 3.5).
- The first speed measurements (`docs/benchmarks.md`): 269 of them on one otherwise idle NVIDIA GH200, every one clean,
  with the top numbers in the README.
- Every park and layer runs on the CPU, with the same results as on the GPU (`docs/devices.md`).
  - `VecModesPark`, `VecCombatPark` and `VecComposedPark` share one tick on tensors, which `evolve()` runs on any device
    but a CUDA one. The combat park's tick on tensors now holds FOLLOW, ORBIT and WARP. Over these parks, the server
    layers, the space layer and the delayed removal now tick on the CPU.
  - The kernels and the CUDA graphs of these parks refuse a park on the CPU before any library is loaded or built, and
    their messages name the CPU path.
  - Their tick on tensors returns at once for a tick of no time (`tick_interval_ms=0`), as destiny and every kernel do.
  - With collisions off, the composed park's kernel path keeps destiny's collision lists, made from the kernels' counts
    after the tick, as its tick on tensors and the reference keep them. `CombatGraph` refuses a `VecComposedPark`.
  - Without `nvcc`, `g++` compiles the space layer's host code with the flags `nvcc` gives its host compiler. The
    source, and with it every `nvcc` build's cache key, is unchanged. On the qualified platform, the two compilers'
    host code gives the same bits.
  - New tests hold the CPU tick to `ModeBallpark` and `FullBallpark` with no tolerance. A check on one NVIDIA GH200
    compared both GPU paths with the CPU path, every state tensor after every tick (`docs/devices.md`, section 4).
- Two corrections to the documentation, from an outside review:
  - The flight-mode benchmark's rows counted ships as worlds: its worlds hold four interacting ships. The README's
    flight-mode row and the flight-mode tables in `docs/benchmarks.md` now give worlds and ships, world-ticks and
    ship-ticks a second, and milliseconds a batch tick. Every other row already counted whole worlds. The README's
    space-layer row was measured with the proximity sensors off; it now says so, beside the full layer's row.
    The benchmark probes and the report now carry worlds, ships, both rates and the batch tick's milliseconds.
  - A `CombatGraph` replay leaves out the eager tick's domain watch at the tick's start and end, and `check()` only
    raises on flags already set. `docs/limits.md` and the README now give the exact sequence around each replay
    (`watch()`, `tick()`, `watch()`, `check()`) and the watch's exact condition: any world holding two massive balls
    outside missile mode, however far apart. The composed park's graph leaves nothing out.
