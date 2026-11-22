# Changelog

All notable changes are recorded here. Versions follow `0.y.z` until the interface settles. Each release names the
upstream commits it is checked against (destiny, carbonengine/math, DirectXMath) and the platform it is qualified on.

## [Unreleased]

- Registered four theorems connecting scalar movement to a Physlib Newtonian system and proving uniform motion at force balance in both branches. Added an optional experimental target for timing guards and measured falling-sphere speeds, with a separate log for changes a physical interpretation would require. [Physics experiments](lean/experimental/README.md) · [Physical discrepancies](lean/experimental/DISCREPANCIES.md)

- Added ten Lean theorems in four claims, bringing the register to 430. CPU comparisons cover fixed-obstacle selection across tick lengths, grazing sweeps, mixed-stamp queue dispatch and the fused multiply-add control for radial tracking. [Proofs and replay scope](lean/README.md#ten-proofs-behind-four-discrepancy-reports)
- The upstream Lean gate now builds flight and collision replays from the pinned commit even when the supplied Destiny clone has a different HEAD or local edits. Added controls for the first collision sweep of a half-second tick, mirrored bubble replies and the proposed geometry and tracking corrections. [Gate instructions](lean/README.md#verify-locally) · [Candidate fixes](lean/DISCREPANCIES.md#where-the-result-changes-in-destiny-c-and-how-to-fix-it)
- Replayed the collision-history boundary on neighboring inputs, both mixed packets through the public client update, radial tracking across five comparison ratios and capsule crossings on either side of 10 µm. The discrepancy log and its figure now distinguish the stored rebound from the committed tick. [Discrepancy log](lean/DISCREPANCIES.md)
- Added 37 registered Lean theorems in eight claims, bringing the register to 420. New CPU controls cover STOP, orbit steering, limited pair impacts, solid boxes, range callbacks, early cleanup and tracking falloff. A pinned C++ replay checks the unequal-cap collision. [Milestone overview](lean/README.md#proofs-across-seven-engine-behaviours)

- Added 16 Lean theorems in ten claims covering sensor callbacks and network delivery, warp dropout, formation geometry, force-field access, GOTO timing and rewind, tracking arithmetic, mixed-world CPU ticks and piecewise thrust work. A pinned replay checks the warp against compiled Destiny C++ and routing against five original Python methods. [Proofs and evidence](lean/README.md#ten-claims-across-engine-systems)
- Proved exact contact at half a second, both impulses, and separation through the end of three head-on collision ticks. CPU replays cover equal masses, unequal base masses and a one-sided limiter. Lean bounds the final axial position and velocity errors against ordinary drag. [Rebound results](lean/README.md#one-impact-through-the-end-of-a-tick)
- Added an offline CPU checker for first contact of two spheres following ordinary drag. Lean certifies the turning GOTO fixture to one microsecond, a 3.5 m clearance control, a moving pair and the order of two eligible obstacles. The checker reports the engine’s straight-sweep decision alongside each result. [Proofs and usage](lean/README.md#first-contact-along-a-turn)
- Added a CPU clearance checker for straight collision sweeps against continuous ordinary drag. Lean proves trajectory uniqueness, bounds departure from the endpoint chord and certifies a positive surface gap from finite endpoint words and motion budgets. A turning GOTO fixture reproduces both sphere-sweep misses in pinned Destiny C++. [Clearance checker and examples](lean/README.md#clearance-throughout-a-tick)
- Proved the work–energy law and strict coasting energy loss for the ideal three-dimensional drag model. Added Physlib unit checks for the drag exponent and collision times, with a binary64 control for the wrong-obstacle discrepancy. [Physics proofs](lean/README.md#energy-and-physical-units) · [Collision replay](lean/DISCREPANCIES.md#col-715-fixed-window)
- Proved how scalar GOTO, STOP, setter and replacement commands compose across ticks, including the cached decay and the complete per-ball state. Added a trajectory error bound for ordinary drag and a five-second binary64 coast checked against the continuous drag equation. [Movement proofs and CPU evidence](lean/README.md#scalar-movement-across-ticks)
- Added the [Lean claim register](lean/CLAIMS.md), its pinned build and proof audit, and the [discrepancy log](lean/DISCREPANCIES.md) with CPU replays.
- Registered 18 claims backed by 20 theorems covering combat tracking, collision replay, server and client behavior, and the software fused multiply-add. The CPU replay reproduces the radial-tracking discrepancy in compiled Destiny. [Claims](lean/CLAIMS.md) · [Tracking discrepancy](lean/DISCREPANCIES.md#cmb-702-tracking)
- Registered 10 claims backed by 20 theorems covering capsule contact, client history, exponential reduction, warp sampling and ordered range queries. The discrepancy log includes a missed capsule crossing in pinned Destiny C++ and two mixed-stamp packet cases in the original Python client. [Law overview](lean/README.md) · [Replays](lean/DISCREPANCIES.md)
- Registered ten more theorems on scalar and tensor drag, collision detection and response, server and client updates, the first exponential FMA and radial tracking. A three-ball tick now reproduces the missed rebound from initial conditions in both Exact Ballpark and pinned Destiny C++. [Ten proof targets](lean/README.md#ten-new-proofs) · [Rebound traces](lean/DISCREPANCIES.md#col-715-history-lookup)
- Corrected the formation-resize docstring to describe how stopping a follower releases its slot on the master.
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
