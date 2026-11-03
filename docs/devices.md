# Devices

This document says where each park and layer runs, what each path needs, and the evidence that the paths agree.

**In short.** Every park and layer runs on the CPU and on a CUDA device, with the same results on both.
- **On the CPU,** the parks and the server layers tick through their PyTorch paths, and the space layer runs its
  compiled host code.
- **On a CUDA device,** they tick through the same PyTorch paths or through the fused kernels, and most ticks can be
  captured as a CUDA graph (section 2).
- **The kernels and the graphs** run on a CUDA device only. They refuse a park on the CPU before anything is loaded or
  built.

## 1. The paths

- **The tick on tensors.** One algorithm, written as PyTorch operations on the park's tensors, for any device.
  - On a park on the CPU, `evolve()` runs it. It needs Python and PyTorch only.
  - On a CUDA device, it is the only path of `VecBallpark` and the client views, and the default of `VecOrientPark`,
    `VecCollidePark` and the server layers.
  - On the flight-mode, combat and composed parks, it is also `evolve_torch()` on a CUDA device, the cross-check of
    their kernels.
- **The space layer's host code.** `VecSpace` is one sequential algorithm a world, written in C++ (`vec_space.cu`). It
  is compiled into host functions, its path on the CPU, and a kernel, its path on a CUDA device (section 3).
- **The fused CUDA kernels.** On a CUDA device, `evolve()` runs the flight-mode, combat and composed parks on their
  kernels. Elsewhere the kernels are chosen:
  - `kernel="cuda"` on `VecOrientPark` (with dynamical orientation), `VecCollidePark` (the gradient method),
    `VecServer` (over `VecModesPark`) and `CombatServer`;
  - `ypr="cuda"` on `VecOrientPark` (the yaw, pitch and roll without dynamical orientation);
  - `CudaSpaceServer`, the server layer over the partition.
- **CUDA graphs.** A tick captured once and replayed, with Python out of the loop (`README.md`, "CUDA graphs").

## 2. Which park and layer runs where

| Park or layer | On the CPU | On a CUDA device: the tick on tensors | On a CUDA device: kernels | CUDA graph |
|---|---|---|---|---|
| `VecBallpark` | the tick on tensors | yes | none | none |
| `VecOrientPark` | the tick on tensors | yes, the default | `kernel="cuda"`; `ypr="cuda"` | `TickGraph`, `YprTickGraph` |
| `VecModesPark` | `evolve()`: the tick on tensors | `evolve_torch()` | `evolve()` | `ModesGraph` |
| `VecCombatPark` | `evolve()`: the tick on tensors, every mode | `evolve_torch()` | `evolve()` | `CombatGraph` |
| `VecComposedPark`, collisions on | `evolve()`: the tick on tensors | `evolve_torch()` | `evolve()` | `ComposedGraph` |
| `VecComposedPark`, collisions off | `evolve()`: the tick on tensors | `evolve_torch()` | `evolve()`: the combat park's kernels, then the collision lists | none |
| `VecCollidePark` | the tick on tensors | yes, the default | `kernel="cuda"` | none of its own |
| `VecSpace` | its host functions | no: its kernel | its kernel | `SpaceGraph` |
| `VecServer`, `CombatServer` | the tick on tensors | yes, the default | `kernel="cuda"` | `ServerGraph` |
| `SpaceServer` | the tick on tensors | yes, the default | `CudaSpaceServer` | `SpaceServerGraph` |
| `ClientView`, `SpaceClientView` | yes | yes | none | none |

**Notes on the table.**
- **Every mode.** The combat park's tick on tensors holds FOLLOW, ORBIT and WARP, as well as GOTO, STOP, MISSILE and
  FORMATION.
- **The composed park with collisions off** ticks, on a CUDA device, on the combat park's kernels. It then makes its
  collision lists from the kernels' counts, as its tick on tensors and the reference keep them.
  - It has no CUDA graph. `ComposedGraph` captures the composed tick, with collisions on.
  - `CombatGraph` captures the combat park's tick only, so it refuses a `VecComposedPark`.
- **Graphs and orientation.** On the combat and composed parks, a graph needs dynamical orientation. Without it, the
  yaw, pitch and roll are computed on the host after each tick (`limits.md`, section 7).
- **The layers follow their park.** Over the flight-mode park on the CPU, the server layers, the space layer and the
  delayed removal (`DelayedRemoval`) run on the park's tick on tensors.
- **The collision park** has no graph of its own. With the gradient method, on its kernel or with `contact_slots`, its
  tick can be captured inside a `SpaceGraph` or a `ServerGraph`.

## 3. What each path needs

| Path | Needs |
|---|---|
| The tick on tensors | Python 3.13 or later and PyTorch, on the CPU or a CUDA device |
| The space layer on the CPU | its host functions, built by `nvcc` (with `cuobjdump` beside it) where `nvcc` is found, else by `g++`; binutils' `objdump`, for the build's check; no GPU |
| The space layer on a CUDA device | `nvcc`. A library built by `g++` holds no kernel, and the layer refuses it there, naming `nvcc` |
| The fused kernels | an NVIDIA GPU, `nvcc` and `cuobjdump`; each library is compiled on first use (`README.md`, "Fused CUDA kernels") |
| The CUDA graphs | what the kernels need; on the combat and composed parks, dynamical orientation |

**Where the compilers are found.**
- `nvcc`: `EXACT_BALLPARK_NVCC`, else `$CUDA_HOME/bin/nvcc`, else `PATH` (`platform.md`).
- `g++`: `EXACT_BALLPARK_CXX`, else `PATH`. It is used only where `nvcc` is not found.

**The `g++` build** (`exact_ballpark.vec_space.build_host`):
- **The same input.** It compiles the same source and generated header as the `nvcc` build.
  - The kernels and their launchers stay out, through the source's own `__CUDACC__` guards.
  - The three CUDA qualifiers outside those guards (`__host__`, `__device__` and `__constant__`) are defined empty on
    the command line.
- **The same flags.** It uses the flags `nvcc` gives its host compiler: `-O3 -std=c++17 -fPIC -ffp-contract=off
  -fno-fast-math`. So there is no contraction into fused multiply-adds, no reassociation and no approximate
  reciprocal.
- **The same check.** The host code is disassembled, and a fused floating-point instruction refuses the library. The
  check knows aarch64's fused instructions only (`platform.md`).
- **Its own cache entry.** It is keyed by the compiler's path and version. The source is unchanged, so no `nvcc`
  build's key changes and nothing is rebuilt.

## 4. The evidence

Every comparison follows `exactness.md`'s rule: values are compared by their bits, with no tolerance.

**The CPU path against the references.** On the CPU, the tick on tensors of the flight-mode, combat and composed parks
is held to the same scalar references as their kernels:
- **The flight-mode park** (`tests/test_movement_vec_modes.py`):
  - against `ModeBallpark`, on the mode scenarios (development seed 1, held-out seed 2);
  - against `FullBallpark`, on the flight scenarios and on the mode scenarios with dynamical orientation (seeds 1
    and 2);
  - a tick of no time (`tick_interval_ms=0`), against both.
- **The flight-mode park's other commands**: `SetBallFree`, `SetBallRigid`, `SetBoid`, `SetBallTroll` and
  `PetrifyTroll`, seeds 1 to 3 (`tests/test_movement_vec_modes_leftovers.py`).
- **OnTick's masked tick**, seeds 1, 7 and 19 (`tests/test_movement_vec_parks_ontick.py`).
- **The combat park**, on every world of its scenarios, follows, orbits and warps included
  (`tests/test_movement_vec_combat.py`, `tests/test_movement_vec_combat_leftovers.py`). FollowBall and Orbit of ships
  are also run under the server layers (`tests/test_movement_vec_server_combat.py`,
  `tests/test_movement_vec_server_ontick.py`).
- **The composed park with collisions off.** Every collision list is compared with `FullBallpark`'s at every step:
  144 worlds of development seed 1, 10,060 lists holding 446 entries (88 of them self-collisions), and the 446
  events (`test_with_collisions_off_the_lists_are_fullballparks`).
- **The layers over the flight-mode park**, each against the scalar server or space layer:
  `tests/test_movement_vec_server.py`, `tests/test_movement_vec_server_g11.py`,
  `tests/test_movement_vec_server_ontick.py`, `tests/test_movement_vec_server_space.py`,
  `tests/test_movement_vec_space.py`, `tests/test_movement_vec_space_primes.py`,
  `tests/test_movement_vec_space_shuffle.py` and `tests/test_movement_vec_space_removal.py`.

**The GPU paths against the CPU path, tensor by tensor.** A digest check ran each scenario family through the same park
on one NVIDIA GH200, in two ways: on the kernels (`evolve()`) and on the tick on tensors (`evolve_torch()`). It
compared a digest of every state tensor after every tick with the CPU path's digests, made beforehand on the CPU.
- **The first round** compared, with no difference:
  - the flight modes (seed 1), on both GPU paths: 414,070 digests over 8,810 ticks;
  - the flight modes with dynamical orientation (seed 2, scenarios 900 to 1,080), on both: 36,942 digests;
  - the combat park (seed 1), on both: 12,214 digests;
  - the flight scenarios (seed 1, scenarios 0 to 200), on the kernels: 115,291 digests over 2,453 ticks;
  - the composed park with collisions off (seed 1, scenarios 0 to 6), on the tick on tensors: 8,120 digests.
- **It found one difference.** With collisions off, the composed park's kernels differed in its collision lists
  alone: `hits` in 85 digests and `hits_n` in 10. The kernels counted the contacts but kept no list. Every other
  tensor was equal. The kernel path now makes the lists from the kernels' counts (section 2).
- **The second round,** after that change, ran on both GPU paths:
  - the flight modes (seeds 1 and 2);
  - the flight scenarios (seeds 1 and 2, scenarios 0 to 800);
  - the flight modes with dynamical orientation (seeds 1 and 2, scenarios 0 to 1,080);
  - the combat park (seeds 1 and 2);
  - the composed park, with collisions on and off (seed 1, scenarios 0 to 12).

  It made 27 runs, every one with exit status 0: 26 digest runs and one run of the composed park's GPU tests
  (4 passed). Each pair of the three paths (the kernels and the tick on tensors on the GPU, and the CPU path)
  agreed on all 3,228,031 tensor digests, with no difference; with collisions off, the composed park's collision
  lists (`hits`, `hits_n`) are now equal too.
- **The script is not in this repository.** The digest check's script is not part of this repository yet. The GPU
  tests below hold the same paths to the references and to each other.

**The GPU tests in this repository** (the `gpu` subset):
- `tests/test_movement_vec_modes.py::test_the_tensor_tick_on_the_card_matches_both_references`: the flight-mode park's
  tick on tensors, on the GPU, against `ModeBallpark` and `FullBallpark`;
- `tests/test_movement_vec_combat.py::test_the_tensor_tick_on_the_gpu_matches_the_reference`: the combat park's tick on
  tensors, on the GPU, every world, warps included;
- `tests/test_movement_vec_composed_cuda.py::test_with_collisions_off_the_kernel_path_keeps_the_tensor_ticks_lists`:
  with collisions off, the kernel path against the tick on tensors on the same GPU, every state tensor after every
  tick, `hits` and `hits_n` among them. It runs unmasked and under OnTick's mask;
- `tests/test_movement_vec_defenders.py`: the kernel path with collisions off against the tick on tensors, every
  tensor every tick, the collision lists included.

On the CPU, `test_with_collisions_off_the_lists_follow_from_the_kernels_counts` checks the lists' construction itself.
After every tick of the tick on tensors, it makes the lists again from the counts the kernels keep. They must give
back the tick's lists bit for bit: 373 ticks, 887 contacts, 88 self-collisions, 0 differences.

**The two host compilers.** One test, which needs `nvcc`, builds the space layer both ways and runs the two side by
side (`tests/test_movement_vec_space.py::test_g_plus_plus_and_nvccs_host_compiler_give_the_same_bits`):
- on the orientation park's seeds 1 and 2 and the flight-mode park's seeds 1 and 7: 1,000 worlds and 88 steps in all;
- every tensor of the layer and of the park is equal after every step: 6,776 tensor comparisons;
- so are the 1,693 events, the 7,272 query answers and the 121,384 branch counts.

Run without `nvcc`, the space layer's tests and the server tests over it pass on the `g++` library. That library is
held to `FullBallpark` with no tolerance, and its planted faults are caught.

**The refusals.** The kernels and the graphs of the flight-mode, combat and composed parks refuse a park on the CPU
before any library is loaded or built, and their messages name the CPU path. (`kernel="cuda"`, `ypr="cuda"` and the
other graphs are refused when they are made, off a CUDA device.) The tests patch the library loader to fail if it is
reached:
- `test_the_kernel_and_its_graph_refuse_a_cpu_park_before_any_build` (`tests/test_movement_vec_modes.py`);
- `test_on_the_cpu_evolve_is_the_tensor_tick_and_the_kernels_refuse_before_any_build`
  (`tests/test_movement_vec_combat.py`);
- `test_the_kernel_path_needs_a_card` and `test_a_combat_graph_refuses_the_composed_park`
  (`tests/test_movement_vec_composed_cuda.py`).

## 5. Speed

- **On the CPU,** the tick on tensors is exact, but slow with dynamical orientation. Its torque search runs as many
  small tensor operations each tick.
- **On a CUDA device,** the kernels are the fast path. With dynamical orientation, the tick on tensors waits for the
  host during a tick, so there it serves as the kernels' cross-check.

The measured speeds are in [`benchmarks.md`](benchmarks.md).
