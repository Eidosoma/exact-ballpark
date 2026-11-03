# The evidence of exactness

This document describes how exact-ballpark's claim of exactness was tested, and on how much. It covers:
- the scenario generators;
- the held-out seeds;
- the faults planted on purpose, which the tests must catch;
- the CUDA-graph checks;
- the volumes compared, feature by feature.

`numerics.md` states the rules the code follows. `platform.md` states where the evidence holds. `limits.md` lists
what it does not cover.

## 1. The claim, as a chain

Each link of the chain is a separate set of tests.

| Link | Compared | Tests |
|---|---|---|
| 1 | **The scalar reference against destiny itself.** The conformance kit compiles destiny's C++ from the pinned source, and destiny's Python server and client layers are held to destiny's own published tests (`provenance.md`). | the `oracle` subset (`tests/test_movement_*_oracle.py`, and the published-test files `tests/test_destiny_server_layer.py`, `tests/test_destiny_client_layer.py`) |
| 2 | **The batched parks against the scalar reference**, on the CPU and on the GPU. | the `cpu` and `gpu` subsets |
| 3 | **The CUDA kernels against the batched parks' torch paths**, every tensor the park holds after every tick, and against the reference directly. | the `gpu` subset |
| 4 | **A tick replayed from a CUDA graph against the same tick run eagerly.** | the `gpu` subset |

Some tests also skip a link: kernels run directly against destiny's compiled code, with no reference in between.

Link 2 on the CPU for the flight-mode, combat and composed parks, a check of the GPU paths against the CPU path, and
the space layer's two host compilers are in `devices.md`, section 4.

**The comparison rule is the same everywhere:**
- `double` and `float` values are compared by their bits;
- two NaNs count as equal;
- integers, flags and lists are compared exactly;
- event lists and query answers are compared in order;
- byte streams are compared byte for byte.

There is no tolerance anywhere. No comparison in the final runs of any feature found a difference, and none needed a
tolerance.

## 2. How the evidence is produced

### 2.1 Scenario generators

Each feature has a generator in `tests/` that draws worlds from a seed. A world holds a few balls, from 2 to 6 in
most families and up to 70 in the crowded proximity tests. Each step applies zero or more records of destiny's API
to each world:
- commands;
- setters, including refused values (NaN, zero, negative, non-finite);
- additions and removals;
- cloaks;
- launches;
- queries.

The generators aim at the branches that matter. For example:
- ships crossing the faces of the spatial partition;
- several contacts in one tick;
- warps that finish within the run;
- missiles whose targets warp away;
- followers with ids both below and above their targets' (destiny's phase order depends on it).

**Coverage is checked, not assumed.** The reference keeps its own branch counters. Tests hold each counter above a
floor (`test_the_scenarios_reach_every_branch` and its siblings), so a generator that stops reaching a branch fails
the suite.

The same records are applied to the reference world by world, and to a batched park as one masked command per kind.

### 2.2 Development seeds and held-out seeds

- **Development.** The code and the generator are developed on one or two development seeds.
- **Freezing.** Before any held-out seed is drawn, the generator is frozen: its SHA-256 is recorded.
- **The first run counts.** The held-out seed's first run is recorded as it came, pass or fail.
- **Mistakes stay development.** Where a seed meant as held out ran before its generator was frozen, it was counted
  as a development seed, and a new held-out seed was drawn.
- **Failures stay.** Failed runs are kept, and no seed is replaced.

**Domains.** A world outside the scalar reference's own domain is counted apart and never compared, for example a
call whose result destiny leaves undefined. A world a batched park refuses is flagged by that park, counted apart,
and never counted as equal.

### 2.3 Planted faults

Every feature's tests include faults planted on purpose. Each fault must make a named comparison fail.
- **In Python**, a fault is planted as a subclass or a patch that changes one rule.
- **In CUDA**, a fault is planted in a copy of the kernel's source or in its generated header. The copy is built with
  the source scan and the PTX check recorded but not enforced (`numerics.md`), so that the fault reaches the
  comparison. Faults that use CUDA's own math functions must also be refused by the source scan or the PTX check.

The planted faults fall into four kinds:
- **Arithmetic:**
  - a multiply-add contracted;
  - CUDA's own `exp`, `sin`, `sinf`, `expf`, `log` or `atan2` used in place of the transcription;
  - one bit of one table entry flipped;
  - a fused step of the C library computed unfused;
  - a polynomial's branch bound moved.
- **Order:**
  - collision contacts in reverse order, or ordered by id alone;
  - a sensor list appended to instead of kept in libstdc++'s order;
  - re-filings in slot order instead of ascending id;
  - followers released before their own turn whatever their ids.
- **destiny's rules:**
  - the light-ball rule, the recursion or an early return left out;
  - `>` for `>=`;
  - the extra integration step at a warp's drop-out left out;
  - dead bubbles evolved;
  - `CloakBall` applied with the other actions;
  - the duplicate test left out;
  - an uncloaked ship made massive in warp proper.
- **Bookkeeping:**
  - a stale index left in a park's row;
  - a removed ball kept massive;
  - a formation slot kept after its follower is released.

The tests record how large each fault is. For example, on the collision kernel's development worlds:
- a contracted multiply-add changes 2,171 (tick, tensor) pairs;
- CUDA's own `exp` changes 709;
- the reversed contact order changes 1,589;
- one bit of the last word of `exp`'s table changes 852 (985 of the 12,006 `exp` arguments reach that word).

### 2.4 CUDA graphs

A tick captured as a CUDA graph and replayed must give the same bits as the same tick run eagerly, and as the torch
path. The commands, setters, launches, cloaks and warps between ticks are applied between replays and copied into the
graph's buffers. The combat park's replay holds its kernels but not its eager tick's domain watch, which the caller
runs around it (`limits.md`, section 7). Each park with a kernel has such a test:

| Park | What was replayed |
|---|---|
| collisions | a park's ticks, replayed 12 times |
| dynamical orientation | 23,310 ship-ticks |
| flight modes | 80 worlds of the mode scenarios and 60 with dynamical orientation, with commands and setters between ticks |
| space | 120 worlds × 12 ticks |
| server, one bubble a world | 64 worlds × 10 ticks of a fixed schedule |
| server over the partition | 64 worlds × 10 ticks |
| combat | 286 worlds, 20,478 ball-steps |
| yaw, pitch and roll on the device | 777 ships × 30 ticks |
| delayed removal | 80 worlds |
| composed park | 256 worlds, 31,261 ball-steps, 6,901,559 values |
| OnTick's masked graphs (flight-mode, orientation, combat and composed parks) | every call, against the reference's OnTick. A masked graph with every row in the mask equals the unmasked graph, every state tensor every tick. With half the worlds in the mask, the others keep every state tensor. |

### 2.5 Independence from the batch

A world's result must not depend on the worlds beside it:
- the same world is run alone, among others, and with the batch reversed;
- worlds are run at large widths: up to 262,144 ships for the orientation kernel, 10,000 worlds for the one-bubble
  server kernel;
- kernels are launched with different block sizes.

Every copy must end where its original ends. A padding bug of exactly this kind was found and fixed during
development.

### 2.6 The default machine code

When a kernel gains an optional feature behind a compile-time switch, the default build is compiled again. Its PTX and
SASS are compared with the build before the change and must be identical. The tests of the parks that use only the
default build then run the same machine code as before.

## 3. Volumes, feature by feature

A ball-step is one ball's state after one tick. It holds position, velocity and their previous values, the goto
point, the thrust, the collision terms, the mode, the speed fraction, the partition's filing and the collision list,
with more fields where the park has them: up to 56 fields with dynamical orientation and the commands. Every row
below had 0 differences.

### 3.1 The C library's functions

| Function | Compared with | Volume |
|---|---|---:|
| `exp` and its emulated fused multiply-add, on tensors (`exact_ballpark.vec_exp`), CPU | the C library's `exp` and `fma` | 6,400,084 arguments; 6,400,000 triples |
| the same, GPU | the same | 6,000,063 arguments; 6,000,000 triples |
| `exp` inside the collision kernel, over every path (tiny arguments, subnormal and near-overflow results, infinities, NaN, random bit patterns) | `vec_exp.exp` and the C library | 750,042 arguments |
| the native `__fma_rn` | `vec_exp.fma` (CPU and GPU) and `math.fma` | 400,000 triples |
| `expf`, `logf`, `sinf`, `cosf`, `acosf`, `atanf` (`exact_ballpark.vec_libm`), CPU and GPU, and the orientation kernel's own copies | the C library | every `float` bit pattern: 4,294,967,296 per function on each |
| `atan2f` and `pow` in the orientation kernel | `vec_libm` and the C library | 60,321 and 90,529 pairs |
| `__fma_rn` and `__fmaf_rn` | the emulations and the C library's `fma` and `fmaf` | 200,000 and 20,589 triples |
| `sin`, `cos`, `log` (`exact_ballpark.dlibm`), CPU, over every branch | the C library | 2,450,019 arguments per function |
| `sin`, `cos`, `log`, `exp` on the GPU: every 2^36-th `double` bit pattern, windows of 2^21 consecutive patterns around 19 branch bounds and their negatives, and random patterns | the C library | 415,236,096 arguments per function |
| `asin` (`exact_ballpark.dlibm_inv`), CPU | the C library | 286,921,780 arguments |
| `atan2` (`exact_ballpark.dlibm_inv`), CPU | the C library | 25,098,788 pairs |
| `sin`, `cos`, `asin`, `asin(sin(x))`, and `sincos`'s sine and cosine, on the GPU | the C library | 393,740,288 arguments each |
| `atan2` on the GPU | the C library | 172,638,208 pairs |
| `sincos` (`exact_ballpark.dlibm.sincos`), CPU: every branch, and every argument within 2,048 bit patterns of the 202 points where `sincos` and `sin` can part, both signs | the C library's `sincos` | 60,000 arguments, plus the windows |
| `powf` (`exact_ballpark.dlibm_powf`), CPU: every pairing of 42 special values; random pairs over every branch; arguments in a mushroom's domain at `y = 0.25` | the C library | 1,764; 150,000; 410,096 |
| `powf` on the GPU: every `float` `x` at `y = 0.25f`; random pairs with signalling and quiet NaNs | the C library | 4,294,967,296; 67,108,906 pairs |
| `pow` in the turret accuracy function, through its overflow and underflow edges | the C library | 20,019 exponents |

The tests that read the library's tables also hold every table and constant to the bytes of the C library they run
on, when it is the qualified one.

### 3.2 Goto, stop, the setters and dynamical orientation

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecOrientPark`, torch path, CPU | the reference | development 1 (620 scenarios); held out 2 | 83,076; 77,470 ball-steps |
| the same on the GPU | the reference | development 1 (534 scenarios); held out 2 (529) | 71,241; 65,552 ball-steps |
| the orientation kernel (one thread a ship) | the reference | development 1 (360 scenarios); held out 2 | 47,726; 45,877 ball-steps |
| the orientation kernel | the torch path on the GPU, ships turning hard with every setter in flight, widths 1 to 262,144 | | 1,112,065 ship-ticks, and 2,850,816 more (131,137,536 values) |
| the reference | destiny compiled with DirectXMath | the turning scenarios, seeds 1 and 2; the full slice, seeds 1 and 2 | 287,839 and 312,350 ball-state rows |
| the quaternion from yaw, pitch and roll (`exact_ballpark.full`) | the compiled DirectXMath function alone | | 300,003 angles |
| yaw, pitch and roll on the GPU (`VecOrientPark(ypr="cuda")`) | the reference | development 1 (260 scenarios); held out 2 (260) | 35,350; 31,593 ball-steps |
| `VecBallpark` and the reference's simplest park, corner cases (NaN, −0, negative and tiny values) | `FullBallpark` | 15 cases, every field every tick, CPU and GPU | |

Tests: `tests/test_movement_ref.py`, `tests/test_movement_oracle.py`, `tests/test_movement_vec.py`,
`tests/test_movement_m2_corners.py`, `tests/test_movement_vec_orient.py`, `tests/test_movement_vec_orient_cuda.py`,
`tests/test_movement_turn_oracle.py`, `tests/test_movement_full_oracle.py`, `tests/test_movement_vec_ypr_cuda.py`.

### 3.3 Flight modes: follow, orbit, warp, and the commands

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| the flight-mode kernel, follow, both orbit laws, warp, warp-in, commands during warp, guards | `ModeBallpark` (22 fields) | development 1; held out 2 (1,268 scenarios each) | 141,525; 132,512 ball-steps |
| the same, with every setter in flight and dynamical orientation on and off | `FullBallpark` (54 fields) | development 1; held out 2 (800 scenarios each) | 110,128; 105,275 ball-steps |
| the mode scenarios with dynamical orientation | `FullBallpark` | development 1 (1,080 scenarios); held out 2 (1,077) | 118,187; 108,628 ball-steps |
| the kernel, with no reference in between | destiny's compiled code | development 1 | 300 and 317 scenarios |
| the reference on the flight scenarios | destiny's compiled code | 1 and 2 | 442,218 rows |
| `SetBallFree`, `SetBallRigid`, `SetBoid`, `SetBallTroll`, `SetBallInteractive`, and the centre and surface distance queries, on the GPU | `FullBallpark` | development 1; held out 2 and 3 | 79,922 ball-steps × 56 fields (4,475,632 values); 32,300 queries |
| the reference on the same scenarios | destiny's compiled code | 1, 2 and 3 | 79,922 ball-steps; 32,300 queries |

Tests: `tests/test_movement_modes_ref.py`, `tests/test_movement_modes_oracle.py`, `tests/test_movement_flight_oracle.py`,
`tests/test_movement_vec_modes.py`, `tests/test_movement_vec_modes_leftovers.py`.

### 3.4 Collisions

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecCollidePark`, gradient method (destiny's default), CPU | the reference | 1 to 4 (960 worlds) | 51,390 ball-steps |
| the same, GPU | the reference and the CPU | 1 and 2 (720 worlds) | 38,707 ball-steps |
| `VecCollidePark`, iterative method, CPU; GPU | the reference | 1 and 2 (480 worlds); 80 worlds | 26,360; 4,623 ball-steps |
| the batched park, with no reference in between: gradient; iterative | destiny's compiled code | 3 and 4 (160 worlds; 48 worlds) | 8,437; 2,683 ball-steps |
| the collision kernel, every park tensor after every tick | the torch path (CPU and GPU) and the reference | 1 to 3 | every tensor, every tick |
| static collidables: capsules and oriented boxes | the reference (15 values a ball row) | development 1 and 7; held out 11 | 1,264 + 1,308 + 1,245 ball rows |
| the reference on those worlds | destiny's compiled code, every row kind | 1, 7 and 11 | every row |
| static collidables in the kernel, every park tensor in lockstep | the torch path | 1 and 7 | 3,000 and 3,360 (tick, tensor) pairs |
| minis and force fields (24 values a ball row) | the reference, four ways | development 1 and 7; held out 11 | 6,751 + 5,974 + 5,408 ball rows |
| minis and force fields in the kernel, in lockstep | the torch path | 1, 7 and 11 | 8,487, 7,826 and 6,769 (tick, tensor) pairs |
| collisions within bubbles: the space layer over `VecCollidePark` | the master's `FullBallpark` | development 1 and 7; held out 11 | 59,280, 56,160 and 56,576 ball fields |
| the same on the GPU, the layer's kernel over the collision kernel | the master's `FullBallpark` | 1; held out 11 | 28,236 and 27,144 ball fields |
| global balls, which every bubble's balls may meet | the master's `FullBallpark` | development 1 and 7; held out 13 | 53,040, 55,120 and 55,536 ball fields |
| the layer's ball rows with global balls, with no reference in between | destiny's compiled code | 1; held out 13 | 97,920 and 102,528 fields |

Tests: `tests/test_movement_vec_collide.py`, `tests/test_movement_vec_collide_cuda.py`, `tests/test_movement_vec_exp.py`,
`tests/test_movement_space_oracle.py`, `tests/test_movement_vec_static_collidables.py`,
`tests/test_movement_vec_minis_and_fields.py`, `tests/test_movement_vec_space_collide.py`,
`tests/test_movement_vec_global_balls.py`.

### 3.5 Space: the partition, bubbles, sensors and queries

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecSpace`, host path, over `VecOrientPark`: the filing, bubbles, dictionaries, sensors, tracking, notification ranges, queries | the master's `FullBallpark` | 1, 2 and held out 7 (300 worlds × 20 steps each) | 944,125 ball fields; 108,000 world fields; 1,409 events; 6,709 answers |
| the space kernel (one thread a world) over the flight-mode and orientation parks | the master's `FullBallpark` | 1 and 7 | 1,129,921 ball fields; 129,600 world fields; 1,717 events; 7,128 answers |
| the kernel | the host path | | every tensor of the layer and the park, every step |
| crowds of 14 to 70 balls: libstdc++'s rehash policy in the reference | destiny's compiled code | 1 to 4 | 65,737 rows |
| the same crowds, host path; kernel | the reference, with each map's bucket count and next resize | 1 to 4 | 1,908 world-steps (891,514 ball fields); 3,808 world-steps (1,755,351 ball fields) |
| the reference's hash tables | libstdc++ itself, through a probe compiled with the conformance kit's compiler and flags | 1 to 4 | 305 primes; 20,620 bucket-count inputs; 7,145 rehash states; 96,697 container operations |
| shuffled sensors (the C library's `rand()` per world), host path | the reference, with every world's generator state | four seeds, one held out | 5,821 draws; 7,680 generator states; 397,774 ball fields |
| the same, kernel | the reference | three seeds, one held out | 4,363 draws |
| shuffled sensors, host path, with no reference in between | destiny's compiled code | the same four seeds | 948,538 ball-row fields |
| the delayed removal (`DelayedRemoval`), over `VecOrientPark` on the CPU | the reference | development 1; held out 5 | 140,933 and 147,238 ball fields; 1,009 and 1,020 delayed removals |
| the same over `VecModesPark` on the GPU | the reference | 1 and 5 (150 worlds × 16 steps each) | every field, every step |

The delayed removal is exact against the reference, never against destiny (`limits.md`).

Tests: `tests/test_movement_space_ref.py`, `tests/test_movement_space_oracle.py`, `tests/test_movement_space_hashtable.py`,
`tests/test_movement_vec_space.py`, `tests/test_movement_vec_space_primes.py`, `tests/test_movement_vec_space_shuffle.py`,
`tests/test_movement_vec_space_removal.py`.

### 3.6 The server layer and the clients

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| the scalar server layer | destiny's published server tests | | 112 tests (two of them skipped) |
| the scalar client layer | destiny's published client tests (`test_ticker.py`, `test_mergestateintohistory.py`) | | 9 and 11 tests |
| a session: one server, the network and one client park per pilot | each client's park against the server's at the client's time | 1,000 sessions, four delay pairs | 89,250 client-ticks |
| `VecServer` (one bubble a world) over the orientation, flight-mode and collision parks, CPU | the scalar server layer | development 1; held out 2 | 27,244 ball-ticks; 14,640 client update lists |
| the same with every park evolving, and the server kernel, GPU | the scalar server layer | 1 and 2 | 23,324 ball-ticks; 10,964 update lists |
| `SpaceServer` (several bubbles a world), CPU | the scalar server layer | 1, 2 and held out 7 | 23,652 client updates; 58,144 ship-ticks |
| the same over `VecModesPark`, GPU | the scalar server layer | 1 and 7 | 10,092 client updates; 25,168 ship-ticks |
| `CudaSpaceServer` (its kernel), every tensor and every decoded update | `SpaceServer`'s torch path | 1; held out 7 | 3,971 and 4,058 queued actions; 3,035 and 3,193 client updates |
| `CudaSpaceServer` | the scalar server layer | held out 7 | 3,193 client updates; 7,664 ship-ticks |
| the pending-action queries and the undo of a pending cloak, in three server layers | the scalar server layer | four seeds, one held out | 19,185 answers; 4,904 undos; 11,906 client updates; 25,824 ship rows |
| OnTick's intervals, each world at its own clock, the space server over the collision park | the scalar server layer, driven through its park's OnTick | development 1 and 7; held out 19 | 1,526, 1,442 and 1,609 updates |
| a client's view with global balls | the session's client parks | development 1 to 12; held out 101 to 112 | 1,731 and 1,723 client-ticks |

Client update lists are compared in a canonical form that keeps every argument's type, every float's bits, every byte
of every stream and every packaged action.

Tests: `tests/test_destiny_server_layer.py`, `tests/test_destiny_client_layer.py`, `tests/test_movement_vec_server.py`,
`tests/test_movement_vec_server_cuda.py`, `tests/test_movement_vec_server_space.py`,
`tests/test_movement_vec_server_space_cuda.py`, `tests/test_movement_vec_server_pending.py`,
`tests/test_movement_vec_ontick.py`, `tests/test_movement_vec_global_edges.py`, `tests/test_movement_vec_global_graphs.py`.

### 3.7 Combat: missiles, formations, cloaking, mushrooms, and the accuracy function

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecCombatPark`, torch path, CPU | `FullBallpark` | development 1 (540 worlds); held out 2 (548) | 43,480 and 49,440 ball-steps; 2,081 and 1,980 collision events |
| the combat kernels, GPU | `FullBallpark` | development 1 (573 worlds); held out 2 (575) | 45,645 and 51,360 ball-steps |
| the kernels replayed from a CUDA graph | `FullBallpark` | 1 | 286 worlds; 20,478 ball-steps |
| the reference on the same scenarios | destiny's compiled code | 1 and 2 (576 scenarios each) | 45,720 and 51,441 ball rows; 5,949 and 5,669 events |
| the accuracy helpers (`exact_ballpark.vec_accuracy`), CPU, GPU and the device kernel | the reference | 1 and 2 | 2,478 and 2,476 calls of each helper |
| mushrooms, trolls and the commands in the combat park, GPU | `FullBallpark` | development 1; held out 2 and 3 | 12,305, 12,675 and 11,127 ball-steps |
| the reference on those scenarios | destiny's compiled code | 1, 2 and 3 | 14,098, 13,476 and 12,865 ball rows |
| `LaunchMissile` through the server layer (`CombatServer`), CPU | the scalar server layer | development 1; held out 5 | 4,858 and 4,911 client update lists |
| the combat server's kernel, every tensor every tick | its torch path | development 1 and 2; held out 7 | 60 worlds × 12 ticks each, with and without dynamical orientation |
| the combat server's kernel | the scalar server layer | development 1; held out 9 | 4,818 and 5,028 client update lists |

Tests: `tests/test_movement_vec_combat.py`, `tests/test_movement_vec_accuracy.py`,
`tests/test_movement_vec_combat_leftovers.py`, `tests/test_movement_vec_server_combat.py`,
`tests/test_movement_vec_server_combat_cuda.py`.

### 3.8 The composed park: flight modes, missiles and collisions together

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecComposedPark`, torch path, CPU and GPU | `FullBallpark` | development 1 (248 worlds); held out 2 (251) | 32,344 and 30,103 ball-steps; 1,663 and 1,354 collision events |
| the reference on the same scenarios, every world | destiny's compiled code | 1 and 2 (256 worlds each) | 33,327 and 30,613 ball rows |
| the composed kernel, every one of the park's tensors after every tick | the torch path | all sets, including long ticks with warp-ins (development 11, held out 3) | 1,171 worlds; 106,466 ball-steps; 23,966,900 values |
| the composed kernel replayed from a CUDA graph | the torch path, and eager kernel ticks | 1 and 2 | 31,261 ball-steps; 6,901,559 values |
| defender missiles, torch path | `FullBallpark` | development 21 and 22; held out 23 | 38,609, 39,808 and 39,817 ball-steps |
| the reference on those scenarios | destiny's compiled code | 21, 22 and 23 | 38,756, 40,390 and 40,640 ball rows |
| defender missiles in the kernel | the torch path | development 21; held out 23 | 2,067,219 and 2,497,319 values |
| mushrooms and the commands, kernel path | `FullBallpark` | development 1; held out 2 and 3 | 18,869, 18,980 and 16,424 ball-steps |
| the reference on those scenarios | destiny's compiled code | 1, 2 and 3 | 20,543, 19,597 and 17,360 ball rows |

The composed park's comparison of every tensor found two places where the kernel did not hold the torch path's bits.
Both were fixed before the numbers above were taken:
- a released missile's owner index, an internal row index that no comparison with the reference could see;
- the filing of a massive ball dropping out of warp, in worlds the park refuses.

Tests: `tests/test_movement_vec_composed.py`, `tests/test_movement_vec_composed_cuda.py`,
`tests/test_movement_vec_defenders.py`, `tests/test_movement_vec_composed_leftovers.py`.

### 3.9 OnTick's per-world intervals in the parks

Each world runs destiny's OnTick at its own clock. Its start, and each call's advance, from none to 7.4 intervals with
a jitter, are drawn per world. The scenarios come from the frozen generators of the earlier sections. After every
call, the following are compared with `FullBallpark.on_tick`, one park a world:
- every field of every ball;
- which balls exist;
- the collision events and the answers;
- the clocks.

The runs count each case of the schedule: calls too early, first calls, catch-ups and the master's
resynchronisations.

| Path | Compared with | Seeds | Volume |
|---|---|---|---:|
| `VecOrientPark`, torch path, CPU, 40 OnTick scenarios a seed | `FullBallpark.on_tick` | development 1 and 7; held out 19 | 5,033, 5,122 and 4,567 ball rows |
| `VecCombatPark`, torch path, CPU, the combat scenarios | `FullBallpark.on_tick` | 1 and 7; held out 19 | 2,640, 1,921 and 2,946 ball-steps; 193, 158 and 318 events |
| `VecCombatPark`, torch path, CPU, the mushroom and command scenarios | `FullBallpark.on_tick` | 1 and 7; held out 19 | 1,388, 1,725 and 1,314 ball-steps |
| `VecComposedPark`, torch path, CPU | `FullBallpark.on_tick` | 1 and 7; held out 19 | 4,446, 4,044 and 4,116 ball-steps; 357, 357 and 464 events |
| the flight-mode kernel, eager and as masked graph replays, GPU, 40 OnTick scenarios a seed, dynamical orientation on and off | `FullBallpark.on_tick` | 1, 7 and 19 | every field after every call |
| the orientation kernel and the device yaw, pitch and roll, eager and as masked graph replays | `FullBallpark.on_tick` | 1 and 19 | every field after every call |
| the combat and composed kernels and their masked graphs | `FullBallpark.on_tick` | 1 and 19 | every world, warps included |
| the reference on the OnTick scenarios of the orientation and flight sections | destiny's compiled code | 1 | 5,033 and 7,631 ball rows of 54 fields; 1,448 and 2,784 clocks |
| the orientation park, and the flight-mode park on the GPU, with no reference in between | destiny's compiled code | 1 | every row and clock |
| the server layers' OnTick, torch paths, CPU (16 worlds and 14 calls a run): `VecServer` over the orientation and collision parks, `CombatServer`, and `SpaceServer` over the orientation park | the scalar server layer, driven through `FullBallpark.on_tick` | development 1 and 7; held out 31 | 15 runs; 13,286 ball rows; 5,553 updates |
| the same on the GPU, the one-bubble and combat server kernels included, and `SpaceServer` over `VecModesPark` | the scalar server layer | 1, 7 and 31 | 27 runs; 24,108 ball rows; 10,147 updates |
| the one-bubble and combat server kernels under the mask | their torch paths, every tensor after every call | 1, 7 and 31 | 12 runs |

**The masked tick** is also checked on twin parks, one ticked under a mask and the other not:
- inside the mask, every per-row tensor is the unmasked tick's;
- outside it, every per-row tensor is as it was before the tick;
- a mask of every row is the unmasked tick;
- an empty mask changes nothing.

**Planted faults.** Seven are caught on the CPU:
- `BringOutTheDeads` not called;
- a masked tick's events not limited to its rows;
- a world without a tick losing its clock;
- no resynchronisation on the master;
- the first call running every interval;
- the remainder dropped;
- the rows outside the mask ticked.

One more is caught on the GPU: a masked graph without its put-back.

The server layers' OnTick adds ten more, each caught: seven on the CPU, two on the GPU and one in the partition layer.
Among them:
- a world without a tick losing its queue;
- the last queued action reset, which breaks the duplicate test;
- every world's pre-tick run;
- every world evolved.

Tests: `tests/test_movement_vec_parks_ontick.py`, `tests/test_movement_vec_server_ontick.py`.

## 4. Faults the tests did not catch at first

Some planted faults went unnoticed at first. In every case the tests were strengthened, never the code under test, and
the fault was then caught.
- **A range query skipping a ball that had just left its bubble.** The queries rarely met such a ball. The worlds
  gained moves across a face, with queries there.
- **A warp's drop-out not re-filed.**
  - First, no warp finished within the runs. The generators then used longer warps and slower ships.
  - Then the fault still changed no state, because a ship crawls in its last warp ticks.
  - Only a constructed warp whose exit lies 5 cm below a box face shows it, and that case is now a test.
- **Four faults in the client layer, and one in the server kernel's duplicate test.**
  - The client checks were strengthened.
  - The generator now repeats a call that differs only in its local argument.
- **Faults in `powf`'s last bits.** They went unnoticed on 40,000 arguments, because `powf` rounds a `double` result
  to `float` once. They were replaced by faults that move the result (`numerics.md`, section 7).
- **The drop-out's re-filing at rest in the composed park.**
  - It was first seen only on a hand-built world.
  - A new generator with long ticks and short warp-ins now shows it in 156 of 223 development worlds, and in 132 of 195
    worlds of its held-out seed.

## 5. What this evidence does not show

- **Other platforms.** Every comparison ran on one platform (`platform.md`). The `sm_100` binaries and the
  `compute_90` PTX are built but have never run.
- **Worlds outside a park's domain.** They are refused and flagged, never compared as equal (`limits.md`).
- **Other builds of destiny.** The claim is about destiny's public source at the pinned commit, compiled by the
  conformance kit. It says nothing about other builds of destiny, on other compilers, C libraries or platforms.
- **Speed.** No speed is claimed here (`benchmarks/`).
