# Limits

This document lists what exact-ballpark does not cover, or covers only on some paths. Section 1 covers the port as a
whole, section 2 the scalar reference, and the later sections each park in turn.

**How refusals work.** A batched park checks its domain as it runs. A world that leaves the domain gets a flag on its
rows, and the park's `check()` raises. A flagged world is never computed approximately and passed off as exact. The
tests count flagged worlds apart and never count them as equal. One exception: a replay of the combat park's CUDA
graph leaves out that park's domain watch, which the caller runs around it (section 7).

## 1. Across the port

- **One qualified platform** (`platform.md`). The `sm_100` binaries and the `compute_90` PTX are built but have never
  been run.
- **Devices** (`devices.md`). Every park and layer runs on the CPU and on a CUDA device. The fused kernels and the CUDA
  graphs run on a CUDA device only, and refuse a park on the CPU before anything is built. On the CPU, the space layer
  runs its host code, compiled by `nvcc` or, where `nvcc` is absent, by `g++`.
- **Many small, independent worlds.** The batched parks step many worlds at once, each with a few balls. The space
  layer holds at most 64 balls a world.
- **OnTick's per-world intervals.** destiny's OnTick lets each world tick on its own clock. A call earlier than a
  whole interval does nothing. On the master, a backlog of more than five intervals is cut to one. A park's first call
  evolves once. After that, every whole interval is a tick, and the remainder is carried.
  - **Where they are batched:**
    - the flight-mode, orientation, combat and composed parks, on their torch paths, their kernels and their CUDA
      graphs, through a masked tick (`on_tick`, `evolve(active=...)`);
    - the one-bubble server layer (`VecServer`) and the combat server layer (`CombatServer`), on their torch paths and
      their kernels. The queue is split by world on the host. A world without a tick keeps its queue, its stamp, its
      balls and its clock, and sends nothing;
    - the server over the partition (`SpaceServer`), over every park, on its torch path.
  - **Where they are not batched:**
    - the space server's kernel (`CudaSpaceServer`), which refuses `on_tick`: its kernel runs each world's clean-up and
      bubble update itself;
    - a server layer's per-world tick as a CUDA graph: the queue's split and each world's first pre-tick are decided
      on the host.
  - **What a masked tick costs.** `on_tick` runs the worlds' intervals in rounds. A round in which every world ticks
    is the ordinary, unmasked tick. A round that leaves some worlds out is a masked tick: it runs the park's whole
    tick for every row, then puts back the rows outside the mask. That means a copy of each state tensor before the
    tick and a selection written back after it.

    The cost was measured once, on the flight-mode park with dynamical orientation, at 8,192 and 65,536 rows with
    half of the worlds in the mask, on a GPU shared with other work, so the figures are provisional. A masked replay
    of a CUDA graph took about 3.5 to 4.5 times as long as an unmasked replay. An eager masked tick took 8 to 17
    times as long as an unmasked one.
- **Speed.** No absolute speed is claimed (`benchmarks/README.md`). The masked tick's relative cost above is the
  only cost these documents state, and it is provisional.
- **Six of destiny's rules are implemented but not passing** (`rules.md`):

| Rule | Why it does not pass |
|---|---|
| TIME-713 | destiny cleans up removed balls in an order that follows heap addresses. No port can reproduce that order, so the reference uses ascending id. The batched delayed removal is exact against the reference, never against destiny. |
| TIME-708 | A client ball's display interpolation is not transcribed. It is display code, not compiled by the conformance kit. |
| MOV-731, MOV-737 | Each rule's two halves are held to two different oracles: destiny's compiled C++, and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine. |
| SEN-712, TIME-712 | Parts of each rule are not covered by destiny's published tests: the events of a ball in no known bubble or in a dead bubble, and the pre-tick's follower clean-up. Those parts are checked only against the transcription. |

## 2. The scalar reference

The scalar reference consists of `exact_ballpark.ref`, `modes`, `space`, `full`, `server` and `client`.

**Undefined behaviour is refused.** Where destiny's result is undefined, the reference raises instead of guessing
(`provenance.md`, section 3.1).

**Inputs where Python and C part ways.** Python raises where C computes on. Examples:
- the cosine of an infinity, after `SetMaxSpeed(+inf)` in the new orbit law;
- a NaN or infinite mass or agility, which destiny's guards store, followed by the old orbit law's integer conversion
  of a NaN, which C leaves undefined;
- a massive ball in warp proper, whose warp constants take the logarithm of zero.

Such inputs are outside the reference's domain, and no generated scenario reaches them. The kernels compute C's IEEE
result there, and aarch64's conversion of NaN to 0, but this is not compared with destiny.

**Not modelled:**
- NaN payloads, `errno` and the floating-point exception flags;
- the hooks destiny's engine calls into a game's own code (setting a client's full state, the game's handling of bubble
  transitions);
- signals and logging;
- profiling.

**destiny's Python server and client layers** were transcribed but never run. Their stand-ins (`provenance.md`,
section 3.1) are labelled in the code.

## 3. Flight: `VecBallpark`, `VecOrientPark` and `VecModesPark`

**`exact_ballpark.vec.VecBallpark`** holds goto and stop only, without collisions or orientation.

**`exact_ballpark.vec_orient.VecOrientPark`**, with its orientation kernel, holds:
- goto, stop and the commands that set a goto point;
- the setters;
- dynamical orientation;
- OnTick's per-world intervals (`on_tick`), which the flight-mode, combat and composed parks inherit.

Not in this park:
- follow, orbit and warp;
- missiles and formations;
- collisions and the partition;
- OnTick's time stamps: the park keeps no per-world current time, since its tick reads none;
- the server's view.

**Yaw, pitch and roll without dynamical orientation.** By default they are computed on the host through the C
library, two round trips a tick, so such a park cannot be captured as a CUDA graph. With `ypr="cuda"` they are
computed on the device, but only without dynamical orientation and only on a CUDA device. Where a yaw's change is
infinite, the host path raises and the device gives C's NaN. No scenario goes there.

**`exact_ballpark.vec_modes.VecModesPark`**, with the flight-mode kernel, holds:
- goto and stop;
- follow;
- orbit under both laws;
- warp, warp-in and the follower releases;
- the setters;
- dynamical orientation;
- `SetBallFree`, `SetBallRigid`, `SetBoid`, `SetBallTroll` and `SetBallInteractive`;
- OnTick's per-world intervals.

Not in this park:
- collisions;
- the partition, bubbles and sensors (every bubble is live, and no events are posted);
- missiles, formations and mushrooms;
- the yaw, pitch and roll without dynamical orientation, which are display values only (`exact_ballpark.vec_ypr_cuda`
  computes them);
- OnTick's time stamps, so `SetBallFree`'s time is not kept;
- a delayed removal of its own: its `on_tick` brings out no moribund balls (`DelayedRemoval`, section 5, removes them
  under the space layer).

A world's balls must all be in the park and share one clock. A follower's target must be in its world.

**The flight-mode park's tick on tensors** (`evolve_torch`, which `evolve()` runs on any device but a CUDA one) is the
kernel's algorithm. On the CPU it is held to the same references as the kernel (`devices.md`, section 4).

**destiny's behaviour is reproduced, not judged.** Where its public code does something surprising, for example in
the commands it accepts during a warp, the port does the same.

## 4. Collisions: `VecCollidePark` and its kernel

**`exact_ballpark.vec_collide.VecCollidePark`** holds:
- goto and stop;
- both of destiny's collision methods, gradient (the default) and iterative;
- fixed spheres;
- static collidables: capsules and oriented boxes;
- minis: miniballs, minicapsules and miniboxes;
- force fields and harmonics;
- global balls, over the space layer.

**The kernel** (`exact_ballpark.vec_collide_cuda`) runs the gradient method only. The iterative method stays on the
torch path, and the kernel refuses it.

**Static collidables, minis and force fields:**
- They work with the gradient method only.
- They are refused on the master's view with bubbles.
- They are fixed at the load. Removing or re-adding a static collidable during a run is not covered, and neither is
  making or removing a parent's minis when it is fixed or freed during a run.

Not in this park:
- follow, orbit, warp and missiles (the composed park has them with collisions, section 8);
- proximity sensors, whose checks re-file their balls;
- moribund balls;
- balls whose filing would clamp at the partition's edge or span its 27 top boxes;
- dynamical orientation (off, as destiny's default).

## 5. Space: `VecSpace` and the delayed removal

**`exact_ballpark.vec_space.VecSpace`** holds:
- the partition's filing;
- bubbles and their dictionaries;
- interactive and keep-alive sets, and dead bubbles;
- proximity sensors, including shuffled ones, with target tracking and notification ranges;
- the visibility and range queries;
- global balls.

It runs over the flight-mode, orientation and collision parks, on the CPU and the GPU:
- **On the CPU,** it runs host functions compiled from the same C++ source as its kernel, by `nvcc` or, where `nvcc` is
  absent, by `g++` (`devices.md`, section 3).
- **On a CUDA device,** it runs its kernel, so it needs `nvcc`. A library built by `g++` holds no kernel, and the layer
  refuses it there.

**Its domain:**
- free or fixed balls with positive ids;
- no ball wider than a top box;
- no ball at the edge of the partition's grid, where destiny clamps;
- no space junk, harmonics or force fields;
- no static collidables;
- over the flight-mode and orientation parks, no two massive balls meeting, since those parks have no collisions.

**Bounds:**
- at most 64 balls a world;
- at most 256 boxes a world, measured as each ball's boxes with their ancestors, summed, plus 8. Past it destiny can
  read a released box's memory, so the world is flagged.
- by default, at most 256 events a tick.

**`exact_ballpark.vec_space_removal.DelayedRemoval`** (balls kept moribund until destiny's `BringOutTheDeads`):
- It runs over the flight-mode and orientation parks only.
- It removes in ascending id, so it is exact against the reference, never against destiny (TIME-713).

Not covered by the delayed removal:
- removals inside a tick (missiles' contacts; the combat park has those);
- the removal of static collidables and minis;
- the server layer's view of a removed ball.

## 6. The server layers and the clients

**`exact_ballpark.vec_server.VecServer`**, the server layer with one bubble a world, and its kernel:
- one live, static bubble a world; a world whose bubble dies is refused.

Not in this layer:
- more bubbles, and bubble transitions;
- global balls;
- removals and additions after the load;
- missiles (the combat server layer has them);
- harmonics and force fields;
- formations;
- the master's sensor events.

Its kernel runs over the flight-mode park only, and takes at most 256 queued actions a tick.

**`exact_ballpark.vec_server_space.SpaceServer`**, the server layer over the partition (several bubbles a world), and
its kernel `CudaSpaceServer`:
- **Not in this layer:**
  - re-initialising bubbles during play (a ball in two interactive sets is flagged);
  - missiles;
  - `SetBallTroll` (refused);
  - removals, so also `RemoveGlobalBall` with its removals;
  - OnTick's per-world intervals in its kernel: `CudaSpaceServer.on_tick` refuses them. The torch path holds them over
    every park (section 1).
- **Global balls** are held through `AddBall`'s flag, and through `make_ball_global` and `make_ball_local` on the
  torch path.
- **The kernel** runs over the flight-mode park only. It refuses `make_ball_global` and `make_ball_local`. Its queue
  takes at most 256 actions a tick, including the follower clean-up's.
- **Over the collision park** the layer runs its torch path only. Its tick reads a count on the host, so it is not one
  CUDA graph.

**`exact_ballpark.vec_server_combat.CombatServer`**, the server layer over the combat park, and its kernel:
- one bubble a world;
- not over the composed park;
- no defender missiles;
- no `AddBall` or `RemoveBall` through the server layer.

**The clients:**
- **The one-bubble client view** is a delay line. It is exact for whole-tick delays and in-order delivery. A late
  update, with the client's prediction, rewind and clock slip that follow, is not modelled.
- **The client view over the partition** is exact for synced clients and the balls they hold fresh.
  - A diverged client's park is flagged, not simulated.
  - It records one server tick a call, so a world's first OnTick, or several intervals in one call, are not modelled.
- **Where destiny's own client and server layers disagree** about the same events, the batched views give the server's
  state.

## 7. Combat: `VecCombatPark` and its kernels

**`exact_ballpark.vec_combat.VecCombatPark`** holds:
- missiles: launch, straight flight then pursuit, contacts, release and removal;
- formations;
- cloaking;
- mushrooms and trolls;
- the commands of section 3;
- immediate additions and removals;
- OnTick's per-world intervals, with destiny's `BringOutTheDeads` at each world's call.

Its kernels run them on a CUDA device. Its tick on tensors runs them on any device, FOLLOW, ORBIT and WARP included.

**Its domain:**
- the gradient method;
- no two balls meet except a normal missile and its target, since there is no other collision code (the composed
  park, section 8, has it);
- no defender missiles;
- on the master, one bubble a world with an interactive ball;
- no call whose result destiny leaves undefined;
- no delayed removal; a mushroom removed in a tick is the one moribund state it holds;
- no second mushroom on a client while the first is in the park.

**Also:**
- Formations of more than 16 slots are refused, as in destiny.
- There is one formation table for the whole park, so a `LoadFormations` reloads every world at once.
- Without dynamical orientation, the yaw, pitch and roll are computed on the host by default, and the park's CUDA
  graph needs dynamical orientation.
- **A CUDA graph of the park (`CombatGraph`) replays its kernels only.** The eager tick (`evolve`) runs the park's
  domain watch, `watch()`, at the start and at the end of every tick, around its kernels. A replay leaves both out.
  - **What `watch()` flags:** every world that holds two or more balls that are in the park (not an empty row, not
    moribund), massive and not in missile mode, free or fixed, however far apart they are. This park has no
    collision code for them, so it refuses the configuration, not a contact. The flag goes on all of the world's
    rows and stays until the park is loaded again.
  - **What `check()` does:** it raises on the flags already set. On the master it also tests, itself, for a world
    with balls in more than one bubble or without an interactive ball. It does not run `watch()`: right after a load,
    a world holding two massive balls has no flag yet, and `check()` passes.
  - **So, around each replay, in this order:**

    ```python
    park.watch()    # the eager tick's watch at its start
    graph.tick()    # the replay: the park's kernels (graph.tick(active) with OnTick's mask)
    park.watch()    # and at its end
    park.check()    # raises if any world is outside the domain
    ```

    Without OnTick's mask, this sets the same flags as the eager tick. With it (a graph made with `masked=True`),
    `watch()` also flags a world outside the mask, which the eager masked tick leaves as it was: stricter, never
    laxer.
- **`CombatGraph` captures this park's tick only.** It refuses a `VecComposedPark`, whose tick with collisions on is
  `ComposedGraph`'s, and whose tick with collisions off makes collision lists after the kernels, which a replay would
  leave out.

## 8. The composed park: `VecComposedPark` and its kernel

**`exact_ballpark.vec_composed.VecComposedPark`** holds flight modes, missiles (defender missiles included),
formations, cloaking, mushrooms, trolls and the commands, all together with ship-to-ship collisions. It also holds
OnTick's per-world intervals, as the combat park does. Its CUDA graph (`ComposedGraph`) needs collisions on, and
then leaves nothing of the eager tick out. With collisions, two massive balls are computed, so the park's `watch()`
does nothing. The refusals below are flagged inside the tick by its kernels (a massive ball in warp proper; a ball at
the partition's edge, or wider than a top box), or by the cloak command before it. As after an eager tick, a flagged
world raises only at `check()`. With collisions off there is no graph: the park ticks eagerly (`evolve`), and its
`watch()` flags two massive balls as the combat park's does.

**With collisions off,** the park ticks eagerly on a CUDA device: the combat park's kernels, then the collision lists,
made from the kernels' counts as its tick on tensors and the reference keep them. Its CUDA graph (`ComposedGraph`)
needs collisions on.

**Its domain is the combat park's, less two refusals:** two massive balls may meet, and defender missiles are
computed. In addition, it refuses:
- a massive ball in warp proper;
- a ball at the partition's edge, or wider than a top box;
- on the master, a normal cloak, whose proximity sensor re-files its ball.

Not in this park:
- the iterative method;
- static collidables, minis and force fields (its build refuses the static collidables' code);
- space junk;
- defender missiles with collisions off;
- a CUDA graph with collisions off;
- proximity sensors;
- more than one bubble a world: apart from a missile's one candidate, its balls' candidates are a client's view;
- the delayed removal;
- a second mushroom on a client while the first is in the park.

## 9. Not ported

- The display code of a client's ball and its interpolation (TIME-708).
- A fused kernel for the iterative collision method. The method itself can be used on the CPU and the GPU, through the
  torch path (section 4).
- Space junk.
