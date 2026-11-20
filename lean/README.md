# Exact Ballpark laws in Lean

What happens when a ship stops thrusting, two swept spheres touch, or a server tick arrives late? Exact Ballpark gives each behavior a rule ID. This project links **105 engine laws** to **238 kernel-checked theorems**. Each entry records the part of the law covered, its assumptions and the evidence from the engine. [Claim register](CLAIMS.md) · [Engine rule catalogue](../docs/rules.md)

[![Exact Ballpark law map. Movement has 41 laws, warp 16, collisions 19, space and sensors 12, time and events 15, and combat 2.](assets/law-map.webp)](assets/law-map.webp)

## The six law families

| Family and rule IDs | What the laws govern | One result Lean checks |
| --- | --- | --- |
| **Movement and flight** · 41 · MOV-701–741 | Drag and thrust, GOTO and STOP, following and orbiting, orientation, property setters and numerical primitives. | Over a positive interval with fixed positive effective mass and friction and constant acceleration, the gap to terminal velocity shrinks by exactly the exponential decay factor. [Movement bounds](ExactBallparkProofs/Movement/Bounds.lean) |
| **Warp** · 16 · TRV-701–716 | The command to align, the transition into warp, acceleration, cruise, deceleration and the return to ordinary flight. | A destination strictly closer than 100 km selects GOTO. At or beyond that boundary, repeating the current warp command preserves the ship’s alignment state. [Warp commands](ExactBallparkProofs/Warp/Commands.lean) |
| **Collisions** · 19 · COL-701–719 | Candidate selection, straight sweeps, first contact, overlap correction, elastic response and iterative impacts. | For two spheres moving along straight chords over the reals, the solver returns the earliest contact in the tick, zero for initial overlap, or a miss. [Swept spheres](ExactBallparkProofs/Collision/Sweep.lean) |
| **Space and sensors** · 12 · SEN-701–712 | Partition levels, connected bubbles, proximity events, visibility, cloaking and which clients receive an update. | Starting from a seed box, the bubble traversal labels exactly the initially unassigned boxes reachable through adjacent cells. [Bubble traversal](ExactBallparkProofs/Space/Bubbles.lean) |
| **Time, server and clients** · 15 · TIME-701–715 | Tick duration, backlog, update order, command queues, removal and client synchronization. | In the modeled tick, every ball’s thrust calculation precedes collision responses, and every new state is calculated before the first commit. [Tick schedule](ExactBallparkProofs/Time/Schedule.lean) |
| **Combat** · 2 · CMB-701–702 | How relative motion across the line of sight, tracking and range determine hit probability. | For distinct positions on the ordinary accuracy branch, the computed probability lies in (0, 1]. With zero transversal speed and a target inside optimal range, it is exactly one. [Accuracy](ExactBallparkProofs/Combat/Accuracy.lean) |

The latest ten proofs follow collision candidates from spatial filing to the time-of-impact decision. For ordinary interior filings, the scalar tree and the tensor sort return the same ordered IDs. CPU probes compare **54 ordered candidate lists** and their packed keys across eight levels and both tensor sorting paths. They also run the face tests on the stored binary64 inputs. The movement bound explains when the filing radius covers the entire regular drag trajectory. The collision proofs cover minimum-time selection, ties, partial windows and repeated free-ball hits. [Candidate order](ExactBallparkProofs/Space/CandidateOrder.lean) · [Filing radius](ExactBallparkProofs/Space/FilingRadius.lean) · [Collision search](ExactBallparkProofs/Collision/SearchWindow.lean)

The formation proofs cover flight commands under MOV-726. If a ship requests a slot beyond the end of a formation, it keeps following its old leader. If the slot fits but the new leader is cloaked or in another bubble, the ship stops. Shrinking a formation on the master releases the slots of the ships it stops. A reservation with no ship attached can survive the shrink and block that slot when the formation grows again. The CPU check compares Lean with the scalar engine on 147 reservations, 63 releases, 280 joins and 50 resize sequences. [Formation proofs](ExactBallparkProofs/Movement/Formation.lean) · [Resize documentation discrepancy](DISCREPANCIES.md#formation-resize-documentation)

The server and client proofs follow an update from server batching to client rewind. A client with both private and bubble events receives the private updates first, with a flag telling it to wait for the bubble. On the client, an eleventh snapshot trims the cache to its original base and the five newest streams. If two saved streams have the requested stamp, a rewind restores the last eligible one in storage order. CPU checks compare Lean with the scalar engine on 50 outgoing batches and 133 snapshot operations. [Server batches](ExactBallparkProofs/Space/Batches.lean) · [Client snapshots](ExactBallparkProofs/Time/Snapshots.lean)

## When the engine changes a collision or movement outcome

The log ranks two collision-search discrepancies and seven rounding cases by their observed gameplay consequences and describes the likelihood of each trigger. [Gameplay impact review](DISCREPANCIES.md#what-a-player-might-notice)

**COL-715 · A farther obstacle can win after a rebound.** In a one-second encounter with three fixed obstacles, changing just the time comparison reversed the ship’s final direction. The original scalar and pinned C++ engines agreed on the final position and velocity bits. [Four-ball replay](DISCREPANCIES.md#col-715-fixed-window)

**COL-715 · A lookup can select the pre-impact velocity.** With an impact at 1/16 of a second, rounding the sum of its time and the lookup tolerance changes which record the engine reads at one representable query. In a scalar replay with a supplied second contact, the ordinary lookup missed a rebound and the ship ended the tick 1.22 m from the control. The impact-history proofs also cover insertion, removal, invalidation order and reconstruction of the tick endpoint. [History proofs](ExactBallparkProofs/Collision/ImpactHistory.lean) · [Lookup discrepancy and replay](DISCREPANCIES.md#col-715-history-lookup)

**COL-715 · A second free-ball hit can disappear.** With a supplied turn in the other ball’s history, the engine kept the old start positions after suppressing a repeat. The two-chord search missed a hit that the final chord alone reported at 0.708901778 s. [Repeat-contact replay](DISCREPANCIES.md#col-715-free-repeat)

**COL-705 · A missed sphere can register a hit.** A ship sweeps from (−2²⁸, 2, 0) to (2²⁸, 2, 0) past a stationary sphere at the origin, with combined radius one metre. The closest approach is two metres. In binary64, cancellation makes the quadratic discriminant zero. Lean proves the real miss and the rounded contact at half a tick. The scalar Python check reproduces that contact. [Finite sweep proof](ExactBallparkProofs/Collision/FiniteSweep.lean)

Move the same sweep out to five metres and Lean can certify the miss from the endpoint bits alone. An offline checker bounds the rounding error in the discriminant. When its entire interval lies below zero, both the modeled binary64 solver and the exact straight-chord solver return −1. At the original two-metre offset the interval contains zero and the checker returns “uncertain.” A positive discriminant still requires a check that the first root falls within the tick. [Sweep certificate](ExactBallparkProofs/Collision/SweepCertificate.lean)

The timing proof checks bounds on the first root against both tick endpoints. For a sweep from x = 4 to the next binary64 value above 1, Lean places first contact strictly after the tick. Scalar Python and pinned Destiny C++ both report contact at exactly 1. At exact tangency the opposite error is possible. In a six-metre sweep that touches halfway through the tick, the rounded discriminant is −1008 and both engines return a miss. [Timing proof](ExactBallparkProofs/Collision/Timing.lean) · [Discrepancy log](DISCREPANCIES.md)

**COL-718 · The impact limiter can admit an impulse just above its nominal bound.** For a maximum speed of 4097, five times that speed is 20485. The engine rounds the squared threshold in binary32 and accepts a change of 20485 + 2⁻¹³ along one axis. [Ideal response proofs](ExactBallparkProofs/Collision/Impact.lean) · [Mixed-precision witness](ExactBallparkProofs/Collision/FiniteImpact.lean) · [CPU replay](DISCREPANCIES.md#col-718-impact-limit)

**MOV-703 · Position rounding depends on thrust and coordinate scale.** For the extreme prescribed acceleration in the Lean witness, the stored position is +0 m, the first-order branch gives 1 m over the reals, and the regular drag trajectory gives about 524,289 m. With full GOTO thrust at a configured 10,000 m/s, the measured branch error was 0.583 pm from the origin and 30.518 µm from one AU. [Branch proof, CPU measurements and rounding bound](DISCREPANCIES.md#mov-703-position)

**TRV-709 · Short warps can have a tiny negative cruise duration.** At factor 1000, trips of 100,000 and 100,001 metres both have zero cruise in exact arithmetic. The engine rounds them to opposite sides of zero. Both sampled ships exited warp on the same tick as their zero-cruise controls. [Warp discrepancy and replay](DISCREPANCIES.md#trv-709-short-cruise)

## What connects a theorem to the engine

| Python source | How it enters Lean | Scope |
| --- | --- | --- |
| `Ballpark.integrate` | A restricted translator reads the guards, cached decay factor and order of arithmetic for one coordinate. | The extracted real program equals the integrator used in the trajectory proofs. A second translation models binary64 operations with the exponential supplied as an input function. |
| `ModeBallpark.warp_constants` | A translator reads the scalar expressions and constants. | All nine returned real-valued expressions agree with the independently defined warp parameters. |
| `BaseTicker.tick` and `Actions.remove_ball` | A translator extracts ordered calls and the guard for global removal notifications. | The six pre-tick hooks run in order. After a successful lookup, removal ends with the parent call after any global notification, assuming the calls return. |
| Other registered methods | Contributors transcribe the relevant expressions or transitions and pin the reviewed source by hash. | Each theorem states the property of that model and the inputs or state it requires. |

The Python-to-Lean translators are trusted. Applying a real-number theorem to a Python execution requires its operations to be defined. CPU probes compare the scalar integrator’s position and velocity bits, including signed zero, and exercise the mass constructor and agility setter on halfway cases and NaNs. Compiled CPU and GPU conformance has its own platform-specific evidence. [Numerical contract](../docs/numerics.md) · [Platform qualification](../docs/platform.md) · [Conformance evidence](../docs/exactness.md)

## Verify locally

Use Python 3.13 or later with the package’s PyTorch dependency and Elan’s Lake from the repository root. On a fresh checkout, fetch the compiled mathlib dependencies used by this project. The following Linux commands use eight CPUs. Choose CPU IDs from the process’s allowed affinity set on other machines.

```bash
cd lean
mapfile -t modules < <(rg --no-filename '^public import Mathlib\.' ExactBallparkProofs | cut -d' ' -f3 | sort -u)
taskset -c 0-7 lake exe cache get "${modules[@]}"
taskset -c 0-7 python3 scripts/check.py
```

The project pins Lean 4.36.0 and the mathlib revision in the Lake manifest. The checker prefers `.lake/toolchain/bin/lake`, falls back to Lake on PATH, and checks the selected Lean version against `lean-toolchain`. It runs the incremental build with warnings treated as errors, checks reviewed source hashes, regenerates the source models in memory, validates the theorem registry, audits transitive axioms, runs small scalar comparisons and checks the generated page. Use `python3 scripts/check.py --write` to refresh the page after changing a proof or claim.

## Extending the proofs

1. Choose a remaining obligation in the [register](CLAIMS.md). Add a source-shaped definition or extend an existing source translation. Import the module from `ExactBallparkProofs.lean`.
2. State a theorem with explicit input, mode, numerical and cache assumptions. Record the declaration, source symbols, rule IDs and remaining obligations in `claims.json`. Use `ruleTheorems` to associate individual rules with their supporting declarations when a claim group spans several laws.
3. Run the checker. The allowed transitive axioms are `propext`, `Quot.sound` and `Classical.choice`. The audit checks project theorems outside the registry, rejects project-declared axioms and verifies that every proof module is imported. For a source change, review the affected code and its correspondence before updating the source hash.
4. For an implementation claim, identify the operations and backend it covers. Outstanding work includes full scalar state transitions, floating-point geometry near contact, whole elementary functions, and correspondence to the batched and fused kernels. Device qualification is recorded separately in the engine’s [platform](../docs/platform.md) and [evidence](../docs/exactness.md) documents.
