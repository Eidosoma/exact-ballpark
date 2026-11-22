# Engine discrepancy log

[![13 replayed cases by gameplay severity. Two high-severity collision replays end in opposite directions. Seven moderate cases cover collisions, client updates and tracking. Two low and two negligible cases cover grazing contact, impulse, position and cruise rounding.](assets/discrepancy-log.webp)](assets/discrepancy-log.webp)

This log records **13 engine behaviors** reproduced on x86-64 Linux and **one documentation correction** for MOV-726. Two collision-search decisions, eight rounding cases and a capsule motion cutoff have matching scalar Python and compiled Destiny outputs. Two packet-ordering cases reproduce in the original Destiny Python client and Exact Ballpark’s client. The formation entry has a scalar replay and a reading of the C++ source. The control experiments below identify which inputs or comparisons they change.

The scalar replay used **CPython 3.13.18** and a working tree based on Exact Ballpark `098653233eb0f9910a3f02564e36b0641cbd4e0f`. Among the reviewed engine files, only `full.py` differs from that commit, in the corrected formation docstring. The C++ replay used Destiny `114e89fa584f0e1a81eb0a679aabf41af8870072` with GCC 11.4.0 and `-O2 -ffp-contract=off -fno-fast-math`. The runtime evidence here applies to this CPU platform. [Working source hashes](claims.json) · [CPU replay](scripts/replay_discrepancies.py) · [Platform evidence](../docs/exactness.md)

## What a player might notice

The ranking describes the gameplay consequence in the stated scenario. **High** means a replay changed the ship’s direction. **Moderate** covers a distinct collision gained, lost or assigned to an earlier tick, an accuracy query outside the probability range, and client queue behavior that can delay or discard updates. **Low** covers a missed tangent with zero normal approach speed or a small excess impulse. **Negligible** covers the measured GOTO position rounding and the short warps that followed the same phases and exited on the same tick as their controls. MOV-703 also has a separate numerical stress case.

The likelihood column gives the operating mode and input scales needed to reach each case. The replays measure outcomes on selected inputs. Live encounter rates would require telemetry. A separate COL-705 example measures how far the documented straight sweep can pass from a turning ship’s continuous trajectory. [Turning flight and clear control](#col-705-turning-path)

| Severity | Case | In plain terms | Likelihood of noticeable impact | How it could show up in play |
| --- | --- | --- | --- | --- |
| High | [COL-715 · wrong fixed obstacle](#col-715-fixed-window) | After a bump, the ship can select a farther obstacle along its path. | **Reachable in an ordinary four-ball tick with iterative collisions enabled.** It needs two eligible fixed obstacles in the remaining window. Frequency in live encounters is unknown. | The engine and the earliest-hit control ended the tick moving in opposite directions. |
| High | [COL-715 · wrong side of a bump](#col-715-history-lookup) | A ship can use its velocity from before the first bump to respond to the second. | **Reachable from three starting ball states.** The 100 ms replay uses the optional iterative mode and finely spaced contact times. Live frequency is unmeasured. | The engine missed the rebound at the second contact. Changing just the lookup made the same ship finish 1.325268 m away, moving in the opposite direction. |
| Moderate | [COL-715 · missed second hit](#col-715-free-repeat) | After two ships separate and one turns back, the second impact can disappear. | **Conditional on iterative collisions and a return within the same tick.** The replay supplies impact history. Its occurrence rate during ordinary play is unknown. | Both engines missed a distinct contact. A game response to that contact could have changed the subsequent paths. |
| Moderate | [TIME-715 · extra bubble wait](#time-715-extra-wait) | One private update can leave two timestamps waiting for a bubble reply. | A session with ordinary commands reproduced this after postponing a server send for one tick. Sending after each tick cleared the queue. Live incidence is unknown. | The client reports a fatal desync for the mixed packet. With the replay’s returning handler, the later command stays queued until a matching reply arrives. |
| Moderate | [TIME-715 · mixed update after reset](#time-715-reset-order) | Reversing two events in the same update can admit an obsolete command or discard a fresh one. | Requires an ordinary packet containing events from both sides of the last state reset. Live incidence is unknown. | A movement command sent after the reset can disappear. If the fresh event arrives first, an older command can reach the dispatch queue. The replay measures admission. |
| Moderate | [COL-712 · shallow capsule crossing](#col-712-shallow-crossing) | A ship crosses a capsule wall by less than one micrometre and the collision check misses it. | Requires the optional iterative path and very little motion across the capsule axis. The approach can also have an axial component. | The contact can be skipped for this tick. The replay measures the intersection test. A subsequent tick starting inside may detect overlap. |
| Moderate | [COL-705 · false contact](#col-705-long-sweep) | Two spheres pass with a metre to spare, but the collision check reports a touch. | **Likely low for ordinary flight.** The example sweeps 536,870,912 m in a tick past a target with a combined radius of just 1 m. | If that pair reaches the collision solver, it could be processed as a contact. Whether it produces a bump depends on the response calculation and the other contacts that tick. |
| Moderate | [COL-705 · early contact](#col-705-tick-end) | A ship ends the tick barely clear. The engine counts contact in that tick. | **Likely low visually.** In the example the gap is about 2.22 × 10⁻¹⁶ m. The timing can still matter when two contacts compete. | Contact handling could run before the next movement command, or before another collision. That ordering change can matter more than the tiny gap itself. |
| Moderate | [CMB-702 · radial tracking](#cmb-702-tracking) | A target moving along the line of sight can have an undefined reported angular speed. In the stress case its hit probability is infinite. | A nearly radial velocity can trigger the angular result. Overflow also requires a sufficiently large tracking ratio, here 10³⁰. Live incidence is unmeasured. | A caller using the result as a probability can receive a value above one. The replay measures the query output. |
| Low | [COL-705 · missed graze](#col-705-tangency) | A path just touches a very large sphere and the engine reports a miss. | **Likely low for the demonstrated geometry.** It requires exact grazing at a combined radius of 268,435,456 m. Perfect tangency also has zero incoming speed along the normal. | A contact record could be absent. A visible bounce is a weaker expectation because the ship was already moving along the surface. |
| Low | [COL-718 · speed after a bump](#col-718-impact-limit) | The engine accepts a slightly larger change in velocity than the ideal collision limit. | **Very low in this example.** This limiter runs in the optional iterative collision mode. The excess is about 0.12 mm/s. | If maintained for one second, the excess alone would move the ship about 0.12 mm. A bit-exact replay can distinguish it immediately. |
| Negligible | [MOV-703 · special-branch rounding](#mov-703-position) | A one-second GOTO ends a tiny distance short of the branch’s predicted position. | **Small at the tested thrust scale.** At default friction the branch requires effective mass above 10¹⁶ for a one-second step. | The measured error was 0.583 pm from the origin and 30.518 µm from one AU. The separate metre-loss stress case prescribes an extreme acceleration. |
| Negligible | [TRV-709 · short warp](#trv-709-short-cruise) | The cruise phase should last zero seconds. The engine stores a tiny positive or negative duration. | **Very low for the sampled flights.** The arithmetic difference was present at all 1,000 tested distances. The two trajectory replays kept the same phase and exit tick. | A diagnostic can show a negative duration. In the two sampled flights the largest position change attributable to these cruise fields was under 0.000000000015 m. |

The wrong-obstacle and history-lookup replays start full ticks from four and three ball states respectively. The missed second free-ball hit uses supplied impact history. [Fixed and free collision replays](scripts/contact_replay.py) · [History replay](scripts/replay_discrepancies.py)

## Where the result changes in Exact Ballpark and how to fix it

Each proposal would change compatibility with the pinned upstream behavior on the affected inputs.

| Case | Where the result changes | Candidate fix and expected result |
| --- | --- | --- |
| **High · COL-715 · wrong obstacle** | The fixed-ball loop compares the new fraction `toi` with the stored absolute tick time `item.time`. | Compare `t < item.time` after computing `t`. Keep the fraction for the ignore test and the normal. In the half-tick witness either order then selects ball 3. [SpaceBallpark._next_collision_static](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1834-L1840) |
| **High · COL-715 · missed rebound** | In `find_index`, adding `FUZZY_DELTA` rounds the cutoff up to the query time. The strict comparison then stops one record early. [CollisionBallProperties.find_index](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L488-L492) | Keep the strict comparison. When the rounded sum equals the query, compare the exact sum of the stored timestamp and tolerance against the query. The control replay does this with exact fractions and recovers the rebound. |
| **Moderate · COL-715 · missed second hit** | An ignored hit executes `continue` above the assignments to `b0, o0`. The next interval starts at the old positions. | Advance the two position anchors on the ignored path before continuing. In the supplied history the next chord then finds the returning ball. A compatibility change would also need the tensor path’s corresponding anchor policy. [SpaceBallpark._next_collision_free](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1867-L1879) · [VecCollidePark._response](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/vec_collide.py#L2053-L2058) |
| **Moderate · TIME-715 · extra bubble wait** | Appending into empty history copies `wait_for_bubble` to every sorted group. [merge_state_into_history](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/client.py#L244-L271) | Associate the wait with its expected reply stamp. For the one-reply-at-the-earliest-stamp case, clear the flag after the first tail append. Replay a later reply as a separate test. |
| **Moderate · TIME-715 · mixed update after reset** | The ordinary packet guard checks only `state[0][0]` before merging every event. [Ticker._flush_state](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/client.py#L315-L330) | Filter ordinary events against the last SetState stamp, retaining order within each surviving group. Skip the merge if all events are stale. Both orders of the replay should then keep only stamp 11. |
| **Moderate · COL-712 · shallow capsule crossing** | The outside-barrel path rejects all sub-threshold perpendicular displacements before computing a root. [FullBallpark._capsule_check](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/full.py#L2839-L2856) | Distinguish zero perpendicular displacement from a short crossing. Use a numerically stable short-step or higher-precision fallback for the latter and validate the axial foot and cap hits. |
| **Moderate · COL-705 · false contact** | Squaring the long separation loses the small sideways offset. The resulting discriminant rounds to zero and creates an apparent tangent hit. [SpaceBallpark.collide_two_spheres](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1633-L1643) | Bound the rounding error. When the bound includes zero, recompute the discriminant from the original endpoint and radius values using exact arithmetic. Its sign is negative here, giving a miss. |
| **Moderate · COL-705 · early contact** | Subtracting the endpoints rounds the displacement to −3. The root calculated from that displacement passes the inclusive end-of-tick test at exactly 1. [SpaceBallpark.collide_two_spheres](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1637-L1647) | When the root bounds straddle a tick boundary, use the original endpoints to decide whether the first root lies in `[0, 1]` before rounding the returned time. This rejects the logged root just after 1 and still accepts a true endpoint contact. [Root-bound proof](ExactBallparkProofs/Collision/Timing.lean) |
| **Moderate · CMB-702 · radial tracking** | Scalar and tensor accuracy subtract the squared radial projection from squared total speed. Cancellation can give a negative result on finite inputs. [FullBallpark.get_accuracy](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/full.py#L1977-L1988) | For a gameplay-corrected variant, guard a finite negative squared speed before both the square root and the exponent in each backend. Retain the existing behavior in a Destiny-compatible replay. |
| **Low · COL-705 · missed graze** | The dot product rounds `R² + 9` to `R² + 16` before the solver subtracts `R²`. This changes an exact zero discriminant to −1008, triggering the early return. [SpaceBallpark.collide_two_spheres](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1638-L1643) | Use the same discriminant fallback as for false contact, including when the computed value is negative but its sign is uncertain. Exact zero yields a repeated root at half a tick here. |
| **Low · COL-718 · excess impulse** | The first binary32 product, 25 × 4097 = 102425, is exact. Multiplying by 4097 again rounds 419635225 up to 419635232. The comparison then admits the entire impulse. [SpaceBallpark._add_impact](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1948-L1951) | Promote the stored maximum speed to binary64 before either multiplication and use the existing scaling formula. With a binary64 leading factor, the numerator is exact for every finite stored binary32 maximum speed and the limiter engages on this input. Division and final vector scaling still round. |
| **Negligible · MOV-703 · special-branch rounding** | After first-order substitution, the two acceleration terms should cancel. Rounding the intermediate sum erases the incoming velocity term in the stress case. | Evaluate `p[i] + v[i] * t` directly. This returns 1 m for the stress input and preserves the branch’s first-order approximation. Recovering the regular drag trajectory would require changing the branch law. [Ballpark.integrate](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/ref.py#L182-L192) · [SpaceBallpark.integrate](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/space.py#L1618-L1627) |
| **Negligible · TRV-709 · short-warp cruise** | After selecting the distance-limited peak speed, `warp_constants` recomputes cruise time by subtracting nearly equal terms. The remainder can have either sign. [ModeBallpark.warp_constants](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/modes.py#L610-L616) | Record when the distance-limited peak is at or below the requested speed. In that branch, set cruise duration and cruise distance to zero directly. This gives zero for both logged trips and retains the cruise calculation for longer trips. |

<a id="where-the-result-changes-in-destiny-c-and-how-to-fix-it"></a>

## Where the result changes in Destiny and how to fix it

Destiny implements the collision and arithmetic rules in C++ and the client queue in Python. The source links identify the original functions at the replay revision.

| Case | Where the upstream result changes | Candidate upstream fix and expected result |
| --- | --- | --- |
| **High · COL-715 · wrong obstacle** | `GetNextCollisionStatic` computes the absolute `toi` and compares the original fraction `timeOfImpact` with the incumbent’s tick time. | Use `toi < collisionItem.timeOfImpact` in the overwrite guard. Preserve the strict tie policy and the fraction used for the normal. [Fixed-ball guard](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L2146-L2166) |
| **High · COL-715 · missed rebound** | `CollisionBallProperties::FindLocationsIndex` adds `m_fuzzyDelta` to the record time in `double`. The rounded sum equals the query, the loop stops, and the index steps back to the pre-impact record. Both position and rotation lookup use this helper. [History cutoff](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/CollisionBallProperties.cpp#L216-L250) | Keep the strict comparison. On equality, use an error-free sum such as `TwoSum` to recover the signed rounding residual. Advance only if the residual is negative, meaning the exact cutoff precedes the query. The logged query then selects record 1. |
| **Moderate · COL-715 · missed second hit** | `GetNextCollisionFree` continues after `IgnoreCollision`, bypassing the update of `ballP0` and `otherP0`. | Let an ignored hit fall through to the shared end-of-segment update while leaving the incumbent untouched. The next sweep starts at the separating positions and finds the later contact. [Free-ball loop](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L2268-L2300) |
| **Moderate · TIME-715 · extra bubble wait** | The tail append in Destiny’s Python client copies the same wait flag into every remaining time group. Its tick returns as soon as it sees a waiting head. [Client tail append](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/python/destiny/net/client/_util.py#L57-L68) · [Client wait guard](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/python/destiny/net/client/_ticker.py#L93-L102) | If a private packet expects one reply at its earliest stamp, consume the flag after appending that group. If the protocol can designate another stamp, carry the expected reply stamp explicitly. |
| **Moderate · TIME-715 · mixed update after reset** | The original Python ticker checks the first event against the most recent SetState, then sends the full packet to the merge. [Client admission](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/python/destiny/net/client/_ticker.py#L71-L84) | For ordinary packets, drop each group older than the most recent reset, retain eligible events in order and skip merging an empty result. Keep the reset path’s handling of later entries. |
| **Moderate · COL-712 · shallow capsule crossing** | `Capsule::CheckCollision` returns false as soon as squared perpendicular travel is below 10⁻¹⁰. [Capsule motion cutoff](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Capsule.cpp#L142-L159) | For a short nonzero perpendicular step, check the radial quadratic over the remaining chord with a stable root calculation. Retain the axial-foot and end-cap tests. The two replayed chords should both return ½. |
| **Moderate · COL-705 · false contact** | `CollideTwoSpheres` loses the sideways clearance in the `Vector3d` dot product. By the time `Quadratic` receives `a, b, c`, the discriminant evaluates to zero. [Sphere sweep](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Collision.cpp#L108-L129) · [Dot product](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Vector3d.h#L48-L49) | Put the fallback in `CollideTwoSpheres`, where all four endpoints and the radius are available. If a rounding-error bound leaves the discriminant sign uncertain, recompute its sign exactly from those inputs. Here it is negative, so return the existing miss value `-1.0`. |
| **Moderate · COL-705 · early contact** | In `CollideTwoSpheres`, the endpoint subtraction in `dpq` rounds to −3 on the sweep axis. The earlier root `s2` then passes the inclusive test at 1. [Root interval test](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Collision.cpp#L118-L130) | When the first root is too close to either tick boundary to classify reliably, test its membership in `[0, 1]` using exact coefficients reconstructed from the endpoints. Round the reported time after that decision. The logged sweep returns `-1.0` and a true endpoint hit still returns `1.0`. |
| **Moderate · CMB-702 · radial tracking** | The subtraction in `Ballpark::GetAccuracy` can make the squared transversal speed negative. Its square root and its contribution to the power then violate the mathematical range. [Tracking subtraction](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L5205-L5217) | A finite negative result can be clamped to zero before both uses. Qualify the change across tracking ratios and range falloff. |
| **Low · COL-705 · missed graze** | The squared separation rounds up before the radius squared is subtracted. `Quadratic` receives 16 for the constant coefficient instead of 9 and exits on a discriminant of −1008. [Coefficient construction](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Collision.cpp#L110-L123) · [Quadratic miss guard](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Collision.cpp#L150-L156) | Use the same endpoint-based fallback for an uncertain negative discriminant. Exact zero is a tangent contact. Evaluate its repeated root `-b / (2*a)` and apply the tick bounds. This sweep returns `0.5`. |
| **Low · COL-718 · excess impulse** | In `Ballpark::AddImpact`, `mMaxVel` is a `float`. With the integer literal `25`, both numerator multiplications run in float precision. Division by the double `speed2` promotes the already-rounded value. [Maximum speed field](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ball.h#L126) · [Impact speed limiter](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L2597-L2604) | Change the leading literal to `25.0`. Both products then use binary64. The significand of 25 × `mMaxVel`² needs at most 5 + 24 + 24 = 53 bits, and the exponent range fits binary64 even for binary32 subnormals. The numerator is exact for every finite binary32 input. The existing `fraction < 1` branch then scales the impulse. |
| **Negligible · MOV-703 · special-branch rounding** | `Ballpark::Integrate` substitutes the linear Taylor approximation for the decay in the position expression. Vector division multiplies by a reciprocal. The addition inside `a * t + v * k / m * t` loses the velocity term in the stress case. | Set `p += v * t` before updating velocity. This evaluates the first-order position using the incoming velocity. The omitted acceleration contribution belongs to the regular drag law. [Integrator](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L759-L774) · [Vector division](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Vector3d.h#L45-L46) |
| **Negligible · TRV-709 · short-warp cruise** | `Ballpark::SetupWarpConstants` lowers `warpSpeed` with `std::min`, then calculates `cruiseDuration` by subtracting nearly equal doubles. It multiplies the residual by `warpSpeed` to get `cruiseDistance`. [Warp cruise duration](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L4245-L4257) | Save the requested speed before taking the minimum. When the distance-limited peak is at or below it, assign `0.0` to both cruise outputs. Continue calculating the acceleration and deceleration outputs with the selected peak. Both logged trips then have zero cruise. |

## Replayed outputs

| Rule and observable | Mathematical or chronological result | Scalar Python and pinned C++ | Output bits |
| --- | --- | --- | --- |
| COL-715 · fixed obstacle order | Select the nearer obstacle in both orders | Near-first traversal selects the farther obstacle. Far-first traversal selects the nearer one. | `3fe8000000000000`, `3fe4000000000000` |
| COL-715 · history lookup and x velocity | Select post-impact record 1 | Select pre-impact record 0, return +0 m/s | `0000000000000000` |
| COL-715 · free-ball second contact | Recontact on the final chord | Searching both intervals misses. Searching the final interval alone returns a hit. | `bff0000000000000`, `3fe6af52c7e6490c` |
| COL-712 · shallow capsule crossing and control | Both chords first touch halfway through the tick | Short chord misses (−1 in the replay). Faster control hits at ½. | `bff0000000000000`, `3fe0000000000000` |
| COL-705 · long sweep | Miss | Contact at half a tick | `3fe0000000000000` |
| COL-705 · tick endpoint | First contact after the tick | Contact at the end of the tick | `3ff0000000000000` |
| COL-705 · exact tangency | Contact at half a tick | Miss | `bff0000000000000` |
| COL-718 · impact velocity limiter | Magnitude at most 20485 | 20485 + 2⁻¹³ | `40d4014002000000` |
| CMB-702 · radial accuracy and surface distance | A finite-input squared angular speed is nonnegative and the hit probability is at most one | +∞ probability, 11 m surface distance and NaN angular speed (compared by class) | `7ff0000000000000`, `4026000000000000` |
| MOV-703 · final position | First-order branch advances 1 m on the stress input | +0 m | `0000000000000000` |
| MOV-703 · GOTO positions | First-order branch advances 100 m from each origin | Both final coordinates round slightly below their exact branch values | `4058ffffffffffd7`, `42416a5d2d67ffff` |
| TRV-709 · cruise duration at 100,000 and 100,001 m | 0 s at both distances | +2.18 × 10⁻¹⁶ s and −1.83 × 10⁻¹⁶ s | `3caf75c000000000`, `bcaa6dc000000000` |

<a id="col-715-fixed-window"></a>

## COL-715 · a farther fixed obstacle can replace the earlier hit

Over the second half of a tick a ship sweeps from x = 0 to 8 m. Fixed balls 3 and 4 are centred at 3 and 5 m, with a combined contact radius of 1 m for each pair. Their hit fractions are ¼ and ½. Their absolute times are 0.625 and 0.75 s. When the nearer ball is visited first, the test for the farther ball compares 0.5 against 0.625 and accepts it. All four numbers are exactly representable. [Selection proof](ExactBallparkProofs/Collision/SearchWindow.lean)

| Enumeration | Selected ID | Absolute time bits in Python and Destiny |
| --- | ---: | --- |
| Fixed balls, near first | 4 | `3fe8000000000000` |
| Fixed balls, far first | 3 | `3fe4000000000000` |

Physlib’s unit types let Lean reject the engine’s fraction-to-seconds comparison. The control converts each candidate to seconds before comparing it with the incumbent. Lean proves the control selects ball 3 at 0.625 s in both orders, and the CPU gate checks both IDs and timestamp words against the scalar control. [Typed comparison and binary64 witness](ExactBallparkProofs/Physics/Units.lean) · [Scalar replay](scripts/contact_replay.py)

A one-second tick reaches the same decision after a rebound. Ball 1 starts at x = 105 m travelling at −16 m/s. Three fixed balls start at 100, 107 and 108 m. All share y = z = 100 m, radius 0.5 m, mass 16 and agility 1, with friction 1 and zero speed fraction. Scalar Python and pinned C++ reproduce the engine row bit for bit. The scalar control changes the fixed-ball comparison to select the earliest absolute time. The ship’s final positions in the two runs are **0.300768 m apart**. It finishes moving left in the engine run and right in the control. [Full-tick replay and control](scripts/contact_replay.py)

| Selection | Final x | Final x velocity |
| --- | ---: | ---: |
| Engine comparison | 101.867349 m | −14.688277 m/s |
| Earliest-time control | 102.168117 m | +15.030609 m/s |

<a id="col-715-history-lookup"></a>

## COL-715 · rounding the lookup cutoff selects the pre-impact record

Start a stationary ball with unit mass, agility and friction. Apply a +1 m/s impulse on the x-axis at t = 1/16, then complete the one-second tick. The history holds three records, at 0, 1/16 and 1. Query the state at binary64 word `3fb0068db8bac711`. The lookup scans past an entry only if its timestamp plus the stored tolerance (word `3f1a36e2eb1c432d`) is strictly less than the query.

Over the exact values of those words the sum is smaller than the query by 211/2⁶⁶ seconds, so the lookup selects record 1 with the impulse. In binary64 the sum rounds up to equal the query. The scan stops and the engine integrates from record 0, returning x velocity +0. At the next query word, `3fb0068db8bac712`, both implementations select record 1. Lean checks the two rounded indices and the exact comparison. The x86-64 replay checks the indices and the velocity at the boundary. [Lookup proof](ExactBallparkProofs/Collision/ImpactHistory.lean) · [CPU replay](scripts/replay_discrepancies.py)

To measure the consequence, a second scalar probe supplies a static contact at that same query time, with the surface normal pointing against the ship’s +x velocity. The ordinary lookup gives the second contact a zero x impulse. The table retains its binary64 sign bit as −0 m/s, which applies zero change in velocity. The control evaluates the lookup’s timestamp-plus-tolerance comparison using exact fractions of the same stored words, then uses the same scalar engine to calculate the response, insert it into the history and finish the tick.

| Lookup at the second contact | Second change in x velocity | End-of-tick x | End-of-tick x velocity |
| --- | ---: | ---: | ---: |
| Engine binary64 cutoff | −0 m/s | +0.608394 m | +0.391606 m/s |
| Exact cutoff in that lookup only | −1.99980 m/s | −0.608194 m | −0.391606 m/s |

The supplied contact isolates the response and gives 1.22 m of separation in that one-second experiment. [Scalar control experiment](scripts/replay_discrepancies.py)

**A three-ball tick reaches the same cutoff through collision detection.** A free ball starts with x velocity 16,384 m/s and approaches two fixed spheres in a 100 ms tick. All three have radius 1/64 m. The sweep finds the first contact at 0.09970000000005665 s and the second at 0.09980000000005665 s. The gap between the second time and the exact cutoff is again 211/2⁶⁶ seconds. The ordinary lookup reads record 0 and applies zero impulse at that contact. It then detects the second obstacle again. The control reads record 1, rebounds immediately and finishes after two contacts. The small radii and tuned initial positions make this a stress case. [Starting states and scalar replay](scripts/contact_replay.py) · [Lean trace](ExactBallparkProofs/Collision/ReboundCutoff.lean)

| Three-ball run | Contacts processed | Final x and vx | Binary64 words for x and vx |
| --- | ---: | --- | --- |
| Engine lookup | 3 | −0.994747 m, −3307.872551 m/s | `bfefd4f82e6ee100`, `c0a9d7bebefb4e15` |
| Exact-cutoff control | 2 | +0.330521 m, +3307.872551 m/s | `3fd52741d8968a00`, `40a9d7bebefb4e15` |

The ship’s final positions differ by **1.325268 m**. Lean and scalar Python agree on each contact time, selected record, applied impulse and both endpoint words in both runs. Pinned Destiny C++ agrees with scalar Python on the unmodified engine’s stored contact history and endpoint. [Word checks](scripts/arithmetic_checks.py) · [Compiled replay](scripts/contact_replay.py)

<a id="col-715-free-repeat"></a>

## COL-715 · an ignored repeat can hide the next free-ball contact

A stationary ball at x = 0 is touching another at x = 1 m. They have radius 0.5 m, unit mass and unit agility, with friction 1. The other ball starts at +4 m/s. The replay supplies a prior contact at time zero and an impulse of −12 m/s on the other ball at 0.5 s. The balls separate, then approach again. With both subintervals present, the engine ignores the first touch and tests the next chord from those same touching positions. It sees another repeat and returns a miss. Starting the actual selector at 0.5 s uses the correct positions and reports ball 2 at 0.708901778 s. The two searches use the same history and sphere solver. [Search-step proof](ExactBallparkProofs/Collision/SearchWindow.lean) · [Scalar and pinned C++ replay](scripts/contact_replay.py)

| Search | Time bits in Python and Destiny |
| --- | --- |
| Both free-ball subintervals | `bff0000000000000` |
| Final subinterval alone | `3fe6af52c7e6490c` |

<a id="time-715-extra-wait"></a>

## TIME-715 · one private packet can block a later stamp

Start with empty history and a private packet containing stamps 11 and 10 in that order. The merge sorts them and marks both as waiting. A bubble reply for 10 releases the first group. Advancing the clock from 12 to 100 still leaves 11 blocked. A later reply for 11 releases it. Exact Ballpark and the pinned Destiny Python functions produce the same trace. The fixture records dispatch, with the clock and action handlers supplied by the test. [Client proof](ExactBallparkProofs/Time/MixedHistory.lean) · [Replay](scripts/mixed_history_checks.py)

| Input | Observed stamps |
| --- | --- |
| Private packet | [10, 11] |
| Reply at 10, then advance the clock to 100 | [10] applied, [11] waiting |
| Reply at 11 | [10, 11] applied, queue empty |

A server-to-client replay reaches this case with ordinary mass-change commands. At stamp 3 the server records a private change for a cloaked pilot and a bubble change for a visible ship. At stamp 4 it records another private change for the pilot. Sending each tick leaves the client queue empty. Postponing the first send combines private stamps 3 and 4 with a bubble reply for stamp 3. The client reports one fatal desync. The session’s handler records it and returns, leaving stamp 4 waiting. This establishes one producer schedule that reaches the condition. [Server and client replay](scripts/mixed_history_checks.py) · [Two-stamp proof](ExactBallparkProofs/Time/BatchStamps.lean)

<a id="time-715-reset-order"></a>

## TIME-715 · packet order changes which commands survive a reset

After a SetState at 10, send one ordinary packet with events at 9 and 11. The client inspects the first raw stamp before it groups the events. With 11 first it admits both, queuing 9 ahead of 11. With 9 first it drops the whole packet. Each result below is the pending queue after the reset group has been consumed. A stale event’s eventual effect depends on the action and available snapshots. [Admission theorem](ExactBallparkProofs/Time/MixedHistory.lean) · [Pinned client replay](scripts/mixed_history_checks.py)

| Order in the incoming packet | Observed queue |
| --- | --- |
| Fresh event first | [9, 11] |
| Stale event first | [] |

<a id="col-712-shallow-crossing"></a>

## COL-712 · a shallow crossing of a capsule wall returns a miss

Use a capsule running from (0, 0, −1) to (0, 0, 1) m and give both it and the moving sphere radius ½ m. The sphere moves along x from 1 + 2⁻²⁰ to 1 − 2⁻²⁰ m. It first touches the straight side at half a tick and ends about 0.954 µm inside. Squared perpendicular travel is 2⁻³⁸ m². The starting position, ending position and squared travel are exactly representable. The binary64 cutoff for 10⁻¹⁰ has word `3ddb7cdfd9d7bdbb`. The squared travel falls below it, so the engine skips the quadratic. A control using 2⁻¹⁶ m on each side clears the cutoff and both engines return ½. The normal approach distance controls this branch even when the path is perpendicular to the capsule axis. [Capsule geometry and finite witness](ExactBallparkProofs/Collision/Capsule.lean) · [CPU replay](scripts/replay_discrepancies.py)

<a id="col-705-long-sweep"></a>

## COL-705 · a clear long sweep reports contact

Sweep from (−2²⁸, 2, 0) to (2²⁸, 2, 0) past a stationary centre at the origin with combined radius 1. At closest approach the centres are 2 metres apart, leaving 1 metre of clearance. Binary64 rounds away the small contribution to the constant coefficient, and the discriminant cancels to zero. Both engines return 0.5. Changing the offset to 5 metres gives a certified miss. [Real and binary64 witnesses](ExactBallparkProofs/Collision/FiniteSweep.lean) · [Miss certificate](ExactBallparkProofs/Collision/SweepCertificate.lean)

<a id="col-705-tick-end"></a>

## COL-705 · a sweep ending just outside reports contact at the tick boundary

Sweep along the x-axis from 4 to 1 + 2⁻⁵² past a stationary centre at the origin with combined radius 1. The endpoint is the next binary64 number above 1. The exact first root is 3/(3 − 2⁻⁵²), strictly greater than 1. The rounded displacement is −3, so the engine accepts the earlier root at exactly 1. The positive discriminant alone cannot detect this discrepancy. [Timing certificate and endpoint witness](ExactBallparkProofs/Collision/Timing.lean)

<a id="cmb-702-tracking"></a>

## CMB-702 · radial tracking can produce an infinite hit probability

Place two balls of radius one metre at (0, 0, 0) and (3, 4, 12) m. Give the second ball velocity (0.03, 0.04, 0.12) m/s and leave the first at rest. The decimal velocity points along the line of sight. On the binary64 inputs, subtracting the squared radial projection from the squared velocity gives a negative squared angular speed, word `bbd83c977ab2bedd`. The helper consequently reports NaN for angular speed. This part of the example needs no extreme weapon setting.

Set optimal range and optimal signature radius to 10³⁰ and falloff, tracking and signature radius to one. The surface distance is 11 m, inside optimal range. The negative tracking term changes the sign of the exponent, yielding binary64 word `484e2a40f9945fd6`. The resulting hit probability is +∞. The exact geometric probability is at most one. A caller comparing a uniform draw with this result would always pass that comparison. The shot-resolution behavior of a game client was outside the replay. Lean certifies the two intermediate words. The scalar and pinned C++ replays capture those words at the square-root and power calls. The PyTorch CPU lane also reproduces the final outputs. This fixes the inputs of a regression already present in the engine suite. [Finite tracking proof](ExactBallparkProofs/Combat/Execution.lean) · [CPU regression](../tests/test_movement_vec_accuracy.py) · [Replay](scripts/replay_discrepancies.py)

An offline Lean check compares the two rounded squared speeds before subtraction. It rejects this replay and certifies a nonnegative square-root input for both the perpendicular and exact radial controls. The engine still returns an infinite hit probability on the failing replay. [Input guard](ExactBallparkProofs/Combat/InputMargin.lean) · [CPU controls](scripts/law_trace_checks.py)

<a id="col-705-tangency"></a>

## COL-705 · an exact grazing contact returns a miss

Let the combined radius R be 2²⁸ metres. One centre stays at the origin while the other travels from (−3, R, 0) to (3, R, 0). The exact squared separation minus R² is (6t − 3)². It vanishes only at t = ½. The scalar solver first computes R² + 9. At this magnitude adjacent binary64 values are 16 apart, so it rounds to R² + 16. Subtracting R² gives a constant coefficient of 16. The other coefficients are 36 and −36. The discriminant becomes −1008 and the solver returns −1. Lean checks the exact coefficients, first-contact time and rounded result from the encoded endpoints. [Tangency proof](ExactBallparkProofs/Collision/Timing.lean) · [CPU replay](scripts/replay_discrepancies.py)

<a id="col-718-impact-limit"></a>

## COL-718 · rounding raises the impact velocity threshold

Set the binary32 maximum speed to 4097 and the incoming velocity change to (20485 + 2⁻¹³, 0, 0). The exact squared threshold is 25 × 4097² = 419635225. The first product is exactly 102425 in binary32. Only the second multiplication rounds, producing 419635232 before division by the binary64 squared magnitude. The input magnitude falls between the ideal and rounded thresholds after taking square roots, so the scalar park stores the entire change. The x coordinate exceeds five times the configured maximum speed by 0.0001220703125. This measures the impact velocity change applied to a ball initially at rest. [Magnitude bound](ExactBallparkProofs/Collision/Impact.lean) · [Binary64 output proof](ExactBallparkProofs/Collision/FiniteImpact.lean)

For an ordinary activation of the same limiter, the head-on replay uses a maximum speed of 0.5 m/s on the left ship. The proposed −4 m/s impulse becomes −1.5625 m/s while the other ship receives +4 m/s. Both engines agree on these words. Lean proves the spheres stay clear for the rest of the tick even though they finish moving closer together. [One-sided limiter control](README.md#one-impact-through-the-end-of-a-tick)

A second control isolates the effect of giving each ship its own limit. Two 1 kg ships enter contact at −100 and −90 m/s. Their raw changes are +10 and −10 m/s. Limits of 1 and 2 m/s on the ships’ maximum-speed fields yield applied changes of +2.5 and −10 m/s. The pair loses 7.5 kg·m/s of momentum and gains 703.125 J of kinetic energy in this frame. Both threshold numerators are exact. This behaviour follows from the documented per-body limiter even in real arithmetic. [Pair proof](ExactBallparkProofs/Collision/LimitedPair.lean) · [Scalar replay](scripts/boundary_checks.py) · [Destiny replay](scripts/contact_replay.py)

<a id="mov-703-position"></a>

## MOV-703 · position rounding in the near-zero-friction branch

**What the metre measures.** Start at the origin with incoming x velocity 1 m/s, prescribe x acceleration 2²⁰ m/s² for one second, set effective mass `m = 2⁴⁰` and friction `k = 1`. The cached decay word is `3fefffffffffe000`. This implies terminal speed `m·a/k = 2⁶⁰`, about 1.15 × 10¹⁸ m/s. Destiny substitutes the first-order Taylor expansion of the decay into its position update. The branch expression simplifies over the reals to `p + v·t`, giving **1 m**. In binary64, adding 2⁻⁴⁰ to 2²⁰ rounds back to 2²⁰ and the stored position is **+0 m**. Lean checks the branch identity and the encoded stress input. [Real branch identity](ExactBallparkProofs/Movement/SpecialBranch.lean) · [Binary64 witness](ExactBallparkProofs/Movement/Correspondence.lean) · [Destiny’s Taylor substitution](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L763-L773)

**What the regular drag law predicts.** With the same constant acceleration and drag, the regular trajectory advances approximately **524,289 m**. Its first term beyond `p + v·t` is `½(a − kv/m)t²`. For this witness the first two alternating acceleration terms bracket the regular result between 524,288.999999 m and 524,289 m. Evaluating `p + v·t` directly recovers the metre lost to rounding of the branch expression. It retains the truncation. [Regular trajectory](ExactBallparkProofs/Movement/Integrator.lean) · [Rational bracket](scripts/replay_discrepancies.py)

**What changes at GOTO thrust.** At full untapered GOTO thrust the ideal terminal speed is `speed_fraction × max_vel`. With default friction 10⁶ and a one-second tick, effective mass must exceed 10¹⁶ to take this branch. Two CPU replays use mass 1.1 × 10¹⁶, agility and speed fraction 1, maximum speed 10,000 m/s and incoming speed 100 m/s. The target is 65,536 m ahead in both runs. The exact branch adds 100 m. From x = 0 the stored position is short by **0.583 pm**. From x = 149,597,870,700 m (one AU) it is short by **30.518 µm**, one binary64 spacing there. Both reference endpoints `p + 100` are representable. Scalar Python and compiled Destiny agree on both output words. [Ballpark.__init__](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/ref.py#L80-L84) · [Ballpark.goto_thrust](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/ref.py#L166-L175) · [CPU replay](scripts/replay_discrepancies.py)

**How the error scales.** For one coordinate, write `u = 2⁻⁵³` and `V = ma/k`. For positive mass and friction, with round-to-nearest binary64 operations whose exact intermediate results are normal or zero and whose rounded results are finite, the absolute error against `p + v·t` is bounded by `u(6|V·t| + 13|v·t| + 5|p|)`. This follows by expanding the actual expression and bounding the two acceleration terms before their common factors. The replay checks the coefficient inequalities with exact rational arithmetic. At the origin with the stress case’s incoming speed and duration, losing a metre requires terminal speed above **1.5 × 10¹⁵ m/s** under those assumptions. The coordinate term explains why a picometre estimate at the origin cannot describe every position in the game. [Ballpark.integrate](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/ref.py#L182-L188) · [Roundoff factors](scripts/replay_discrepancies.py)

<details>
<summary>Derivation of the one-coordinate rounding bound</summary>

Let `L = 1 − u` and `H = 1 + u`. Each rounded operation contributes a multiplicative factor in [L, H]. Expanding the source expression gives two contributions of opposite sign with magnitude `|V·t|`. One has three factors before the terms meet and the other has two. They then share five factors and divide by the rounding factor of `k²`. Their residual coefficient is at most `(H⁵/L) max(H³ − L², H² − L³) < 6u`.

The coefficient on `v·t` has eleven factors divided by one, so it lies in [L¹¹/H, H¹¹/L] and differs from 1 by less than 13u. In the coefficient on `p`, the same stored `k²` occurs in numerator and denominator and cancels. The remaining four factors give [L⁴, H⁴], within 5u of 1. The triangle inequality yields the absolute bound above. [Source expression](ExactBallparkProofs/Movement/Reference.lean) · [Exact rational checks](scripts/replay_discrepancies.py)

</details>

Lean now derives this bound from nearest-even rounding at each of the 16 arithmetic operations, including reuse of the squared friction. The 53-bit model applies on IEEE executions whose operations share those rounding grids. [Rounding proof](ExactBallparkProofs/Movement/Roundoff.lean)

<a id="trv-709-short-cruise"></a>

## TRV-709 · a zero-length cruise can round either way

At warp factor 1000 the ship cannot reach its requested top speed on either of these trips. Lean proves that the reduced peak gives exactly zero cruise time throughout the interval from 100,000 to 100,001 metres. In the source expression the engine subtracts the acceleration and deceleration terms and then adds a final term. Rounding those operations produces opposite signs at the two ends of the interval. Scalar Python and the pinned C++ function agree on both words. [Real and binary64 proofs](ExactBallparkProofs/Warp/Boundaries.lean)

In a scalar check at factor 1000 across every whole metre from 100,000 to 100,999, cruise duration rounded positive 677 times and negative 323 times. This is a fixed grid of nearby distances. To see what a ship would do, the replay also samples the actual scalar warp-distance function once per second for the two logged flights. Each begins in active warp on the x axis with maximum ordinary speed 200 m/s. A control run sets only the cruise duration and cruise distance to zero. All four runs left warp at second 27 with the same sequence of phases. The largest position difference from the corresponding control was 1.46 × 10⁻¹¹ m at 100,000 m and 7.28 × 10⁻¹² m at 100,001 m. These comparisons cover the active-warp function from entry through dropout. [CPU replay and control](scripts/replay_discrepancies.py)

## Formation resize documentation

**MOV-726 · Formation resize documentation.** At Exact Ballpark commit `0986532` the scalar docstring said a shrink never freed the removed slots. The working copy corrects that wording. In Destiny commit `114e89fa`, the comment at `Ballpark.cpp:6719` still says the shrink clears higher slots. Its loop starts at the old length and stops below the smaller new length, so it runs zero times. Stopping a follower first releases its occupied slot on the master. The client keeps the bit, and an unclaimed reservation survives on either side. The evidence here is a scalar replay and inspection of the two sources. [FullBallpark.set_ball_formation](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/full.py#L1863-L1866) · [FullBallpark._stop](https://github.com/Eidosoma/exact-ballpark/blob/098653233eb0f9910a3f02564e36b0641cbd4e0f/src/exact_ballpark/full.py#L1850-L1857) · [Working source hashes](claims.json) · [Destiny shrink comment and loop](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L6714-L6724) · [Destiny STOP](https://github.com/carbonengine/destiny/blob/114e89fa584f0e1a81eb0a679aabf41af8870072/src/Ballpark.cpp#L4629-L4633)

In a scalar replay, a leader had all four slots reserved, formation followers in slots 0 and 3, and no ships in slots 1 or 2. Shrinking to one slot stopped the followers in slot 3. After expanding back to four, the next reservation returned slot 3 on the master. On the client it returned −1 because all four bits were still set. This is a documentation discrepancy with direct implications for predicting slot availability. The Lean model matched both scalar outcomes. [Formation proofs](ExactBallparkProofs/Movement/Formation.lean) · [CPU comparison](scripts/formation_checks.py)

<a id="col-705-turning-path"></a>

## COL-705 model comparison · a turning ship can cross an obstacle between endpoints

COL-705 explicitly uses straight paths for time of impact. A tight turn can carry the continuous drag trajectory inside an obstacle even when the straight sweep passes clear. This is a consequence of the documented game rule. Its relevance to an encounter depends on the size of the turn relative to the available clearance. [Engine rule catalogue](../docs/rules.md) · [Trajectory and clearance proof](ExactBallparkProofs/Collision/TurningFlight.lean)

One ship starts at the origin at (10, 0, 0) m/s and receives a GOTO to (0, 100, 0) m. Effective mass, friction, agility and speed fraction are all one, maximum speed is 10 m/s, and the tick lasts one second. The applied acceleration is (0, 10, 0) m/s². Both spheres have radius 0.25 m. Scalar Python and the compiled Destiny functions agree on the cache, thrust, half-tick position, endpoint and both miss words. [Scalar fixture](scripts/curve_checks.py) · [Pinned C++ replay](scripts/contact_replay.py)

| Obstacle centre | Straight-sweep result | Continuous result |
| --- | --- | --- |
| (4, 1, 0) m | The minimum centre distance along the stored chord is **1.147695 m**. The engine returns `bff0000000000000` (miss). | At half a second the scalar partial step gives **0.092357 m** centre distance. Lean proves the real distance is below 0.1 m, inside the 0.5 m contact radius. |
| (4, −4, 0) m | The engine returns `bff0000000000000` (miss). | Lean proves at least **3.094137999999 m** between the surfaces throughout the tick, including endpoint rounding. |

Lean now encloses the **first** touch at (4, 1, 0) m in **[0.435661, 0.435662] s**. The certificate covers every earlier time. An independent 60-digit decimal calculation puts the crossing at **0.435661309892 s**. Scalar Python and pinned Destiny agree on all three partial-step position words at the binary64 times nearest the two decimal endpoints. For the lower obstacle, the same initial-state calculation proves a **3.5 m** surface gap throughout. Giving the obstacle at (4, 1, 0) m an initial velocity of (−0.5, −0.25, 0) m/s, a downward applied acceleration of 1 m/s² and an effective mass of two makes both bodies turn. Their first contact is in **[0.404889, 0.404890] s**. Both engines again return a chord miss. [First-contact certificates](ExactBallparkProofs/Collision/ContactExamples.lean) · [Run the checker](README.md#first-contact-along-a-turn)

The full-tick x and y endpoint words are `401948ea1e23ea7d` and `400d6e2bc3b82b06`. For the clear control, Lean proves that 15 m/s² bounds the ship’s net acceleration and that the combined three-coordinate endpoint error is at most 10⁻¹² m. A shorter sweep reduces the curvature allowance quadratically when the acceleration bound stays fixed. [Run the offline checker](README.md#clearance-throughout-a-tick)

## Reproduce on a CPU

From the repository root, use Python 3.13+ and a local Destiny clone containing the pinned commit. The probe reads the source blobs directly from that commit, compiles the upstream accuracy query, integrator, GOTO thrust and time-factor setup, sphere sweep, capsule check, quadratic, history lookup and position functions, warp constants and the velocity-limiting block of `Ballpark::AddImpact` with the original vector operators, runs the scalar Python functions on the same inputs and compares the returned binary64 words with this log. The client replay runs the pinned merge, admission and pre-tick Python functions on the same packet traces. The contact replay creates a temporary checkout of the pin, builds the existing space harness and runs one full four-ball tick through Destiny’s iterative collision loop.

```bash
python3 -B lean/scripts/replay_discrepancies.py /path/to/destiny
```

The Lean gate checks that each local Python range falls inside its named function or method. In a source archive it checks Python against the registered file hashes. Each Destiny link must point to the registered file and include the line for its operation. With a Destiny clone, the replay also checks the range against the file length and reads the expected source text at that line in the pinned commit. [Source citation checks](scripts/discrepancy_sources.py)

Run `python3 lean/scripts/check.py` for the Lean proofs, scalar regressions and consistency of the severity tables. For each new entry, record the exact input words or representable values, the mathematical result, the observed engine bits, and the revision and platform of each replay. [Replay source](scripts/replay_discrepancies.py) · [Proof register](CLAIMS.md)
