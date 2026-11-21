# Lean claim register

278 kernel-checked theorems have claims linked to 105 of the 105 catalogue rules. A link covers the scope stated below.

Engine revision `098653233eb0f9910a3f02564e36b0641cbd4e0f`. The register identifies manually transcribed models and source-extracted definitions. The Python-to-Lean extractors are trusted. The Lean kernel checks the resulting theorems. The full.py digest includes a docstring correction to the pinned formation source.

The engine records device qualification on NVIDIA GH200, aarch64 and glibc 2.34. This proof gate runs on the host CPU. [Platform](../docs/platform.md) · [Evidence](../docs/exactness.md)

## Catalogue coverage

| Family | Rules | With a scoped claim | Awaiting a first claim |
| --- | ---: | ---: | ---: |
| MOV | 41 | 41 | 0 |
| TRV | 16 | 16 | 0 |
| COL | 19 | 19 | 0 |
| SEN | 12 | 12 | 0 |
| TIME | 15 | 15 | 0 |
| CMB | 2 | 2 | 0 |

The rule descriptions, upstream locations and per-rule test files are in the [engine catalogue](../docs/rules.md).

| Rule | Engine law | Scope and supporting theorems |
| --- | --- | --- |
| MOV-701 | Motion is viscous with friction k = 1e6 and effective mass = mass x agility | [refresh_positive](#claim-1), [source_regular, velocity_hasDerivAt](#claim-3), [decay_bounds, velocity_contraction, speed_envelope, velocity_tends_to_terminal](#claim-5), [extracted_integrator](#claim-49) |
| MOV-702 | Closed-form exponential integration of position and velocity | [source_regular, flow_zero, decay_hasDerivAt, position_hasDerivAt, velocity_hasDerivAt, flow_compose](#claim-3), [integrate_zero, integrate_regular, integrate_velocity](#claim-4), [decay_bounds, velocity_contraction, speed_envelope, velocity_tends_to_terminal](#claim-5), [extracted_integrator](#claim-49), [full_tick_velocity_bits](#claim-50), [zero_duration_bits](#claim-52) |
| MOV-703 | Near-zero friction switches to a branch where acceleration does not move the ball | [special_position_eq, integrate_special](#claim-4), [extracted_integrator](#claim-49), [special_rounding_witness](#claim-50), [special_position_error](#claim-105) |
| MOV-704 | Each ball caches its one-tick decay factor exp(-k dt / (mass agility)) | [refresh_positive, stepFactor_eq](#claim-1), [addBall_valid, update_valid, history_valid, velocity_after_updates](#claim-2), [integrate_regular, integrate_velocity](#claim-4), [extracted_integrator](#claim-49), [full_tick_velocity_bits](#claim-50) |
| MOV-705 | Maximum thrust is k x speedFraction x maxVel / (mass x agility) | [goto_terminal_bound, goto_speed_bound](#claim-6) |
| MOV-706 | Goto thrust is full thrust toward the target with an (L/D)^4 homing taper | [taper_bounds, goto_terminal_bound, goto_speed_bound](#claim-6) |
| MOV-707 | Goto has no braking term, so a ship overshoots its goto point | [goto_overshoot](#claim-26) |
| MOV-708 | GotoPoint and GotoDirection set a goto point, cancel follows and default speed | [goto_cancels](#claim-28) |
| MOV-709 | Stop cancels follow-type modes and leaves the ball coasting to rest | [stop_clears](#claim-28) |
| MOV-710 | STOP mode damps the vertical velocity by a hard-coded 0.93/1.07 per tick | [stop_speed_sq](#claim-7) |
| MOV-711 | AddBall clamps physical properties and starts every ball in STOP | [addBall_valid](#claim-2), [clamp_negative, clamp_other, clamp_nan, clamp_negative_zero, signed_zeros_distinct](#claim-8) |
| MOV-712 | Radius, max speed, agility, speed fraction and follow range are 32-bit floats | [agility_accepts_iff, agility_half_subnormal, agility_above_half_subnormal](#claim-11) |
| MOV-713 | Property setters validate their inputs | [agility_accepts_iff, agility_half_subnormal, agility_above_half_subnormal](#claim-11) |
| MOV-714 | Follow (keep at range) aims at the point at the desired surface range | [unit_norm, follow_range](#claim-27) |
| MOV-715 | Follow and orbit need a live, uncloaked target in the same bubble | [follow_iff](#claim-28) |
| MOV-716 | Orbit uses the old-style law unless the global useNewOrbit setting is on | [orbit_switch](#claim-32) |
| MOV-717 | Old-style orbit plane precesses with tick count and ball ID | [orbit_id_period](#claim-27) |
| MOV-718 | Old-style orbit blends radial and tangent thrust with a 40000 m^2 Gaussian | [orbit_weight_bounds](#claim-27) |
| MOV-719 | New-style orbit plane follows the orbiter's velocity around the predicted target | [predicted_direction](#claim-33) |
| MOV-720 | New-style orbit steers along a logarithmic spiral toward the orbit radius | [spiral_radius](#claim-27) |
| MOV-721 | GotoThrustFollow blends reach-the-point and match-the-velocity accelerations | [solve_guidance](#claim-27) |
| MOV-722 | OrbitThrust brakes toward a point and switches off below 10% of max acceleration | [orbit_brake_cap](#claim-27) |
| MOV-723 | Missiles fly straight for 800 ms, then pursue a lead point at full thrust | [pursuit_center](#claim-34) |
| MOV-724 | MissileFollow aims a missile at the target centre at full speed fraction | [pursuit_center](#claim-34), [pursuit_offset_witness](#claim-55) |
| MOV-725 | A launched missile starts at the launcher with launcher velocity plus 150 m/s | [launch_speeds](#claim-34) |
| MOV-726 | Formation followers aim at the leader's rotated slot offset | [formation_distance](#claim-35), [reserve_first, reserve_failure, reserve_release, release_empty](#claim-87), [ignored_iff, refused_stops, joined_link](#claim-88), [stopped_iff, resize_bit, regrow_skips_reserved](#claim-89) |
| MOV-727 | The agility turn-angle cap is disabled, so thrust can point anywhere | [cap_preserves](#claim-32) |
| MOV-728 | Optional dynamical orientation steers thrust along the ship's heading | [orientation_bound](#claim-35), [normalized_heading](#claim-100) |
| MOV-729 | Without dynamical orientation, heading is a smoothed function of velocity | [smooth_error](#claim-35) |
| MOV-730 | Followers are released when their target warps, cloaks or is removed | [fly_on_iff](#claim-28) |
| MOV-731 | Followers of a ball in another bubble are stopped at the next pre-tick | [remote_iff](#claim-40), [remote_pending_stops](#claim-107) |
| MOV-732 | Only free balls move; fixing a ball stops it and spawns its collision parts | [fixed_capsule](#claim-47) |
| MOV-733 | Ball flags free, massive, global, interactive and space junk | [flags_roundtrip](#claim-48) |
| MOV-734 | Thirteen ball modes; six compute thrust, STOP damps, the rest are inert | [thrust_modes](#claim-32) |
| MOV-735 | Mode changes raise DoModeChange; leaving warp raises OnExitWarp | [exit_iff, exiting_order](#claim-28) |
| MOV-736 | A troll ball turns rigid after a delay | [troll_due](#claim-36) |
| MOV-737 | Force fields let balls with a matching harmonic, corporation or alliance through | [field_iff](#claim-47), [configured_field_filter](#claim-108) |
| MOV-738 | Destiny's exp is the C library's; exact on tensors and the GPU for glibc 2.34 on aarch64 | [exp_tiny_iff, exp_large_iff](#claim-22), [two_sum_exact](#claim-103), [split_reconstructs](#claim-112), [dekker_product_exact](#claim-113), [add_odd_grid](#claim-114), [fma_single_round](#claim-115), [table_address, scale_reconstruction, special_rescaling](#claim-120), [reduction_index_bound, tiny_correct_round](#claim-121) |
| MOV-739 | Destiny's float functions are the C library's; exact for glibc 2.34 on aarch64 | [powf_signaling_iff](#claim-23) |
| MOV-740 | Destiny's double sin, cos and log are the C library's; exact for glibc 2.34 on aarch64 | [log_near_one_iff](#claim-24) |
| MOV-741 | Destiny's double atan2 and asin are the C library's; exact for glibc 2.34 on aarch64 | [atan2_sign](#claim-25) |
| TRV-701 | WarpTo to a point closer than 100 km becomes a goto | [short_iff](#claim-12) |
| TRV-702 | Repeating a warp command to the same point does nothing | [repeated_warp](#claim-12) |
| TRV-703 | WarpTo stops the ship and enters the WARP alignment state | [long_starts_alignment](#claim-12) |
| TRV-704 | Warp factor is the top warp speed in milli-AU per second | [rate_bounds](#claim-14), [source_constants](#claim-31) |
| TRV-705 | While aligning for warp the ship thrusts toward the destination like a goto | [waiting_ticks](#claim-37) |
| TRV-706 | Warp starts at 75% of max speed within about 8 degrees, or after 180 ticks | [alignment_after_waits](#claim-13) |
| TRV-707 | Warp activation drops collisions and offsets the exit point by minRange | [exit_distance](#claim-13) |
| TRV-708 | Warp speed, acceleration and deceleration rates scale with the warp factor | [rate_bounds](#claim-14), [source_constants](#claim-31) |
| TRV-709 | Short warps lower the top speed so the cruise phase has zero length | [short_cruise_zero](#claim-14), [source_constants](#claim-31), [positive_cruise_iff](#claim-72), [short_cruise_words](#claim-74), [cruise_exit_margin](#claim-101) |
| TRV-710 | Warp phase durations and distances | [cruise_nonnegative, phase_distances_sum](#claim-14), [source_constants](#claim-31) |
| TRV-711 | Warp proper follows an analytic exponential-then-cruise path | [phase_joins](#claim-15), [acceleration_sample](#claim-122), [placement_ray](#claim-124) |
| TRV-712 | Warp deceleration decays exponentially | [phase_decel_iff, decel_velocity](#claim-15), [deceleration_extension](#claim-123), [placement_ray](#claim-124) |
| TRV-713 | Ships drop out of warp below min(maxVel/2, 100 m/s) | [dropout_iff](#claim-15), [dropout_time_iff](#claim-73), [cruise_exit_margin](#claim-101) |
| TRV-714 | The exit tick integrates one extra step from the drop-out point | [exit_regular, exit_special](#claim-37) |
| TRV-715 | EntityWarpIn starts a warp already in progress | [entity_elapsed](#claim-37) |
| TRV-716 | The ballpark warpSpeed attribute is not used by the warp model | [source_constants](#claim-31) |
| COL-701 | Two collision methods; the per-ball gradient method is the default | [gradient_iff](#claim-43) |
| COL-702 | Only free, massive, live balls respond; non-massive balls pass through | [gradient_iff](#claim-43) |
| COL-703 | Collision candidates exclude cloaked, non-massive, missile and other-bubble balls | [missile_target, ordinary_candidate](#claim-43), [related_iff](#claim-95), [first_visit_once](#claim-96), [ordered_candidates](#claim-97) |
| COL-704 | Missiles ignore owners; non-junk balls ignore space junk | [owner_junk_skip](#claim-43) |
| COL-705 | Time of impact assumes straight paths over the tick | [gap_quadratic, contact_correct](#claim-9), [long_chord_clear, long_chord_real_miss, long_chord_rounded_hit](#claim-57), [finite_negative_iff, discriminant_error, certified_sign, certified_miss](#claim-58), [decoded_gap, certified_chord_miss, horizontal_margin_examples, positive_outside_tick](#claim-59), [entry_comparisons, timing_sound, tick_boundary_examples, late_root_rounded_hit](#claim-60), [tangent_contact_iff, tangent_rounded_miss, tangent_tick_boundaries](#claim-61) |
| COL-706 | Collision inertia is mass x agility, fixed balls 1e34, velocity exchange base mass | [elastic_exchange](#claim-17) |
| COL-707 | Hitting a fixed ball reflects the normal velocity elastically | [reflection](#claim-17), [scaled_reflection](#claim-51) |
| COL-708 | Two free balls exchange normal velocity as a 1-D elastic collision | [elastic_exchange](#claim-17) |
| COL-709 | Overlapping balls are pushed apart by the overlap plus 1 m, split by inertia | [overlap_split](#claim-18), [overlap_clearance, overlap_inertia_center](#claim-64) |
| COL-710 | Free-free overlaps are re-tested from separated positions, lighter ball boosted | [lighter_boost](#claim-45), [overlap_clearance, overlap_inertia_center](#claim-64) |
| COL-711 | The applied collision response is 0.85 of the computed acceleration | [response_tie](#claim-45), [scaled_reflection](#claim-51) |
| COL-712 | Static capsules and boxes reflect or eject balls with the same 0.85 response | [excluded_chord](#claim-10), [scaled_reflection, static_branches](#claim-51), [support_geometry, crossing_iff, parallel_sweep, face_membership](#claim-71), [axis_distance, bounding_rejection, cylinder_quadratic, first_barrel_contact](#claim-116), [shallow_crossing_missed](#claim-117) |
| COL-713 | Fixed balls carry axis-aligned miniballs, minicapsules and miniboxes | [fixed_capsule](#claim-47) |
| COL-714 | Iterative collisions re-examine colliding balls for up to 20 passes | [iteration_cap](#claim-44), [remove_first_matching_impact, remove_reset_at_zero, finish_window](#claim-68) |
| COL-715 | Iterative search order is static, then fixed, then higher-ID free balls | [unique_pair](#claim-45), [ignore_recent_colliders, index_prefix, position_from_prefix](#claim-69), [rounded_history_lookup](#claim-70), [full_tick_first_min, partial_tick_selection](#claim-90), [static_priority](#claim-91), [free_subintervals, ignored_segment](#claim-92), [four_ball_endpoint, returning_ball_words](#claim-102), [compensated_history_cutoff](#claim-104) |
| COL-716 | Iterative impacts on static objects are perfectly elastic impulses | [reflection](#claim-17), [static_velocity_components, start_contact_acceleration](#claim-62) |
| COL-717 | Iterative impacts between free balls are elastic with base masses | [iterative_exchange](#claim-17), [start_contact_acceleration](#claim-62), [pair_separates, pair_preserves_tangent, pair_momentum, pair_kinetic_energy](#claim-63) |
| COL-718 | Iterative velocity changes above 5 x maxVel are scaled by the squared ratio | [limitImpact_norm](#claim-18), [limited_speed_bound](#claim-65), [rounded_limiter_exceeds_bound](#claim-66), [insert_cutoff_and_retry, insert_after_predecessor, insert_before_all](#claim-67), [promoted_limiter_exact](#claim-106) |
| COL-719 | Collision events are raised only for missiles; orphaned missiles hit themselves | [orphan_with_owner](#claim-44), [dispatch_moving, dispatch_silent](#claim-54) |
| SEN-701 | Proximity sensors check every period seconds, default 2.0 s | [sensor_schedule](#claim-20), [first_due_poll](#claim-77) |
| SEN-702 | A ball is inside a sensor when centre distance <= range + both radii | [sensor_inside, crossing, repeated_sample](#claim-20), [vanished_target, present_target](#claim-78) |
| SEN-703 | Target tracking fires when the centre distance crosses range + own radius | [tracking_boundary, crossing, repeated_sample](#claim-20) |
| SEN-704 | Notification range reports arrival at the goto point or follow distance | [follow_band_iff, crossing](#claim-20) |
| SEN-705 | Space is partitioned into 8 box levels from 7,864,320 m down to 480 m | [finest_width](#claim-21) |
| SEN-706 | A ball is filed at the finest level whose box is at least 3x its inflated radius | [finest_level](#claim-21), [radius_covers_trajectory](#claim-93), [face_edge_corner](#claim-94) |
| SEN-707 | Bubbles are clusters of top-level boxes joined through their 26 neighbours | [neighbor_offsets, neighbor_iff](#claim-46), [bubble_component, corner_neighbors](#claim-56) |
| SEN-708 | Balls interact only within their bubble | [ordinary_candidate, missile_target](#claim-43) |
| SEN-709 | Line of sight is blocked by the first massive, uncloaked sphere on the segment | [visibility_entry, first_occluder](#claim-46), [ray_first_entry](#claim-111) |
| SEN-710 | Range queries count balls whose centre is within range plus their own radius | [range_abs](#claim-46), [ordered_scan, query_by_id](#claim-125) |
| SEN-711 | Cloaking makes a ball non-massive and drops its followers | [cloak_effect](#claim-47), [warp_and_sensor, erase_and_reenter](#claim-79) |
| SEN-712 | Clients receive each tick's events only for balls in their bubble | [recipient_split](#claim-42), [bubble_destination, character_destination](#claim-80), [pilot_add_iff, pilot_remove_iff, pilot_disjoint](#claim-81), [single_recipient, dual_recipient](#claim-82), [character_wait, client_trace](#claim-83), [private_bubble_history](#claim-110) |
| TIME-701 | A ballpark tick is 1000 ms by default and dt = tickInterval/1000 s | [default_tick_seconds](#claim-38) |
| TIME-702 | OnTick evolves once per whole tick interval elapsed and carries the remainder | [rounds_remainder, early, ordinary_tick](#claim-16), [clock_window, next_client_deadline](#claim-75) |
| TIME-703 | The server drops a backlog of more than 5 ticks and evolves only one | [master_resync](#claim-16), [master_boundary](#claim-76) |
| TIME-704 | The first OnTick evolves exactly once without tick events | [first_due](#claim-16) |
| TIME-705 | Each tick runs DoPreTick, Evolve, DoPostTick; dead balls are removed before | [rounds_trace, ordinary_tick](#claim-16) |
| TIME-706 | Evolve computes all thrusts, then collisions and integration, then commits | [phase_ids](#claim-39), [schedule_monotone](#claim-53) |
| TIME-707 | Free balls are always processed in ascending ID order | [phase_ordered](#claim-39) |
| TIME-708 | Clients display ball positions two ticks behind their clock | [render_two_ticks](#claim-16) |
| TIME-709 | Server movement commands are queued and applied at the next pre-tick | [last_enqueued, distinct_actions_kept](#claim-29) |
| TIME-710 | An action identical to the last queued action is dropped | [duplicate_idempotent](#claim-29), [remote_pending_stops](#claim-107) |
| TIME-711 | Ball removal through Actions is immediate, not queued | [removal_calls](#claim-41) |
| TIME-712 | Pre-tick order is follower cleanup, stamp, flush, apply, bubbles, cloaks | [pretick_calls](#claim-41), [remote_pending_stops](#claim-107) |
| TIME-713 | Removed balls linger as moribund and are cleaned up between ticks | [cleanup_budget](#claim-40), [removal_prefix_masks](#claim-109) |
| TIME-714 | Balls in dead bubbles are not evolved on the server | [dead_iff](#claim-40) |
| TIME-715 | A client applies each server update at its stamp, waiting, running ahead or rewinding to it | [sync_reaches_stamp, wait_iff](#claim-30), [eleventh_store, stores_bounded, post_deadline](#claim-84), [rebase_contents](#claim-85), [rewind_stream, sync_failure](#claim-86), [private_bubble_history](#claim-110), [merged_stamps, merged_payload](#claim-118), [waiting_tail, drain_frontier, reset_packet_order](#claim-119) |
| CMB-701 | Angular velocity is transversal speed over centre distance | [angular_agrees](#claim-19), [rotating_observer](#claim-99) |
| CMB-702 | Destiny's accuracy helper is 2 to the power of tracking and falloff terms | [accuracy_bounds, optimal_stationary](#claim-19), [accuracy_overflow_input, accuracy_early_lanes](#claim-98) |

<a id="claim-1"></a>

## Effective mass and the cached tick factor

**Real arithmetic.** The cache refresh theorem assumes positive mass × agility. A full tick requires the stored decay for that interval. A partial tick recomputes it. The accuracy of libm exp remains open.

Engine symbols `Ballpark._refresh_time_factor, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Integrator.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.refresh_positive` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.stepFactor_eq` | propext, Classical.choice, Quot.sound |

<a id="claim-2"></a>

## Cache maintenance through mass and agility setters

**Real arithmetic.** One ball with fixed friction and tick interval. Agility inputs are the finite values after conversion. Construction and any finite history of these two setters preserve the cache, including rejected changes and zero effective mass. Positive final effective mass and nonzero friction suffice for the velocity formula. IEEE rounding and exceptional values remain open.

Engine symbols `Ballpark.add_ball, Ballpark.set_mass, Ballpark.set_agility, Ballpark._refresh_time_factor, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-704 passing (compiled code), MOV-711 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Cache.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.CacheState.addBall_valid` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.CacheState.update_valid` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.CacheState.history_valid` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.CacheState.velocity_after_updates` | propext, Classical.choice, Quot.sound |

<a id="claim-3"></a>

## Regular constant-force trajectory

**Real arithmetic.** One Cartesian component, constant acceleration and coefficients. The source identity requires nonzero friction. The drag ODE assumes nonzero effective mass and friction. Composition concerns the regular mathematical flow.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Integrator.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.source_regular` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.flow_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.decay_hasDerivAt` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.position_hasDerivAt` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.velocity_hasDerivAt` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.flow_compose` | propext, Classical.choice, Quot.sound |

<a id="claim-4"></a>

## Dispatcher and near-zero-friction position

**Real arithmetic.** Zero time returns the input unconditionally. The ordinary branch includes equality at the threshold. At a nonzero special step with nonzero effective mass and friction, position is p + v t and velocity uses the selected factor. Rounded cancellation is outside this claim.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/SpecialBranch.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.integrate_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.integrate_regular` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.special_position_eq` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.integrate_special` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.integrate_velocity` | propext, Classical.choice, Quot.sound |

<a id="claim-5"></a>

## Three-dimensional velocity bounds

**Real arithmetic.** Euclidean speed under constant acceleration. Positive effective mass and friction and nonnegative time bound decay by one. Both initial and terminal speeds must lie inside the claimed envelope.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Bounds.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.decay_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.velocity_contraction` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.speed_envelope` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.velocity_tends_to_terminal` | propext, Classical.choice, Quot.sound |

<a id="claim-6"></a>

## GOTO taper and thrust

**Real arithmetic.** Nonnegative tick and requested speed (speed fraction × max velocity), positive effective mass and friction. The taper uses squared Euclidean distance. Normalizing the offset, including zero, and applying drag preserves the requested speed envelope if the initial speed is inside it.

Engine symbols `_normalized, Ballpark.goto_thrust, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-705 passing (compiled code), MOV-706 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Modes.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.taper_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.goto_terminal_bound` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.goto_speed_bound` | propext, Classical.choice, Quot.sound |

<a id="claim-7"></a>

## STOP vertical damping and ordinary drag

**Real arithmetic.** Three velocity components, zero mode acceleration and zero collision impulse. The two encoded binary64 constants are interpreted exactly over the reals. The bound is on squared Euclidean speed after damping and drag.

Engine symbols `STOP_DAMP, Ballpark.evolve, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-710 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Modes.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.stop_speed_sq` | propext, Classical.choice, Quot.sound |

<a id="claim-8"></a>

## Binary64 mass clamp

**Lean Float logical model (binary64).** Mass field only. The strict comparison maps negatives to positive zero, retains negative zero and preserves the NaN class. The equality convention treats any two NaNs alike. Binary32 conversion and other setters remain open.

Engine symbols `Ballpark.add_ball`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-711 passing (compiled code).

Host correspondence check — Lean and Python agree on 24 binary64 inputs.

| [Theorems](ExactBallparkProofs/Numerics/Clamp.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.clamp_negative` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.clamp_other` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.clamp_nan` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.clamp_negative_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.signed_zeros_distinct` | propext, Classical.choice, Quot.sound |

<a id="claim-9"></a>

## First contact of two swept spheres

**Real arithmetic.** Straight interpolation on both endpoint chords and a physical combined radius ≥ 0. Covers initial overlap, stationary relative motion, tangency, endpoint hits and a complete miss. Sign decisions in binary64 remain open.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Sweep.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.gap_quadratic` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.contact_correct` | propext, Classical.choice, Quot.sound |

<a id="claim-10"></a>

## Box exclusion along a straight sweep

**Real arithmetic.** Broad-phase filter used by the oriented-box part of COL-712. For a valid box and nonnegative radius, one strict separating side persists for the full chord, including endpoints. Contact response remains open.

Engine symbols `AABB.can_exclude`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Sweep.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.excluded_chord` | propext, Classical.choice, Quot.sound |

<a id="claim-11"></a>

## Agility rounding before validation

**Lean core binary64 and binary32 logical models.** Binary64 input is unpacked, rounded to nearest-even binary32 and classified before the write. Negative values and both zeros are rejected. NaN and positive infinity are accepted when conversion returns. Includes both signs of the halfway-to-minimum-subnormal input and their next magnitudes. Host overflow exceptions and NaN payloads are outside the model.

Engine symbols `f32, Ballpark.set_agility`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-712 passing (compiled code), MOV-713 passing (compiled code).

Host correspondence check — Lean and Python agree on 24 binary64 inputs.

| [Theorems](ExactBallparkProofs/Numerics/Agility.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.agility_accepts_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.agility_half_subnormal` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.agility_above_half_subnormal` | propext, Classical.choice, Quot.sound |

<a id="claim-12"></a>

## Warp destination and alignment state

**Real arithmetic.** An existing ball with finite destination. The 100 km guard precedes repeated-target detection. Long duplicate commands preserve all modeled fields. Fresh long commands install the clamped factor and range and start alignment at stamp -1. GOTO bookkeeping and emitted events remain separate.

Engine symbols `ModeBallpark.warp_to`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-701 passing (compiled code), TRV-702 passing (compiled code), TRV-703 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Commands.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.short_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.repeated_warp` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.long_starts_alignment` | propext, Classical.choice, Quot.sound |

<a id="claim-13"></a>

## Alignment and warp exit geometry

**Real arithmetic.** Given the computed direction cosine and nonnegative speed and max velocity, both motion thresholds are strict. The forced path is available after 180 unsuccessful calls from stamp -1. Distinct endpoints and nonnegative minimum range give exactly that distance between the original target and offset exit.

Engine symbols `ModeBallpark._is_aligned_for_warp, ModeBallpark._evolve_warp, ModeBallpark._real_warp`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-706 passing (compiled code), TRV-707 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Commands.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.alignment_after_waits` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.exit_distance` | propext, Classical.choice, Quot.sound |

<a id="claim-14"></a>

## Warp rates, peak and phase distances

**Real arithmetic.** Positive warp factor and rates, nonnegative distance. The encoded rate constants determine requested peak speed. The deceleration rate caps at two. Limiting the peak makes cruise duration nonnegative and the short-warp branch makes it zero. Phase distances add to the requested distance.

Engine symbols `ModeBallpark.warp_constants`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-704 passing (compiled code), TRV-708 passing (compiled code), TRV-709 passing (compiled code), TRV-710 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Profile.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.rate_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.short_cruise_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.cruise_nonnegative` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.phase_distances_sum` | propext, Classical.choice, Quot.sound |

<a id="claim-15"></a>

## Warp trajectory joins and drop-out

**Real arithmetic.** Axial traveled distance under positive rates and peak. The exponential, cruise and deceleration distance branches meet, with nominal deceleration ending at the target distance. The strict deceleration and speed tests are preserved. Reported speed can be held above the acceleration curve.

Engine symbols `ModeBallpark.warp_constants, ModeBallpark._warp_distance`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-711 passing (compiled code), TRV-712 passing (compiled code), TRV-713 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Profile.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.phase_joins` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.phase_decel_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.dropout_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.decel_velocity` | propext, Classical.choice, Quot.sound |

<a id="claim-16"></a>

## Tick backlog, timestamps and render delay

**Integer clock and ordered calls.** Integer 100 ns timestamps and positive interval. Covers early calls, the first due call, server resynchronization above five whole intervals, ordered event/evolve calls, and exact remainder for the ordinary loop. Cleanup is a call in the trace. Handler and deletion effects remain outside this projection.

Engine symbols `FullBallpark.on_tick, FullBallpark.shifted_time`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — TIME-702 passing (compiled code), TIME-703 passing (compiled code), TIME-704 passing (compiled code), TIME-705 passing (compiled code), TIME-708 implemented, not passing (compiled code). The client ball's display interpolation is not transcribed..

| [Theorems](ExactBallparkProofs/Time/Tick.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.rounds_remainder` | propext |
| `ExactBallpark.Time.rounds_trace` | propext, Quot.sound |
| `ExactBallpark.Time.early` | propext |
| `ExactBallpark.Time.first_due` | propext |
| `ExactBallpark.Time.master_resync` | propext, Quot.sound |
| `ExactBallpark.Time.ordinary_tick` | propext, Quot.sound |
| `ExactBallpark.Time.render_two_ticks` | propext |

<a id="claim-17"></a>

## Elastic normal response

**Real arithmetic.** Unit contact normal for reflection. Positive base masses for one-dimensional exchange. Before drag, response scaling and AddImpact limiting, reflection preserves speed and reverses the normal projection. An applied pair impulse reverses relative normal speed and preserves normal momentum and kinetic energy.

Engine symbols `SpaceBallpark._potential, SpaceBallpark._impact_static, SpaceBallpark._impact_free`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-706 passing (compiled code), COL-707 passing (compiled code), COL-708 passing (compiled code), COL-716 passing (compiled code), COL-717 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Response.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.reflection` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.elastic_exchange` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.iterative_exchange` | propext, Classical.choice, Quot.sound |

<a id="claim-18"></a>

## Overlap displacement and impact limiting

**Real arithmetic.** Positive collision inertias split overlap plus one metre. The limited nonzero impulse has magnitude (5 maxVel)² / original magnitude. Real interpretation before binary32 rounding of the numerator.

Engine symbols `SpaceBallpark._potential, SpaceBallpark._add_impact`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-709 passing (compiled code), COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Response.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.overlap_split` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.limitImpact_norm` | propext, Classical.choice, Quot.sound |

<a id="claim-19"></a>

## Transversal angular speed and accuracy bounds

**Real arithmetic.** Distinct positions and ordinary relative velocities with dynamical orientation off. The independent squared-speed expression equals squared angular speed from the transverse projection. On the successful accuracy branch the chance lies in (0,1], and a stationary relative target inside optimal range has chance one. Early returns and numeric cancellation remain separate.

Engine symbols `FullBallpark.relative_velocities, FullBallpark.get_accuracy`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — CMB-701 passing (compiled code), CMB-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Combat/Accuracy.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.angular_agrees` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.accuracy_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.optimal_stationary` | propext, Classical.choice, Quot.sound |

<a id="claim-20"></a>

## Sensor timing, edge events and range boundaries

**Real arithmetic.** Live eligible candidates. A sensor checks on reaching its period and includes the outer distance boundary. Tracking uses a strict outer boundary and follow notifications use an open band. Radii in the sensor comparison are supplied after the source float additions. An unchanged inside/outside sample emits no repeated edge event.

Engine symbols `SpaceBallpark._check_sensor, SpaceBallpark._check_target_tracking, SpaceBallpark._check_notification_range`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-701 passing (compiled code), SEN-702 passing (compiled code), SEN-703 passing (compiled code), SEN-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Proximity.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.crossing` | None |
| `ExactBallpark.Space.repeated_sample` | propext |
| `ExactBallpark.Space.sensor_inside` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.tracking_boundary` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.follow_band_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.sensor_schedule` | propext, Classical.choice, Quot.sound |

<a id="claim-21"></a>

## Finest admissible partition level

**Real arithmetic.** Eight widths from 7864320 m to 480 m, a factor of four apart. The argument is the filing radius supplied to level_for. An oversized object falls back to level zero. Every positive level has the exact inclusive three-radius threshold.

Engine symbols `LEVEL_WIDTH, SpaceBallpark.level_for`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-705 passing (compiled code), SEN-706 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Partition.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.finest_level` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.finest_width` | propext, Classical.choice, Quot.sound |

<a id="claim-22"></a>

## Exponential dispatch by magnitude

**64-bit words.** All binary64 encodings. The signed shift and mask implement the two exact unsigned-magnitude intervals for the tiny and large branches. Table evaluation, rounding and device execution remain open.

Engine symbols `exp`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/LibmGuards.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.exp_tiny_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.exp_large_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-23"></a>

## Signaling NaN detection in powf

**32-bit words.** All binary32 encodings. The test is exactly an all-ones exponent, clear quiet bit and nonzero remaining payload. The resulting powf arithmetic remains open.

Engine symbols `_signaling`. [Source](../src/exact_ballpark/dlibm_powf.py)

Recorded upstream evidence — MOV-739 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/LibmGuards.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.powf_signaling_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-24"></a>

## Logarithm range reduction guard

**64-bit words.** All binary64 encodings. The wrapping addition recognizes the inclusive near-one bit interval. Polynomial and other branches remain open.

Engine symbols `log`. [Source](../src/exact_ballpark/dlibm.py)

Recorded upstream evidence — MOV-740 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/LibmGuards.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.log_near_one_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-25"></a>

## Atan2 sign restoration

**64-bit words.** All binary64 encodings, including signed zeros and NaN payloads. The returned sign comes from y and the other 63 bits come from z. Only this final primitive is covered.

Engine symbols `_signed`. [Source](../src/exact_ballpark/dlibm_inv.py)

Recorded upstream evidence — MOV-741 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/LibmGuards.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.atan2_sign` | propext, Classical.choice, Quot.sound |

<a id="claim-26"></a>

## A GOTO overshoot witness

**Real arithmetic.** A ball at x=0 moving toward x=1 at speed 2, with unit speed request, tick, friction and effective mass. The taper and regular integration yield a position strictly beyond the target.

Engine symbols `Ballpark.goto_thrust, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-707 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Guidance.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.goto_overshoot` | propext, Classical.choice, Quot.sound |

<a id="claim-27"></a>

## Follow geometry and orbit steering

**Real arithmetic.** The range offset has magnitude |r|, including the coincident fallback axis. The old orbit angles repeat across IDs separated by 65536 and its truncated Gaussian stays in [0,1]. The new orbit-plane target has the spiral radius in an orthonormal frame. The two uncapped linear guidance solves reach their prescribed endpoint component or velocity when denominators are nonzero. Positive-limit orbit braking bounds the output norm.

Engine symbols `ModeBallpark._evolve_follow, _evolve_old_orbit, _evolve_new_orbit, _goto_thrust_follow, _orbit_thrust`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-714 passing (compiled code), MOV-717 passing (compiled code), MOV-718 passing (compiled code), MOV-720 passing (compiled code), MOV-721 passing (compiled code), MOV-722 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Guidance.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.unit_norm` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.follow_range` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.orbit_id_period` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.orbit_weight_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.spiral_radius` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.solve_guidance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.orbit_brake_cap` | propext, Classical.choice, Quot.sound |

<a id="claim-28"></a>

## Follow cleanup, GOTO defaults and mode events

**Discrete transitions with real speed fractions.** Projection onto one existing ball with consistent follow links and finite command values. Checks STOP cleanup and idempotence, GOTO speed default, target eligibility, fly-on exceptions, and ordered exit/mode-change events. Target follower sets, formation slots and command rejection for nonfinite coordinates are separate.

Engine symbols `ModeBallpark._stop, _goto_point, _follow, _stop_all_followers, _set_mode`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-708 passing (test vectors), MOV-709 passing (test vectors), MOV-715 passing (compiled code), MOV-730 passing (compiled code), MOV-735 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Control.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Control.stop_clears` | propext |
| `ExactBallpark.Control.goto_cancels` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Control.follow_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Control.fly_on_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Control.exit_iff` | propext |
| `ExactBallpark.Control.exiting_order` | propext |

<a id="claim-29"></a>

## Server queue duplicate suppression

**Lists with decidable equality.** A history element is the complete compared value, including ball id, stamp and event. For values with reflexive decidable equality, a duplicate tail is idempotent and a distinct next action is appended. Python custom equality and NaN tuple behavior are outside the model.

Engine symbols `Actions._add_to_system_history`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — TIME-709 passing (test vectors), TIME-710 passing (test vectors).

| [Theorems](ExactBallparkProofs/Time/History.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.last_enqueued` | propext |
| `ExactBallpark.Time.duplicate_idempotent` | propext |
| `ExactBallpark.Time.distinct_actions_kept` | propext |

<a id="claim-30"></a>

## Client wait and successful synchronization

**Integer clock and ordered snapshots.** A waiting bubble defers dispatch. Without that flag only future stamps one and two ticks ahead are deferred at the queue head. On successful synchronization with usable snapshots and one tick per evolve, the resulting clock equals the requested stamp.

Engine symbols `Ticker.do_pre_tick, Ticker.synchronize_to_simulation_time`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

| [Theorems](ExactBallparkProofs/Time/History.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.sync_reaches_stamp` | propext, Quot.sound |
| `ExactBallpark.Time.wait_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-31"></a>

## Python warp constants and parameter dependence

**Real arithmetic.** The build gate regenerates this pure helper’s real expression from its Python AST and checks all nine tuple fields against the analytic definitions. Only warp factor, distance and module constants enter the extracted function. The dependency statement is local to this helper. The extractor is trusted and intermediate floating-point rounding is outside the model.

Engine symbols `ModeBallpark.warp_constants`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-704 passing (compiled code), TRV-708 passing (compiled code), TRV-709 passing (compiled code), TRV-710 passing (compiled code), TRV-716 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Profile.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.source_constants` | propext, Classical.choice, Quot.sound |

<a id="claim-32"></a>

## Behavior dispatch and angular cap

**Discrete dispatch and real vectors.** Exhaustive classification of all 13 modes for both orientation settings. Six select a thrust callback, ORBIT selects the configured law, and CapAcceleration returns the requested vector. The callbacks have independent contracts.

Engine symbols `FullBallpark._behaviour, _cap_acceleration`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-716 passing (compiled code), MOV-727 passing (compiled code), MOV-734 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Flight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.thrust_modes` | propext |
| `ExactBallpark.Movement.orbit_switch` | None |
| `ExactBallpark.Movement.cap_preserves` | propext, Classical.choice, Quot.sound |

<a id="claim-33"></a>

## Predicted target direction

**Real arithmetic.** Nonzero offset and dynamical orientation off. The computed direction times the distance reaches the target advanced by its velocity times dt. The cross product and default normal branch have further obligations.

Engine symbols `ModeBallpark._orbital_normal`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-719 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Flight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.predicted_direction` | propext, Classical.choice, Quot.sound |

<a id="claim-34"></a>

## Game flight targets and launch speed

**Real arithmetic.** The 800 ms straight phase includes equality. On the pursuit branch the radii cancel if their sum was stored exactly, including coincident positions. For a valid aimed launch with distinct target the initial speed is one. On an ordinary moving launch the added speed has magnitude max(150, launcher.max_vel). Endpoint storage, float narrowing and later commands remain separate.

Engine symbols `FullBallpark._evolve_missile, missile_follow, launch_missile`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-723 passing (compiled code), MOV-724 passing (compiled code), MOV-725 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Flight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.pursuit_center` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.launch_speeds` | propext, Classical.choice, Quot.sound |

<a id="claim-35"></a>

## Formation and orientation bounds

**Real arithmetic.** A valid formation slot and norm-preserving rotation, before vector narrowing. For the above-threshold orientation branch with unit predicted heading and nonnegative requested magnitude, projected thrust stays within the request. For nonnegative smoothing parameter the target-angle error contracts by 1/(1+mm/2) after angle wrapping. Quaternion and torque calculation remain separate.

Engine symbols `FullBallpark._evolve_formation, _orient, _yaw_pitch_roll`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-726 passing (compiled code), MOV-728 passing (compiled code), MOV-729 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Flight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.formation_distance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.orientation_bound` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.smooth_error` | propext, Classical.choice, Quot.sound |

<a id="claim-36"></a>

## Troll readiness and minimum delay

**Integer ticks and modes.** A ball kept in TROLL mode becomes ready exactly when elapsed integer ticks reach both the requested delay and one. Petrification side effects remain separate.

Engine symbols `FullBallpark.set_ball_troll, _troll_ready`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-736 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Flight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.troll_due` | propext |

<a id="claim-37"></a>

## Alignment countdown, backdating and exit tick

**Integer transitions and real integration.** From a negative stamp each unsuccessful wait decrements before the GOTO callback. For nonnegative clock, EntityWarpIn starts min(now,5) ticks into warp on the warp path. Dropout uses its returned position and velocity in the full-tick integrator, including both position branches, with cache and nonzero premises in the theorems.

Engine symbols `ModeBallpark._evolve_warp, entity_warp_in, evolve`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-705 passing (compiled code), TRV-714 passing (compiled code), TRV-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Transitions.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.waiting_ticks` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.entity_elapsed` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.exit_regular` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.exit_special` | propext, Classical.choice, Quot.sound |

<a id="claim-38"></a>

## Default tick duration

**Lean logical binary64.** The actual binary64 encoding of 1000 multiplied by the encoding of 0.001 rounds to exactly one second. Other tick lengths can incur rounding.

Engine symbols `Ballpark.__init__`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TIME-701 passing (compiled code).

| [Theorems](ExactBallparkProofs/Time/Schedule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.default_tick_seconds` | propext, Classical.choice, Quot.sound |

<a id="claim-39"></a>

## Phase order and ascending free IDs

**Ordered lists of integer IDs.** Ordered call projection using a fixed live predicate for the three passes, and a fixed free-ID snapshot. Filtering each phase recovers the same ascending live IDs. Actual handler state changes, mushroom commits and device scheduling remain separate.

Engine symbols `FullBallpark.evolve`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — TIME-706 passing (compiled code), TIME-707 passing (compiled code).

| [Theorems](ExactBallparkProofs/Time/Schedule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.phase_ids` | propext, Quot.sound |
| `ExactBallpark.Time.phase_ordered` | propext, Quot.sound |

<a id="claim-40"></a>

## Dead bubbles, remote followers and cleanup quota

**Discrete state.** Dead-bubble exclusion needs a master, non-warp mode and both interest counts zero. Remote follower IDs must be free with a follow mode and either missing or other-bubble target. For actual moribund entries outside Evolve, expired entries consume no quota while near-due deletion preserves the seven-entry budget. Server delivery of STOP and heap order remain separate.

Engine symbols `SpaceBallpark.in_dead_bubble, remote_followers, bring_out_the_deads`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — TIME-713 implemented, not passing (compiled code). Destiny cleans up moribund balls in an order that follows heap addresses, which no port can reproduce. The reference uses ascending id., TIME-714 passing (compiled code), MOV-731 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done..

| [Theorems](ExactBallparkProofs/Time/Schedule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.dead_iff` | propext, Quot.sound |
| `ExactBallpark.Time.remote_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.cleanup_budget` | propext, Classical.choice, Quot.sound |

<a id="claim-41"></a>

## Immediate removal and pre-tick ordering

**Source-extracted call traces.** Gate extracts ordered calls from the two Python methods. Successful removal always calls the parent last, preceded by notification exactly for global balls. Pre-tick calls the six named hooks in order. Effects inside hooks, exceptions and transport are separate.

Engine symbols `Actions.remove_ball, BaseTicker.tick`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — TIME-711 passing (test vectors), TIME-712 implemented, not passing (test vectors). Destiny's published tests do not cover the pre-tick's first step (the follower clean-up). That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Time/Server.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.removal_calls` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.pretick_calls` | None |

<a id="claim-42"></a>

## Bubble update recipients

**Sets of integer IDs.** The union of interested clients across a bubble’s interactive balls splits exhaustively and disjointly according to character history. Separate singlecasts and network delivery are outside this claim.

Engine symbols `ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Time/Server.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.recipient_split` | propext, Classical.choice, Quot.sound |

<a id="claim-43"></a>

## Gradient choice and candidate exceptions

**Discrete state.** Free massive live actors in the eligible bubble use the gradient when iteration is disabled. After broad-phase lookup and other eligibility filters, a non-defender missile sees its same-bubble target directly. Ordinary candidates exclude cloaks and missiles, with the master global-ball exception. Gradient skips the owner and non-junk versus junk.

Engine symbols `FullBallpark.evolve, _collision_candidates, _nearby, _gradient`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-701 passing (compiled code), COL-702 passing (compiled code), COL-703 passing (compiled code), COL-704 passing (compiled code), SEN-708 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Selection.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.gradient_iff` | None |
| `ExactBallpark.Collision.missile_target` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ordinary_candidate` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.owner_junk_skip` | propext, Classical.choice, Quot.sound |

<a id="claim-44"></a>

## Twenty-pass iteration and orphan detection

**Bounded recursion and discrete state.** The work-list loop enters one to twenty passes for arbitrary successive frontier results and terminates after one for an empty first frontier. With a known owner, orphan detection uses cloak or a different bubble while the missile has a bubble. Delivery of collision events has additional motion and master guards.

Engine symbols `FullBallpark._collision_responses, _orphaned`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-714 passing (compiled code), COL-719 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Selection.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.iteration_cap` | propext, Quot.sound |
| `ExactBallpark.Collision.orphan_with_owner` | propext, Classical.choice, Quot.sound |

<a id="claim-45"></a>

## Collision tie strength, mass-ratio and ID ordering

**Discrete selection and real arithmetic.** Equal-time non-skipped response has the greater norm after scaling by the exact binary64 value of 0.85. Positive first inertia converts the strict lighter-ball threshold to m2 > 25*m1. For distinct ordinary free IDs the search order admits exactly one ordered pair. Recursive and static candidate geometry remain separate.

Engine symbols `SpaceBallpark._potential, _next_collision_free`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-710 passing (compiled code), COL-711 passing (compiled code), COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Selection.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.response_tie` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.lighter_boost` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.unique_pair` | propext, Classical.choice, Quot.sound |

<a id="claim-46"></a>

## Bubble neighbors, visibility and range

**Real arithmetic.** The neighbor loop enumerates all 26 distinct adjacent offsets, excluding the center. In one valid bubble, the first eligible blocker in dictionary order has nonnegative discriminant and entry strictly between endpoints. Eligible range candidates with nonnegative distance satisfy distance ≤ |range+radius|, including negative range inputs. Partition reachability, eligibility filtering and rounded grazing cases remain separate.

Engine symbols `SpaceBallpark._nearby_bubbles, _traverse, check_visibility, ball_ids_in_range`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-707 passing (compiled code), SEN-709 passing (compiled code), SEN-710 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Queries.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.neighbor_offsets` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.neighbor_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.first_occluder` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.visibility_entry` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.range_abs` | propext, Classical.choice, Quot.sound |

<a id="claim-47"></a>

## Cloak, field and fixed collision-part rules

**Discrete state and real vectors.** Successful positive cloak releases followers and clears massive, with a sensor call only for normal cloak on the master. The harmonic, corporation and alliance filter includes both sentinel exceptions. A fixed parent translates both capsule endpoints preserving the axis vector; free parents do not instantiate it. Other mini shapes, allocation and sensor effects are outside these claims.

Engine symbols `FullBallpark.cloak_ball, _nearby, _add_actual_minicapsule, _add_minis`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — SEN-711 passing (compiled code), MOV-737 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done., MOV-732 passing (compiled code), COL-713 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Queries.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.cloak_effect` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.field_iff` | propext |
| `ExactBallpark.Space.fixed_capsule` | propext, Classical.choice, Quot.sound |

<a id="claim-48"></a>

## Ball flag byte round trip

**Eight-bit words.** All 256 combinations of five state flags and three mini-presence bits recover their original values in the eight reader masks. The claim concerns the flag byte.

Engine symbols `DestinyPark._write_ball_to_stream, _read_ball_from_stream`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — MOV-733 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/Flags.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.flags_roundtrip` | propext, Classical.choice, Quot.sound |

<a id="claim-49"></a>

## Source-extracted movement integrator

**Real arithmetic.** A restricted translator reads the actual Python arithmetic, indexed comprehensions, guards, reciprocal multiplications and cached-factor override. Its single-coordinate real program agrees with the existing integrator for every choice of inputs. Real division in Lean is total; applying the result to an actual Python return requires defined source operations. The source translation is trusted.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Correspondence.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.extracted_integrator` | propext, Classical.choice, Quot.sound |

<a id="claim-50"></a>

## Finite integrator operations and compatibility-branch witness

**Lean logical binary64.** The source-extracted binary64 model uses an explicit exp function and both position branches. For every nonzero full tick whose duration compares equal to itself, velocity bits follow the stated ordered arithmetic on the supplied factor. The MOV-703 stress input sets p=0, v=k=t=dt=1, effective mass 2^40 and acceleration 2^20, implying terminal speed 2^60. The position operations store +0 m. The 1 m baseline is the exact-real value of the first-order branch. The test uses cached factor 0x3fefffffffffe000. Host exp and the device remain separate.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-704 passing (compiled code).

Host correspondence check — 24 coordinate updates match Python for position and velocity bits.

| [Theorems](ExactBallparkProofs/Movement/Correspondence.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.full_tick_velocity_bits` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.special_rounding_witness` | propext, Classical.choice, Quot.sound |

<a id="claim-51"></a>

## Static reaction and scaled drag correction

**Real arithmetic.** For nonzero inertia and friction and a cached factor other than one, applying a fraction of the source collision acceleration interpolates exactly between free end velocity and the reflected substep result, coordinate by coordinate. This includes the source response scale. A zero time ejects, a positive time reflects except for an outward-accelerating capsule contact, and the sentinel skips. The selected contact point, normal and ejection depth have their own geometric obligations.

Engine symbols `FullBallpark._reflect, _capsule_react, _obox_react`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-707 passing (compiled code), COL-711 passing (compiled code), COL-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Response.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.scaled_reflection` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.static_branches` | propext, Classical.choice, Quot.sound |

<a id="claim-52"></a>

## Zero-duration bit preservation

**Lean logical binary64.** For every binary64 position and velocity bit pattern, a duration comparing equal to zero returns both original bit patterns, independently of all other arithmetic inputs and the exponential implementation.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Correspondence.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.zero_duration_bits` | propext, Classical.choice, Quot.sound |

<a id="claim-53"></a>

## Phase barriers across different ball IDs

**Ordered calls.** For every snapshot of free IDs and live predicate, all behavior calls precede the optional responses pass, all calculations precede every commit, and proximity and clock advance come last. This projection tracks the ordering of these phase calls.

Engine symbols `FullBallpark.evolve`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — TIME-706 passing (compiled code).

| [Theorems](ExactBallparkProofs/Time/Schedule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.schedule_monotone` | propext, Classical.choice, Quot.sound |

<a id="claim-54"></a>

## Missile collision event order

**Discrete events.** After storage and iterative collider collection, only a missile crossing the movement threshold dispatches the list in order. A moving master-side orphan appends its own ID last. The moved Boolean is the result of the source displacement guard.

Engine symbols `FullBallpark.evolve, _orphaned`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-719 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Selection.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.dispatch_moving` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.dispatch_silent` | propext, Classical.choice, Quot.sound |

<a id="claim-55"></a>

## Rounded follow range can miss exact radius cancellation

**Lean logical binary32 and binary64.** For binary32 radii 1 and 2^-24, storing the negative rounded sum and then adding the two promoted radii leaves positive 2^-24 in binary64. This is a witness about the radial offset before projection into the pursuit target.

Engine symbols `FullBallpark.missile_follow, _evolve_missile`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-724 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/Agility.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.pursuit_offset_witness` | propext, Classical.choice, Quot.sound |

<a id="claim-56"></a>

## A bubble traversal fills its connected component

**Finite graph and terminating depth-first traversal.** For a finite set of initially unlabeled top-level box keys, stepping to all 26 adjacent keys and marking each box before recursing terminates. A box is newly assigned exactly when a path from the seed reaches it using initially unlabeled boxes. Missing and previously labeled boxes are excluded from the pool. Fixed topology, one seed distinct from the unassigned sentinel, and the source’s level-zero grid clamp after key addition. The 26 corner queries include the center and reach eight distinct keys. Other bubble bookkeeping and Python recursion limits remain separate.

Engine symbols `SpaceBallpark._traverse, _check_box`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-707 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Bubbles.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.bubble_component` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.corner_neighbors` | propext, Classical.choice, Quot.sound |

<a id="claim-57"></a>

## Long chord with a spurious binary64 contact

**Real arithmetic and Lean logical binary64.** One moving center travels from (-2^28,2,0) to (2^28,2,0), the other remains at the origin, and the combined radius is one. Every exact-real squared gap is at least three, so the real algorithm returns -1. The same exactly representable endpoints in the ordered binary64 program return t=0.5. This qualifies any general earliest-contact claim about the rounded predicate.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

Host correspondence check — the Python scalar reference returns the same binary64 midpoint.

| [Theorems](ExactBallparkProofs/Collision/FiniteSweep.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.long_chord_clear` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.long_chord_real_miss` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.long_chord_rounded_hit` | propext, Classical.choice, Quot.sound |

<a id="claim-58"></a>

## When rounding preserves the discriminant sign

**Lean logical binary64 and rational error bounds.** Given finite decoded quadratic coefficients and bounds on their absolute errors, the exact discriminant lies within a computed margin of the ordered binary64 result. A strict margin certifies agreement on the negative-discriminant guard. With a positive leading coefficient, a negative certificate proves that the binary64 solver returns the miss sentinel and the exact gap polynomial stays positive at every real time.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Discriminant.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.finite_negative_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.discriminant_error` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.certified_sign` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.certified_miss` | propext, Classical.choice, Quot.sound |

<a id="claim-59"></a>

## A certified miss from encoded sweep endpoints

**Lean logical binary64 and exact straight chords.** The offline certificate derives all three error bounds from the four binary64 endpoints and the supplied combined radius. Whenever it reports a negative discriminant, both the modeled binary64 solver and the exact straight-chord solver return -1. Early overlap, nonfinite data and an interval containing zero give no certificate. At a five-metre offset the long chord is a certified miss. At two metres the result is uncertain. A positive discriminant still requires checking the root against the tick interval.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

Host correspondence check — the scalar solver returns −1 for the five-metre offset and for the sweep whose roots lie outside the tick.

| [Theorems](ExactBallparkProofs/Collision/SweepCertificate.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.decoded_gap` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.certified_chord_miss` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.horizontal_margin_examples` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.positive_outside_tick` | propext, Classical.choice, Quot.sound |

<a id="claim-60"></a>

## Bounding first contact at the tick boundaries

**Exact straight chords and Lean logical binary64.** For finite encoded endpoints with positive exact initial gap, positive relative speed squared and nonnegative discriminant, the checker validates a supplied rational bracket for the earlier exact root. It decides membership in the closed tick interval and records the decoded binary64 return when finite. The certified examples cover an early hit, an exact endpoint hit, and a late exact root that binary64 accepts at the endpoint.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Timing.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.entry_comparisons` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.timing_sound` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.tick_boundary_examples` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.late_root_rounded_hit` | propext, Classical.choice, Quot.sound |

<a id="claim-61"></a>

## Exact grazing contact and a binary64 miss

**Exact straight chords and Lean logical binary64.** For a moving quadratic sweep with zero exact discriminant, the gap reaches zero only at the vertex and is positive everywhere else. A six-metre tangent sweep at combined radius 2^28 touches halfway through the tick. The encoded endpoints give a binary64 discriminant of −1008 and a miss. Additional tangent sweeps exercise contact exactly at the end and beyond the end of a tick.

Engine symbols `SpaceBallpark.collide_two_spheres, _sub, _dot`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Timing.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.tangent_contact_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.tangent_rounded_miss` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.tangent_tick_boundaries` | propext, Classical.choice, Quot.sound |

<a id="claim-62"></a>

## Normal and tangential components at a static or start contact

**Real arithmetic in three dimensions.** For a unit normal, the guarded static impulse makes the outward projection equal to the absolute incoming projection and preserves every tangential component. When the source applies the start-contact correction, the total normal acceleration equals maxVel times friction divided by effective inertia and the tangential acceleration is unchanged. The engine interpretation of that division assumes nonzero mass and agility.

Engine symbols `SpaceBallpark._impact_static, SpaceBallpark._impact_free`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-716 passing (compiled code), COL-717 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Impact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.static_velocity_components` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.start_contact_acceleration` | propext, Classical.choice, Quot.sound |

<a id="claim-63"></a>

## Three-dimensional conservation through the free-pair impulse

**Real arithmetic in three dimensions.** The velocity update includes the source impulse guard, opposite normals and positive base masses. It preserves vector momentum. With a unit normal it preserves kinetic energy and yields an outward relative normal speed equal to the incoming magnitude. Each tangential component is preserved. These are the local velocity changes before AddImpact limiting and later integration.

Engine symbols `SpaceBallpark._impact_free`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-717 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Impact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.pair_separates` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.pair_preserves_tangent` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.pair_momentum` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.pair_kinetic_energy` | propext, Classical.choice, Quot.sound |

<a id="claim-64"></a>

## Clearance and centre of inertia in a free-free overlap re-test

**Real arithmetic in three dimensions.** For positive collision inertias, the two temporary positions preserve the inertia-weighted centre. Given a nonnegative combined radius and a unit normal aligned with the displacement, the new centre distance is the combined radius plus one metre. Coincident centres satisfy the alignment premise at zero distance with either unit fallback direction. The source restores the original positions after its recursive test.

Engine symbols `SpaceBallpark._potential`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-709 passing (compiled code), COL-710 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Impact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.overlap_clearance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.overlap_inertia_center` | propext, Classical.choice, Quot.sound |

<a id="claim-65"></a>

## Magnitude bound for the ideal impact limiter

**Real arithmetic in three dimensions.** For every velocity-change vector and nonnegative maximum speed, the real limiter output has magnitude at most the smaller of its original magnitude and five times maxVel. This includes zero input. The binary32 threshold used in the engine is covered by the following witness.

Engine symbols `SpaceBallpark._add_impact`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Impact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.limited_speed_bound` | propext, Classical.choice, Quot.sound |

<a id="claim-66"></a>

## Rounded impact threshold admits a change above five times maxVel

**Lean logical binary32 and binary64.** For binary32 maxVel = 4097 and a velocity change (20485 + 2⁻¹³, 0, 0), the binary32 numerator rounds from 419635225 to 419635232. The mixed-precision limiter leaves the x component above the ideal bound, with binary64 word 40d4014002000000.

Engine symbols `SpaceBallpark._add_impact, VELOCITY_LIMIT, _lensq`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/FiniteImpact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.rounded_limiter_exceeds_bound` | propext, Classical.choice, Quot.sound |

<a id="claim-67"></a>

## Inserting an earlier impact and cancelling the suffix

**Impact history with logical binary64 time and a supplied translational integrator.** The end-time guard rejects equality. Replaying a comparable impact time directly after insertion has no effect. The backward scan retains the last qualifying predecessor, including an equal-time record, then integrates from it and applies the velocity change. With no predecessor it applies the change to the starting state. Invalidations omit sentinel collider −1 and follow newest-first order. The model covers a single path before recursive notifications to other balls.

Engine symbols `CollisionBallProperties.add_impact, SpaceBallpark._add_impact`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ImpactHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.History.insert_cutoff_and_retry` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.insert_after_predecessor` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.insert_before_all` | propext, Classical.choice, Quot.sound |

<a id="claim-68"></a>

## Removing an impact and rebuilding the tick endpoint

**Impact history with logical binary64 time and a supplied translational integrator.** Removal uses the first matching collider within the strict binary64 time tolerance. A later match retains the earlier prefix and reports subsequent non-sentinel colliders oldest first. Both a first-record match and a missing match reset the first record to the initial velocity, preserving its position. For a nonempty path, CalculateEnd preserves an already complete window, appends an integrated endpoint when needed, or signals the modeled capacity failure. The supplied capacity is the actual capacity after truncation. The undefined C++ reallocation path has no claimed C++ result.

Engine symbols `CollisionBallProperties.remove_impact, CollisionBallProperties.calculate_end`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-714 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ImpactHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.History.remove_first_matching_impact` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.remove_reset_at_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.finish_window` | propext, Classical.choice, Quot.sound |

<a id="claim-69"></a>

## Duplicate suppression and predecessor lookup

**Impact history with logical binary64 time and a supplied translational integrator.** For the backward duplicate scan, the endpoint is skipped. A collider matches within the contiguous window ending at the penultimate location, including equality at the lower edge, provided the reported time passes the initial guard. For lookup, the strict scan chooses the predecessor or clamps to index zero. On a nonempty path the selected motion is copied or integrated according to the inclusive absolute-difference guard. Theorems use the stored binary64 tolerance and preserve the order of addition and subtraction.

Engine symbols `CollisionBallProperties.ignore_collision, CollisionBallProperties.find_index, CollisionBallProperties.position, FUZZY_DELTA`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ImpactHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.History.ignore_recent_colliders` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.index_prefix` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.History.position_from_prefix` | propext, Classical.choice, Quot.sound |

<a id="claim-70"></a>

## A rounded lookup chooses the pre-impact state

**Impact history with logical binary64 time and a supplied translational integrator.** With stored times 0, 1/16 and 1, query word 3fb0068db8bac711 selects index zero. Applying the same strict guard to the exactly decoded values selects index one. The next binary64 query selects index one in the rounded model. At the boundary, direct Python and pinned C++ position queries returned positive-zero x velocity for the recorded unit impulse. The theorem covers the indices and the replay covers the observed velocity.

Engine symbols `CollisionBallProperties.find_index, CollisionBallProperties.position, FUZZY_DELTA`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ImpactHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.History.rounded_history_lookup` | propext, Classical.choice, Quot.sound |

<a id="claim-71"></a>

## Sphere against a box face

**Real Euclidean space in three dimensions.** For a unit normal and nonnegative radius the support point lies one radius from the centre with the expected normal projection. A nonparallel chord hits a plane exactly at its in-range root, including either endpoint. A parallel chord returns neither time nor point, even if coplanar. On a nondegenerate triangle in its plane, the barycentric face test excludes the edge opposite the first vertex. The caller tests rejected face points separately. These claims cover the box-face primitives before collision response.

Engine symbols `_point_on_ball, _segment_plane, Triangle.contains, _sphere_triangle`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Triangle.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.Triangle.support_geometry` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Triangle.crossing_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Triangle.parallel_sweep` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Triangle.face_membership` | propext, Classical.choice, Quot.sound |

<a id="claim-72"></a>

## Distance threshold for a positive cruise phase

**Real arithmetic.** For nonnegative distance and positive rates and requested speed, cruise duration is positive exactly when the distance-limited peak exceeds the requested speed. Equality gives zero duration.

Engine symbols `ModeBallpark.warp_constants`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-709 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Boundaries.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.positive_cruise_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-73"></a>

## Local time at warp dropout

**Real arithmetic and mathematical exp/log.** During deceleration at positive peak speed, deceleration rate and maximum ship speed, dropout occurs strictly after log(peak/cutoff)/rate in local phase time. The two speed thresholds and their equality boundary are included. Tick sampling and the host exponential are outside the claim.

Engine symbols `ModeBallpark._warp_distance`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-713 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Boundaries.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.dropout_time_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-74"></a>

## Two short warps round zero cruise in opposite directions

**Real arithmetic and Lean logical binary64.** At warp factor 1000 the exact cruise is zero from 100000 through 100001 metres. At the two endpoints, the binary64 duration has words 3caf75c000000000 and bcaa6dc000000000. The finite projection uses the stored rate constants and omits the unused log fields.

Engine symbols `ModeBallpark.warp_constants`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-709 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Boundaries.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.short_cruise_words` | propext, Classical.choice, Quot.sound |

<a id="claim-75"></a>

## Tick clock bounds and the next client deadline

**Integer clock in 100 ns units.** For any positive interval and monotone poll, on either master or client including initialization, the reported time lies between the prior time and the poll with less than one interval left over. For an initialized client the next deadline is inclusive and evolves at the previous stored timestamp.

Engine symbols `FullBallpark.on_tick`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — TIME-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Time/Poll.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.clock_window` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.next_client_deadline` | propext, Quot.sound |

<a id="claim-76"></a>

## Master backlog at the five-to-six tick boundary

**Integer clock and ordered call trace.** For an initialized master and any subinterval remainder, five whole intervals yield five ordered evolutions and preserve the remainder. Six yield one evolution at poll time minus one interval and reset the stored clock to the poll time.

Engine symbols `FullBallpark.on_tick`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — TIME-703 passing (compiled code).

| [Theorems](ExactBallparkProofs/Time/Poll.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.master_boundary` | propext, Quot.sound |

<a id="claim-77"></a>

## First due sensor sample and discarded overshoot

**Real arithmetic and a discrete trace.** For n waiting polls, nonnegative fixed dt and an elapsed clock still below the period after n increments, reaching or crossing the period at the next poll gives exactly n waits followed by one check. The returned clock is zero even after overshooting. Repeated binary64 clock addition is outside this real model.

Engine symbols `SpaceBallpark._check_sensor`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-701 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/SensorScan.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.SensorScan.first_due_poll` | propext, Classical.choice, Quot.sound |

<a id="claim-78"></a>

## Proximity membership and events across a complete scan

**Candidate list and mixed binary32/binary64 arithmetic.** Projects both passes onto one target in a complete due scan with stable membership inputs and no mid-scan deactivation. An absent remembered non-self target produces one leave. A once-listed live non-self target produces exactly the entry or exit implied by its previous membership and the strict outside guard. Self and listed moribund targets keep their prior membership. The model narrows the two radius sums in source order and takes the computed binary64 squared distance and the candidate list as inputs.

Engine symbols `SpaceBallpark._check_sensor`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-702 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/SensorScan.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.SensorScan.vanished_target` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.SensorScan.present_target` | propext, Classical.choice, Quot.sound |

<a id="claim-79"></a>

## Uncloak during alignment or active warp, and the next sensor entry

**Discrete state and mixed-precision sensor predicate.** For an existing ship the cloak clears. Active warp preserves its massiveness flag, while alignment and ordinary flight restore it. Only master-side normal uncloak disables its sensor. For an observer that already holds the ship, erasure requires a registered live proximity observer in the same bubble outside active warp. Observer eligibility is taken after any sensor deregistration. A cleared observer emits a fresh entry on a subsequent complete scan if the ship is still live and inside. Partition candidate discovery is supplied.

Engine symbols `FullBallpark.uncloak_ball, ModeBallpark._is_warping, SpaceBallpark._check_sensor`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — SEN-711 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Uncloak.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Uncloak.warp_and_sensor` | propext, Quot.sound |
| `ExactBallpark.Space.Uncloak.erase_and_reenter` | propext, Classical.choice, Quot.sound |

<a id="claim-80"></a>

## Ordinary event routing after a bubble change

**Integer IDs, registration flags and an optional character mapping.** For an entry reaching default distribution, bubble history receives it exactly for a positive existing uncloaked ball in its current registered, interactive bubble. A cloaked ball on the same valid path routes only to the mapped character if one exists. This includes a missing ball with a valid RemoveBall fallback. Known and interactive mean dictionary key presence. Special handlers, downstream batching and network delivery have separate obligations.

Engine symbols `BaseTicker._find_bubble_id_for_event, BaseTicker._is_valid_event_bubble, Ticker._apply_history`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Space/Distribution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Distribution.bubble_destination` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.character_destination` | propext, Classical.choice, Quot.sound |

<a id="claim-81"></a>

## Additions and removals for a pilot moving between bubbles

**Per-ID projection of finite bubble dictionaries.** For a pilot on the changed-bubble branch, the two membership equivalences cover every action code, the local-ID cutoff, optional ball flags, keys present in both snapshots and unavailable bubbles. They preserve the source’s different global/cloak checks for old removals and the distinction between an absent and an empty old bubble. A single ID cannot be both added and removed in this per-pilot update. List order, stable-pilot lists and serialized delivery are outside this projection.

Engine symbols `DestinyPark.AdditionsAndDeletions`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Space/Distribution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Distribution.pilot_add_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.pilot_remove_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.pilot_disjoint` | propext, Classical.choice, Quot.sound |

<a id="claim-82"></a>

## Clients addressed by single- and dual-update bubble batches

**Finite client sets and ordered event tokens.** For a bubble-history entry, the recipients are the union of interests through its interactive balls. Each interested client gets the ONE group exactly when it has no mapped nonempty character history, otherwise the TWO group. Both groups carry the original ordered payload and a false wait flag. Missing interactive-bubble keys have an empty audience.

Engine symbols `ParkUpdateBatcher.get_clients_with_character_history, ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Space/Batches.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Batches.single_recipient` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Batches.dual_recipient` | propext, Classical.choice, Quot.sound |

<a id="claim-83"></a>

## Private update flags and per-client packet order

**Ordered history entries, finite recipient sets and projected network calls.** A nonempty mapped character history waits exactly when that client also receives a bubble packet in this batch. Its count field is TWO while waiting and ONE otherwise. Across the three successful interface calls, a client sees its character entries first, then one packet per addressed bubble entry in iteration order, including shared interests or several characters per client. This counts the encoded flags separately from the number of packets a client can receive. Transport scheduling and serialization are outside the model.

Engine symbols `ParkUpdateBatcher.get_clients_with_character_history, ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Space/Batches.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Batches.character_wait` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Batches.client_trace` | propext, Classical.choice, Quot.sound |

<a id="claim-84"></a>

## Snapshot retention and periodic save deadlines

**Integer stamps and distinct tokens for snapshot streams.** At the eleventh running save, the client keeps its original base plus the five most recent streams. Any sequence of running or paused stores from a cache of at most ten stays within ten. Without a pending rebase, the post-tick stamp must exceed the last stored stamp by ten to invoke a save. Rebase takes precedence. The comparison uses the supplied post-tick stamp and storage uses the park clock.

Engine symbols `Ticker.store_state, Ticker.do_post_tick`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

| [Theorems](ExactBallparkProofs/Time/Snapshots.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.Snapshots.eleventh_store` | propext, Quot.sound |
| `ExactBallpark.Time.Snapshots.stores_bounded` | propext, Quot.sound |
| `ExactBallpark.Time.Snapshots.post_deadline` | propext, Classical.choice, Quot.sound |

<a id="claim-85"></a>

## Rebase with or without a new snapshot

**Snapshot identity, integer park clock and running flag.** With a new base requested, the only previous stream eligible for retention is the final entry when it is a mid-tick snapshot from exactly one tick earlier. A running park appends the current full snapshot and updates the last-saved stamp. A paused park can retain the eligible mid-tick stream without appending. Releasing the cache clears all streams and preserves its last-saved stamp.

Engine symbols `Ticker.flush_simulation_history, Ticker.store_state`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

| [Theorems](ExactBallparkProofs/Time/Snapshots.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.Snapshots.rebase_contents` | propext, Classical.choice, Quot.sound |

<a id="claim-86"></a>

## Which stream the client restores for a late update

**Ordered readable streams and integer simulation steps.** For a rewind, the last eligible stream in list order wins, including ties in stamp or an unordered cache. The modeled evolve count is the requested stamp minus that stream’s stamp, agreeing with the existing stamp-only model. Synchronization fails exactly for a past request with no stream at or before it. The stream tokens represent truthy, readable snapshots, and stream loading and evolve calls are assumed to succeed.

Engine symbols `Ticker.synchronize_to_simulation_time`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

| [Theorems](ExactBallparkProofs/Time/Snapshots.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.Snapshots.rewind_stream` | propext, Quot.sound |
| `ExactBallpark.Time.Snapshots.sync_failure` | propext, Classical.choice, Quot.sound |

<a id="claim-87"></a>

## Formation slot allocation and release

**Ordered slot scan, bit membership and unsigned 64-bit subtraction.** For a present leader with a resolved formation, a successful reservation sets only the lowest free bit. A full, empty or oversized formation returns failure and preserves the mask. Releasing a successful reservation immediately restores every bit. With a zero-length formation, subtracting one from the unsigned size wraps. Releasing an index below 16 then clears that bit, while any other size_t reaches the out-of-range bitset reset. The latter outcome models the scalar exception. The mask represents the 16 stored bits and the release index is taken after argument conversion.

Engine symbols `FullBallpark.reserve_formation_slot, FullBallpark.free_formation_slot`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-726 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Formation.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Formation.reserve_first` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.reserve_failure` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.reserve_release` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.release_empty` | propext, Classical.choice, Quot.sound |

<a id="claim-88"></a>

## When a formation join preserves or cancels the previous command

**Command guards and projected follower state.** A missing source, self-target, unavailable destination formation or signed slot at or above its size skips STOP. After those guards pass, a different bubble or cloaked leader leaves the ship stopped. A successful join applies STOP and records the destination and signed slot in FORMATION mode. Inputs model the slot after signed-char narrowing, a length representable as a C++ int, and a successful internal STOP. The upper-bound check admits negative slots. The result covers the join command only. Subsequent indexing by a negative slot has no defined C++ trajectory.

Engine symbols `FullBallpark.formation_follow, FullBallpark._stop`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-726 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Formation.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Formation.ignored_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.refused_stops` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.joined_link` | propext, Classical.choice, Quot.sound |

<a id="claim-89"></a>

## Reservations and followers across a formation resize

**Ordered follower list and the leader’s occupied slot bits.** When switching between assigned formation IDs with valid lengths, shrinking stops exactly the formation followers whose signed slot is at or above the new length. For the mask results, all listed formation followers are live, point to this leader and hold valid slots within the old formation of at most 16 slots. The master clears the bits of the stopped ships and the client retains them. An unclaimed reservation still blocks allocation after shrinking and growing back, in the absence of intervening commands. Explicitly leaving a formation takes a separate reset path.

Engine symbols `FullBallpark.set_ball_formation, FullBallpark._stop, FullBallpark.reserve_formation_slot`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-726 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Formation.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Formation.stopped_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.resize_bit` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Formation.regrow_skips_reserved` | propext, Classical.choice, Quot.sound |

<a id="claim-90"></a>

## Which fixed obstacle wins an iterative search

**Real time comparisons and an encoded binary64 witness.** With an empty incumbent, the full unit window selects the first minimum among eligible nonnegative sphere hits. For a partial window the source compares the new fraction to the incumbent’s absolute time. The theorem characterizes exactly when that decision differs from comparing two absolute times. A kernel-evaluated binary64 sphere sweep on [0.5, 1] computes both winning IDs and times in the two enumeration orders.

Engine symbols `SpaceBallpark._next_collision_static, SpaceBallpark.collide_two_spheres`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

CPU replay — the scalar and pinned C++ selectors choose different obstacles when the same pair is enumerated in the opposite order. [Inputs and full-tick control](DISCREPANCIES.md#col-715-fixed-window)

| [Theorems](ExactBallparkProofs/Collision/SearchWindow.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.SearchWindow.full_tick_first_min` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SearchWindow.partial_tick_selection` | propext, Classical.choice, Quot.sound |

<a id="claim-91"></a>

## Static shape priority in a collision window

**Real-valued timestamps and source control flow.** Given eligible shape hits at or after the window start, the first minimum wins the static pass. A hit at the start preempts the fixed and free passes. Shape intersection tests and filtering supply the input list.

Engine symbols `FullBallpark._next_collision_static, FullBallpark._collision_response`. [Source](../src/exact_ballpark/full.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/SearchWindow.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.SearchWindow.static_priority` | propext, Classical.choice, Quot.sound |

<a id="claim-92"></a>

## Free-ball sweep intervals and ignored repeats

**Ordered real times and parameterized sweep operations.** Sorted partner history contributes exactly the interior cut times up to the smaller endpoint. The intervals cover the common window including its endpoints when it has positive length. On an otherwise winning, ignored repeat the clock advances while both start positions carry over. The second-step result states the resulting time for a subsequent accepted hit with an empty incumbent. CPU replay supplies a reversal in the partner history.

Engine symbols `SpaceBallpark._next_collision_free, CollisionBallProperties.ignore_collision`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/SearchWindow.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.SearchWindow.free_subintervals` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SearchWindow.ignored_segment` | propext, Classical.choice, Quot.sound |

<a id="claim-93"></a>

## The filing radius covers a regular drag trajectory

**Euclidean three-space and exact real arithmetic.** For a free ball with fixed positive effective mass and friction, constant acceleration, a correct full-tick decay cache and 0 ≤ t ≤ dt, the displacement from its starting position plus the body radius stays within its filing radius. Both the initial speed and the terminal speed for the applied force are at most speed_fraction times max_vel. This models the regular branch between impulses.

Engine symbols `SpaceBallpark.inflated_radius, SpaceBallpark.integrate`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-706 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/FilingRadius.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.radius_covers_trajectory` | propext, Classical.choice, Quot.sound |

<a id="claim-94"></a>

## Which neighboring cells receive a ball

**Binary64 face guards and integer cell coordinates.** For ordinary unclamped interior filings, the explicit insert calls give exactly the Cartesian product of the one or two choices on each axis, hence 1, 2, 4 or 8 distinct cells. The low face is tested first. Equal boundaries alone do not activate either strict guard. The separate 27-cell top-level case is outside this model.

Engine symbols `SpaceBallpark._ball_insert_in_boxes`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-706 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Filing.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Filing.face_edge_corner` | propext, Classical.choice, Quot.sound |

<a id="claim-95"></a>

## Which filed balls share a cell or an ancestor

**Integer coordinates and Cartesian filings.** For two ordinary filings, the tensor per-axis disjunction succeeds exactly when a cell of the finer ball has an ancestor among the cells of the coarser ball. Equal levels use the cell itself. ID, bubble, eligibility and special-entity filters are separate inputs.

Engine symbols `VecCollidePark._related`. [Source](../src/exact_ballpark/vec_collide.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-703 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Filing.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Filing.related_iff` | propext, Classical.choice, Quot.sound |

<a id="claim-96"></a>

## First visit to a candidate across overlapping boxes

**Ordered eligible visits and a handled-ID set.** After the ordinary eligibility filters, a walk emits each encountered ID once. Splitting the input at the first visit to an ID preserves the output preceding that visit and records that exact occurrence, even when later boxes contain the same ball.

Engine symbols `FullBallpark._nearby, FullBallpark._explode_children`. [Source](../src/exact_ballpark/full.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-703 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/CandidateOrder.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.CandidateOrder.first_visit_once` | propext, Classical.choice, Quot.sound |

<a id="claim-97"></a>

## Scalar and batched candidate order

**Lexicographic tree addresses and exact packed integer sort keys.** Given the same eligible ordinary interior filings, the scalar model traverses numbered boxes, near-to-far ancestors, descendants in child-before-parent order, then the box itself, with ascending IDs within a box. Keeping the first visit gives the entire same ID list as minimizing the packed address per ID and sorting by address and ID. Membership is equivalent to a related filing. Base-65 ordering agrees with lexicographic path digits at fixed length, and every valid packed address is below the tensor sentinel 2^62. Correspondence uses positive distinct IDs and levels 0–7. Edge clamps, huge top-level balls, static shapes, miniballs, eligibility filters and device execution are separate obligations.

Engine symbols `VecCollidePark._related, VecCollidePark._order_keys, VecCollidePark._sorted, FullBallpark._nearby`. [Source](../src/exact_ballpark/vec_collide.py) · [full.py](../src/exact_ballpark/full.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-703 passing (compiled code).

CPU correspondence — 54 ordered candidate lists and their active packed keys agree with scalar Python traversal and PyTorch on the CPU across all eight tree levels.

| [Theorems](ExactBallparkProofs/Space/CandidateOrder.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.CandidateOrder.ordered_candidates` | propext, Classical.choice, Quot.sound |

<a id="claim-98"></a>

## Radial tracking cancellation and scalar early returns

**Logical binary64 and Boolean control.** The fixed radial witness proves the negative squared angular speed and exponent input to the host power function. The mask theorem compares guarded returns word for word with the same ordinary output supplied. CPU execution checks pow and the tensor lane.

Engine symbols `FullBallpark.get_accuracy, vec_accuracy.get_accuracy`. [Source](../src/exact_ballpark/full.py) · [vec_accuracy.py](../src/exact_ballpark/vec_accuracy.py)

Recorded upstream evidence — CMB-702 passing (compiled code).

The scalar and PyTorch CPU checks cover 8 combat lanes, including the same-ball guard. For the radial case, the scalar and pinned C++ square-root and power inputs match the certified words. The engine returns infinite accuracy, 11 m and NaN angular speed. [Lane checks](scripts/arithmetic_checks.py) · [Pinned C++ replay](DISCREPANCIES.md#cmb-702-tracking)

| [Theorems](ExactBallparkProofs/Combat/Execution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.accuracy_overflow_input` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.accuracy_early_lanes` | propext, Classical.choice, Quot.sound |

<a id="claim-99"></a>

## Angular speed in a rotating observer frame

**Real vector arithmetic.** Positive separation, supplied transversal velocity, observer angular velocity and unit sight direction. The theorem identifies the cancelling tangential velocity.

Engine symbols `FullBallpark.relative_velocities, vec_accuracy.relative_velocities`. [Source](../src/exact_ballpark/full.py) · [vec_accuracy.py](../src/exact_ballpark/vec_accuracy.py)

Recorded upstream evidence — CMB-701 passing (compiled code).

| [Theorems](ExactBallparkProofs/Combat/Execution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.rotating_observer` | propext, Classical.choice, Quot.sound |

<a id="claim-100"></a>

## Normalized quaternion steering keeps thrust on its envelope

**Real arithmetic.** Nonzero quaternion, nonnegative requested acceleration and an arbitrary supplied cosine. The forward row comes from the normalized rotation. Binary32 quaternion conversion is outside this proof.

Engine symbols `q_len, FullBallpark._orient`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-728 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Heading.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.normalized_heading` | propext, Classical.choice, Quot.sound |

<a id="claim-101"></a>

## Stability of the sampled warp exit under a cruise-duration perturbation

**Real arithmetic.** Fixed acceleration and deceleration durations, positive peak, rate and maximum speed. Perturb only the cruise duration. Every eligible sample has the stated strict time margin at all three decision boundaries.

Engine symbols `ModeBallpark._warp_distance`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-709 passing (compiled code), TRV-713 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/ExitMargin.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.cruise_exit_margin` | propext, Classical.choice, Quot.sound |

<a id="claim-102"></a>

## Four-ball rebound and returning free-ball search

**Logical binary64.** Fixed one-dimensional fixtures. The four-ball loop replays source order and the absolute-time control with nine supplied exp words. The free search uses supplied endpoints and prior-contact state.

Engine symbols `SpaceBallpark._next_collision_static, SpaceBallpark._next_collision_free, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — COL-715 passing (compiled code).

The scalar replay matches the certified x and vx words for the engine and the earliest-hit control, and both words for the free-ball return. [Witness comparison](scripts/arithmetic_checks.py)

| [Theorems](ExactBallparkProofs/Collision/ReplayTick.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReplayTick.four_ball_endpoint` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReplayTick.returning_ball_words` | propext, Classical.choice, Quot.sound |

<a id="claim-103"></a>

## Error-free addition in the CPU FMA helper

**Binary nearest-even, 53 significand bits.** Every represented input pair, either sign and order. FloatLib supplies the unbounded-exponent format and nearest-even semantics. IEEE execution requires matching intermediate grids. This value identity does not distinguish signed zeros.

Engine symbols `two_sum`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/TwoSum.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.two_sum_exact` | propext, Classical.choice, Quot.sound |

<a id="claim-104"></a>

## A compensated strict cutoff for impact histories

**Binary nearest-even, 53 significand bits.** For representable times, tolerance and query, comparing the rounded sum and its signed residual retains exactly the ideal strict prefix. This verifies the proposed correction to the current rounded guard.

Engine symbols `CollisionBallProperties.find_index, two_sum`. [Source](../src/exact_ballpark/space.py) · [vec_exp.py](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/TwoSum.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.compensated_history_cutoff` | propext, Classical.choice, Quot.sound |

<a id="claim-105"></a>

## Error in the exceptional drag position update

**Binary nearest-even, 53 significand bits.** Positive effective mass and friction on the taken special branch. FloatLib supplies rounding bounds for each of the 16 operations, including both uses of the same stored squared friction. The error is relative to p + v t on one coordinate. IEEE application requires matching rounding grids.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-703 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Roundoff.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.Roundoff.special_position_error` | propext, Classical.choice, Quot.sound |

<a id="claim-106"></a>

## Exact promoted numerator for the impact limiter

**Binary32 dyadics and binary64 nearest-even.** Any finite binary32 input in signed significand and dyadic exponent form, including subnormals and zero. Promoting before both multiplications makes the numerator 25 × speed² exact. The current limiter evaluates it in binary32.

Engine symbols `SpaceBallpark._add_impact, FullBallpark._add_impact`. [Source](../src/exact_ballpark/space.py) · [full.py](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/Product.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.promoted_limiter_exact` | propext, Classical.choice, Quot.sound |

<a id="claim-107"></a>

## Remote followers become pending server stops

**Discrete queue and mode transitions.** A unique set of free IDs, a fixed system stamp and the initial queue. Missing or different-bubble targets select followers in ascending order. Only the first request can match the previous tail. Applying its Stop to a still-following ship clears target, owner and stamp. Notification callbacks and changes by intervening events are outside the projection.

Engine symbols `Actions.clean_up_followers, Actions.stop, Actions._add_to_system_history, SpaceBallpark.remote_followers`. [Source](../src/exact_ballpark/server.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — MOV-731 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done., TIME-710 passing (test vectors), TIME-712 implemented, not passing (test vectors). Destiny's published tests do not cover the pre-tick's first step (the follower clean-up). That part is checked only against the transcription of destiny's Python server layer, which has not been run itself..

| [Theorems](ExactBallparkProofs/Time/ServerLaws.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.ServerLaws.remote_pending_stops` | propext, Classical.choice, Quot.sound |

<a id="claim-108"></a>

## Force-field configuration through the candidate filter

**Discrete action and candidate semantics.** A present, uncloaked force field after both parent calls. The query excludes non-massive objects, checks harmonics and has passed the other candidate filters. With no enabled tag the field is non-massive.

Engine symbols `Actions.set_ball_harmonic, SpaceBallpark.set_harmonic, SpaceBallpark._nearby_balls`. [Source](../src/exact_ballpark/server.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — MOV-737 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done..

| [Theorems](ExactBallparkProofs/Time/ServerLaws.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.ServerLaws.configured_field_filter` | propext, Classical.choice, Quot.sound |

<a id="claim-109"></a>

## The seven-ball removal budget across scalar and batched lanes

**Integer time and ordered Boolean masks.** Outside evolution, same ascending ID order, exact signed tick differences and prefix counts within int64. Overdue and stale entries consume no early quota. Padding has pending false. Destiny's pointer visitation order is external.

Engine symbols `SpaceBallpark.bring_out_the_deads, DelayedRemoval.due`. [Source](../src/exact_ballpark/space.py) · [vec_space_removal.py](../src/exact_ballpark/vec_space_removal.py)

Recorded upstream evidence — TIME-713 implemented, not passing (compiled code). Destiny cleans up moribund balls in an order that follows heap addresses, which no port can reproduce. The reference uses ascending id..

| [Theorems](ExactBallparkProofs/Time/RemovalMasks.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.RemovalMasks.removal_prefix_masks` | propext, Classical.choice, Quot.sound |

<a id="claim-110"></a>

## Server packets merge at a single client history stamp

**Discrete packet and history transitions.** One intended recipient, one nonempty private and one nonempty bubble payload, in send order, at one admitted stamp. A group already present keeps its earlier events. The earlier and later history entries keep their payloads and wait flags. The model bypasses SetState resets and packet expansion.

Engine symbols `ParkUpdateBatcher.send_batch, merge_state_into_history`. [Source](../src/exact_ballpark/server.py) · [client.py](../src/exact_ballpark/client.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. That part is checked only against the transcription of destiny's Python server layer, which has not been run itself., TIME-715 passing (test vectors).

| [Theorems](ExactBallparkProofs/Space/MergeDelivery.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Batches.private_bubble_history` | propext, Classical.choice, Quot.sound |

<a id="claim-111"></a>

## Strict first sphere entry in a visibility ray

**Real Euclidean geometry.** Distinct sight endpoints, nonnegative obstacle radius and one eligible sphere. The entry must be the first surface intersection on the entire oriented line and strictly between the endpoints. Interior tangencies qualify. Starting inside the sphere does not block.

Engine symbols `SpaceBallpark.check_visibility`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-709 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/Occlusion.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.ray_first_entry` | propext, Classical.choice, Quot.sound |

<a id="claim-112"></a>

## Reconstruction and residual scale of Veltkamp's split

**Binary nearest-even, 53 significand bits.** Every represented operand of magnitude below 2⁹⁹⁶. The actual splitter is 2²⁷+1. The two rounded outputs sum exactly to the operand and the low component obeys an explicit magnitude bound. This is a value model with unbounded exponents.

Engine symbols `split`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/Product.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.split_reconstructs` | propext, Classical.choice, Quot.sound |

<a id="claim-113"></a>

## Dekker's product with certified partial widths

**Binary nearest-even, 53 significand bits.** Both inputs satisfy the split range. Four exact partial products and three intermediate differences admit 53-bit dyadic significands. The theorem derives the final residual, including its representability, from those conditions. A supplied cached split must match splitting the operand.

Engine symbols `split, two_product`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/Product.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.dekker_product_exact` | propext, Classical.choice, Quot.sound |

<a id="claim-114"></a>

## Odd rounding from a signed residual and the stored low bit

**Binary nearest-even and signed word steps on a specified grid.** The nearest result is decoded on the adjacent grid containing the exact sum, with an even encoding offset. Exact values stay fixed. For an inexact sum, the source parity and sign tests select the odd neighbor. The model concerns finite values.

Engine symbols `two_sum, add_odd`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/OddFma.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.add_odd_grid` | propext, Classical.choice, Quot.sound |

<a id="claim-115"></a>

## One rounding for the composed software FMA

**Composition over two binary grids.** Product-width hypotheses hold, both split operands are in range, the low grid is at least two bits finer, the high sum is an even multiple of that unit, and the ideal result and final add input share a coarse grid. Signed zero, overflow, subnormal underflow and other binade configurations require separate results.

Engine symbols `fma, two_product, two_sum, add_odd`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

CPU comparison on the registered source — 68 software FMAs (67 inside the grid conditions, 14 with inexact low sums). The FMA oracle is CPython math.fma. Exact dyadic arithmetic checks the product widths and both rounding grids for each sampled call. [Probe](scripts/arithmetic_checks.py)

| [Theorems](ExactBallparkProofs/Numerics/OddFma.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.Compensated.fma_single_round` | propext, Classical.choice, Quot.sound |

<a id="claim-116"></a>

## Capsule rejection, radial distance and first barrel contact

**Real Euclidean geometry.** Straight sweep, nonzero capsule axis and nonnegative combined radius for first contact. The bounding test uses a supplied enclosing radius. The barrel result applies when the exterior quadratic branch is reached, its earlier root is accepted and its projected foot lies on the finite axis. Contact with a cap outside this branch and the subsequent response require separate reasoning.

Engine symbols `FullBallpark._capsule_check, Capsule.bounding_radius`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/Capsule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.Capsule.axis_distance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Capsule.bounding_rejection` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Capsule.cylinder_quadratic` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.Capsule.first_barrel_contact` | propext, Classical.choice, Quot.sound |

<a id="claim-117"></a>

## A sub-micrometre approach missed by the capsule cutoff

**Binary64 guards and exact straight-chord geometry.** The capsule is aligned with z and both radii are one half. The witness starts and ends 2⁻²⁰ m on opposite sides of the combined surface. Its exact first contact is at half a tick. The projected source guards reach the near-parallel rejection with stored binary64 words.

Engine symbols `FullBallpark._capsule_check`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-712 passing (compiled code).

Scalar Python and the pinned Destiny C++ capsule check miss the shallow crossing and return half a tick on the faster control. [Capsule replay](DISCREPANCIES.md#col-712-shallow-crossing)

| [Theorems](ExactBallparkProofs/Collision/Capsule.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.Capsule.shallow_crossing_missed` | propext, Classical.choice, Quot.sound |

<a id="claim-118"></a>

## Sorted client history and stable payload order

**Discrete client history.** Existing history and the incoming groups each have strictly increasing stamps. The client has already grouped a nonempty raw packet by timestamp, retaining its original order within each group. The resulting ordered stamp set is their union and each payload is the old list followed by the incoming list, including duplicates.

Engine symbols `merge_state_into_history`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

Lean and the scalar client agree on the complete ordered payloads and wait flags for 112 mixed-stamp histories. [CPU probe](scripts/mixed_history_checks.py)

| [Theorems](ExactBallparkProofs/Time/MixedHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.MixedHistory.merged_stamps` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.MixedHistory.merged_payload` | propext, Classical.choice, Quot.sound |

<a id="claim-119"></a>

## Waiting, dispatch and reset for mixed-stamp packets

**Discrete packet and clock transitions.** The append case starts with empty history. One equal-stamp reply clears its group and later groups keep waiting. Tick dispatch assumes handlers return, represents time after each dispatch by a supplied clock function and advances a successful extra parent evolve by one. Reset is recognized by the first raw event. Admission of an ordinary packet depends on that first stamp before grouping.

Engine symbols `merge_state_into_history, Ticker._flush_state, Ticker.do_pre_tick`. [Source](../src/exact_ballpark/client.py)

Recorded upstream evidence — TIME-715 passing (test vectors).

The pinned Destiny Python client reproduces the extra wait and the reset-order trace. [Packet replays](DISCREPANCIES.md#time-715-extra-wait)

| [Theorems](ExactBallparkProofs/Time/MixedHistory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.MixedHistory.waiting_tail` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.MixedHistory.drain_frontier` | propext, Quot.sound |
| `ExactBallpark.Time.MixedHistory.reset_packet_order` | propext, Quot.sound |

<a id="claim-120"></a>

## Exponential table addressing and scale words

**Signed integers, 64-bit words and real powers of two.** The table has 128 tail/scale pairs. Normal reconstruction assumes a 52-bit fraction for its positive scale and an in-range result exponent. The two special shifts and their sign-bit dispatch use the proved index range. Their powers of two cancel over the reals before the polynomial, overflow and subnormal result handling.

Engine symbols `exp_table, exp`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

The CPU exponential reads every tail and scale slot on 640 non-tiny inputs. Each returned word matches a 100-digit decimal reference rounded to binary64. [Sampling checks](scripts/sampling_checks.py)

| [Theorems](ExactBallparkProofs/Numerics/ExpReduction.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.ExpReduction.table_address` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.ExpReduction.scale_reconstruction` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.ExpReduction.special_rescaling` | propext, Classical.choice, Quot.sound |

<a id="claim-121"></a>

## Exponential reduction bounds and tiny-input rounding

**FloatLib binary nearest-even with 53 significand bits.** The reduction uses the exact stored INVLN2N constant on sanitized inputs of magnitude at most 1024. Rounded multiply and half addition followed by truncation leave room for the signed shift product. For |x| < 2⁻⁵⁴ the normal result of rounding x+1 and of rounding exp(x) is exactly one. This covers that branch of the exponential.

Engine symbols `exp, INVLN2N`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

| [Theorems](ExactBallparkProofs/Numerics/ExpReduction.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.ExpReduction.reduction_index_bound` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Numerics.ExpReduction.tiny_correct_round` | propext, Classical.choice, Quot.sound |

<a id="claim-122"></a>

## Speed continuity during warp acceleration

**Real trajectory and reported speed.** During the acceleration branch the scalar trajectory has derivative a·exp(a·t). The reported speed agrees with that derivative exactly when the incoming speed is at most the analytic speed.

Engine symbols `ModeBallpark._warp_distance`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-711 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Sampling.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.Sampling.acceleration_sample` | propext, Classical.choice, Quot.sound |

<a id="claim-123"></a>

## The last second of warp deceleration

**Real local deceleration coordinate.** Positive deceleration and peak speed. For a sample δ seconds past nominal deceleration, with 0 ≤ δ < 1, the signed remainder is exp(−bδ)−1 metres and the speed is b·exp(−bδ). This is conditional on the sampler reaching this phase before a prior dropout.

Engine symbols `ModeBallpark.warp_constants, ModeBallpark._warp_distance`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — TRV-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Sampling.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.Sampling.deceleration_extension` | propext, Classical.choice, Quot.sound |

<a id="claim-124"></a>

## Direction after sampling either side of the warp exit

**Real Euclidean geometry.** The current position differs from the exit, and the branch places the ship on the normalized sightline at a signed remainder. Its distance is the remainder's absolute value. Recomputing the sightline preserves, zeroes or reverses the direction according to the sign.

Engine symbols `ModeBallpark._warp_distance, _normalized`. [Source](../src/exact_ballpark/modes.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TRV-711 passing (compiled code), TRV-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Warp/Sampling.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.Sampling.placement_ray` | propext, Classical.choice, Quot.sound |

<a id="claim-125"></a>

## Ordered range results for ID and coordinate queries

**Ordered registry lookup and real Euclidean distance.** Successful argument parsing and a resolved bubble with one object per dictionary key. The scan retains live, eligible candidates in traversal order at squared center distance at most (range + candidate radius)². At the source ball's coordinates, when both overloads resolve the same bubble, the ID query removes exactly that source. Vacant keys contribute no result.

Engine symbols `SpaceBallpark.ball_ids_in_range`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-710 passing (compiled code).

| [Theorems](ExactBallparkProofs/Space/RangeScan.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.RangeScan.ordered_scan` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.RangeScan.query_by_id` | propext, Classical.choice, Quot.sound |
