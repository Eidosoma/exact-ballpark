# Lean claim register

420 kernel-checked theorems have claims linked to 105 of the 105 catalogue rules. A link covers the scope stated below.

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
| MOV-701 | Motion is viscous with friction k = 1e6 and effective mass = mass x agility | [refresh_positive](#claim-1), [source_regular, velocity_hasDerivAt](#claim-3), [decay_bounds, velocity_contraction, speed_envelope, velocity_tends_to_terminal](#claim-5), [extracted_integrator](#claim-49), [regular_trajectory_error](#claim-140), [five_second_coast](#claim-141), [velocity3_hasDerivAt, drag_energy_rate, drag_work_energy, coast_energy, coast_energy_strictAnti](#claim-142), [drag_exponent_decay, refresh_factor_units](#claim-143), [trajectory_unique](#claim-146), [acceleration_bound](#claim-147), [manoeuvre_work_energy](#claim-169) |
| MOV-702 | Closed-form exponential integration of position and velocity | [source_regular, flow_zero, decay_hasDerivAt, position_hasDerivAt, velocity_hasDerivAt, flow_compose](#claim-3), [integrate_zero, integrate_regular, integrate_velocity](#claim-4), [decay_bounds, velocity_contraction, speed_envelope, velocity_tends_to_terminal](#claim-5), [extracted_integrator](#claim-49), [full_tick_velocity_bits](#claim-50), [zero_duration_bits](#claim-52), [coordinate_words](#claim-126), [regular_step_error](#claim-128), [tick_lookup](#claim-137), [regular_trajectory_error](#claim-140), [five_second_coast](#claim-141), [extracted_velocity3, velocity3_hasDerivAt, drag_energy_rate, drag_work_energy, coast_energy, coast_energy_strictAnti](#claim-142), [pos_zero, vel_zero, position_derivative, extracted_position, trajectory_unique](#claim-146), [acceleration_bound, flight_chord_deviation](#claim-147), [pair_no_contact, pair_clearance](#claim-148), [position_error_from_axes, rounded_endpoint_budget](#claim-149), [turning_overlap, goto_sweep_inputs, turning_clearance](#claim-150), [decay_enclosed, weights_monotone, position_enclosed](#claim-152), [ordinary_throughout, squared_distance_enclosed](#claim-153), [turning_window, clear_control, two_turning_bodies](#claim-154), [ordinary_inputs, incoming](#claim-155), [restart_state, remainder](#claim-156), [half_decay, positive_gap, shared_drag_separation](#claim-157), [replay_words](#claim-158), [ideal_endpoint_enclosed, endpoint_error](#claim-159), [manoeuvre_work_energy](#claim-169), [axis_position, rest_point](#claim-170) |
| MOV-703 | Near-zero friction switches to a branch where acceleration does not move the ball | [special_position_eq, integrate_special](#claim-4), [extracted_integrator](#claim-49), [special_rounding_witness](#claim-50), [special_position_error](#claim-105), [coordinate_words](#claim-126), [tick_lookup](#claim-137), [extracted_velocity3](#claim-142), [ordinary_throughout](#claim-153) |
| MOV-704 | Each ball caches its one-tick decay factor exp(-k dt / (mass agility)) | [refresh_positive, stepFactor_eq](#claim-1), [addBall_valid, update_valid, history_valid, velocity_after_updates](#claim-2), [integrate_regular, integrate_velocity](#claim-4), [extracted_integrator](#claim-49), [full_tick_velocity_bits](#claim-50), [coordinate_words](#claim-126), [regular_step_error](#claim-128), [edit_preserves_cache](#claim-136), [trace_refines](#claim-138), [trace_preserves_cache, cache_from_empty](#claim-139), [regular_trajectory_error](#claim-140), [five_second_coast](#claim-141), [extracted_velocity3](#claim-142), [drag_exponent_decay, refresh_factor_units](#claim-143), [extracted_position](#claim-146), [rounded_endpoint_budget](#claim-149), [goto_sweep_inputs](#claim-150), [decay_enclosed](#claim-152), [replay_words](#claim-158) |
| MOV-705 | Maximum thrust is k x speedFraction x maxVel / (mass x agility) | [goto_terminal_bound, goto_speed_bound](#claim-6), [tick_lookup](#claim-137), [goto_sweep_inputs](#claim-150), [ordinary_inputs](#claim-155), [clipped_orbit_speed](#claim-172) |
| MOV-706 | Goto thrust is full thrust toward the target with an (L/D)^4 homing taper | [taper_bounds, goto_terminal_bound, goto_speed_bound](#claim-6), [tick_lookup](#claim-137), [goto_sweep_inputs](#claim-150), [goto_first_commit, goto_moves_at_five](#claim-165) |
| MOV-707 | Goto has no braking term, so a ship overshoots its goto point | [goto_overshoot](#claim-26) |
| MOV-708 | GotoPoint and GotoDirection set a goto point, cancel follows and default speed | [goto_cancels](#claim-28), [edit_preserves_cache](#claim-136), [trace_refines](#claim-138), [goto_sweep_inputs](#claim-150) |
| MOV-709 | Stop cancels follow-type modes and leaves the ball coasting to rest | [stop_clears](#claim-28), [edit_preserves_cache](#claim-136), [trace_refines](#claim-138), [axis_velocity, axis_position, rest_point, stop_loss_budget](#claim-170) |
| MOV-710 | STOP mode damps the vertical velocity by a hard-coded 0.93/1.07 per tick | [stop_speed_sq](#claim-7), [tick_lookup](#claim-137), [trace_refines](#claim-138), [axis_velocity, rest_point, stop_loss_budget, split_stop](#claim-170) |
| MOV-711 | AddBall clamps physical properties and starts every ball in STOP | [addBall_valid](#claim-2), [clamp_negative, clamp_other, clamp_nan, clamp_negative_zero, signed_zeros_distinct](#claim-8), [edit_preserves_cache](#claim-136), [run_accepts_iff, trace_refines](#claim-138) |
| MOV-712 | Radius, max speed, agility, speed fraction and follow range are 32-bit floats | [agility_accepts_iff, agility_half_subnormal, agility_above_half_subnormal](#claim-11), [edit_preserves_cache](#claim-136), [run_accepts_iff, trace_refines](#claim-138) |
| MOV-713 | Property setters validate their inputs | [agility_accepts_iff, agility_half_subnormal, agility_above_half_subnormal](#claim-11) |
| MOV-714 | Follow (keep at range) aims at the point at the desired surface range | [unit_norm, follow_range](#claim-27) |
| MOV-715 | Follow and orbit need a live, uncloaked target in the same bubble | [follow_iff](#claim-28) |
| MOV-716 | Orbit uses the old-style law unless the global useNewOrbit setting is on | [orbit_switch](#claim-32) |
| MOV-717 | Old-style orbit plane precesses with tick count and ball ID | [orbit_id_period](#claim-27) |
| MOV-718 | Old-style orbit blends radial and tangent thrust with a 40000 m^2 Gaussian | [orbit_weight_bounds](#claim-27), [weight_order, weight_cutoff, outer_thrust_norm](#claim-171) |
| MOV-719 | New-style orbit plane follows the orbiter's velocity around the predicted target | [predicted_direction](#claim-33) |
| MOV-720 | New-style orbit steers along a logarithmic spiral toward the orbit radius | [spiral_radius](#claim-27) |
| MOV-721 | GotoThrustFollow blends reach-the-point and match-the-velocity accelerations | [solve_guidance](#claim-27), [guidance_denominators, blend_residuals, clipped_orbit_speed](#claim-172) |
| MOV-722 | OrbitThrust brakes toward a point and switches off below 10% of max acceleration | [orbit_brake_cap](#claim-27) |
| MOV-723 | Missiles fly straight for 800 ms, then pursue a lead point at full thrust | [pursuit_center](#claim-34) |
| MOV-724 | MissileFollow aims a missile at the target centre at full speed fraction | [pursuit_center](#claim-34), [pursuit_offset_witness](#claim-55) |
| MOV-725 | A launched missile starts at the launcher with launcher velocity plus 150 m/s | [launch_speeds](#claim-34) |
| MOV-726 | Formation followers aim at the leader's rotated slot offset | [formation_distance](#claim-35), [reserve_first, reserve_failure, reserve_release, release_empty](#claim-87), [ignored_iff, refused_stops, joined_link](#claim-88), [stopped_iff, resize_bit, regrow_skips_reserved](#claim-89), [rotateSlot_norm, rotated_formation_distance](#claim-163) |
| MOV-727 | The agility turn-angle cap is disabled, so thrust can point anywhere | [cap_preserves](#claim-32) |
| MOV-728 | Optional dynamical orientation steers thrust along the ship's heading | [orientation_bound](#claim-35), [normalized_heading](#claim-100) |
| MOV-729 | Without dynamical orientation, heading is a smoothed function of velocity | [smooth_error](#claim-35) |
| MOV-730 | Followers are released when their target warps, cloaks or is removed | [fly_on_iff](#claim-28) |
| MOV-731 | Followers of a ball in another bubble are stopped at the next pre-tick | [remote_iff](#claim-40), [remote_pending_stops](#claim-107), [stopped_before_bubbles](#claim-132) |
| MOV-732 | Only free balls move; fixing a ball stops it and spawns its collision parts | [fixed_capsule](#claim-47) |
| MOV-733 | Ball flags free, massive, global, interactive and space junk | [flags_roundtrip](#claim-48) |
| MOV-734 | Thirteen ball modes; six compute thrust, STOP damps, the rest are inert | [thrust_modes](#claim-32) |
| MOV-735 | Mode changes raise DoModeChange; leaving warp raises OnExitWarp | [exit_iff, exiting_order](#claim-28) |
| MOV-736 | A troll ball turns rigid after a delay | [troll_due](#claim-36) |
| MOV-737 | Force fields let balls with a matching harmonic, corporation or alliance through | [field_iff](#claim-47), [configured_field_filter](#claim-108), [configured_membership](#claim-164) |
| MOV-738 | Destiny's exp is the C library's; exact on tensors and the GPU for glibc 2.34 on aarch64 | [exp_tiny_iff, exp_large_iff](#claim-22), [two_sum_exact](#claim-103), [split_reconstructs](#claim-112), [dekker_product_exact](#claim-113), [add_odd_grid](#claim-114), [fma_single_round](#claim-115), [table_address, scale_reconstruction, special_rescaling](#claim-120), [reduction_index_bound, tiny_correct_round](#claim-121), [first_reduction_exact](#claim-134) |
| MOV-739 | Destiny's float functions are the C library's; exact for glibc 2.34 on aarch64 | [powf_signaling_iff](#claim-23) |
| MOV-740 | Destiny's double sin, cos and log are the C library's; exact for glibc 2.34 on aarch64 | [log_near_one_iff](#claim-24) |
| MOV-741 | Destiny's double atan2 and asin are the C library's; exact for glibc 2.34 on aarch64 | [atan2_sign](#claim-25) |
| TRV-701 | WarpTo to a point closer than 100 km becomes a goto | [short_iff](#claim-12) |
| TRV-702 | Repeating a warp command to the same point does nothing | [repeated_warp](#claim-12) |
| TRV-703 | WarpTo stops the ship and enters the WARP alignment state | [long_starts_alignment](#claim-12), [accepted_warp_commits, dropout_handoff](#claim-162) |
| TRV-704 | Warp factor is the top warp speed in milli-AU per second | [rate_bounds](#claim-14), [source_constants](#claim-31) |
| TRV-705 | While aligning for warp the ship thrusts toward the destination like a goto | [waiting_ticks](#claim-37) |
| TRV-706 | Warp starts at 75% of max speed within about 8 degrees, or after 180 ticks | [alignment_after_waits](#claim-13), [accepted_warp_commits, dropout_handoff](#claim-162) |
| TRV-707 | Warp activation drops collisions and offsets the exit point by minRange | [exit_distance](#claim-13) |
| TRV-708 | Warp speed, acceleration and deceleration rates scale with the warp factor | [rate_bounds](#claim-14), [source_constants](#claim-31) |
| TRV-709 | Short warps lower the top speed so the cruise phase has zero length | [short_cruise_zero](#claim-14), [source_constants](#claim-31), [positive_cruise_iff](#claim-72), [short_cruise_words](#claim-74), [cruise_exit_margin](#claim-101) |
| TRV-710 | Warp phase durations and distances | [cruise_nonnegative, phase_distances_sum](#claim-14), [source_constants](#claim-31) |
| TRV-711 | Warp proper follows an analytic exponential-then-cruise path | [phase_joins](#claim-15), [acceleration_sample](#claim-122), [placement_ray](#claim-124), [accepted_warp_commits, dropout_handoff](#claim-162) |
| TRV-712 | Warp deceleration decays exponentially | [phase_decel_iff, decel_velocity](#claim-15), [deceleration_extension](#claim-123), [placement_ray](#claim-124) |
| TRV-713 | Ships drop out of warp below min(maxVel/2, 100 m/s) | [dropout_iff](#claim-15), [dropout_time_iff](#claim-73), [cruise_exit_margin](#claim-101), [accepted_warp_commits, dropout_handoff](#claim-162) |
| TRV-714 | The exit tick integrates one extra step from the drop-out point | [exit_regular, exit_special](#claim-37), [accepted_warp_commits, dropout_handoff](#claim-162) |
| TRV-715 | EntityWarpIn starts a warp already in progress | [entity_elapsed](#claim-37) |
| TRV-716 | The ballpark warpSpeed attribute is not used by the warp model | [source_constants](#claim-31) |
| COL-701 | Two collision methods; the per-ball gradient method is the default | [gradient_iff](#claim-43), [gradient_winner](#claim-130) |
| COL-702 | Only free, massive, live balls respond; non-massive balls pass through | [gradient_iff](#claim-43) |
| COL-703 | Collision candidates exclude cloaked, non-massive, missile and other-bubble balls | [missile_target, ordinary_candidate](#claim-43), [related_iff](#claim-95), [first_visit_once](#claim-96), [ordered_candidates](#claim-97), [sweep_candidate](#claim-129) |
| COL-704 | Missiles ignore owners; non-junk balls ignore space junk | [owner_junk_skip](#claim-43) |
| COL-705 | Time of impact assumes straight paths over the tick | [gap_quadratic, contact_correct](#claim-9), [long_chord_clear, long_chord_real_miss, long_chord_rounded_hit](#claim-57), [finite_negative_iff, discriminant_error, certified_sign, certified_miss](#claim-58), [decoded_gap, certified_chord_miss, horizontal_margin_examples, positive_outside_tick](#claim-59), [entry_comparisons, timing_sound, tick_boundary_examples, late_root_rounded_hit](#claim-60), [tangent_contact_iff, tangent_rounded_miss, tangent_tick_boundaries](#claim-61), [detected_rebound](#claim-127), [certified_chord_hit](#claim-131), [chord_deviation, flight_chord_deviation](#claim-147), [pair_no_contact, pair_clearance](#claim-148), [certified_clearance](#claim-149), [turning_overlap, turning_words, turning_clearance](#claim-150), [exists_first, initial_contact, first_unique, first_on_surface, first_in_window](#claim-151), [squared_distance_enclosed, prefix_clear, whole_interval_margin, verified_result, classify_sound](#claim-153), [turning_window, clear_control, two_turning_bodies, moving_pair_window, late_window_rejected](#claim-154), [incoming_distance, exact_first_contact](#claim-155), [no_recontact, only_contact](#claim-157), [replay_words](#claim-158), [excluded_solid](#claim-174) |
| COL-706 | Collision inertia is mass x agility, fixed balls 1e34, velocity exchange base mass | [elastic_exchange](#claim-17), [raw_impulse](#claim-156) |
| COL-707 | Hitting a fixed ball reflects the normal velocity elastically | [reflection](#claim-17), [scaled_reflection](#claim-51) |
| COL-708 | Two free balls exchange normal velocity as a 1-D elastic collision | [elastic_exchange](#claim-17) |
| COL-709 | Overlapping balls are pushed apart by the overlap plus 1 m, split by inertia | [overlap_split](#claim-18), [overlap_clearance, overlap_inertia_center](#claim-64) |
| COL-710 | Free-free overlaps are re-tested from separated positions, lighter ball boosted | [lighter_boost](#claim-45), [overlap_clearance, overlap_inertia_center](#claim-64) |
| COL-711 | The applied collision response is 0.85 of the computed acceleration | [response_tie](#claim-45), [scaled_reflection](#claim-51), [gradient_winner](#claim-130) |
| COL-712 | Static capsules and boxes reflect or eject balls with the same 0.85 response | [excluded_chord](#claim-10), [scaled_reflection, static_branches](#claim-51), [support_geometry, crossing_iff, parallel_sweep, face_membership](#claim-71), [axis_distance, bounding_rejection, cylinder_quadratic, first_barrel_contact](#claim-116), [shallow_crossing_missed](#claim-117), [grown_bounds, axis_extents, solid_enclosed, excluded_solid](#claim-174) |
| COL-713 | Fixed balls carry axis-aligned miniballs, minicapsules and miniboxes | [fixed_capsule](#claim-47), [axis_extents, solid_enclosed, diagonal_radius](#claim-174) |
| COL-714 | Iterative collisions re-examine colliding balls for up to 20 passes | [iteration_cap](#claim-44), [remove_first_matching_impact, remove_reset_at_zero, finish_window](#claim-68) |
| COL-715 | Iterative search order is static, then fixed, then higher-ID free balls | [unique_pair](#claim-45), [ignore_recent_colliders, index_prefix, position_from_prefix](#claim-69), [rounded_history_lookup](#claim-70), [full_tick_first_min, partial_tick_selection](#claim-90), [static_priority](#claim-91), [free_subintervals, ignored_segment](#claim-92), [four_ball_endpoint, returning_ball_words](#claim-102), [compensated_history_cutoff](#claim-104), [detected_rebound](#claim-127), [contact_time_value, contact_time_lt_iff, half_tick_absolute_selection](#claim-144), [earlier64_eq, half_tick_selection_words](#claim-145), [disjoint_first_contacts](#claim-151), [obstacle_order](#claim-154), [replay_words](#claim-158) |
| COL-716 | Iterative impacts on static objects are perfectly elastic impulses | [reflection](#claim-17), [static_velocity_components, start_contact_acceleration](#claim-62) |
| COL-717 | Iterative impacts between free balls are elastic with base masses | [iterative_exchange](#claim-17), [start_contact_acceleration](#claim-62), [pair_separates, pair_preserves_tangent, pair_momentum, pair_kinetic_energy](#claim-63), [raw_impulse, restart_state, momentum_change](#claim-156), [replay_words](#claim-158), [endpoint_error](#claim-159), [from_pair_velocity, momentum_iff, separates_iff, common_fraction_loss](#claim-173) |
| COL-718 | Iterative velocity changes above 5 x maxVel are scaled by the squared ratio | [limitImpact_norm](#claim-18), [limited_speed_bound](#claim-65), [rounded_limiter_exceeds_bound](#claim-66), [insert_cutoff_and_retry, insert_after_predecessor, insert_before_all](#claim-67), [promoted_limiter_exact](#claim-106), [detected_rebound](#claim-127), [applied_impulse, momentum_change](#claim-156), [limited_closing_at_end](#claim-157), [replay_words](#claim-158), [endpoint_error](#claim-159), [limit_normal, from_pair_velocity, momentum_iff, separates_iff, common_fraction_loss, asymmetric_witness](#claim-173) |
| COL-719 | Collision events are raised only for missiles; orphaned missiles hit themselves | [orphan_with_owner](#claim-44), [dispatch_moving, dispatch_silent](#claim-54) |
| SEN-701 | Proximity sensors check every period seconds, default 2.0 s | [sensor_schedule](#claim-20), [first_due_poll](#claim-77), [timed_crossing](#claim-160) |
| SEN-702 | A ball is inside a sensor when centre distance <= range + both radii | [sensor_inside, crossing, repeated_sample](#claim-20), [vanished_target, present_target](#claim-78), [timed_crossing](#claim-160) |
| SEN-703 | Target tracking fires when the centre distance crosses range + own radius | [tracking_boundary, crossing, repeated_sample](#claim-20), [trace_channels, edge_parity, boundary_words](#claim-175) |
| SEN-704 | Notification range reports arrival at the goto point or follow distance | [follow_band_iff, crossing](#claim-20), [trace_channels, edge_parity, notification_rearms, stop_exit_once, boundary_words](#claim-175) |
| SEN-705 | Space is partitioned into 8 box levels from 7,864,320 m down to 480 m | [finest_width](#claim-21) |
| SEN-706 | A ball is filed at the finest level whose box is at least 3x its inflated radius | [finest_level](#claim-21), [radius_covers_trajectory](#claim-93), [face_edge_corner](#claim-94), [sweep_candidate](#claim-129) |
| SEN-707 | Bubbles are clusters of top-level boxes joined through their 26 neighbours | [neighbor_offsets, neighbor_iff](#claim-46), [bubble_component, corner_neighbors](#claim-56) |
| SEN-708 | Balls interact only within their bubble | [ordinary_candidate, missile_target](#claim-43) |
| SEN-709 | Line of sight is blocked by the first massive, uncloaked sphere on the segment | [visibility_entry, first_occluder](#claim-46), [ray_first_entry](#claim-111) |
| SEN-710 | Range queries count balls whose centre is within range plus their own radius | [range_abs](#claim-46), [ordered_scan, query_by_id](#claim-125) |
| SEN-711 | Cloaking makes a ball non-massive and drops its followers | [cloak_effect](#claim-47), [warp_and_sensor, erase_and_reenter](#claim-79) |
| SEN-712 | Clients receive each tick's events only for balls in their bubble | [recipient_split](#claim-42), [bubble_destination, character_destination](#claim-80), [pilot_add_iff, pilot_remove_iff, pilot_disjoint](#claim-81), [single_recipient, dual_recipient](#claim-82), [character_wait, client_trace](#claim-83), [private_bubble_history](#claim-110), [delayed_private_flush](#claim-133), [callback_recipients, crossing_delivered](#claim-161) |
| TIME-701 | A ballpark tick is 1000 ms by default and dt = tickInterval/1000 s | [default_tick_seconds](#claim-38) |
| TIME-702 | OnTick evolves once per whole tick interval elapsed and carries the remainder | [rounds_remainder, early, ordinary_tick](#claim-16), [clock_window, next_client_deadline](#claim-75), [masked_world_trace, poll_replays_single](#claim-168), [repeat_early_poll](#claim-176) |
| TIME-703 | The server drops a backlog of more than 5 ticks and evolves only one | [master_resync](#claim-16), [master_boundary](#claim-76) |
| TIME-704 | The first OnTick evolves exactly once without tick events | [first_due](#claim-16) |
| TIME-705 | Each tick runs DoPreTick, Evolve, DoPostTick; dead balls are removed before | [rounds_trace, ordinary_tick](#claim-16), [repeat_early_poll](#claim-176) |
| TIME-706 | Evolve computes all thrusts, then collisions and integration, then commits | [phase_ids](#claim-39), [schedule_monotone](#claim-53), [tick_lookup](#claim-137), [trace_refines, trace_append](#claim-138), [five_second_coast](#claim-141), [masked_world_trace, poll_replays_single](#claim-168) |
| TIME-707 | Free balls are always processed in ascending ID order | [phase_ordered](#claim-39), [tick_order](#claim-137) |
| TIME-708 | Clients display ball positions two ticks behind their clock | [render_two_ticks](#claim-16) |
| TIME-709 | Server movement commands are queued and applied at the next pre-tick | [last_enqueued, distinct_actions_kept](#claim-29), [goto_first_commit, goto_moves_at_five](#claim-165), [rewind_goto_same_stamp](#claim-166) |
| TIME-710 | An action identical to the last queued action is dropped | [duplicate_idempotent](#claim-29), [remote_pending_stops](#claim-107), [stopped_before_bubbles](#claim-132) |
| TIME-711 | Ball removal through Actions is immediate, not queued | [removal_calls](#claim-41) |
| TIME-712 | Pre-tick order is follower cleanup, stamp, flush, apply, bubbles, cloaks | [pretick_calls](#claim-41), [remote_pending_stops](#claim-107), [stopped_before_bubbles](#claim-132) |
| TIME-713 | Removed balls linger as moribund and are cleaned up between ticks | [cleanup_budget](#claim-40), [removal_prefix_masks](#claim-109), [row_decision, early_ids, removal_count, cohort_deadline, repeat_early_poll](#claim-176) |
| TIME-714 | Balls in dead bubbles are not evolved on the server | [dead_iff](#claim-40) |
| TIME-715 | A client applies each server update at its stamp, waiting, running ahead or rewinding to it | [sync_reaches_stamp, wait_iff](#claim-30), [eleventh_store, stores_bounded, post_deadline](#claim-84), [rebase_contents](#claim-85), [rewind_stream, sync_failure](#claim-86), [private_bubble_history](#claim-110), [merged_stamps, merged_payload](#claim-118), [waiting_tail, drain_frontier, reset_packet_order](#claim-119), [delayed_private_flush](#claim-133), [rewind_goto_same_stamp](#claim-166) |
| CMB-701 | Angular velocity is transversal speed over centre distance | [angular_agrees](#claim-19), [rotating_observer](#claim-99), [radial_invariance](#claim-177) |
| CMB-702 | Destiny's accuracy helper is 2 to the power of tracking and falloff terms | [accuracy_bounds, optimal_stationary](#claim-19), [accuracy_overflow_input, accuracy_early_lanes](#claim-98), [sideways_margin](#claim-135), [tracking_input_nonnegative, tracking_margin_controls](#claim-167), [chance_factors, range_order, half_chance_iff, falloff_landmarks](#claim-177) |

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

Recorded upstream evidence — TIME-711 passing (test vectors), TIME-712 implemented, not passing (test vectors). Destiny's published tests do not cover the pre-tick's first step (the follower clean-up). The pre-tick follower clean-up is checked against the transcription..

| [Theorems](ExactBallparkProofs/Time/Server.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.removal_calls` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.pretick_calls` | None |

<a id="claim-42"></a>

## Bubble update recipients

**Sets of integer IDs.** The union of interested clients across a bubble’s interactive balls splits exhaustively and disjointly according to character history. Separate singlecasts and network delivery are outside this claim.

Engine symbols `ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

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

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

| [Theorems](ExactBallparkProofs/Space/Distribution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Distribution.bubble_destination` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.character_destination` | propext, Classical.choice, Quot.sound |

<a id="claim-81"></a>

## Additions and removals for a pilot moving between bubbles

**Per-ID projection of finite bubble dictionaries.** For a pilot on the changed-bubble branch, the two membership equivalences cover every action code, the local-ID cutoff, optional ball flags, keys present in both snapshots and unavailable bubbles. They preserve the source’s different global/cloak checks for old removals and the distinction between an absent and an empty old bubble. A single ID cannot be both added and removed in this per-pilot update. List order, stable-pilot lists and serialized delivery are outside this projection.

Engine symbols `DestinyPark.AdditionsAndDeletions`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

| [Theorems](ExactBallparkProofs/Space/Distribution.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Distribution.pilot_add_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.pilot_remove_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Distribution.pilot_disjoint` | propext, Classical.choice, Quot.sound |

<a id="claim-82"></a>

## Clients addressed by single- and dual-update bubble batches

**Finite client sets and ordered event tokens.** For a bubble-history entry, the recipients are the union of interests through its interactive balls. Each interested client gets the ONE group exactly when it has no mapped nonempty character history, otherwise the TWO group. Both groups carry the original ordered payload and a false wait flag. Missing interactive-bubble keys have an empty audience.

Engine symbols `ParkUpdateBatcher.get_clients_with_character_history, ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

| [Theorems](ExactBallparkProofs/Space/Batches.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.Batches.single_recipient` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.Batches.dual_recipient` | propext, Classical.choice, Quot.sound |

<a id="claim-83"></a>

## Private update flags and per-client packet order

**Ordered history entries, finite recipient sets and projected network calls.** A nonempty mapped character history waits exactly when that client also receives a bubble packet in this batch. Its count field is TWO while waiting and ONE otherwise. Across the three successful interface calls, a client sees its character entries first, then one packet per addressed bubble entry in iteration order, including shared interests or several characters per client. This counts the encoded flags separately from the number of packets a client can receive. Transport scheduling and serialization are outside the model.

Engine symbols `ParkUpdateBatcher.get_clients_with_character_history, ParkUpdateBatcher.send_batch`. [Source](../src/exact_ballpark/server.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

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

Recorded upstream evidence — MOV-731 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done., TIME-710 passing (test vectors), TIME-712 implemented, not passing (test vectors). Destiny's published tests do not cover the pre-tick's first step (the follower clean-up). The pre-tick follower clean-up is checked against the transcription..

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

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport., TIME-715 passing (test vectors).

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

<a id="claim-126"></a>

## Scalar and batched movement produce the same coordinate words

**Lean binary64 coordinate programs.** For every coordinate and cache word, the tensor masks select the same position and velocity words as the source-extracted scalar program. Only a nonzero partial tick needs an exponential result with matching bits. The cached full-tick tensor expression also agrees for a nonzero, non-NaN tick. The result covers the operation models and their signed-zero returns.

Engine symbols `VecCollidePark._integrate_t, VecCollidePark._integrate_dt, Ballpark.integrate`. [Source](../src/exact_ballpark/vec_collide.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-704 passing (compiled code).

The gate compares 24 position and velocity pairs through scalar Python and PyTorch CPU, including both sides of the strict branch cutoff. It also exercises the cached full-tick tensor entry point. [Integrator comparison](scripts/check.py)

| [Theorems](ExactBallparkProofs/Movement/TensorLane.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.TensorLane.coordinate_words` | propext, Classical.choice, Quot.sound |

<a id="claim-127"></a>

## A missed rebound from three starting ball states

**Kernel-evaluated binary64 iterative tick.** One free ball and two fixed spheres on a line, zero thrust, a 100 ms tick, eight supplied exponential words and the fixture’s scalar candidate order. Starting with an empty impact history, Lean checks each detected obstacle, time, selected record, impulse and final x and vx. The control changes only the cutoff comparison to exact arithmetic on the stored words.

Engine symbols `SpaceBallpark._next_collision_static, SpaceBallpark._impact_static, CollisionBallProperties.find_index, FullBallpark.evolve`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py) · [full.py](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-705 passing (compiled code), COL-715 passing (compiled code), COL-718 passing (compiled code).

Both scalar runs match the certified traces and endpoints. A separately compiled pinned Destiny run matches the unmodified engine’s stored contact history and endpoint words. [Replay and measurements](DISCREPANCIES.md#col-715-history-lookup)

| [Theorems](ExactBallparkProofs/Collision/ReboundCutoff.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundCutoff.detected_rebound` | propext, Classical.choice, Quot.sound |

<a id="claim-128"></a>

## Forward error of the ordinary drag step

**Source-extracted arithmetic with FloatLib 53-bit nearest-even rounding.** For the ordinary branch with positive friction, a covered decay-factor error ε and fixed inputs, the bound accounts for each rounded operation before adding the sensitivity of position or velocity to ε. The physical trajectory uses positive effective mass. The arithmetic model has unbounded exponents and applies to binary64 value operations when their grids agree and results stay finite. The cache accuracy is an input.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-704 passing (compiled code).

The gate re-extracts the ordinary expression from Python. Lean applies the bound directly to that expression evaluated with rounding after each source operation. [Translator](scripts/integrator_reference.py)

| [Theorems](ExactBallparkProofs/Movement/RegularError.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.RegularError.regular_step_error` | propext, Classical.choice, Quot.sound |

<a id="claim-129"></a>

## An interior sweep overlap survives the collision broad phase

**Real travel, bounded face-test error and discrete filing.** For two eligible ordinary filings with unclamped, aligned cells, a common point inside both swept bodies by the stated face-rounding margins survives in both candidate lists in either query direction. Travel may be bounded directly or by the regular drag theorem with positive coefficients, a valid cache and the stated speed and thrust limits. Each filing reach is at most a third of its level’s cell width. Boundary tangencies and special top-level filings remain open.

Engine symbols `SpaceBallpark.inflated_radius, SpaceBallpark._ball_insert_in_boxes, FullBallpark._nearby, VecCollidePark._related`. [Source](../src/exact_ballpark/space.py) · [full.py](../src/exact_ballpark/full.py) · [vec_collide.py](../src/exact_ballpark/vec_collide.py)

Recorded upstream evidence — SEN-706 passing (compiled code), COL-703 passing (compiled code).

Scalar and PyTorch CPU probes check a moving face crossing, a corner overlap and a mixed-level pair in both directions. [Spatial probes](scripts/candidate_checks.py)

| [Theorems](ExactBallparkProofs/Space/SweepCandidates.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.SweepCandidates.sweep_candidate` | propext, Classical.choice, Quot.sound |

<a id="claim-130"></a>

## Which response the default gradient retains

**Real-valued fold over eligible ordinary ball hits.** With an empty incumbent and supplied successful positive-time ball responses, the fold picks the latest contact fraction, then the largest squared norm after the stored response scale, then the first candidate in traversal order. The selected ID identifies the source of the vector in the model. The engine stores the vector and time. Overlap recursion and static shapes are separate paths.

Engine symbols `SpaceBallpark._potential, FullBallpark._gradient`. [Source](../src/exact_ballpark/space.py) · [full.py](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-701 passing (compiled code), COL-711 passing (compiled code).

CPU runs use actual candidate traversal and sphere contacts in both orders, including equal-strength contacts with different response directions. [Gradient checks](scripts/geometry_checks.py)

| [Theorems](ExactBallparkProofs/Collision/GradientWinner.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.GradientWinner.gradient_winner` | propext, Classical.choice, Quot.sound |

<a id="claim-131"></a>

## A certified interior hit time for the binary64 chord solver

**Binary64 solver and exact rational bisection.** Given the endpoint and radius words and a chosen depth d, a successful certificate encloses both the exact first contact of the straight chords and the decoded binary64 result strictly inside the tick, within 2⁻ᵈ of each other. The checker can decline a case. It requires finite inputs, nonnegative radius, initial clearance and nonzero relative motion.

Engine symbols `SpaceBallpark.collide_two_spheres`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code).

Host probes check the reported word, bracket width and endpoints for interior hits and verify that a known false hit and ambiguous tick endpoints receive no certificate. [Certificate checks](scripts/geometry_checks.py)

| [Theorems](ExactBallparkProofs/Collision/HitCertificate.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.certified_chord_hit` | propext, Classical.choice, Quot.sound |

<a id="claim-132"></a>

## A remote follower has stopped before the bubble update

**Master pre-tick queue and ordinary follower state.** For an eligible remote follower, successful queued actions, and the ordinary master Stop path, the state at the bubble hook equals applying the old queue then that follower’s Stop. A duplicate tail Stop has the same effect. A still-following ship releases its current leader’s membership and clears its target, pointer, owner, effect stamp and range. Formation slots, external callbacks and a missing master pointer are outside the model.

Engine symbols `Actions.clean_up_followers, Actions.flush_history, Ticker._pre_update_bubbles, ModeBallpark._stop`. [Source](../src/exact_ballpark/server.py) · [space.py](../src/exact_ballpark/space.py) · [modes.py](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-731 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done., TIME-710 passing (test vectors), TIME-712 implemented, not passing (test vectors). Destiny's published tests do not cover the pre-tick's first step (the follower clean-up). The pre-tick follower clean-up is checked against the transcription..

The gate observes the actual state on entry to the bubble hook with a prior mass change and with a duplicate Stop. It checks both ships in the leader’s original follower set. [Server checks](scripts/mixed_history_checks.py)

| [Theorems](ExactBallparkProofs/Time/FollowerTick.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.FollowerTick.stopped_before_bubbles` | propext, Classical.choice, Quot.sound |

<a id="claim-133"></a>

## Sending private and bubble updates across two stamps

**Server packet delivery and client history merge.** One dual recipient, an initially empty client history and ordinary admitted events. A private packet carries an earlier group and an optional later group. A bubble packet at the earlier stamp releases that group in private-then-bubble order. Only the later group, when present, stays waiting. Production cadence and a fatal-desync handler’s recovery policy are external.

Engine symbols `Ticker._apply_history, ParkUpdateBatcher.send_batch, merge_state_into_history`. [Source](../src/exact_ballpark/server.py) · [client.py](../src/exact_ballpark/client.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport., TIME-715 passing (test vectors).

A three-ball session sends actual mass changes for a cloaked pilot and a visible ship. Sending each tick clears the client queue. Postponing one send reports a fatal desync and leaves the later change waiting when the handler returns. [Producer replay](scripts/mixed_history_checks.py)

| [Theorems](ExactBallparkProofs/Time/BatchStamps.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.BatchStamps.delayed_private_flush` | propext, Classical.choice, Quot.sound |

<a id="claim-134"></a>

## The first exponential reduction FMA on negative drag inputs

**FloatLib binary precision 53 with gradual underflow.** For every stored binary64 value x in [−1, −2⁻⁵⁴], the rounded multiply, half adjustment and truncation determine the index. With the source’s split high coefficient, the emulated first FMA returns exactly x + kd·NEGLN2HIN, which is also the once-rounded value. Its two low tails vanish. This is a value theorem for this call and interval. Later FMA calls, final exp rounding and the sign bit of a zero require separate reasoning.

Engine symbols `exp, fma, split, two_product, two_sum, add_odd, INVLN2N, NEGLN2HIN`. [Source](../src/exact_ballpark/vec_exp.py)

Recorded upstream evidence — MOV-738 passing (compiled code).

The gate observes the first call inside the PyTorch CPU exponential at all 186 reached indices, including transition neighbors and exact cancellations. It checks the stored constants, exact dyadic result and the result bits against CPython math.fma. [Arithmetic checks](scripts/arithmetic_checks.py)

| [Theorems](ExactBallparkProofs/Numerics/ExpFirstFma.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Numerics.ExpFirstFma.first_reduction_exact` | propext, Classical.choice, Quot.sound |

<a id="claim-135"></a>

## When transversal motion keeps the tracking square root in range

**Exact rational error certificate for the binary64 tracking input.** For finite relative velocity and nonzero sightline, the certificate measures errors in the two squared speeds and bounds the final subtraction and division. With a positive stored squared distance, a sideways-speed squared at least as large as the returned threshold guarantees a nonnegative input to the angular-speed square root. The threshold is in squared velocity units. This covers the successful scalar tracking branch.

Engine symbols `FullBallpark.get_accuracy, get_accuracy`. [Source](../src/exact_ballpark/full.py) · [vec_accuracy.py](../src/exact_ballpark/vec_accuracy.py)

Recorded upstream evidence — CMB-702 passing (compiled code).

Four CPU fixtures compare the actual square-root input with Lean, including a clear transversal, small margins on either side of the test, and the logged radial input. [Tracking checks](scripts/geometry_checks.py)

| [Theorems](ExactBallparkProofs/Combat/TrackingMargin.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.TrackingMargin.sideways_margin` | propext, Classical.choice, Quot.sound |

<a id="claim-136"></a>

## Cache writes during scalar movement commands

**Binary64 ball state and logical binary32 property conversion.** Construction or replacement establishes the cached decay. From a valid cache, each modeled edit preserves equality to the source calculation for the new mass and agility, including refused writes and NaN inputs. Configuration and the supplied exponential stay fixed. The command dispatch and property guards are transcribed from the scalar API.

Engine symbols `Ballpark.add_ball, set_mass, set_agility, set_max_velocity, set_radius, set_speed_fraction, stop, goto_point, goto_direction, _refresh_time_factor`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-704 passing (compiled code), MOV-708 passing (test vectors), MOV-709 passing (test vectors), MOV-711 passing (compiled code), MOV-712 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/ScalarCommands.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.Scalar.edit_preserves_cache` | propext, Classical.choice, Quot.sound |

<a id="claim-137"></a>

## One scalar tick preserves IDs and agrees with immediate per-ball integration

**Source-extracted three-pass tick and a single-ball binary64 update.** For a duration that compares equal to itself, every ID lookup after the staged tick equals an immediate update of that ball with the cached-factor lane. Both paths use the same mode preparation. Zero duration preserves the prepared coordinates, including STOP damping. The ordered list of IDs is unchanged. The proof concerns observable state at the boundary of a returning collision-free tick.

Engine symbols `Ballpark.evolve, goto_thrust, integrate, _normalized`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-705 passing (compiled code), MOV-706 passing (compiled code), MOV-710 passing (compiled code), TIME-706 passing (compiled code), TIME-707 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/ScalarTrace.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.Scalar.tick_lookup` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.Scalar.tick_order` | propext, Classical.choice, Quot.sound |

<a id="claim-138"></a>

## Scalar GOTO and STOP across a finite command trace

**Binary64 transition system with an explicit admitted execution domain.** From any modeled park, the runner succeeds exactly when the fixed configuration and each successive action meet the stated guards. At any admitted prefix each ball equals a run of its own commands and all ticks. The stamp advances once per tick. Running two consecutive segments agrees with running their concatenation, including rejection of an out-of-domain suffix. The theorem uses one common exponential function. Runtime correspondence covers calls on API-created, unaliased ball state in this domain.

Engine symbols `Ballpark.evolve and the scalar ball creation, command and property APIs`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-704 passing (compiled code), MOV-708 passing (test vectors), MOV-709 passing (test vectors), MOV-710 passing (compiled code), MOV-711 passing (compiled code), MOV-712 passing (compiled code), TIME-706 passing (compiled code).

CPython and Lean agree on the clock, IDs, modes and every stored field after 202 movement trace prefixes. The probe compares non-NaN words exactly and checks 3 rejected suffixes against scalar exceptions. [Trace replay](scripts/scalar_trace_checks.py)

| [Theorems](ExactBallparkProofs/Movement/ScalarTrace.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.Scalar.run_accepts_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.Scalar.trace_refines` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.Scalar.trace_append` | propext, Classical.choice, Quot.sound |

<a id="claim-139"></a>

## The cached decay stays valid across complete scalar traces

**Binary64 cache invariant over arbitrary finite admitted traces.** Starting with a valid park, every live ball has exactly the source cache value for its current mass and agility after the trace. The empty park discharges the initial premise, including for traces that replace an existing ID. This establishes consistency with the supplied exponential function. Its accuracy is a separate input to trajectory error bounds.

Engine symbols `Ballpark._refresh_time_factor, add_ball, set_mass, set_agility, evolve`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/ScalarTrace.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.Scalar.trace_preserves_cache` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.Scalar.cache_from_empty` | propext, Classical.choice, Quot.sound |

<a id="claim-140"></a>

## Accumulated error along the ordinary drag trajectory

**Ordinary source expression under 53-bit nearest-even rounding with unbounded exponents.** For any finite sequence of positive-mass, positive-friction, positive-duration ordinary steps, the per-coordinate budget bounds both position and velocity against the piecewise constant-force drag ODE. Both paths use the same supplied acceleration and coefficients in each interval. The budget uses the incoming rounded state, a bound on each decay-factor error, and the incoming position and speed errors. Applying it to binary64 requires finite operations on matching rounding grids and selection of the ordinary branch.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/RegularTrajectory.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.RegularTrajectory.regular_trajectory_error` | propext, Classical.choice, Quot.sound |

<a id="claim-141"></a>

## Five-second coast against the continuous drag equation

**Five binary64 ticks, exact rational decoding and a certified real exponential enclosure.** Starting at the origin with horizontal velocity 100 m/s, effective mass and friction both 1 and one-second ticks, the finite scalar model finishes within 1e-14 m in position and 1e-15 m/s in velocity of the continuous horizontal drag solution. The sole elementary-function input is the supplied binary64 word for exp(-1). The proof evaluates the whole modeled command trace, decodes both returned values and compares them to the exact solution at five seconds.

Engine symbols `Ballpark.add_ball, _refresh_time_factor, evolve, integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), MOV-704 passing (compiled code), TIME-706 passing (compiled code).

The CPU gate compares the actual scalar cache word and both exact endpoint ratios to the Lean witness. A separate x86-64 replay with GCC 11.4 and Destiny at 114e89fa matches all six scalar motion words at each tick. [Coast replay](scripts/scalar_trace_checks.py) · [Harness and command](README.md#verify-locally)

| [Theorems](ExactBallparkProofs/Movement/CoastWitness.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.CoastWitness.five_second_coast` | propext, Classical.choice, Quot.sound |

<a id="claim-142"></a>

## Work and kinetic energy along the three-dimensional drag trajectory

**Real three-dimensional velocity from the translated scalar integrator.** For fixed positive effective mass m = mass × agility, positive friction k and constant applied acceleration a, kinetic energy m‖v‖²/2 changes at the rate m⟪v,a⟫ − k‖v‖². The integral of that power gives the energy difference over any interval. With zero thrust the energy scales by the squared decay factor and strictly decreases for every nonzero initial velocity. The velocity connection covers both position branches given an exact full-tick cache. Interpreting integrated power as work along the position trajectory uses the ordinary branch where p′ = v. Command damping and collision impulses lie outside this interval.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), MOV-703 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/Energy.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.extracted_velocity3` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.velocity3_hasDerivAt` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.drag_energy_rate` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.drag_work_energy` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.coast_energy` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.coast_energy_strictAnti` | propext, Classical.choice, Quot.sound |

<a id="claim-143"></a>

## Physical dimensions of the drag exponent and cached factor

**Physlib quantities over real numbers.** Giving friction dimension mass/time, duration dimension time and agility dimension one makes −k Δt / (mass × agility) dimensionless. Its real exponential equals the integrator decay. Positive effective mass selects this value in the real cache refresh.

Engine symbols `Ballpark._refresh_time_factor, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Physics/Units.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Physics.drag_exponent_decay` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Physics.refresh_factor_units` | propext, Classical.choice, Quot.sound |

<a id="claim-144"></a>

## From a sweep fraction to a collision time in seconds

**Physlib quantities over real numbers.** Removing units from the typed timestamp gives the source expression. On every positive-length window, two fractions are in strict order exactly when their timestamps are. For the half-tick fixture the absolute-time scan selects ball 3 at 5/8 s in both orders, ahead of ball 4 at 3/4 s. Lean also checks that comparing a dimensionless fraction directly with seconds fails to typecheck.

Engine symbols `SpaceBallpark._next_collision_static, FullBallpark._next_collision_static`. [Source](../src/exact_ballpark/space.py) · [full.py](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Physics/Units.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Physics.contact_time_value` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Physics.contact_time_lt_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Physics.half_tick_absolute_selection` | propext, Classical.choice, Quot.sound |

<a id="claim-145"></a>

## Binary64 control for the fixed-obstacle comparison

**Physlib time and fraction types over Lean binary64.** For all binary64 inputs the typed control computes the same decision as evaluating start + fraction × (end − start) < held in that order. In the half-tick fixture the engine comparison chooses ball 4 or 3 depending on enumeration. Comparing in seconds selects ball 3 at the exact binary64 word for 0.625 s in both orders. The two policies use the same modeled sphere sweeps and timestamp storage.

Engine symbols `SpaceBallpark._next_collision_static`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-715 passing (compiled code).

The CPU gate compares the selected ID and every timestamp bit for both orders against the scalar engine and the earliest-time control. The pinned Destiny replay exercises the engine policy. [Scalar control](scripts/contact_replay.py) · [Recorded discrepancy](DISCREPANCIES.md#col-715-fixed-window)

| [Theorems](ExactBallparkProofs/Physics/Units.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Physics.earlier64_eq` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Physics.half_tick_selection_words` | propext, Classical.choice, Quot.sound |

<a id="claim-146"></a>

## Uniqueness of the ordinary three-dimensional drag trajectory

**Real ordinary integration with constant inputs.** The three position and velocity coordinates start at the supplied state. Position has the modeled velocity as its derivative. With positive effective mass and friction, any trajectory continuous on the interval that satisfies both differential equations from the right before the endpoint and shares the initial state agrees throughout. The real source translation yields this position when the ordinary guard holds and a full tick reads the exact decay.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/DragChord.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.DragChord.pos_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.vel_zero` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.position_derivative` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.extracted_position` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.trajectory_unique` | propext, Classical.choice, Quot.sound |

<a id="claim-147"></a>

## Maximum acceleration and deviation from the endpoint chord

**Real three-dimensional motion over one interval.** For fixed positive effective mass and friction and held thrust, net acceleration has magnitude exp(−kt/m) times its initial magnitude. Any twice differentiable path whose acceleration magnitude is bounded by A stays within A T²/8 of its endpoint chord at the same time throughout a nonnegative duration T. Instantiating the result with the ordinary drag trajectory uses A = ‖a − (k/m)v₀‖. Split the motion at force changes and impulses.

Engine symbols `Ballpark.integrate, SpaceBallpark.collide_two_spheres`. [Source](../src/exact_ballpark/ref.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code), COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/DragChord.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.DragChord.acceleration_bound` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.chord_deviation` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.DragChord.flight_chord_deviation` | propext, Classical.choice, Quot.sound |

<a id="claim-148"></a>

## Two moving spheres stay clear throughout the tick

**Real ordinary trajectories and same-time straight sweeps.** For every nonnegative real duration, strict chord separation above the combined radius plus both curvature bounds excludes contact even at the endpoints. With approximate endpoints, a certified lower bound also subtracts each body's larger start or end Euclidean position error. Each body may have its own effective mass, friction and acceleration.

Engine symbols `SpaceBallpark.collide_two_spheres, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/CurveClearance.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.CurveClearance.pair_no_contact` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.CurveClearance.pair_clearance` | propext, Classical.choice, Quot.sound |

<a id="claim-149"></a>

## Certified clearance in metres from binary64 endpoints

**Logical binary64 sphere solver, exact rational geometry and real motion bounds.** The checker requires the engine miss word and decodes the endpoints and combined radius exactly. It minimizes the squared centre distance on the relative segment, validates a micrometre-rounded lower bound and subtracts the radius, both curvature bounds and both endpoint error allowances. A positive returned margin is guaranteed between the surfaces throughout every ordinary trajectory covered by the supplied budgets. The three-coordinate adapter converts existing ordinary-step rounding budgets to Euclidean error bounds. Applying those budgets to binary64 requires matching rounding grids and certified decay errors.

Engine symbols `SpaceBallpark.collide_two_spheres, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-704 passing (compiled code), COL-705 passing (compiled code).

The CPU gate compares the checker with the scalar sphere solver and an independent rational closest-point calculation. Cases include two moving bodies, closest endpoints, zero relative motion, touching, invalid bounds, nonfinite geometry and a rounded false hit. [CPU comparisons](scripts/curve_checks.py) · [Run the checker](README.md#clearance-throughout-a-tick)

| [Theorems](ExactBallparkProofs/Collision/CurveClearance.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.CurveClearance.position_error_from_axes` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.CurveClearance.rounded_endpoint_budget` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.CurveClearance.certified_clearance` | propext, Classical.choice, Quot.sound |

<a id="claim-150"></a>

## A turning GOTO ship and a certified clear control

**One full scalar-model tick, exact endpoint words and a certified continuous drag trajectory.** The ship starts at the origin at 10 m/s along x and receives a GOTO along y with mass, agility and friction all one. Given the recorded full-tick decay word, the modeled scalar command trace produces the cached factor, thrust and endpoint words used by the sweep. The ordinary position branch applies throughout. The obstacle at (4,1,0) is more than 0.5 m from the ideal chord everywhere and less than 0.1 m from the true centre at half a second. Both binary64 sweeps report a miss. Moving the obstacle to (4,−4,0) yields 3.094137999999 m of certified surface clearance, with the actual acceleration and endpoint budgets proved for both bodies.

Engine symbols `Ballpark.add_ball, goto_point, goto_thrust, evolve, integrate, SpaceBallpark.collide_two_spheres`. [Source](../src/exact_ballpark/ref.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-704 passing (compiled code), MOV-705 passing (compiled code), MOV-706 passing (compiled code), MOV-708 passing (test vectors), COL-705 passing (compiled code).

On x86-64 Linux, CPython and the pinned Destiny C++ functions agree on the cache, three thrust words, all three half-tick and endpoint coordinates, and both solver results. [Fixture and replay](DISCREPANCIES.md#col-705-turning-path)

| [Theorems](ExactBallparkProofs/Collision/TurningFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.TurningFlight.turning_overlap` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.TurningFlight.goto_sweep_inputs` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.TurningFlight.turning_words` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.TurningFlight.turning_clearance` | propext, Classical.choice, Quot.sound |

<a id="claim-151"></a>

## Earliest contact of continuous sphere paths

**Continuous centre distance on a closed time interval.** On a common clock, a nonempty set of sphere contacts has a unique earliest time. Initial overlap gives time zero. A positive first time lies on the surface. Strict clearance before a lower endpoint and overlap at the upper endpoint enclose the first time even when distance later increases. Strictly disjoint windows order two eligible contacts before any response changes the trajectory.

Engine symbols `SpaceBallpark.collide_two_spheres, SpaceBallpark._next_collision_static`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-705 passing (compiled code), COL-715 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/FirstContact.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.FirstContact.exists_first` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.FirstContact.initial_contact` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.FirstContact.first_unique` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.FirstContact.first_on_surface` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.FirstContact.first_in_window` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.FirstContact.disjoint_first_contacts` | propext, Classical.choice, Quot.sound |

<a id="claim-152"></a>

## Intermediate positions from a drag initial state

**Rational enclosures of the real ordinary drag solution.** From positive effective mass and friction and constant applied acceleration, the checker bounds exp(−kt/m) and all three position coordinates throughout every nonnegative time slab. The coefficients on initial and terminal velocity are monotone even when a velocity component reverses. A twelve-term exponential series uses mathlib’s remainder bound, range reduction through 64 decay times and a monotone tail bound beyond 64. The ideal decay factor here is evaluated from time and coefficients. A measured cache word or host exp can have rounding error.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-704 passing (compiled code).

| [Theorems](ExactBallparkProofs/Movement/PositionEnclosure.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.PositionEnclosure.decay_enclosed` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.PositionEnclosure.weights_monotone` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.PositionEnclosure.position_enclosed` | propext, Classical.choice, Quot.sound |

<a id="claim-153"></a>

## A CPU certificate for the first contact of two ordinary drag paths

**Exact rational tests for whole-interval real geometry.** The guard is checked over the full duration using the exact source coefficient. Coordinate boxes enclose the squared Euclidean separation of both moving centres. Every accepted prefix covers contiguous time slabs with strict surface clearance. A contact certificate also bounds the separation by the exact combined radius at its upper endpoint. A reported whole-interval surface margin holds in metres. Unresolved inputs or unsuccessful searches make no geometric claim. The two COL-705 trajectories here continue under their initial forces until the first impact.

Engine symbols `SpaceBallpark.collide_two_spheres, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-703 passing (compiled code), COL-705 passing (compiled code).

The CPU gate compares the separate binary64 chord word with the scalar engine and checks curved-path results for turning, grazing, two moving bodies, interval subdivision and invalid inputs. [Checker and examples](README.md#first-contact-along-a-turn)

| [Theorems](ExactBallparkProofs/Collision/ContactCertificate.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ContactCertificate.ordinary_throughout` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactCertificate.squared_distance_enclosed` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactCertificate.prefix_clear` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactCertificate.whole_interval_margin` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactCertificate.verified_result` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactCertificate.classify_sound` | propext, Classical.choice, Quot.sound |

<a id="claim-154"></a>

## First-contact times and obstacle order from initial states

**Kernel-evaluated rational certificates and ideal 3D drag trajectories.** On the existing one-second GOTO fixture, first contact with the obstacle at (4,1,0) m occurs in [0.435661, 0.435662] s. The control at (4,−4,0) m stays at least 3.5 m clear. When the second body also turns with a different drag time, their first touch is in [0.404889, 0.404890] s. Two ships approaching at 2 m/s each first meet in [0.499999, 0.5] s. A later overlap sample fails first-contact verification. On the fixed-obstacle partial tick, disjoint windows put the obstacle at x = 3 m before the one at x = 5 m. Both candidate paths are evaluated before a response to either obstacle.

Engine symbols `SpaceBallpark.collide_two_spheres, SpaceBallpark._next_collision_static, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), COL-705 passing (compiled code), COL-715 passing (compiled code).

The scalar engine and pinned Destiny agree on all three partial-step coordinates at the binary64 times nearest the turning window edges. Both engines report a chord miss for the fixed obstacle and for the independently turning second ship. The independent 60-digit decimal calculation estimates the first contact at 0.435661309892 s. [Turning replay](DISCREPANCIES.md#col-705-turning-path)

| [Theorems](ExactBallparkProofs/Collision/ContactExamples.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ContactExamples.turning_window` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactExamples.clear_control` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactExamples.two_turning_bodies` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactExamples.moving_pair_window` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactExamples.late_window_rejected` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ContactExamples.obstacle_order` | propext, Classical.choice, Quot.sound |

<a id="claim-155"></a>

## Exact incoming state of the half-second head-on contact

**Real ordinary drag for the isolated two-sphere fixture.** Two free spheres of radius 0.5 m begin at x = 0 and 3 m. Opposite full-strength GOTO thrusts sustain speeds of 2 m/s. Each effective mass and the friction are one in all three variants, including the left constructor speed fraction of four in the limited case. The ordinary position guard holds for the entire one-second tick. Lean identifies the previous continuous-contact fixture with these axial paths and proves that their first touch is exactly 0.5 s, at x = 1 and 2 m.

Engine symbols `SpaceBallpark.collide_two_spheres, Ballpark.integrate, Ballpark.goto_thrust`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-705 passing (compiled code), COL-705 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ReboundFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundFlight.ordinary_inputs` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.incoming` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.incoming_distance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.exact_first_contact` | propext, Classical.choice, Quot.sound |

<a id="claim-156"></a>

## Impulse, velocity limiter and restart at positive contact time

**Three-dimensional real translational impulse followed by ordinary drag.** At positive contact time, the two-body elastic update uses the normal from the actual centre positions and both base masses. Each body applies its own squared-ratio limiter. The restart keeps position continuous, adds the applied velocity change and retains the original acceleration. Lean gives all positions and velocities on the remainder. In the 1:2 base-mass example the applied changes are −16/3 and +8/3 m/s. In the equal-mass limited example they are −25/16 and +4 m/s. The latter changes the pair momentum by +39/16 kg m/s at impact.

Engine symbols `SpaceBallpark._impact_free, SpaceBallpark._add_impact, CollisionBallProperties.add_impact, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), COL-706 passing (compiled code), COL-717 passing (compiled code), COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ReboundFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundFlight.restart_state` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.remainder` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.raw_impulse` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.momentum_change` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.applied_impulse` | propext, Classical.choice, Quot.sound |

<a id="claim-157"></a>

## A single touch through the end of an iterative collision tick

**Signed separation on collinear real drag trajectories.** For two collinear ordinary flights starting at contact with common positive drag parameters, a nonnegative excess of outgoing relative velocity over terminal relative velocity and a strictly positive signed endpoint gap suffice for strict separation throughout the open remainder. The proof uses convexity of the exponential and an independent rational enclosure of exp(−1/2). All three joined paths touch only at 0.5 s on the closed tick. The limited pair is already closing again at one second.

Engine symbols `SpaceBallpark._next_collision_free, SpaceBallpark._impact_free, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), COL-705 passing (compiled code), COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ReboundFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundFlight.half_decay` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.positive_gap` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.shared_drag_separation` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.no_recontact` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.only_contact` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundFlight.limited_closing_at_end` | propext, Classical.choice, Quot.sound |

<a id="claim-158"></a>

## Binary64 replay from head-on contact to both committed endpoints

**Logical binary32 limiter and binary64 translational replay.** The projection admits one eligible free pair, no static candidates, spin, initial overlap or other interactions. It calculates the GOTO thrust and cached decay, solves the endpoint chord, computes both impulses, inserts the two records, checks the zero-time duplicate on the next search and samples both final histories. The exponential model supplies the two host words at −1 and −1/2. Each returned history has one reciprocal contact. The kernel checks three complete fixture traces, including transverse signed zeros.

Engine symbols `SpaceBallpark._next_collision_free, SpaceBallpark._impact_free, SpaceBallpark._add_impact, CollisionBallProperties.position, SpaceBallpark._position_velocity`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py) · [modes.py](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-704 passing (compiled code), COL-705 passing (compiled code), COL-715 passing (compiled code), COL-717 passing (compiled code), COL-718 passing (compiled code).

On x86-64 Linux, the scalar Evolve entry point and compiled pinned Destiny agree with Lean on 29 words for each of the six ball records. These cover the cache, contact time, thrust, incoming state, raw and applied impulses, stored contact state and committed tick state. A second C++ park exposes the original impulse helpers so intermediate changes can be observed. [Rebounds and reproduction](README.md#one-impact-through-the-end-of-a-tick)

| [Theorems](ExactBallparkProofs/Collision/ReboundTick.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundTick.replay_words` | propext, Classical.choice, Quot.sound |

<a id="claim-159"></a>

## Error of the committed state after the rebound

**Rational enclosures compared with decoded binary64 words.** For each of the three replays, all four axial outputs are compared with the real trajectory after the same impulse and limiter. A rational interval for exp(−1/2) bounds each ideal value. The kernel decodes the committed word and proves absolute x-position error at most 10⁻¹⁵ m and x-velocity error at most 10⁻¹⁵ m/s. The bound is specific to these inputs and the two supplied exponential words.

Engine symbols `SpaceBallpark._position_velocity, CollisionBallProperties.calculate_end, Ballpark.integrate`. [Source](../src/exact_ballpark/space.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), COL-717 passing (compiled code), COL-718 passing (compiled code).

| [Theorems](ExactBallparkProofs/Collision/ReboundAccuracy.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.ReboundAccuracy.ideal_endpoint_enclosed` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.ReboundAccuracy.endpoint_error` | propext, Classical.choice, Quot.sound |

<a id="claim-160"></a>

## Timed proximity crossing at the surface boundary

**Binary32 thresholds, binary64 distances and elapsed time.** Eight one-second polls at a two-second period, with both radii 1 m and sensor range 10 m. The live target remains a candidate throughout. The projected scan admits exact contact at 12 m on poll four, retains membership on early polls and the inside repeat, and emits a leave on poll eight. The callback leaves the sensor active.

Engine symbols `SpaceBallpark.add_proximity_sensor, _check_sensor`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-701 passing (compiled code), SEN-702 passing (compiled code).

A three-pilot session refiles the moving target and checks the actual elapsed word, membership and ordered callbacks on every poll. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Space/SensorDelivery.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.SensorDelivery.timed_crossing` | propext, Classical.choice, Quot.sound |

<a id="claim-161"></a>

## From proximity callbacks to the recipient of a queued response

**Discrete actions, bubble membership and client interest sets.** An ordinary action for a positive, uncloaked owner, no private traffic or competing actions. A DoProximity callback alone gives an empty outgoing history. For the application that queues SetBallRadius(1) on enter and SetBallRadius(2) on leave, a client interested in an interactive ball in the known, live bubble receives the two responses on relative ticks five and nine. Every other client receives neither response.

Engine symbols `Ticker._apply_history, BaseTicker._is_valid_event_bubble, ParkUpdateBatcher.send_batch, SpaceBallpark._check_sensor`. [Source](../src/exact_ballpark/server.py) · [space.py](../src/exact_ballpark/space.py) · [client.py](../src/exact_ballpark/client.py)

Recorded upstream evidence — SEN-712 implemented, not passing (test vectors). Destiny's published tests do not cover the events of a ball in no known bubble or in a dead bubble. A separate Lean CPU replay exercises the pinned Python routing and sending methods for visible owners in known, dead and missing bubbles with a fixture transport..

Two sessions exercise default and explicit responses, including a distant pilot. With a pinned Destiny checkout, the same routing fixtures execute the original five Python methods for history expansion, lookup, validation, routing and sending. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Space/SensorDelivery.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.SensorDelivery.callback_recipients` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.SensorDelivery.crossing_delivered` | propext, Classical.choice, Quot.sound |

<a id="claim-162"></a>

## Accepted warp through dropout and the following STOP tick

**Binary64 axial flight with supplied host log and exp words.** A fresh client-side warp over 131072 m at factor 4000, with minimum range zero, initial axial speed 80 m/s, maximum speed 100 m/s and one-second intervals. With no contacts, the aligned ship activates at stamp zero, drops below 50 m/s during stamp nine and takes its extra integration step. Stamp ten starts from that committed pair and advances ordinary STOP. The table supplies 13 elementary-function argument/result pairs used by this flight.

Engine symbols `ModeBallpark.warp_to, _is_aligned_for_warp, _warp_distance, evolve, Ballpark.integrate`. [Source](../src/exact_ballpark/modes.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TRV-703 passing (compiled code), TRV-706 passing (compiled code), TRV-711 passing (compiled code), TRV-713 passing (compiled code), TRV-714 passing (compiled code).

All eleven committed x and vx words and modes match ModeBallpark. The optional pinned C++ flight oracle also matches the words and the activation, deactivation and exit event sequence. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Warp/Journey.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Warp.Journey.accepted_warp_commits` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Warp.Journey.dropout_handoff` | propext, Classical.choice, Quot.sound |

<a id="claim-163"></a>

## Unit quaternion rotation preserves formation-slot distance

**Real interpretation of the quaternion matrix and transformed slot.** For every unit quaternion and every slot vector, multiplying the row vector by the source rotation matrix preserves its Euclidean norm. Substitution into formationTarget removes the previous isometry premise and includes the zero-slot fallback. Rounded binary32 rotation and normalization error are outside the distance identity.

Engine symbols `m_rotation_q, transform_coord, FullBallpark._rotated_vector, _evolve_formation`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — MOV-726 passing (compiled code).

The real formation behaviour with a dependency on the leader is exercised using identity, axis-cycle and half-turn rotations and a zero slot. The probe checks the stored binary64 target after the binary32 transformation. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Movement/FormationRotation.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.rotateSlot_norm` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.rotated_formation_distance` | propext, Classical.choice, Quot.sound |

<a id="claim-164"></a>

## Force-field configuration and the next ordered candidate query

**Queued field actions and the handled-bit candidate walk.** After both parent writes to a present field, the remaining spatial visits pass the common eligibility tests with harmonic checks enabled. For any visit list, the returned IDs are unique. An enabled field appears exactly when visited by a ship with no permitted harmonic, corporation, alliance or master override. Other visited IDs retain membership.

Engine symbols `Actions.set_ball_harmonic, Ticker._apply_park_parent_methods_except_cloak_ball, SpaceBallpark.set_harmonic, _nearby_balls`. [Source](../src/exact_ballpark/server.py) · [space.py](../src/exact_ballpark/space.py)

Recorded upstream evidence — MOV-737 implemented, not passing (compiled code). Its two halves are held to two different oracles, destiny's compiled C++ and the published tests of its Python server layer. Holding them as one rule needs destiny's Python layer run against the compiled engine, which has not been done..

Two ships query the same spatial boxes after a real server pre-tick, across five access settings and a queued disable. An ordinary massive neighbour remains in the list. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Space/FieldQuery.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.FieldQuery.configured_membership` | propext, Classical.choice, Quot.sound |

<a id="claim-165"></a>

## First committed movement after a queued GOTO

**Server stamps and the binary64 single-ball GOTO/STOP transition.** After startup, one accepted GotoDirection and no competing actions, with deterministic full ticks and a fixed cache. For an issue at n and uplink delay d, the first d ticks equal the command-free trajectory. The event bears n+d and the first affected commit uses the GOTO at n+d+1. A stationary witness issued at 3 with one tick of delay actually moves at 5.

Engine symbols `Actions.go_to_direction, Ticker.tick, Session.command, Session.tick, Ballpark.goto_direction, Ballpark.evolve`. [Source](../src/exact_ballpark/server.py) · [client.py](../src/exact_ballpark/client.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TIME-709 passing (test vectors), MOV-706 passing (compiled code).

The Session transport supplies the event stamp. The replay checks mode and all six position and velocity words against Lean at each observed stamp. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Time/CommandFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.CommandFlight.goto_first_commit` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.CommandFlight.goto_moves_at_five` | propext, Classical.choice, Quot.sound |

<a id="claim-166"></a>

## Restoring a retained client snapshot recovers a GOTO trajectory

**Retained stream selection and deterministic binary64 movement.** The single public GOTO arrives behind the client clock. The last eligible retained stream decodes to the movement state at its own stamp and all subsequent saved streams are newer than the command. Identical configurations, cached factors, command and full ticks, with no intervening commands or contacts, give identical modeled ball states at a common stamp. The predicted state may be arbitrary.

Engine symbols `Ticker.synchronize_to_simulation_time, Ticker._real_flush_state, Session.tick, Ballpark.evolve`. [Source](../src/exact_ballpark/client.py) · [server.py](../src/exact_ballpark/server.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TIME-715 passing (test vectors), TIME-709 passing (test vectors).

A control and a session with one delayed packet compare position, velocity and mode at the same simulation stamp. The delayed client first predicts a different state, rewinds successfully and subsequently remains one stamp behind wall-clock delivery. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Time/CommandFlight.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.CommandFlight.rewind_goto_same_stamp` | propext, Classical.choice, Quot.sound |

<a id="claim-167"></a>

## Input guard for the sign of the tracking square-root argument

**Logical IEEE binary64 subtraction, division and rounding.** The guard requires finite nonnegative rounded total and projected squared speeds and a finite positive squared distance. It aligns their significands at the smaller binary exponent and checks that total covers projection before cancellation. On acceptance, the actual rounded quotient compares greater than or equal to zero, including underflow to zero. The sign claim allows positive overflow and excludes a guarantee about the subsequent power or hit probability.

Engine symbols `FullBallpark.get_accuracy, _dot, _lensq`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — CMB-702 passing (compiled code).

The scalar replay captures the word passed to the actual square root. The known failing radial input is rejected. A perpendicular velocity and an exactly radial zero-margin control are accepted. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Combat/InputMargin.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.InputMargin.tracking_input_nonnegative` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.InputMargin.tracking_margin_controls` | propext, Classical.choice, Quot.sound |

<a id="claim-168"></a>

## A world advances exactly its own due movement rounds

**Per-row state selection and deterministic binary64 GOTO/STOP ticks.** For independent free GOTO/STOP rows, each mask selects the proposed single-row result or its saved fields. Across any list of masks the row equals a solo run of its selected rounds. In ordinary catch-up rounds, selecting world counts greater than the round index runs each world exactly its own count. No removal, coupled flight, rotation, contact or global launch-counter claim.

Engine symbols `VecServer.on_tick, ModesAdapter.evolve, VecModesPark.evolve_torch, masked_tick`. [Source](../src/exact_ballpark/vec_server.py) · [vec_modes.py](../src/exact_ballpark/vec_modes.py) · [vec_orient.py](../src/exact_ballpark/vec_orient.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — TIME-702 passing (compiled code), TIME-706 passing (compiled code).

Two CPU worlds are compared word for word with isolated parks and the Lean rows over five polls. The schedule includes first calls, alternating single-world updates, simultaneous updates and signed zero. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Movement/WorldMask.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.WorldMask.masked_world_trace` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.WorldMask.poll_replays_single` | propext, Classical.choice, Quot.sound |

<a id="claim-169"></a>

## Work and drag loss over a sequence of thrust changes

**Three-dimensional real drag trajectory and integrals of power.** For any finite sequence of nonnegative-duration intervals, with fixed positive effective mass and friction and arbitrary constant acceleration on each interval, pass the ending velocity to the next interval. Final minus initial kinetic energy equals accumulated thrust work minus accumulated drag loss, and drag loss is nonnegative. Instantaneous velocity changes from STOP damping or collision impulses are outside these intervals. Interpreting work along position assumes the ordinary trajectory.

Engine symbols `Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-701 passing (compiled code), MOV-702 passing (compiled code).

A three-interval turn and coast are independently integrated at 70 decimal digits. The scalar integrator is compared with the resulting component velocities at each endpoint. [CPU replay](scripts/law_trace_checks.py)

| [Theorems](ExactBallparkProofs/Movement/Manoeuvre.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.manoeuvre_work_energy` | propext, Classical.choice, Quot.sound |

<a id="claim-170"></a>

## How far STOP carries each axis before coming to rest

**Real coordinate updates with the two exact binary64 STOP constants.** Collision-free scalar STOP prepares vertical velocity once per tick. The position results use ordinary integration, fixed positive effective mass and drag, positive duration and the matching exponential cache. The closed forms apply to any number of such ticks. Loss in kinetic energy separates into vertical damping and isotropic drag. Splitting a tick changes nonzero vertical velocity.

Engine symbols `Ballpark.evolve, Ballpark.integrate`. [Source](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-702 passing (compiled code), MOV-709 passing (test vectors), MOV-710 passing (compiled code).

Scalar STOP runs twelve ticks each at 250, 500 and 1000 ms. Every coordinate is compared with a 70-digit geometric sum and decay, including the two half-ticks against one whole tick. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Movement/StopDistance.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.StopDistance.axis_velocity` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.StopDistance.axis_position` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.StopDistance.rest_point` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.StopDistance.stop_loss_budget` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.StopDistance.split_stop` | propext, Classical.choice, Quot.sound |

<a id="claim-171"></a>

## How the old orbit law truncates and combines its steering directions

**Real Gaussian, seven-place truncation and normalized steering vectors.** The truncated radial weight decreases weakly with absolute radius error. It becomes zero precisely when the squared error exceeds 40000 log(10⁷). On the outer branch, when both normalized steering directions have unit length, their blend has unit length, including the zero-radicand case. The result does not assume a unit direction at a degenerate tangent.

Engine symbols `ModeBallpark._evolve_old_orbit, _trunc7`. [Source](../src/exact_ballpark/modes.py)

Recorded upstream evidence — MOV-718 passing (compiled code).

Six offsets exercise the engine’s outer tangent branch. The gate reads its actual truncation result on both sides of the cutoff and measures the returned thrust norm. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Movement/OrbitSteering.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.OrbitSteering.weight_order` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.OrbitSteering.weight_cutoff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.OrbitSteering.outer_thrust_norm` | propext, Classical.choice, Quot.sound |

<a id="claim-172"></a>

## How new orbit trades position error for velocity error

**Real inverse controllers, blending and magnitude clipping.** For positive friction divided by effective mass and positive duration, exact decay makes both inverses defined. Before clipping, the residual in position and the residual in velocity are the two explicitly weighted errors from the other controller. A nonnegative speed and the magnitude cap preserve the conditional drag speed envelope. The low-target-speed fallback and finite rounding have separate behavior.

Engine symbols `ModeBallpark._goto_thrust_follow, Ballpark.integrate`. [Source](../src/exact_ballpark/modes.py) · [ref.py](../src/exact_ballpark/ref.py)

Recorded upstream evidence — MOV-721 passing (compiled code), MOV-705 passing (compiled code).

Four controller calls cover three uncapped weights and one clipped approach. The gate compares acceleration components and feeds the uncapped outputs back into the scalar integrator. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Movement/OrbitSteering.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Movement.OrbitSteering.guidance_denominators` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.OrbitSteering.blend_residuals` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Movement.OrbitSteering.clipped_orbit_speed` | propext, Classical.choice, Quot.sound |

<a id="claim-173"></a>

## Conservation and separation after independently limited normal impulses

**Real normal impulses with individual speed limits and an exact rational witness.** An approaching axial pair uses positive base masses and a unit normal. Each body has its own retained fraction of the elastic impulse. The theorems give the exact momentum condition, the relative-velocity threshold for rebound, and the work lost with a common fraction from zero to one. The example measures kinetic energy in the stated reference frame immediately after the impulse. The real cap omits intermediate binary32 rounding, which is exact on the example.

Engine symbols `SpaceBallpark._impact_free, SpaceBallpark._add_impact, CollisionBallProperties.add_impact`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — COL-717 passing (compiled code), COL-718 passing (compiled code).

At initial contact the scalar engine stores −97.5 and −100 m/s. The normal changes and the derived momentum and energy match the rational Lean witness. The optional pinned run compiles Destiny’s selector, impact, limiter and history methods. [Compiled replay](scripts/contact_replay.py) [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Collision/LimitedPair.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.LimitedPair.limit_normal` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.LimitedPair.from_pair_velocity` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.LimitedPair.momentum_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.LimitedPair.separates_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.LimitedPair.common_fraction_loss` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.LimitedPair.asymmetric_witness` | propext, Classical.choice, Quot.sound |

<a id="claim-174"></a>

## A straight sphere sweep excluded by a box misses its entire solid

**Real three-dimensional geometry and the source corner enumeration.** Growing the box from its eight corners gives tight per-axis extrema. Every point in the solid spanned by the three arms is inside those extrema. Strict broad-phase rejection keeps a sphere of nonnegative radius clear of that entire solid throughout its straight sweep. The half-diagonal radius result requires pairwise perpendicular arms and precedes the binary32 cast.

Engine symbols `BoxShape.__init__, BoxShape.center, BoxShape.bounding_radius, OrientedBox.initialize, AABB.update, AABB.can_exclude, FullBallpark._add_actual_minibox`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — COL-705 passing (compiled code), COL-712 passing (compiled code), COL-713 passing (compiled code).

A rotated rectangular box and a skew box exercise corner growth, 27 solid points per shape, and tangency and one-word-outside controls on each of the six faces. The rectangular control also checks the unrounded bounding radius. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Collision/SolidBox.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Collision.SolidBox.grown_bounds` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SolidBox.axis_extents` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SolidBox.solid_enclosed` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SolidBox.excluded_solid` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Collision.SolidBox.diagonal_radius` | propext, Classical.choice, Quot.sound |

<a id="claim-175"></a>

## Range and target-tracking events across polls, resets and mode changes

**Binary64 comparisons and ordered state transitions.** For an eligible ball with supplied squared distances and widened stored fields, each channel follows successive changes in its own active samples. Between resets, odd event count is exactly a membership change. Writing the same accepted range re-arms entry. STOP leaves an active notification range once. The finite trace covers strict GOTO and follow boundaries, the clamped inner radius and the tracking-enable threshold.

Engine symbols `SpaceBallpark._check_for_proximity, _check_notification_range, _check_target_tracking, set_notification_range`. [Source](../src/exact_ballpark/space.py)

Recorded upstream evidence — SEN-703 passing (compiled code), SEN-704 passing (compiled code).

Through the real spatial proximity dispatcher the eight-poll trace matches both flags and the order and payload of every callback. The same scene tests a setter re-arm and repeated STOP polls. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Space/RangeAlerts.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Space.RangeAlerts.trace_channels` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.RangeAlerts.edge_parity` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.RangeAlerts.notification_rearms` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.RangeAlerts.stop_exit_once` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Space.RangeAlerts.boundary_words` | propext, Classical.choice, Quot.sound |

<a id="claim-176"></a>

## Which delayed removals occur before a tick advances

**Integer deadlines, ordered scalar removals and per-row masks.** Outside Evolve, the first seven eligible early IDs in the supplied traversal are selected in addition to all overdue rows. Stale rows clear only their pending marker. A group with a shared deadline and no competing entries survives according to the derived prefix rule. Repeated early polls can clear the group at one game stamp. Exact Ballpark supplies ascending IDs. Destiny’s heap traversal can choose other identities under the same quota.

Engine symbols `SpaceBallpark.bring_out_the_deads, FullBallpark.on_tick, DelayedRemoval.due`. [Source](../src/exact_ballpark/space.py) · [full.py](../src/exact_ballpark/full.py) · [vec_space_removal.py](../src/exact_ballpark/vec_space_removal.py)

Recorded upstream evidence — TIME-713 implemented, not passing (compiled code). Destiny cleans up moribund balls in an order that follows heap addresses, which no port can reproduce. The reference uses ascending id., TIME-705 passing (compiled code), TIME-702 passing (compiled code).

Nineteen mixed rows check both clear and remove flags. Seventeen delayed balls check all four deadline regions and three actual on_tick calls at the same wall time. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Time/CleanupWindow.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Time.CleanupWindow.row_decision` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.CleanupWindow.early_ids` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.CleanupWindow.removal_count` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.CleanupWindow.cohort_deadline` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Time.CleanupWindow.repeat_early_poll` | propext, Classical.choice, Quot.sound |

<a id="claim-177"></a>

## Surface range, radial drift and the half-chance boundary

**Real nondegenerate tracking and range calculation.** On the ordinary branch, range is measured between the surfaces and clamped at zero. A change in relative velocity parallel to a nonzero line of sight preserves angular speed squared. Range and tracking multiply as probability factors. At fixed angular penalty the probability decreases strictly for a larger surface range above optimal. With positive falloff and radial motion, one and two falloff lengths beyond optimal give one half and one sixteenth. The binary64 cancellation case remains separately recorded.

Engine symbols `FullBallpark.get_accuracy`. [Source](../src/exact_ballpark/full.py)

Recorded upstream evidence — CMB-701 passing (compiled code), CMB-702 passing (compiled code).

Thirty scalar queries vary centre distance, transversal speed and added radial drift. They include overlap, exact optimal, the half-chance boundary and both falloff landmarks. [CPU controls](scripts/boundary_checks.py)

| [Theorems](ExactBallparkProofs/Combat/Falloff.lean) | Transitive axioms |
| --- | --- |
| `ExactBallpark.Combat.Falloff.radial_invariance` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.Falloff.chance_factors` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.Falloff.range_order` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.Falloff.half_chance_iff` | propext, Classical.choice, Quot.sound |
| `ExactBallpark.Combat.Falloff.falloff_landmarks` | propext, Classical.choice, Quot.sound |
