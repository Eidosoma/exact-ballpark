// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's flight modes, missiles and ship-to-ship collisions in one park on a CUDA device (the GPU parity report's G1 and G2, the kernel path): exact_ballpark.vec_composed.VecComposedPark's tick
// (`evolve_torch`, the torch path, its exactness reference) as launches on one stream, with no allocation
// and no host synchronisation, so a tick can be captured in a CUDA graph. Built and loaded by
// exact_ballpark.vec_composed_cuda; held to the torch path bit for bit, every tensor every tick, by
// tests/test_movement_vec_composed_cuda.py.
//
// The tick (Ballpark::Evolve, Ballpark.cpp:421), in the torch path's order:
//  1. phase 1: the prepass and phase-1 kernels (this file includes vec_combat_cuda.cu unedited, and through it
//     vec_modes_cuda.cu and vec_orient_cuda.cu), with `composed_mark_kernel` between them (the followers phase 1
//     releases: the torch path's Stop clears a released missile's owner row; the release did not until the fix in vec_combat_cuda.cu's release_in_phase1, so this clearing is now a repeat, kept, idempotent), then
//     `composed_pre_kernel`: that owner row cleared, phase 1's clearing of mLastC and the
//     collision lists of the live balls, the gradient's mLastCollision of a massive ball (-1), FLAG_WARP_MASSIVE, and
//     the warp path of each ball in warp proper (the warp_distance on a copy of its state: the drop-out point and
//     velocity, whether it drops out this tick, whether it rests there);
//  2. the drop-out's re-filing at rest: the filing kernel on the rows (vec_collide_cuda.cu's xbpc_tick, part 2)
//     twice: the view's (from the state the view holds, into scratch: _gradient_rows's refile) and the rows' (from
//     the drop-out point, in place: evolve_torch's _file(rest)); they differ only for a massive dropper (a refused
//     world: FLAG_WARP_MASSIVE), which the view holds at its warp position; then `composed_check_kernel` (a filing at
//     the partition's edge flags the world);
//  3. `composed_gather_kernel`: the rows gathered into the [world, slot] view (a world's rows by ascending id; a
//     non-massive ball dropping out this tick at its drop-out state; the view's filing);
//  4. the gradient: the gradient kernel built with XBPC_COMPOSED=1 (the composed candidates and COL-711's exception);
//  5. `composed_scatter_kernel`: mLastC, mLastCollision and the collision list of each responder back on its row;
//  6. `composed_commit_kernel`: phases 3 and 4 as the commit kernel, with the drop-out from the precomputed path
//     (massive, sensing, STOP, at its point, one more step), Integrate with mLastG + mLastC, the missiles' dispatch
//     counts and the master's self-collision appended to the list, the clock;
//  7. the re-filing of the balls that moved at least sqrt(0.1) m: the filing kernel on the rows, and the check.
//
// Exactness: as vec_combat_cuda.cu, whose device code this file includes: double operations through the pinned
// intrinsics (dadd, dmul, ... : __dadd_rn ... ) in the torch path's order, compiled with -fmad=false, -prec-div=true,
// -prec-sqrt=true, -ftz=false and no fast math; the PTX read back and checked; no function of CUDA's math library;
// every constant from the generated headers (exact_ballpark.vec_composed_cuda.tables_header reads the reference's
// modules), never typed. This file's own kernels add copies, integer work, one doubling, one sum of mLastG and mLastC,
// and the squared lengths of the existing helpers.
//
// Planted faults, for the tests only (XBPP_FAULT): 1 the drop-out's re-filing at rest left out, 2 mLastC left out of
// the integration, 3 the master's self-collision not appended, 4 a ball dropping out seen at its warp position; the (vec_composed_cuda.G11_FAULTS): 5 a growing mushroom not re-filed, 6 a removed mushroom's later members
// released after their commit, 7 a mushroom's swap not skipped, 8 the petrified trolls not re-filed.
//
// (1 November 2026): the commands and the mushrooms in this tick. The release
// mark takes StopAllFollowers' members (as the phase 1); the commit grows or removes a free mushroom
// at its turn without the swap (combat_commit_kernel's way), marks a grown one for the re-filing with its new radius,
// and releases a removed mushroom's members with higher ids before their own commit; after the host's yaw, pitch and
// roll, `xbpp_after` launches the fifth launch (the members with lower ids; PetrifyTroll) and re-files each
// petrified troll (SetBallFree(false)'s InsertBallInBoxes) with the filing kernel, then checks those filings.

#include "vec_combat_cuda.cu"
#include "vec_composed_cuda_tables.h"

#define XBPP_ABI 2                        // 2: (xbpp_after: the fifth launch and the troll's re-filing)
#ifndef XBPP_FAULT
#define XBPP_FAULT 0
#endif

namespace xbpm {

// ============================================================================================ the park's state

struct Composed {
    Combat c;                             // the tensors (the flight, missiles, formations, cloaks)
    double *last_c, *wp, *wv;             // [n, 3]: mLastC; the warp path's position and velocity (scratch)
    int64_t *hits, *level, *key, *mod;    // [n, H] the collision lists; the partition's filing [n], [n, 3], [n, 3]
    uint8_t *dropped, *rest, *moved;      // [n] scratch: drops out of warp this tick; at rest there; moved >= sqrt(0.1)
    uint8_t* released;                    // [n] scratch: phase 1 releases this follower (its target starts warp)
    double *fpos, *fvel;                  // [n, 3] scratch: the state the view holds (the view's at-rest filing's)
    int64_t *flevel, *fkey, *fmod;        // [n], [n, 3], [n, 3] scratch: the view's filing (the rows' but at rest)
    const int64_t *slot_rows, *row_slot;  // [W * S] each slot's row (-1: empty); [n] each row's slot
    // the view [W, S] (the Park, slot j = w S + s), and the composed candidates' fields
    double *v_pos, *v_vel, *v_last_g, *v_mass, *v_agility, *v_tf, *v_radius, *v_last_c, *v_last_col;
    uint8_t *v_exists, *v_free, *v_massive, *v_junk, *v_cloaked, *v_responder, *v_missile, *v_dropped;
    int64_t *v_id, *v_level, *v_key, *v_mod, *v_mode, *v_owner, *v_follow, *v_bubble, *v_tslot;
    int64_t *v_hits, *v_hits_n, *v_imp_n, *v_imp_slot;
    int64_t H, W, S, Hv;                  // the rows' list width (Hv + 1), worlds, slots, the view's list width
};

static __device__ __forceinline__ bool row_live(const Composed& q, int64_t i) {
    // VecComposedPark's `live`: free, not moribund, existing
    const Modes& s = q.c.m;
    return s.free_[i] && !s.moribund[i] && q.c.exists[i];
}

// ================================================================================================== the kernels

__global__ void composed_mark_kernel(Composed q) {
    // Between the prepass and phase 1: the followers phase 1 releases (combat_phase1_kernel's condition: a member
    // of a row starting warp proper at its turn this tick, StopAllFollowers' own test since follow_ptr
    // names the row and the follow id is its id; here too, for the members SetBoid took out of a
    // follow mode, whose release keeps their follow row and so their owner's row)
    const Modes& s = q.c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    const int64_t t = s.follow_ptr[i];
    q.released[i] = (member(q.c, i, t, s.follow_id[i]) && s.act[t]) ? 1 : 0;
}

__global__ void composed_pre_kernel(Composed q) {
    // After phase 1 (VecComposedPark._phase1's end and evolve_torch's start of phase 3): mLastC and the collision list of
    // a live ball cleared; FLAG_WARP_MASSIVE; a live massive ball's mLastCollision -1; the warp path (_warp_path) of a
    // ball in warp proper on a copy of its state; the drop-out at rest, |v|^2 < 0.1 (_at_rest)
    const Modes& s = q.c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    // StopAllFollowers' Stop of a released follower (combat_release's stop_following: its follow row now -1) clears
    // its owner's row in the torch path (VecCombatPark._stop_rows); a refused release (not finite) keeps it
    if (q.released[i] && s.follow_ptr[i] < 0) q.c.owner_ptr[i] = -1;
    const bool live = row_live(q, i);
    Ball b = load_ball(s, i);
    if (live) {
        d3_store(q.last_c, i, d3_zero());
        for (int64_t k = 0; k < q.H; ++k) q.hits[i * q.H + k] = -1;
        q.c.hits_n[i] = 0;
    }
    const bool warping = live && b.mode == MODE_WARP && b.effect_stamp >= 0;
    if (warping && b.massive) q.c.flags[i] |= P_FLAG_WARP_MASSIVE;
    if (live && b.massive) {
        b.last_collision = -1.0;
        s.last_collision[i] = -1.0;
    }
    D3 p = b.pos, v = b.vel;
    bool drop = false;
    if (warping) {
        Ball w = b;                                                   // a copy: the drop-out is the commit's
        warp_distance(w, &p, &v, dmul(ll2d(b.clock - b.effect_stamp), s.dt));
        drop = w.mode == MODE_STOP;
    }
    d3_store(q.wp, i, p);
    d3_store(q.wv, i, v);
    q.dropped[i] = drop ? 1 : 0;
    const bool vdrop = drop && !b.massive;                        // _gradient_rows's dropped: dropped & ~massive
    d3_store(q.fpos, i, vdrop ? p : b.pos);
    d3_store(q.fvel, i, vdrop ? v : b.vel);
#if XBPP_FAULT == 1
    q.rest[i] = 0;                                                    // planted fault: no re-filing at rest
#else
    q.rest[i] = (drop && !(d3_lensq(v) >= KD(P_REST_SQ))) ? 1 : 0;
#endif
}

__global__ void composed_check_kernel(Composed q, const uint8_t* which) {
    // VecComposedPark._file's refusal (_GradientView.filing_bad) for the rows just filed: a key at the partition's edge
    // or a side past it, or a top-level ball wider than a top box, flags the row (FLAG_PARTITION)
    const Modes& s = q.c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n || !which[i]) return;
    const int64_t lv = q.level[i];
    const int64_t n = P_LEVEL_GRID[lv];
    bool bad = false;
    for (int k = 0; k < 3; ++k) {
        const int64_t key = q.key[3 * i + k], side = key + q.mod[3 * i + k];
        if (key <= 0 || key >= n - 1 || side < 0 || side >= n) bad = true;
    }
    if (lv == 0 && dmul(s.radius[i], 2.0) > KD(P_LEVEL_WIDTH0)) bad = true;
    if (bad) q.c.flags[i] |= P_FLAG_PARTITION;
}

__global__ void composed_gather_kernel(Composed q) {
    // The view (VecComposedPark._fill_view, _gradient_rows, _pair_mask): one thread a slot; an empty slot as the torch
    // path fills it (nothing exists; mode STOP); a ball dropping out this tick at its drop-out state; the gradient's
    // outputs reset (mLastC 0, mLastCollision -1, the lists empty)
    const Modes& s = q.c.m;
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= q.W * q.S) return;
    for (int64_t k = 0; k < q.Hv; ++k) q.v_hits[j * q.Hv + k] = -1;
    q.v_hits_n[j] = 0;
    q.v_imp_n[j] = 0;
    q.v_imp_slot[j] = -1;
    d3_store(q.v_last_c, j, d3_zero());
    q.v_last_col[j] = -1.0;
    const int64_t r = q.slot_rows[j];
    if (r < 0) {
        q.v_exists[j] = q.v_free[j] = q.v_massive[j] = q.v_junk[j] = q.v_cloaked[j] = 0;
        q.v_responder[j] = q.v_missile[j] = q.v_dropped[j] = 0;
        q.v_id[j] = 0;
        q.v_mass[j] = q.v_agility[j] = q.v_tf[j] = q.v_radius[j] = 0.0;
        d3_store(q.v_pos, j, d3_zero());
        d3_store(q.v_vel, j, d3_zero());
        d3_store(q.v_last_g, j, d3_zero());
        q.v_level[j] = 0;
        for (int k = 0; k < 3; ++k) q.v_key[3 * j + k] = q.v_mod[3 * j + k] = 0;
        q.v_mode[j] = MODE_STOP;
        q.v_owner[j] = q.v_follow[j] = q.v_bubble[j] = 0;
        q.v_tslot[j] = -1;
        return;
    }
    const bool exists = q.c.exists[r] && !s.moribund[r];
    const bool live = row_live(q, r);
    const bool massive = s.massive[r] != 0;
    const bool drop = q.dropped[r] != 0 && !massive;              // _gradient_rows's dropped: dropped & ~massive
    q.v_exists[j] = exists ? 1 : 0;
    q.v_free[j] = s.free_[r] ? 1 : 0;
    q.v_massive[j] = massive ? 1 : 0;
    q.v_junk[j] = 0;
    q.v_cloaked[j] = q.c.cloaked[r] != 0 ? 1 : 0;
    q.v_responder[j] = (live && massive) ? 1 : 0;
    q.v_missile[j] = (exists && s.mode[r] == MODE_MISSILE) ? 1 : 0;
    q.v_dropped[j] = drop ? 1 : 0;
    q.v_id[j] = s.ids[r];
    q.v_mass[j] = s.mass[r];
    q.v_agility[j] = s.agility[r];
    q.v_tf[j] = s.time_factor[r];
    q.v_radius[j] = s.radius[r];
#if XBPP_FAULT == 4
    d3_store(q.v_pos, j, d3_load(s.pos, r));                          // planted fault: at its warp position
    d3_store(q.v_vel, j, d3_load(s.vel, r));
#else
    d3_store(q.v_pos, j, drop ? d3_load(q.wp, r) : d3_load(s.pos, r));
    d3_store(q.v_vel, j, drop ? d3_load(q.wv, r) : d3_load(s.vel, r));
#endif
    d3_store(q.v_last_g, j, d3_load(s.last_g, r));
    q.v_level[j] = q.flevel[r];                                     // the view's filing (at rest: the view's state's)
    for (int k = 0; k < 3; ++k) {
        q.v_key[3 * j + k] = q.fkey[3 * r + k];
        q.v_mod[3 * j + k] = q.fmod[3 * r + k];
    }
    q.v_mode[j] = s.mode[r];
    q.v_owner[j] = s.owner_id[r];
    q.v_follow[j] = s.follow_id[r];
    q.v_bubble[j] = q.c.bubble[r];
    const int64_t t = s.follow_ptr[r];
    q.v_tslot[j] = t >= 0 ? q.row_slot[t] : -1;
}

__global__ void composed_scatter_kernel(Composed q) {
    // The gradient's results back on the responders' rows (_gradient_rows): mLastC, mLastCollision, the collision list
    // (the view's width plus the column of the master's self-collision) and its length
    const Modes& s = q.c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    const int64_t j = q.row_slot[i];
    if (!q.v_responder[j]) return;
    d3_store(q.last_c, i, d3_load(q.v_last_c, j));
    s.last_collision[i] = q.v_last_col[j];
    for (int64_t k = 0; k < q.Hv; ++k) q.hits[i * q.H + k] = q.v_hits[j * q.Hv + k];
    for (int64_t k = q.Hv; k < q.H; ++k) q.hits[i * q.H + k] = -1;
    q.c.hits_n[i] = q.v_hits_n[j];
}

__global__ void composed_commit_kernel(Composed q) {
    // Phases 3 and 4 (evolve_torch's end; the commit kernel with collisions): the drop-out from the precomputed
    // warp path, the warp path's state for a ball coasting in warp proper or Integrate with mLastG + mLastC, the commit,
    // the rotation a tick on or the disabled missile's snap, the moved flag (the re-filing), the dispatch's counts, the
    // master's self-collision appended to the list, the clock. (as combat_commit_kernel): a free mushroom grows, or is removed, at its turn without the swap (Ballpark.cpp:562-578: phase 3's
    // integrated state stays in the old members, the rotation's too), a growing one marked for the re-filing with its
    // new radius and its state at the tick's start; a member of a mushroom removed this tick whose turn comes after the
    // mushroom's is released first, from its state before the swap (GotoDirection(mNewVel)), and committed in the mode
    // that leaves it (the others: the fifth launch, combat_after_kernel)
    const Combat& c = q.c;
    const Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    Ball b = load_ball(s, i);
    const bool live = row_live(q, i);
    const bool released = member(c, i, b.follow_ptr, b.follow_id) && c.dying[b.follow_ptr]
                          && s.ids[b.follow_ptr] < b.id;
    int64_t ev_n = 0;
    uint8_t ev_self = 0, moved = 0;
    if (live) {
        const bool warping = b.mode == MODE_WARP && b.effect_stamp >= 0;
        const bool drop = q.dropped[i] != 0;
        if (drop) {                                                   // WarpDistance's drop-out (Stop: STOP alone)
            b.massive = true;
            b.sensor = true;
            b.mode = MODE_STOP;
            b.pos = d3_load(q.wp, i);
            b.vel = d3_load(q.wv, i);
        }
#if XBPP_FAULT == 2
        const D3 lc = d3_zero();                                      // planted fault: mLastC left out
#else
        const D3 lc = d3_load(q.last_c, i);
#endif
        const D3 a = D3{dadd(b.last_g.x, lc.x), dadd(b.last_g.y, lc.y), dadd(b.last_g.z, lc.z)};   // mLastG + mLastC
        const double m = dmul(b.mass, b.agility);
        D3 p = b.pos, v = b.vel;
        if (warping && !drop) {                                       // coasting along the warp path
            p = d3_load(q.wp, i);
            v = d3_load(q.wv, i);
        } else {
            integrate(&p, &v, a, m, s.friction, b.tf, s.dt);
        }
        Q4 r0 = Q4{0.0f, 0.0f, 0.0f, 1.0f}, rot1 = r0;
        V3 w0 = V3{0.0f, 0.0f, 0.0f}, w1 = w0;
        if (s.dynamical) {                                            // the rotation a tick on (phase 3), by its mode then
            r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
            w0 = V3{tof(s.ang_vel[3 * i]), tof(s.ang_vel[3 * i + 1]), tof(s.ang_vel[3 * i + 2])};
            if (!disabled(c, b.mode)) {                               // ApplyTorque a tick on (phase 3)
                const float fmass = tof(b.mass), ag = tof(b.agility), rag = tof(s.rot_agility[i]);
                const float tau = fdiv(fmul(fmul(fmass, ag), rag), s.ffriction);
                const V3 torque = V3{tof(s.torque[3 * i]), tof(s.torque[3 * i + 1]), tof(s.torque[3 * i + 2])};
                xbpo::apply_torque(r0, w0, torque, tof(s.rot_roll[i]), tau, s.fdt, &rot1, &w1);
            } else {                                                  // phase 3 leaves mOldRot as it was
                rot1 = Q4{tof(s.old_rot[4 * i]), tof(s.old_rot[4 * i + 1]), tof(s.old_rot[4 * i + 2]), tof(s.old_rot[4 * i + 3])};
                w1 = V3{tof(s.old_ang_vel[3 * i]), tof(s.old_ang_vel[3 * i + 1]), tof(s.old_ang_vel[3 * i + 2])};
            }
        }
#if XBPP_FAULT != 6
        if (released) release_in_phase1(c, b, i);                    // the mushroom's turn came first
#endif
#if XBPP_FAULT == 7
        if (false) {                                                  // planted fault: a mushroom's swap not skipped
#else
        if (b.mode == MODE_MUSHROOM) {
#endif
            d3_store(s.old_pos, i, p);                                // phase 3's store; phase 4 skips the swap
            d3_store(s.old_vel, i, v);
            if (s.dynamical) {
                s.old_rot[4 * i] = tod(rot1.x);
                s.old_rot[4 * i + 1] = tod(rot1.y);
                s.old_rot[4 * i + 2] = tod(rot1.z);
                s.old_rot[4 * i + 3] = tod(rot1.w);
                s.old_ang_vel[3 * i] = tod(w1.x);
                s.old_ang_vel[3 * i + 1] = tod(w1.y);
                s.old_ang_vel[3 * i + 2] = tod(w1.z);
            }
            if (c.dying[i]) {
                // RemoveBall in Evolve (Ballpark.cpp:5471): Stop(Ball*) (a mushroom is in no follow mode), its
                // followers released (by their own threads), moribund and non-massive with the tick's time as its
                // stamp until BringOutTheDeads (VecCombatPark.bring_out_the_deads); it stays filed, as Destiny's does
                b.mode = MODE_STOP;
                s.moribund[i] = 1;
                b.massive = false;
                b.effect_stamp = b.clock;
            } else {                                                  // mRadius = range*pow(timeFraction,0.25f)
                s.radius[i] = tod(fmul(tof(b.follow_range), g_powf(mushroom_fraction(c, i), KF(F_QUARTER))));
#if XBPP_FAULT != 5
                moved = 1;                                            // InsertBallInBoxes (Ballpark.cpp:571), next launch
#endif
            }
        } else {
            const D3 before = b.pos;
            d3_store(s.old_pos, i, b.pos);
            d3_store(s.old_vel, i, b.vel);
            b.pos = p;
            b.vel = v;
#if XBPP_FAULT == 6
            if (released) release_in_phase1(c, b, i);                // planted fault: released after its commit
#endif
            if (s.dynamical) {
                const bool off = disabled(c, b.mode);                 // phase 4's mode (a release may have changed it)
                const bool rot_ok = ffinite(xbpo::q_lensq(rot1));
                const bool w_ok = ffinite(xbpo::v3_dot(w1, w1));
                Q4 rn = rot_ok ? rot1 : r0, ro = rot_ok ? r0 : rot1;
                V3 wn = w_ok ? w1 : w0, wo = w_ok ? w0 : w1;
                if (off) {                                            // SnapOrientation (phase 4)
                    Q4 sr;
                    if (snap(b.vel, &sr)) {
                        rn = sr;
                        ro = sr;
                        wn = V3{0.0f, 0.0f, 0.0f};
                        wo = wn;
                    }
                }
                s.rot[4 * i] = tod(rn.x);
                s.rot[4 * i + 1] = tod(rn.y);
                s.rot[4 * i + 2] = tod(rn.z);
                s.rot[4 * i + 3] = tod(rn.w);
                s.old_rot[4 * i] = tod(ro.x);
                s.old_rot[4 * i + 1] = tod(ro.y);
                s.old_rot[4 * i + 2] = tod(ro.z);
                s.old_rot[4 * i + 3] = tod(ro.w);
                s.ang_vel[3 * i] = tod(wn.x);
                s.ang_vel[3 * i + 1] = tod(wn.y);
                s.ang_vel[3 * i + 2] = tod(wn.z);
                s.old_ang_vel[3 * i] = tod(wo.x);
                s.old_ang_vel[3 * i + 1] = tod(wo.y);
                s.old_ang_vel[3 * i + 2] = tod(wo.z);
            }
            // phase 4: the re-filing of a ball that moved at least sqrt(0.1) m (the filing kernel, next), and
            // Ball::DispatchCollisions (COL-719) for a missile that moved: its target's id hits_n times; on the master
            // itself when its owner is gone, cloaked or in another bubble, appended to its list after them
            moved = !(d3_lensq(d3_sub(b.pos, before)) < KD(C_DISPATCH_MIN_MOVE_SQ)) ? 1 : 0;
            if (moved && b.mode == MODE_MISSILE) {
                ev_n = c.hits_n[i];
                if (c.master) {
                    const int64_t o = c.owner_ptr[i];
                    const bool gone = !row_ok(c, o) || !c.exists[o];
                    ev_self = (gone || (c.bubble[o] != c.bubble[i] && c.bubble[i] != -1) || c.cloaked[o] != 0) ? 1 : 0;
                }
            }
#if XBPP_FAULT != 3
            if (ev_self) {
                const int64_t k = c.hits_n[i];
                q.hits[i * q.H + (k < q.H - 1 ? k : q.H - 1)] = b.id;
                c.hits_n[i] = k + 1;
            }
#endif
        }
    } else if (released) {
        release_in_phase1(c, b, i);                                   // not evolved, released all the same
    }
    b.clock += 1;
    store_ball(s, i, b);
    s.clock[i] = b.clock;
    c.ev_n[i] = ev_n;
    c.ev_self[i] = ev_self;
    q.moved[i] = moved;
}

__global__ void composed_refile_mark_kernel(Composed q) {
    // After the fifth launch: PetrifyTroll's SetBallFree(false) re-files each troll it petrified
    // (InsertBallInBoxes after Stop and SetBallVelocity(0, 0, 0), out of the free set: full.py's set_free), so the rows
    // combat_after_kernel marks in `petrified` are the filing's next (`moved`, the commit's, already consumed)
    const Modes& s = q.c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
#if XBPP_FAULT == 8
    q.moved[i] = 0;                                                   // planted fault: the petrified not re-filed
#else
    q.moved[i] = q.c.petrified[i];
#endif
}

}  // namespace xbpm

// ====================================================================================== the C interface (ctypes)

extern "C" {

int xbpp_abi(void) { return XBPP_ABI; }

int xbpp_fault(void) { return XBPP_FAULT; }

// One part of the composed tick on `stream` (exact_ballpark.vec_composed_cuda.tick runs the launches between them):
// 0 the prepass, the mark, the phase 1, then composed_pre_kernel; 1 the check of the rows filed at rest, then the gather;
// 2 the scatter, then the commit; 3 the check of the rows re-filed after the commit. No allocation, no host sync.
int xbpp_tick(int part, const xbpm::Composed* q, void* stream, int block) {
    if (block <= 0 || block > 1024 || block % 32 != 0 || part < 0 || part > 3) return -1;
    const int64_t n = q->c.m.n;
    if (n <= 0) return 0;
    if (q->H != q->Hv + 1 || q->W < 1 || q->S < 1) return -1;
    const unsigned grid = grid_for(n, block);
    cudaStream_t st = (cudaStream_t)stream;
    int rc = 0;
    if (part == 0) {
        xbpm::combat_prepass_kernel<<<grid, block, 0, st>>>(q->c);
        if ((rc = launch_check())) return rc;
        xbpm::composed_mark_kernel<<<grid, block, 0, st>>>(*q);
        if ((rc = launch_check())) return rc;
        xbpm::combat_phase1_kernel<<<grid, block, 0, st>>>(q->c);
        if ((rc = launch_check())) return rc;
        xbpm::composed_pre_kernel<<<grid, block, 0, st>>>(*q);
        return launch_check();
    }
    if (part == 1) {
        xbpm::composed_check_kernel<<<grid, block, 0, st>>>(*q, q->rest);
        if ((rc = launch_check())) return rc;
        xbpm::composed_gather_kernel<<<grid_for(q->W * q->S, block), block, 0, st>>>(*q);
        return launch_check();
    }
    if (part == 2) {
        xbpm::composed_scatter_kernel<<<grid, block, 0, st>>>(*q);
        if ((rc = launch_check())) return rc;
        xbpm::composed_commit_kernel<<<grid, block, 0, st>>>(*q);
        return launch_check();
    }
    xbpm::composed_check_kernel<<<grid, block, 0, st>>>(*q, q->moved);
    return launch_check();
}

// after the composed tick and the host's yaw, pitch and roll, the fifth launch on `stream`
// (exact_ballpark.vec_composed_cuda.after runs the filing between the parts): 0 the combat_after_kernel (a removed
// mushroom's members whose turns came first, released; PetrifyTroll, marked in `petrified`), then the re-filing mark
// (`moved` := `petrified`); 1 the check of the rows re-filed. No allocation, no host sync.
int xbpp_after(int part, const xbpm::Composed* q, void* stream, int block) {
    if (block <= 0 || block > 1024 || block % 32 != 0 || part < 0 || part > 1) return -1;
    const int64_t n = q->c.m.n;
    if (n <= 0) return 0;
    const unsigned grid = grid_for(n, block);
    cudaStream_t st = (cudaStream_t)stream;
    int rc = 0;
    if (part == 0) {
        xbpm::combat_after_kernel<<<grid, block, 0, st>>>(q->c);
        if ((rc = launch_check())) return rc;
        xbpm::composed_refile_mark_kernel<<<grid, block, 0, st>>>(*q);
        return launch_check();
    }
    xbpm::composed_check_kernel<<<grid, block, 0, st>>>(*q, q->moved);
    return launch_check();
}

// The kernels as loaded on the current device (loading each, so that no later launch loads one, as a captured launch
// must not): registers, local memory, the binary and PTX versions, the most threads a block
int xbpp_info(int which, int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    cudaError_t e;
    switch (which) {
        case 0: e = cudaFuncGetAttributes(&fa, xbpm::combat_prepass_kernel); break;
        case 1: e = cudaFuncGetAttributes(&fa, xbpm::combat_phase1_kernel); break;
        case 2: e = cudaFuncGetAttributes(&fa, xbpm::composed_pre_kernel); break;
        case 3: e = cudaFuncGetAttributes(&fa, xbpm::composed_check_kernel); break;
        case 4: e = cudaFuncGetAttributes(&fa, xbpm::composed_gather_kernel); break;
        case 5: e = cudaFuncGetAttributes(&fa, xbpm::composed_scatter_kernel); break;
        case 6: e = cudaFuncGetAttributes(&fa, xbpm::composed_commit_kernel); break;
        case 8: e = cudaFuncGetAttributes(&fa, xbpm::combat_after_kernel); break;
        case 9: e = cudaFuncGetAttributes(&fa, xbpm::composed_refile_mark_kernel); break;
        default: e = cudaFuncGetAttributes(&fa, xbpm::composed_mark_kernel); break;
    }
    if (e != cudaSuccess) return (int)e;
    *regs = fa.numRegs;
    *local_bytes = (int)fa.localSizeBytes;
    *binary_version = fa.binaryVersion;
    *ptx_version = fa.ptxVersion;
    *max_threads = fa.maxThreadsPerBlock;
    return 0;
}

}  // extern "C"
