// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's combat pieces on a CUDA device: the flight modes of vec_modes_cuda.cu
// (GOTO, STOP, FOLLOW, ORBIT under both laws, WARP, with dynamical orientation) and the two follow modes it lacks,
// MISSILE (EvolveMissile, Ballpark.cpp:972; MOV-723) and FORMATION (EvolveFormation, Ballpark.cpp:1023; MOV-726), with
// their releases (StopAllFollowers, Ballpark.cpp:5416; MOV-730: a missile flies off non-massive), a missile's contact
// with its target in the gradient method (Partition::GetCollisionCandidates, Ballpark::Gradient and Potential,
// Ballpark.cpp:2745-2990; COL-703, COL-711) and its collision events at the commit (Ball::DispatchCollisions with the
// master's self-collision, Ballpark.cpp:600-639; COL-719); and Destiny's GetAccuracy (Ballpark.cpp:5156; CMB-702) for
// B pairs. Built and loaded by exact_ballpark.vec_combat_cuda; held to the scalar reference (exact_ballpark.full) bit for
// bit by tests/test_movement_vec_combat.py and tests/test_movement_vec_accuracy.py.
//
// A tick is four launches, each one thread a ball, in Ballpark::Evolve's order (Ballpark.cpp:421):
//  1. the prepass (as vec_modes_cuda's): what each ball shows the others during phase 1 (its start position and
//     velocity, its vertical velocity after its own turn, whether it starts warp proper at its turn);
//  2. phase 1: every ball's behaviour (the release by a target starting warp proper before or after its own turn, by
//     ids; the mode's thrust; dynamical orientation unless disabled for missiles), written in place; positions do not
//     change in phase 1, and a ball reads the others only through the prepass's records and fields no ball's phase 1
//     changes (radius, ids, rotation, formation ids, the yaw, pitch and roll);
//  3. the contacts: each massive normal missile's Potential with its one candidate, its target (the gradient's
//     recursion at three depths), counted: the target's id is what the gradient appends to the collision list, and a
//     missile's contact with its target never changes either ball's motion (Ballpark.cpp:2982: the response is not
//     kept for a missile's own target; the target does not see missiles);
//  4. phases 3 and 4 for every ball (the warp path or the integrator, the rotation a tick on, the commit, the rotation
//     snapped for a missile whose dynamical orientation is disabled), the dispatch's counts for each missile that moved
//     at least sqrt(0.1) m, and the world clock.
//
// The domain (exact_ballpark.vec_combat.VecCombatPark.check refuses the rest): the gradient method; no two balls other
// than a missile and its target meet (as vec_modes_cuda: there is no other collision code); normal missiles only (a
// defender missile, negative owner, sees every ball); one bubble a world, live; no space junk.
//
// (1 November 2026): the commands and mushrooms in this tick. A ball's follower set is
// StopAllFollowers' test (Ballpark.cpp:5433-5450: follow_ptr names the row and the follow id is its id; vec_modes_cuda's
// member_of), so a member the Python SetBoid took out of a follow mode is released too, and Stop(Ball*) clears the
// follow fields only in a follow mode. A free mushroom (MUSHROOM, Ballpark.cpp:562-578) grows at its turn in phase 4,
// radius = range x powf(fraction, 0.25f) with glibc's powf (`g_powf`, transcribed from exact_ballpark.dlibm_powf), and
// is removed when grown (RemoveBall in Evolve: stopped, moribund, non-massive, its stamp the tick's time), without the
// swap; the prepass records which mushroom goes (`dying`), its members with higher ids are released before their own
// commit, the others in a fifth launch (`combat_after_kernel`) after the host's yaw, pitch and roll, which then also
// petrifies each troll due (PetrifyTroll, Ballpark.cpp:6327: after the loop, TrollReady checked again) and marks it
// (`petrified`: the host snaps its yaw, pitch and roll, SetBallVelocity's, without dynamical orientation).
//
// Exactness: as vec_modes_cuda.cu, whose device code this file includes unedited (and with it vec_orient_cuda.cu's):
// double operations __dadd_rn ... __dsqrt_rn in the reference's order, float operations __fadd_rn ... __fsqrt_rn where
// Destiny computes in float, conversions explicit, a fused multiply-add only inside the C library's transcribed
// functions (pow, powf, sinf, cosf, acosf, atan2f); no function of CUDA's math library; every constant from the generated
// headers (exact_ballpark.vec_combat_cuda.tables_header, which reads the reference's modules), never typed.

#include "vec_modes_cuda.cu"
#include "vec_combat_cuda_tables.h"

#define XBPC_ABI 2                       // 2: (`dying`, `petrified`, the fifth launch)

namespace xbpm {   // the combat code joins vec_modes_cuda's namespace (its using-declarations pin the arithmetic)

using xbpo::fsqrt;
using xbpo::fmax_;
using xbpo::ffinite;
using xbpo::M3;
using xbpo::fbits;
using xbpo::ffrom;

// ============================================================================================ the park's state

struct Combat {
    xbpm::Modes m;                       // vec_modes_cuda's tensors (the flight)
    uint8_t* exists;                     // the row holds a ball (a removed ball's row is empty)
    int64_t *formation_id, *owner_ptr;   // the formation (char semantics), the row of |owner| (-1: none)
    int64_t *hits_n, *ev_n;              // contacts with the target this tick; target collisions dispatched
    uint8_t* ev_self;                    // the master's self-collision dispatched
    int64_t* bubble;                     // the bubble ids (one a world in the domain; -1 on a client)
    int64_t* cloaked;                    // isCloaked (a byte held in int64)
    double *yaw, *pitch, *roll;          // without dynamical orientation: a leader's rotation (start of the tick)
    int64_t *formation_slots, *flags;    // a leader's slot bits (16); each row's domain flags (FLAG_* of vec_combat)
    uint8_t *dying, *petrified;          // the prepass's record: a mushroom removed this tick; the trolls petrified
    const double* formations;            // [n_formations, max_slots, 3] slot offsets (doubles)
    const int64_t* formation_len;        // [n_formations] slots in each
    int64_t n_formations, max_slots, tick_ms;
    int do_missiles_off, master;
};

// ======================================================================================== helpers (reference order)

static __device__ __forceinline__ bool follow_mode(int64_t m) {
    // followModes (Ballpark.cpp:68)
    return m == MODE_FOLLOW || m == MODE_ORBIT || m == MODE_MISSILE || m == MODE_FORMATION;
}

static __device__ __forceinline__ bool row_ok(const Combat& c, int64_t t) { return t >= 0 && t < c.m.n; }

static __device__ __forceinline__ Q4 q_rotation_ypr(float yaw, float pitch, float roll) {
    // carbonengine/math's RotationQuaternion(yaw, pitch, roll): DirectXMath's XMQuaternionRotationRollPitchYaw scalar
    // path (exact_ballpark.full.q_rotation_ypr): half angles, cosf and sinf, four sums of float products left to right
    const float hp = fmul(pitch, KF(F_HALF)), hy = fmul(yaw, KF(F_HALF)), hr = fmul(roll, KF(F_HALF));
    const float cp = xbpo::g_cosf(hp), sp = xbpo::g_sinf(hp);
    const float cy = xbpo::g_cosf(hy), sy = xbpo::g_sinf(hy);
    const float cr = xbpo::g_cosf(hr), sr = xbpo::g_sinf(hr);
    return Q4{fadd(fmul(fmul(cr, sp), cy), fmul(fmul(sr, cp), sy)), fsub(fmul(fmul(cr, cp), sy), fmul(fmul(sr, sp), cy)),
              fsub(fmul(fmul(sr, cp), cy), fmul(fmul(cr, sp), sy)), fadd(fmul(fmul(cr, cp), cy), fmul(fmul(sr, sp), sy))};
}

static __device__ __forceinline__ V3 rotated_vector(const Combat& c, int64_t t, V3 v) {
    // Ball::GetRotatedVector (Ball.cpp:2191): by the rotation with dynamical orientation, otherwise by the yaw, pitch and
    // roll as floats through RotationQuaternion(yaw, pitch, roll)
    Q4 q;
    if (c.m.dynamical) {
        q = Q4{tof(c.m.rot[4 * t]), tof(c.m.rot[4 * t + 1]), tof(c.m.rot[4 * t + 2]), tof(c.m.rot[4 * t + 3])};
    } else {
        q = q_rotation_ypr(tof(c.yaw[t]), tof(c.pitch[t]), tof(c.roll[t]));
    }
    return xbpo::transform_coord(v, xbpo::m_rotation_q(q));
}

static __device__ __forceinline__ D3 goto_thrust_missile(const Ball& b, D3 target, double friction) {
    // GotoThrust with its missile flag (Ballpark.cpp:1398): toward the target at maxThrust, never tapered
    const D3 a = d3_sub(target, b.pos);
    const double max_thrust = ddiv(dmul(dmul(friction, b.sf), b.max_vel), dmul(b.mass, b.agility));
    return d3_scale(d3_normalized(a), max_thrust);
}

static __device__ __noinline__ D3 evolve_missile(Ball& b, const Combat& c, int64_t t) {
    // EvolveMissile (MOV-723): straight on toward mGoto (tapered: no missile flag) while (time - launch) x tick is at
    // most 800 ms; then the lead point: the target's position plus its velocity times dist / maxVel, offset along the
    // line from it by followRange + both radii (the follow range is minus both radii: the centre), at full thrust
    const xbpm::Modes& s = c.m;
    if ((b.clock - b.effect_stamp) * c.tick_ms <= M_MISSILE_STRAIGHT_MS) return goto_thrust(b, b.gp, s.friction, s.dt);
    const D3 op = d3_load(s.pos0, t);
    D3 ov = d3_load(s.vel0, t);
    if (s.ids[t] < b.id) ov.y = s.vy_after[t];                    // the target's turn came first
    const D3 delta = d3_sub(b.pos, op);
    const double dist = d3_len(delta);
    const double ct = ddiv(dist, b.max_vel);
    D3 target = d3_add(op, d3_scale(ov, ct));
    const double r = dadd(dadd(b.follow_range, b.radius), s.radius[t]);
    if (dist == 0.0) {
        target = D3{dadd(op.x, dmul(1.0, r)), dadd(op.y, dmul(0.0, r)), dadd(op.z, dmul(0.0, r))};
    } else {
        const double inv = ddiv(1.0, dist);                           // Vector3d's operator/ multiplies by it
        target = D3{dadd(target.x, dmul(dmul(delta.x, r), inv)), dadd(target.y, dmul(dmul(delta.y, r), inv)),
                    dadd(target.z, dmul(dmul(delta.z, r), inv))};
    }
    b.gp = target;
    return goto_thrust_missile(b, target, s.friction);
}

static __device__ __noinline__ D3 evolve_formation(Ball& b, const Combat& c, int64_t t) {
    // EvolveFormation (MOV-726): the leader's slot offset, scaled to both radii plus its length, turned by the leader's
    // rotation through a float vector, added to the leader's position; GotoThrust toward it (tapered). Zero (mGoto kept)
    // when the leader has no loaded formation or the slot is past its end
    const xbpm::Modes& s = c.m;
    const int64_t fid = c.formation_id[t];
    if (fid < 0 || fid > c.n_formations - 1) return d3_zero();
    const int64_t slot = (int64_t)(int8_t)(uint8_t)(b.effect_stamp & 0xFF);      // char(mEffectStamp)
    const int64_t len = c.formation_len[fid];
    if (slot > len - 1 || slot < 0) return d3_zero();             // a negative slot is refused by the park's check
    const double* o = c.formations + (fid * c.max_slots + slot) * 3;
    const D3 offset = D3{o[0], o[1], o[2]};
    double length = d3_len(offset);
    if (length == 0.0) length = 1.0;
    const double k = dadd(dadd(b.radius, s.radius[t]), length);
    const double inv = ddiv(1.0, length);
    const V3 vec = V3{tof(dmul(dmul(offset.x, k), inv)), tof(dmul(dmul(offset.y, k), inv)),
                      tof(dmul(dmul(offset.z, k), inv))};
    const V3 rv = rotated_vector(c, t, vec);
    const D3 op = d3_load(s.pos0, t);
    const D3 g = D3{dadd(op.x, tod(rv.x)), dadd(op.y, tod(rv.y)), dadd(op.z, tod(rv.z))};
    b.gp = g;
    return goto_thrust(b, g, s.friction, s.dt);
}

static __device__ __forceinline__ int64_t combat_release(Ball& b, bool apply) {
    // StopAllFollowers on this follower (MOV-730): a missile is made non-massive and, like an interactive orbiter,
    // flies on by GotoDirection(its velocity) (refused when the velocity or the point is not finite: the mode stays);
    // any other follower stops. Returns the mode it is left in.
    const bool missile = b.mode == MODE_MISSILE;
    if (missile || (b.interactive && b.mode == MODE_ORBIT)) {
        if (missile && apply) b.massive = false;
        if (!d3_finite(b.vel)) return b.mode;
        const D3 p = d3_add(b.pos, d3_scale(d3_normalized(b.vel), KD(M_DIR_DIST)));
        if (!d3_finite(p)) return b.mode;
        if (apply) {
            stop_following(b);
            b.gp = p;
            if (b.sf == 0.0) b.sf = 1.0;
            b.mode = MODE_GOTO;
        }
        return MODE_GOTO;
    }
    if (apply) {
        if (follow_mode(b.mode)) stop_following(b);               // Stop(Ball*): only a follow mode leaves the set
        b.mode = MODE_STOP;
    }
    return MODE_STOP;
}

static __device__ __forceinline__ bool member(const Combat& c, int64_t i, int64_t t, int64_t follow_id) {
    // ball i is in row t's follower set as StopAllFollowers acts on it (vec_modes_cuda's member_of: follow_ptr names
    // the row, the follow id is its id; a ball SetBoid took out of a follow mode keeps both), and holds a ball
    return c.exists[i] && member_of(c.m, t, follow_id);
}

static __device__ __forceinline__ void free_on_stop(const Combat& c, const Ball& b, int64_t i) {
    // Stop(Ball*) on the master frees a FORMATION follower's slot in its leader (Ballpark.cpp:4629-4633):
    // Ball::FreeFormationSlot((size_t) mEffectStamp) (Ball.cpp:2324): nothing without a loaded formation or for a slot
    // past its end; bitset<16>::reset of a slot of 16 or more throws in Destiny (flagged: the reference refuses it).
    // Followers of one leader clear their bits with an integer atomic AND: order-free, exact.
    if (!c.master || b.mode != MODE_FORMATION) return;
    const int64_t t = b.follow_ptr;
    if (!row_ok(c, t)) return;
    const int64_t fid = c.formation_id[t];
    if (fid == -1 || fid > c.n_formations - 1) return;
    if (fid < 0) {                                                    // mFormations at a negative index
        c.flags[i] |= FLAG_UNDEFINED;
        return;
    }
    const uint64_t n = (uint64_t)c.formation_len[fid];
    const uint64_t slot = (uint64_t)b.effect_stamp;
    if (slot > n - 1) return;
    if (slot >= FORMATION_SLOT_BITS) {
        c.flags[i] |= FLAG_UNDEFINED;
        return;
    }
    atomicAnd((unsigned long long*)&c.formation_slots[t], ~(1ULL << slot));
}

static __device__ __forceinline__ void release_in_phase1(const Combat& c, Ball& b, int64_t i) {
    // StopAllFollowers on a follower whose target starts warp proper (MOV-730), in phase 1, or a
    // mushroom removed in phase 4 (combat_commit_kernel, combat_after_kernel): the master's slot free,
    // the release, and where Stop ran (its follow row now -1; a release refused for a velocity or point that is not
    // finite keeps it) the owner's row cleared, as the torch path's Stop clears it (VecCombatPark._stop_rows;
    // before, a released missile kept a stale owner row here, unread but unequal)
    free_on_stop(c, b, i);
    combat_release(b, true);
    if (b.follow_ptr < 0) c.owner_ptr[i] = -1;
}

static __device__ __forceinline__ bool disabled(const Combat& c, int64_t mode) {
    // DynamicalOrientationDisabledForBall (Ballpark.cpp:408)
    return c.do_missiles_off && mode == MODE_MISSILE;
}

#include "glibc_powf.cuh"   // LGPL-2.1-or-later: the C library's powf
static __device__ __forceinline__ float mushroom_fraction(const Combat& c, int64_t i) {
    // `float timeFraction = (mCurrentTime - startTime)*mTickInterval*0.001f/(float)(ball->mGoto[0])` (Ballpark.cpp:566):
    // the integer product made a float (as the reference: to double, then float: one rounding below 2^53 ms), times
    // 0.001f, over the lifetime made a float
    const xbpm::Modes& s = c.m;
    const float elapsed = tof(ll2d((s.clock[i] - s.effect_stamp[i]) * c.tick_ms));
    return fdiv(fmul(elapsed, KF(F_MILLI)), tof(s.goto_pt[3 * i]));
}

static __device__ __forceinline__ Ball load_ball(const xbpm::Modes& s, int64_t i) {
    Ball b;
    b.pos = d3_load(s.pos, i);
    b.vel = d3_load(s.vel, i);
    b.gp = d3_load(s.goto_pt, i);
    b.last_g = d3_load(s.last_g, i);
    b.mass = s.mass[i];
    b.tf = s.time_factor[i];
    b.last_collision = s.last_collision[i];
    b.radius = s.radius[i];
    b.max_vel = s.max_vel[i];
    b.agility = s.agility[i];
    b.sf = s.speed_fraction[i];
    b.follow_range = s.follow_range[i];
    b.id = s.ids[i];
    b.mode = s.mode[i];
    b.effect_stamp = s.effect_stamp[i];
    b.follow_id = s.follow_id[i];
    b.owner_id = s.owner_id[i];
    b.follow_ptr = s.follow_ptr[i];
    b.clock = s.clock[i];
    b.massive = s.massive[i] != 0;
    b.interactive = s.interactive[i] != 0;
    b.sensor = s.sensor[i] != 0;
    return b;
}

static __device__ __forceinline__ void store_ball(const xbpm::Modes& s, int64_t i, const Ball& b) {
    d3_store(s.pos, i, b.pos);
    d3_store(s.vel, i, b.vel);
    d3_store(s.goto_pt, i, b.gp);
    d3_store(s.last_g, i, b.last_g);
    s.last_collision[i] = b.last_collision;
    s.speed_fraction[i] = b.sf;
    s.follow_range[i] = b.follow_range;
    s.mode[i] = b.mode;
    s.effect_stamp[i] = b.effect_stamp;
    s.follow_id[i] = b.follow_id;
    s.owner_id[i] = b.owner_id;
    s.follow_ptr[i] = b.follow_ptr;
    s.massive[i] = b.massive ? 1 : 0;
    s.sensor[i] = b.sensor ? 1 : 0;
}

// ---- the gradient's Potential for a missile and its target (Ballpark.cpp:2789; space.py's _potential) ----

static __device__ __forceinline__ D3 integrate_pos(D3 p, D3 v, D3 a, double m, double k, double tf, double dt) {
    // Ballpark::Integrate over the tick (its cached decay factor): the position
    D3 pp = p, vv = v;
    integrate(&pp, &vv, a, m, k, tf, dt);
    return pp;
}

static __device__ __forceinline__ double collide_two_spheres(D3 p0, D3 p1, D3 q0, D3 q1, double cr) {
    // CollideTwoSpheres (Collision.cpp:108; COL-705): 0 if already touching, else the earlier root in [0, 1], else -1
    const D3 p0q0 = d3_sub(p0, q0);
    const double p0q0_2 = d3_lensq(p0q0);
    if (p0q0_2 <= dmul(cr, cr)) return 0.0;
    const D3 dpq = D3{dsub(dsub(p1.x, p0.x), dsub(q1.x, q0.x)), dsub(dsub(p1.y, p0.y), dsub(q1.y, q0.y)),
                      dsub(dsub(p1.z, p0.z), dsub(q1.z, q0.z))};
    const double a = d3_lensq(dpq);
    const double b = dmul(2.0, dadd(dadd(dmul(p0q0.x, dpq.x), dmul(p0q0.y, dpq.y)), dmul(p0q0.z, dpq.z)));
    const double cc = dsub(p0q0_2, dmul(cr, cr));
    if (a == 0.0 && b == 0.0) return -1.0;
    double det = dsub(dmul(b, b), dmul(dmul(4.0, a), cc));
    if (det < 0.0) return -1.0;
    det = dsqrt(det);
    const double s2 = ddiv(dmul(dsub(-b, det), 0.5), a);
    if (s2 >= 0.0 && s2 <= 1.0) return s2;
    return -1.0;
}

struct Pair {
    D3 vel_me, vel_other, g_me, g_other;
    double m1, m2, tf_me, tf_other, cr, k, dt;
    int64_t id_me, id_other;
    bool other_free;
};

static __device__ __noinline__ int64_t contacts(const Pair& q, D3 p, D3 o, int depth) {
    // Potential(missile, target, depth): how many times the target's id is appended to the missile's collision list.
    // s = -1: none; s > 0: one (the response is not kept for a missile's own target, COL-711); s = 0 (touching): with
    // a free target and depth < 2 both are pushed apart along the normal and Potential recurses, then returns early
    // (no append) when they separate faster than 0.01 m/s; otherwise one append at this depth
    const D3 p1 = integrate_pos(p, q.vel_me, q.g_me, q.m1, q.k, q.tf_me, q.dt);
    const D3 q1 = integrate_pos(o, q.vel_other, q.g_other, q.m2, q.k, q.tf_other, q.dt);
    const double s = collide_two_spheres(p, p1, o, q1, q.cr);
    if (s == -1.0) return 0;
    if (s > 0.0) return 1;
    if (!q.other_free || depth >= 2) return 1;
    const D3 p0q0 = d3_sub(o, p);
    const double p0q0_2 = d3_lensq(p0q0);
    D3 normal;
    if (p0q0_2 == 0.0) {
        normal = q.id_me > q.id_other ? D3{1.0, 0.0, 0.0} : D3{-1.0, 0.0, 0.0};
    } else {
        normal = d3_normalized(d3_sub(o, p));
    }
    const double dist = dadd(dsub(q.cr, dsqrt(p0q0_2)), 1.0);
    const double d1 = dmul(ddiv(q.m2, dadd(q.m1, q.m2)), dist);
    const double d2 = dmul(ddiv(q.m1, dadd(q.m1, q.m2)), dist);
    const D3 pp = D3{dsub(p.x, dmul(normal.x, d1)), dsub(p.y, dmul(normal.y, d1)), dsub(p.z, dmul(normal.z, d1))};
    const D3 oo = D3{dadd(o.x, dmul(normal.x, d2)), dadd(o.y, dmul(normal.y, d2)), dadd(o.z, dmul(normal.z, d2))};
    const int64_t inner = contacts(q, pp, oo, depth + 1);
    if (d3_lensq(d3_sub(q.vel_me, q.vel_other)) > KD(C_SEPARATING_SPEED2)) return inner;
    return inner + 1;
}

// ---- SnapOrientation (Ball.cpp:290), for a missile whose dynamical orientation is disabled ----

static __device__ __noinline__ bool snap(D3 vel, Q4* rot) {
    // the rotation turned to the velocity and rolled upright; false (nothing changes) at a speed of 1e-6 m/s or less
    if (!(d3_lensq(vel) > KD(C_SNAP_MIN_SPEED2))) return false;
    const V3 dir = V3{tof(vel.x), tof(vel.y), tof(vel.z)};
    const float len_dir = xbpo::v3_len(dir);
    if (tod(len_dir) < KD(C_SNAP_MIN_LENGTH)) {
        *rot = Q4{0.0f, 0.0f, 0.0f, 1.0f};
        return true;
    }
    const V3 vec = xbpo::v3_cross(V3{0.0f, 0.0f, 1.0f}, dir);
    const float angle = xbpo::g_acosf(fmax_(-1.0f, fmin_(1.0f, fdiv(dir.z, len_dir))));
    Q4 r = xbpo::q_rotation(vec, angle);
    const V3 up = xbpo::transform_coord(V3{0.0f, 1.0f, 0.0f}, xbpo::m_rotation_q(xbpo::q_conj(r)));
    const float up_roll = (up.z < KF(F_UP_LIMIT)) ? xbpo::g_atan2f(-up.x, up.y) : 0.0f;
    const V3 heading = xbpo::transform_coord(V3{0.0f, 0.0f, 1.0f}, xbpo::m_rotation_q(r));
    r = xbpo::q_mul(r, xbpo::q_rotation(heading, up_roll));
    *rot = r;
    return true;
}

// ================================================================================================== the kernels

__global__ void combat_prepass_kernel(Combat c) {
    // vec_modes_cuda's prepass, with the combat release (a missile follower flies on, a formation follower stops)
    const xbpm::Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    s.act[i] = activates(s, i) ? 1 : 0;
    // a free mushroom whose time is up is removed at its turn (Ballpark.cpp:567: !(fraction < 1.0))
    c.dying[i] = (s.free_[i] && !s.moribund[i] && s.mode[i] == MODE_MUSHROOM
                  && !(tod(mushroom_fraction(c, i)) < 1.0)) ? 1 : 0;
    const D3 vel = d3_load(s.vel, i);
    d3_store(s.pos0, i, d3_load(s.pos, i));
    d3_store(s.vel0, i, vel);
    double vy = vel.y;
    if (s.free_[i] && !s.moribund[i] && !s.dynamical) {
        int64_t m = s.mode[i];
        {
            const int64_t t = s.follow_ptr[i];
            if (member(c, i, t, s.follow_id[i]) && s.ids[t] < s.ids[i] && activates(s, t)) {
                Ball f;
                f.mode = m;
                f.interactive = s.interactive[i] != 0;
                f.massive = s.massive[i] != 0;
                f.vel = vel;
                f.pos = d3_load(s.pos, i);
                f.gp = d3_load(s.goto_pt, i);
                f.sf = s.speed_fraction[i];
                m = combat_release(f, false);
            }
        }
        if (m == MODE_STOP) vy = dmul(dsub(vy, dmul(KD(M_STOP_DECAY), vy)), KD(M_STOP_DAMP));
    }
    s.vy_after[i] = vy;
}

__global__ void combat_phase1_kernel(Combat c) {
    // Phase 1 (EvolveBehaviorForBall, Ballpark.cpp:789) for every ball, in place
    const xbpm::Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    Ball b = load_ball(s, i);
    const bool live = s.free_[i] && !s.moribund[i];
    const bool act_self = s.act[i] != 0;
    bool release_before = false, release_after = false;
    {
        const int64_t t = b.follow_ptr;
        if (member(c, i, t, b.follow_id) && s.act[t]) {                // StopAllFollowers' members
            if (s.ids[t] < b.id) release_before = true;
            else release_after = true;
        }
    }
    if (release_before) release_in_phase1(c, b, i);
    if (live) {
        D3 a = d3_zero();
        const int64_t t = b.follow_ptr;
        if (b.mode == MODE_WARP) {
            if (b.effect_stamp >= 0) {
                a = d3_zero();
            } else if (act_self) {
                real_warp(b);
            } else {
                b.effect_stamp -= 1;
                a = goto_thrust(b, b.gp, s.friction, s.dt);
            }
        } else if (b.mode == MODE_GOTO) {
            a = goto_thrust(b, b.gp, s.friction, s.dt);
        } else if (follow_mode(b.mode) && !row_ok(c, t)) {
            a = d3_zero();
        } else if (b.mode == MODE_FOLLOW) {
            a = evolve_follow(b, d3_load(s.pos0, t), s.radius[t], s.friction, s.dt);
        } else if (b.mode == MODE_ORBIT) {
            const D3 op = d3_load(s.pos0, t);
            if (s.new_orbit) {
                D3 ov = d3_load(s.vel0, t);
                if (s.ids[t] < b.id) ov.y = s.vy_after[t];
                Q4 r0 = Q4{0.0f, 0.0f, 0.0f, 1.0f};
                if (s.dynamical) r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
                a = evolve_new_orbit(b, op, ov, s.radius[t], s, r0);
            } else {
                a = evolve_old_orbit(b, op, s.radius[t], s.friction);
            }
        } else if (b.mode == MODE_MISSILE) {
            a = evolve_missile(b, c, t);
        } else if (b.mode == MODE_FORMATION) {
            a = evolve_formation(b, c, t);
        } else if (b.mode == MODE_STOP) {
            if (!s.dynamical) b.vel.y = dmul(dsub(b.vel.y, dmul(KD(M_STOP_DECAY), b.vel.y)), KD(M_STOP_DAMP));
        }
        if (s.dynamical && !disabled(c, b.mode)) {
            const Q4 r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
            const V3 w0 = V3{tof(s.ang_vel[3 * i]), tof(s.ang_vel[3 * i + 1]), tof(s.ang_vel[3 * i + 2])};
            V3 torque;
            float roll;
            Q4 rot1;
            V3 w1;
            a = orient(b, a, s, i, r0, w0, &torque, &roll, &rot1, &w1);
            s.torque[3 * i] = tod(torque.x);
            s.torque[3 * i + 1] = tod(torque.y);
            s.torque[3 * i + 2] = tod(torque.z);
            s.rot_roll[i] = tod(roll);
        }
        b.last_g = a;
    }
    if (release_after) release_in_phase1(c, b, i);
    store_ball(s, i, b);
}

__global__ void combat_contact_kernel(Combat c) {
    // The gradient for each massive normal missile (after every ball's phase 1): its one candidate is its target
    // (Partition::GetCollisionCandidates, in its bubble), less Gradient's skips; the count of target ids appended
    const xbpm::Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    int64_t hits = 0;
    const int64_t t = s.follow_ptr[i];
    if (s.free_[i] && !s.moribund[i] && s.massive[i] && s.mode[i] == MODE_MISSILE && s.owner_id[i] >= 0 && row_ok(c, t)
        && c.bubble[i] == c.bubble[t]) {
        const int64_t mt = s.mode[t];
        const bool skip = s.ids[t] == s.owner_id[i] || (mt == MODE_MINIBALL && s.owner_id[t] == s.owner_id[i])
                          || ((mt == MODE_MISSILE || mt == MODE_MUSHROOM) && s.ids[i] == s.owner_id[t])
                          || (mt == MODE_MISSILE && s.ids[i] == s.follow_id[t]);
        if (!skip) {
            Pair q;
            q.vel_me = d3_load(s.vel, i);
            q.vel_other = d3_load(s.vel, t);
            q.g_me = d3_load(s.last_g, i);
            q.g_other = d3_load(s.last_g, t);
            q.m1 = dmul(s.mass[i], s.agility[i]);
            q.other_free = s.free_[t] != 0;
            q.m2 = q.other_free ? dmul(s.mass[t], s.agility[t]) : KD(C_FIXED_MASS);
            q.tf_me = s.time_factor[i];
            q.tf_other = s.time_factor[t];
            q.cr = dadd(s.radius[i], s.radius[t]);
            q.k = s.friction;
            q.dt = s.dt;
            q.id_me = s.ids[i];
            q.id_other = s.ids[t];
            hits = contacts(q, d3_load(s.pos, i), d3_load(s.pos, t), 0);
        }
    }
    c.hits_n[i] = hits;
}

__global__ void combat_commit_kernel(Combat c) {
    // Phases 3 and 4 for every ball in place, the missiles' dispatch counts, the clock. a free mushroom
    // grows, or is removed, at its turn without the swap (Ballpark.cpp:562-578: phase 3's integrated state stays in the
    // old members, the rotation's too); a member of a mushroom removed this tick whose turn comes after the mushroom's
    // is released first, from its state before the swap (GotoDirection(mNewVel)), and committed in the mode that leaves
    // it (the others: combat_after_kernel)
    const xbpm::Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    Ball b = load_ball(s, i);
    const bool live = s.free_[i] && !s.moribund[i];
    const bool released = member(c, i, b.follow_ptr, b.follow_id) && c.dying[b.follow_ptr]
                          && s.ids[b.follow_ptr] < b.id;
    int64_t ev_n = 0;
    uint8_t ev_self = 0;
    if (live) {
        if (b.massive) b.last_collision = -1.0;                       // the gradient (a missile's response not kept)
        D3 p = b.pos, v = b.vel;
        const D3 a = D3{dadd(b.last_g.x, 0.0), dadd(b.last_g.y, 0.0), dadd(b.last_g.z, 0.0)};   // mLastG + mLastC
        const double m = dmul(b.mass, b.agility);
        if (b.mode == MODE_WARP && b.effect_stamp >= 0) {
            warp_distance(b, &p, &v, dmul(ll2d(b.clock - b.effect_stamp), s.dt));
            if (b.mode == MODE_STOP) {
                b.pos = p;
                b.vel = v;
                integrate(&p, &v, a, m, s.friction, b.tf, s.dt);
            }
        } else {
            integrate(&p, &v, a, m, s.friction, b.tf, s.dt);
        }
        Q4 r0 = Q4{0.0f, 0.0f, 0.0f, 1.0f}, rot1 = r0;
        V3 w0 = V3{0.0f, 0.0f, 0.0f}, w1 = w0;
        if (s.dynamical) {                                            // the rotation a tick on (phase 3), by its mode then
            r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
            w0 = V3{tof(s.ang_vel[3 * i]), tof(s.ang_vel[3 * i + 1]), tof(s.ang_vel[3 * i + 2])};
            if (!disabled(c, b.mode)) {                               // ApplyTorque
                const float fmass = tof(b.mass), ag = tof(b.agility), rag = tof(s.rot_agility[i]);
                const float tau = fdiv(fmul(fmul(fmass, ag), rag), s.ffriction);
                const V3 torque = V3{tof(s.torque[3 * i]), tof(s.torque[3 * i + 1]), tof(s.torque[3 * i + 2])};
                xbpo::apply_torque(r0, w0, torque, tof(s.rot_roll[i]), tau, s.fdt, &rot1, &w1);
            } else {                                                  // phase 3 leaves mOldRot as it was
                rot1 = Q4{tof(s.old_rot[4 * i]), tof(s.old_rot[4 * i + 1]), tof(s.old_rot[4 * i + 2]), tof(s.old_rot[4 * i + 3])};
                w1 = V3{tof(s.old_ang_vel[3 * i]), tof(s.old_ang_vel[3 * i + 1]), tof(s.old_ang_vel[3 * i + 2])};
            }
        }
        if (released) release_in_phase1(c, b, i);                    // the mushroom's turn came first
        if (b.mode == MODE_MUSHROOM) {
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
                // stamp until BringOutTheDeads (VecCombatPark.bring_out_the_deads)
                b.mode = MODE_STOP;
                s.moribund[i] = 1;
                b.massive = false;
                b.effect_stamp = b.clock;
            } else {                                                  // mRadius = range*pow(timeFraction,0.25f)
                s.radius[i] = tod(fmul(tof(b.follow_range), g_powf(mushroom_fraction(c, i), KF(F_QUARTER))));
            }
        } else {
            const D3 before = b.pos;
            d3_store(s.old_pos, i, b.pos);
            d3_store(s.old_vel, i, b.vel);
            b.pos = p;
            b.vel = v;
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
            // Ball::DispatchCollisions (COL-719): a missile that moved at least sqrt(0.1) m; on the master its owner
            // gone, cloaked or in another bubble is a collision with itself, after its target's
            if (b.mode == MODE_MISSILE && !(d3_lensq(d3_sub(b.pos, before)) < KD(C_DISPATCH_MIN_MOVE_SQ))) {
                ev_n = c.hits_n[i];
                if (c.master) {
                    const int64_t o = c.owner_ptr[i];
                    const bool gone = !row_ok(c, o) || !c.exists[o];
                    ev_self = (gone || (c.bubble[o] != c.bubble[i] && c.bubble[i] != -1) || c.cloaked[o] != 0) ? 1 : 0;
                }
            }
        }
    } else if (released) {
        release_in_phase1(c, b, i);                                   // not evolved, released all the same
    }
    b.clock += 1;
    store_ball(s, i, b);
    s.clock[i] = b.clock;
    c.ev_n[i] = ev_n;
    c.ev_self[i] = ev_self;
}

__global__ void combat_after_kernel(Combat c) {
    // After phase 4, after the host's yaw, pitch and roll: the members of a mushroom removed this
    // tick whose turns came before its own, released from their committed state; then PetrifyTroll (Ballpark.cpp:6327)
    // of each troll due, TrollReady checked again (a troll released above is no more one): SetBallFree(false)
    // (Stop(Ball*), a troll in no follow mode; SetBallVelocity(0, 0, 0), the new and old velocity, whose rotation snap
    // leaves a ball at rest as it is, the yaw, pitch and roll's the host's after this launch, for the rows marked in
    // `petrified`; out of the free set; mLastG cleared), SetBallInteractive(false), RIGID
    const xbpm::Modes& s = c.m;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    c.petrified[i] = 0;
    if (!c.exists[i]) return;
    Ball b = load_ball(s, i);
    const int64_t now = b.clock - 1;                                  // the tick's time (the commit moved the clock)
    bool changed = false;
    if (member(c, i, b.follow_ptr, b.follow_id) && c.dying[b.follow_ptr] && s.ids[b.follow_ptr] > b.id) {
        release_in_phase1(c, b, i);
        changed = true;
    }
    if (s.free_[i] && !s.moribund[i] && b.mode == MODE_TROLL && !(b.effect_stamp > now)) {
        b.vel = d3_zero();
        d3_store(s.old_vel, i, d3_zero());
        b.last_g = d3_zero();
        s.free_[i] = 0;
        s.interactive[i] = 0;
        b.mode = MODE_RIGID;
        c.petrified[i] = 1;
        changed = true;
    }
    if (changed) store_ball(s, i, b);
}

__global__ void combat_powf_kernel(int64_t n, const float* x, const float* y, float* out) {
    // g_powf elementwise (for the tests: the device's powf against the C library's)
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = g_powf(x[i], y[i]);
}

// ---- GetAccuracy (Ballpark.cpp:5156; CMB-702) for B pairs: exact_ballpark.vec_accuracy's, one thread a pair ----

__global__ void combat_accuracy_kernel(int64_t n, const double* ps, const double* vs, const double* rs, const double* pd,
                                const double* vd, const double* rd, const double* args, const uint8_t* same,
                                const uint8_t* present, double* acc, double* short_out, double* w_out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double optimal = args[5 * i], falloff = args[5 * i + 1], tracking = args[5 * i + 2];
    const double signature = args[5 * i + 3], osig = args[5 * i + 4];
    acc[i] = 0.0;
    short_out[i] = optimal;
    w_out[i] = tracking;
    if (same[i] || !present[i] || optimal < 0.0) return;
    if (tracking == 0.0 || signature == 0.0 || falloff == 0.0) return;
    const D3 radial = d3_sub(d3_load(pd, i), d3_load(ps, i));
    const double dist = d3_len(radial);
    double sd = dsub(dsub(dist, rd[i]), rs[i]);
    if (sd < 0.0) sd = 0.0;
    if (dist == 0.0) return;
    const D3 combined = d3_sub(d3_load(vd, i), d3_load(vs, i));
    const double radial_speed = ddiv(dadd(dadd(dmul(combined.x, radial.x), dmul(combined.y, radial.y)),
                                          dmul(combined.z, radial.z)), dist);
    const double speed2 = ddiv(dsub(d3_lensq(combined), dmul(radial_speed, radial_speed)), dmul(dist, dist));
    double exponent = ddiv(osig, dmul(tracking, signature));
    exponent = dmul(exponent, dmul(-exponent, speed2));
    if (sd > optimal) {
        const double gap = dsub(sd, optimal);
        exponent = dsub(exponent, ddiv(dmul(gap, gap), dmul(falloff, falloff)));
    }
    acc[i] = xbpo::g_pow(2.0, exponent);
    short_out[i] = sd;
    w_out[i] = dsqrt(speed2);
}

}  // namespace xbpm

// ====================================================================================== the C interface (ctypes)

extern "C" {

int xbpc_abi(void) { return XBPC_ABI; }

// One tick for the park's n balls on `stream`: the four launches; no allocation, no host sync (capturable)
int xbpc_tick(const xbpm::Combat* state, void* stream, int block) {
    if (state->m.n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    const unsigned grid = grid_for(state->m.n, block);
    cudaStream_t st = (cudaStream_t)stream;
    xbpm::combat_prepass_kernel<<<grid, block, 0, st>>>(*state);
    int rc = launch_check();
    if (rc) return rc;
    xbpm::combat_phase1_kernel<<<grid, block, 0, st>>>(*state);
    rc = launch_check();
    if (rc) return rc;
    xbpm::combat_contact_kernel<<<grid, block, 0, st>>>(*state);
    rc = launch_check();
    if (rc) return rc;
    xbpm::combat_commit_kernel<<<grid, block, 0, st>>>(*state);
    return launch_check();
}

// The fifth launch, after the host's yaw, pitch and roll: no allocation, no host sync (capturable)
int xbpc_after(const xbpm::Combat* state, void* stream, int block) {
    if (state->m.n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbpm::combat_after_kernel<<<grid_for(state->m.n, block), block, 0, (cudaStream_t)stream>>>(*state);
    return launch_check();
}

// powf elementwise on the device (the tests')
int xbpc_powf(int64_t n, const float* x, const float* y, float* out, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbpm::combat_powf_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(n, x, y, out);
    return launch_check();
}

int xbpc_accuracy(int64_t n, const double* ps, const double* vs, const double* rs, const double* pd, const double* vd,
                  const double* rd, const double* args, const uint8_t* same, const uint8_t* present, double* acc,
                  double* short_out, double* w_out, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbpm::combat_accuracy_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(n, ps, vs, rs, pd, vd, rd, args, same,
                                                                                   present, acc, short_out, w_out);
    return launch_check();
}

// The kernels as loaded on the current device: registers, local memory, the binary and PTX versions
int xbpc_info(int which, int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    cudaError_t e;
    switch (which) {
        case 0: e = cudaFuncGetAttributes(&fa, xbpm::combat_prepass_kernel); break;
        case 1: e = cudaFuncGetAttributes(&fa, xbpm::combat_phase1_kernel); break;
        case 2: e = cudaFuncGetAttributes(&fa, xbpm::combat_contact_kernel); break;
        case 3: e = cudaFuncGetAttributes(&fa, xbpm::combat_commit_kernel); break;
        case 5: e = cudaFuncGetAttributes(&fa, xbpm::combat_after_kernel); break;
        case 6: e = cudaFuncGetAttributes(&fa, xbpm::combat_powf_kernel); break;
        default: e = cudaFuncGetAttributes(&fa, xbpm::combat_accuracy_kernel); break;
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
