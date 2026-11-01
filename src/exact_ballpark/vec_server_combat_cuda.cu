// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's server layer over the combat park on a CUDA device, one thread a world: what
// exact_ballpark.vec_server_combat.CombatServer's torch path (VecServer over VecCombatPark, master) does in some hundreds
// of launches a tick, as one: the queued actions' duplicate test and each ball's row (rows of the park only: `exists`),
// every action applied in queue order by the combat park's own rules, LaunchMissile among them (PyLaunchMissile,
// Thunkers.cpp:857; MOV-725: the launcher by |owner|, aimed or along its velocity or its rotated (0, 0, -1) at rest,
// massive if asked, SetBallVelocity with its snap, then GotoDirection or MissileFollow), Stop with the master's
// formation-slot free (Ballpark.cpp:4629-4633) and StopAllFollowers with its missile branch (MOV-730: a missile made
// non-massive and flown on along its velocity, as an interactive orbiter; every other follower, a formation follower
// among them, stopped); the bubble's additions; the history (CloakBall applied there, releasing the cloaked ball's
// missiles and formation followers; each event routed to the bubble, the ball's character or nowhere by the
// route_of; the uncloaks' AddBallsToPark notices); and the send flags. Built and loaded by
// exact_ballpark.vec_server_combat_cuda; held bit for bit to the torch path (and through it to exact_ballpark.server over
// FullBallpark) by tests/test_movement_vec_server_combat_cuda.py.
//
// It includes vec_server_cuda.cu (the server kernel, and through it vec_modes_cuda.cu and
// vec_orient_cuda.cu) and vec_ypr_cuda.cu (the transcriptions of the C library's double atan2 and asin),
// all unedited, for their functions: the setters (`xbpv::apply` for the kinds that neither stop, release, snap nor
// launch), its uncloak and its SnapOrientation (`xbpv::snap`, dynamical orientation); the modes' double vectors; the
// orientation's float quaternions and matrices and the C library's float sinf and cosf; atan2 and asin. The commands
// that stop or release a ball are written here again, as VecCombatPark runs them through its own `_stop_rows` and
// `_release_followers` (VecModesPark's commands calling the combat park's overrides).
//
// The phases (as the kernel; one launch does all three, or one each; the second can run over a range of slots, so
// the host can read the park between them for the streams the additions and the uncloaks send):
//   1. the stamps; the queue in order: the duplicate test against the world's last queued action, the ball's (and a
//      FollowBall's or Orbit's target's) row among the world's balls in the park, CloakBall's bubble, every action
//      but CloakBall applied; then the bubble update (BubbleUpdater once per park time: a new user ball gets every
//      uncloaked ball of its bubble, the balls in the park only);
//   2. the history over slots [j0, j1): the route of each queued action, CloakBall applied, the uncloaks' notices;
//   3. the send flags of each row's pilot (a character update; the bubble's narrowcast).
//
// Exactness. The commands' arithmetic is the torch path's, operation for operation: doubles with __dadd_rn ...
// __dsqrt_rn (the modes' d3 helpers), floats with __fadd_rn ... __fmul_rn where the park computes in float (the float
// argument roundings, RotationQuaternion(yaw, pitch, roll), the rotated vector), the C library's float sinf and cosf
// (vec_orient_cuda's g_sinf, g_cosf) and double atan2 and asin (vec_ypr_cuda's g_atan2, g_asin), and C's fmod (exact,
// on the bits here); no function of CUDA's math library; every constant from the generated headers (this module's
// `tables_header`: the two modes, the slots, the flags, the launch speed, the missile's goto distance, 2 pi and the
// two literals the torch path writes without a name, each with its own spelling).

#include <cstddef>

#include "vec_server_cuda.cu"
#include "vec_ypr_cuda.cu"
#include "vec_server_combat_cuda_tables.h"

#define XBPK_ABI 1
#ifndef XBPK_FAULT
#define XBPK_FAULT 0                                     // a planted fault's number (the tests' builds only)
#endif

namespace xbpk {

using xbpm::D3;
using xbpm::Modes;
using xbpm::d3_add;
using xbpm::d3_finite;
using xbpm::d3_len;
using xbpm::d3_lensq;
using xbpm::d3_load;
using xbpm::d3_normalized;
using xbpm::d3_scale;
using xbpm::d3_store;
using xbpm::d3_sub;
using xbpm::dadd;
using xbpm::dbits;
using xbpm::ddiv;
using xbpm::dfinite;
using xbpm::dfrom;
using xbpm::dsub;
using xbpm::tod;
using xbpm::tof;
using xbpo::Q4;
using xbpo::V3;
using xbpo::fadd;
using xbpo::fmul;
using xbpo::fsub;
using xbpv::Park;
using xbpv::Queue;
using xbpv::Srv;
using xbpo::ffrom;

// VecCombatPark's tensors beyond `Modes` and the `Park`
struct Combat {
    uint8_t* exists;                                     // the row holds a ball (an empty row is no ball)
    int64_t *formation_id, *formation_slots, *owner_ptr, *flags;
    double *yaw, *pitch, *roll, *old_yaw, *old_pitch, *old_roll, *roll_speed, *old_roll_speed, *yaw_delta;
    const int64_t* formation_len;                        // [max(1, n_formations)] slots in each formation
    int64_t n_formations;
    int master;                                          // the park's view (the adapter's, checked on the host)
};

static __device__ __forceinline__ float f32(double x) { return tof(x); }
static __device__ __forceinline__ double dneg(double x) { return dfrom(dbits(x) ^ 0x8000000000000000ULL); }
static __device__ __forceinline__ double dabs(double x) { return dfrom(dbits(x) & 0x7fffffffffffffffULL); }

static __device__ __forceinline__ int64_t schar(int64_t n) {
    // a conversion to (signed) char (vec_combat's and vec_server's _char), in unsigned arithmetic (no overflow)
    return (int64_t)(((uint64_t)n + 128ULL) & 0xFFULL) - 128;
}

static __device__ __forceinline__ int64_t iabs(int64_t x) {
    // torch's abs on int64: the most negative value stays itself
    return x < 0 ? (int64_t)(0ULL - (uint64_t)x) : x;
}

static __device__ __forceinline__ bool follow_mode(int64_t md) {
    // followModes (Ballpark.cpp:68): VecCombatPark._following
    return md == MODE_FOLLOW || md == MODE_ORBIT || md == SK_MODE_MISSILE || md == SK_MODE_FORMATION;
}

// ---- the rows of a world ----

static __device__ __forceinline__ int64_t lookup(const Modes& m, const Combat& c, const Srv& s, int64_t w, int64_t id) {
    // the first row of world w whose ball has Destiny id `id` and is in the park (CombatAdapter.lookup and
    // VecCombatPark.lookup: the world's rows in row order, `exists`), or -1
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
#if XBPK_FAULT == 1
        if (x >= 0 && m.ids[x] == id) return x;          // planted: an empty row taken for a ball
#else
        if (x >= 0 && c.exists[x] && m.ids[x] == id) return x;
#endif
    }
    return -1;
}

// ---- Stop(Ball*) and the commands that stop (VecModesPark's commands over VecCombatPark's _stop_rows) ----

static __device__ void free_slot(const Modes& m, const Combat& c, int64_t r) {
    // Ball::FreeFormationSlot((size_t) mEffectStamp) (Ball.cpp:2324) on the leader of FORMATION follower r
    // (VecCombatPark._free_slots, one row): nothing without a loaded formation or for a slot past its end (unsigned
    // comparisons); a negative formation id or a slot of 16 or more flagged (Destiny leaves them undefined)
    const int64_t lead = m.follow_ptr[r];
    const int64_t fid = c.formation_id[lead];
    if (fid == -1 || fid > c.n_formations - 1) return;
    if (fid < 0) {
        c.flags[r] |= SK_FLAG_UNDEFINED;
        return;
    }
    const int64_t slot = m.effect_stamp[r];
    const int64_t len = c.formation_len[fid];
    if ((uint64_t)slot > (uint64_t)(len - 1)) return;   // size_t slot > numberOfSlots - 1
    if (slot < 0 || slot >= SK_FORMATION_SLOTS) {
        c.flags[r] |= SK_FLAG_UNDEFINED;
        return;
    }
    c.formation_slots[lead] &= ~(int64_t)(1ULL << slot);
}

static __device__ void stop_rows(const Modes& m, const Combat& c, int64_t r) {
    // Stop(Ball*) (Ballpark.cpp:4578; VecCombatPark._stop_rows): on the master a FORMATION follower's slot freed in its
    // leader first (4629-4633); a follower (FOLLOW, ORBIT, MISSILE, FORMATION) leaves its target's follower set, its
    // owner's row and follow fields cleared; then STOP
    const int64_t md = m.mode[r];
#if XBPK_FAULT != 2
    if (c.master && md == SK_MODE_FORMATION && m.follow_ptr[r] >= 0) free_slot(m, c, r);
#endif                                                   // (planted 2: the master's slot free left out)
    if (follow_mode(md)) {
        c.owner_ptr[r] = -1;
        m.effect_stamp[r] = 0;
        m.follow_ptr[r] = -1;
        m.follow_id[r] = 0;
        m.owner_id[r] = 0;
        m.follow_range[r] = 0.0;
    }
    m.mode[r] = MODE_STOP;
}

static __device__ __forceinline__ void cmd_stop(const Modes& m, const Combat& c, int64_t r) {
    if (m.mode[r] != MODE_STOP) stop_rows(m, c, r);
}

static __device__ void cmd_goto_point(const Modes& m, const Combat& c, int64_t r, D3 p) {
    // GotoPoint: refused for a non-finite point; Stop, the point, a zero speed fraction made 1, then GOTO
    if (!d3_finite(p)) return;
    stop_rows(m, c, r);
    d3_store(m.goto_pt, r, p);
    if (m.speed_fraction[r] == 0.0) m.speed_fraction[r] = 1.0;
    m.mode[r] = MODE_GOTO;
}

static __device__ void cmd_goto_direction(const Modes& m, const Combat& c, int64_t r, D3 d) {
    // GotoDirection: refused when not finite; a point 1e17 m along the unit direction
    if (!d3_finite(d)) return;
    cmd_goto_point(m, c, r, d3_add(d3_load(m.pos, r), d3_scale(d3_normalized(d), KD(M_DIR_DIST))));
}

static __device__ void cmd_follow(const Modes& m, const Park& p, const Combat& c, int64_t r, int64_t t, double range,
                                  int64_t mode) {
    // FollowBall and Orbit (VecModesPark._follow): the float range refused unless finite; the ball itself, a missing,
    // moribund, cloaked or other-bubble target refused (the target's row is in the ball's world: the lookup's)
    const double rng = tod(f32(range));
    if (!dfinite(rng) || t < 0 || t == r || m.moribund[t] || p.bubble[t] != p.bubble[r] || p.cloaked[t] != 0) return;
    stop_rows(m, c, r);
    m.follow_id[r] = m.ids[t];
    m.follow_ptr[r] = t;
    m.follow_range[r] = rng;
    m.mode[r] = mode;
}

static __device__ void cmd_warp_to(const Modes& m, const Combat& c, int64_t r, D3 point, double mr, int64_t wf) {
    // WarpTo (VecModesPark.warp_to): a short warp a GotoPoint; the same point while warping a no-op; else Stop and
    // aligning
    if (!d3_finite(point)) return;
    if (mr < 0.0) mr = 0.0;
    if (wf <= 0) wf = 1;
    if (d3_lensq(d3_sub(d3_load(m.pos, r), point)) < KD(M_MIN_WARP_DISTANCE_SQ)) {
        cmd_goto_point(m, c, r, point);
        return;
    }
    const D3 g = d3_load(m.goto_pt, r);
    if (m.mode[r] == MODE_WARP && g.x == point.x && g.y == point.y && g.z == point.z) return;
    stop_rows(m, c, r);
    d3_store(m.goto_pt, r, point);
    m.effect_stamp[r] = -1;
    m.follow_range[r] = 0.0;
    m.follow_id[r] = (int64_t)dbits(mr);
    m.owner_id[r] = wf;
    m.mode[r] = MODE_WARP;
}

static __device__ void release_followers(const Modes& m, const Combat& c, const Srv& s, int64_t w, int64_t x) {
    // StopAllFollowers of row x (Ballpark.cpp:5416; MOV-730; VecCombatPark._release_followers): of x's followers in
    // the park (FOLLOW, ORBIT, MISSILE, FORMATION), a missile is made non-massive and, like an interactive orbiter,
    // flies on by GotoDirection along its velocity (refused, it stays, when that is not finite); every other follower
    // stops (a FORMATION follower's slot freed on the master). Each follower's steps touch its own row (and the
    // leader's slot bits), so the world's rows in order give the torch path's masks.
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t f = s.rows_of_world[w * s.N + k];
        if (f < 0) continue;
        const int64_t md = m.mode[f];
#if XBPK_FAULT == 10
        if (!follow_mode(md) || m.follow_ptr[f] != x || !c.exists[f]) continue;   // planted: by the follow modes
#else
        // StopAllFollowers' members as Destiny tests them (VecCombatPark._members):
        // follow_ptr names x and the follow id is x's id, a ball the Python SetBoid took out of a follow mode too
        if (m.follow_ptr[f] != x || m.follow_id[f] != m.ids[x] || !c.exists[f]) continue;
#endif
#if XBPK_FAULT == 3
        const bool missile = false;                      // planted: a missile stopped like any other follower
#else
        const bool missile = md == SK_MODE_MISSILE;
#endif
        if (missile) m.massive[f] = 0;
        if (missile || (m.interactive[f] && md == MODE_ORBIT)) {
            cmd_goto_direction(m, c, f, d3_load(m.vel, f));
        } else {
            stop_rows(m, c, f);
        }
    }
}

static __device__ void cmd_entity_warp_in(const Modes& m, const Combat& c, const Srv& s, int64_t w, int64_t r, D3 point,
                                          int64_t wf) {
    // EntityWarpIn (VecModesPark.entity_warp_in over the combat park): WarpTo with no minimum range, then, unless that
    // made a GOTO, warp proper at once (the followers released), begun five ticks ago at warp speed along the point's
    // direction (the velocity set directly: the yaw, pitch and roll are not snapped, as the torch path)
    cmd_warp_to(m, c, r, point, 0.0, wf);
    if (m.mode[r] == MODE_GOTO) return;
    release_followers(m, c, s, w, r);
    const D3 g = d3_load(m.goto_pt, r);
    const D3 delta = d3_normalized(d3_sub(d3_load(m.pos, r), g));
    const double mr = dfrom((uint64_t)m.follow_id[r]);
    d3_store(m.goto_pt, r, d3_add(g, d3_scale(delta, mr)));
    m.massive[r] = 0;
    m.sensor[r] = 0;
    d3_store(m.vel, r, d3_scale(d3_normalized(point), KD(M_AU)));
    if (m.dynamical) xbpv::snap(m, r);
    const int64_t back = m.clock[r] - SV_WARPIN_BACK;
    m.effect_stamp[r] = back < 0 ? 0 : back;
    m.last_collision[r] = d3_len(d3_sub(d3_load(m.pos, r), d3_load(m.goto_pt, r)));
}

// ---- the yaw, pitch and roll without dynamical orientation (MOV-729): the snap of SetBallVelocity ----

static __device__ double fmod_exact(double x, double y) {
    // C's fmod(x, y) for a finite, non-zero y (2 pi here): x - n y with n = trunc(x / y), exact, on the bits (the
    // significands' long division); x's sign kept (a zero result is x's signed zero); NaN for a NaN or infinite x
    uint64_t ux = dbits(x) & 0x7fffffffffffffffULL, uy = dbits(y) & 0x7fffffffffffffffULL;
    const uint64_t sx = dbits(x) & 0x8000000000000000ULL;
    if (ux >= 0x7ff0000000000000ULL) return dfrom(0x7ff8000000000000ULL | (ux > 0x7ff0000000000000ULL ? dbits(x) : 0));
    if (ux < uy) return x;
    if (ux == uy) return dfrom(sx);
    int ex = (int)(ux >> 52), ey = (int)(uy >> 52);
    uint64_t mx, my;
    if (ex == 0) {                                       // a subnormal x: normalised (its exponent below 1)
        mx = ux;
        ex = 1;
        while (!(mx & (1ULL << 52))) {
            mx <<= 1;
            ex--;
        }
    } else {
        mx = (ux & 0x000fffffffffffffULL) | (1ULL << 52);
    }
    if (ey == 0) {
        my = uy;
        ey = 1;
        while (!(my & (1ULL << 52))) {
            my <<= 1;
            ey--;
        }
    } else {
        my = (uy & 0x000fffffffffffffULL) | (1ULL << 52);
    }
    for (; ex > ey; ex--) {                              // mx 2^ex mod my 2^ey, one bit a step
        if (mx >= my) mx -= my;
        mx <<= 1;
    }
    if (mx >= my) mx -= my;
    if (mx == 0) return dfrom(sx);
    while (!(mx & (1ULL << 52))) {
        mx <<= 1;
        ex--;
    }
    uint64_t out;
    if (ex > 0) {
        out = (mx & 0x000fffffffffffffULL) | ((uint64_t)ex << 52);
    } else {                                             // subnormal: exact (the remainder is below y)
        out = mx >> (1 - ex);
    }
    return dfrom(out | sx);
}

static __device__ void modulo2pi(double a, double b, double* a_out, double* b_out) {
    // Modulo2pi (Ball.cpp:777; VecOrientPark._modulo2pi): both reduced by fmod, then b moved by 2 pi toward a
    const double tp = KD(SK_TWO_PI);
#if XBPK_FAULT == 6
    // planted: the angles not reduced
#else
    b = fmod_exact(b, tp);
    a = fmod_exact(a, tp);
#endif
    const double d0 = dabs(dsub(b, a)), dp = dabs(dsub(dadd(b, tp), a)), dm = dabs(dsub(dsub(b, tp), a));
    double out;
    if (d0 < dp) {
        out = d0 < dm ? b : dsub(b, tp);
    } else {
        out = dp < dm ? dadd(b, tp) : dsub(b, tp);
    }
    *a_out = a;
    *b_out = out;
}

static __device__ void ypr_snap(const Modes& m, const Combat& c, int64_t r) {
    // Ball::CalculateYawPitchRoll with its snap (VecCombatPark._ypr(snap=True), one row): the direction is the velocity
    // (aligning for warp: toward mGoto; in warp proper zero), mGoto - pos at rest; the yaw atan2(x, z), the pitch
    // -asin(y / |d|) clamped to [-1, 1] (0 for a zero direction); both moved within 2 pi of the old values; the roll,
    // its speed and the yaw's change zeroed
    const D3 v = d3_load(m.vel, r);
    const double vel2 = d3_lensq(v);
    const D3 a = d3_sub(d3_load(m.goto_pt, r), d3_load(m.pos, r));
    const bool warp = m.mode[r] == MODE_WARP;
    const bool aligning = warp && m.effect_stamp[r] < 0;
    D3 dir = warp ? (aligning ? a : D3{0.0, 0.0, 0.0}) : v;
    if (vel2 == 0.0) dir = a;
    const double len = d3_len(dir);
    double sine = ddiv(dir.y, len > 0.0 ? len : 1.0);
    sine = -1.0 < sine ? sine : -1.0;                    // _fmax(-1, sine): where(-1 < sine, sine, -1)
    sine = sine < 1.0 ? sine : 1.0;                      // _fmin(1, sine): where(sine < 1, sine, 1)
    double yaw = xbpy::g_atan2(dir.x, dir.z);
    double pitch = dneg(xbpy::g_asin(sine));
    if (!(len > 0.0)) pitch = 0.0;
    double old_yaw, old_pitch;
    modulo2pi(c.yaw[r], yaw, &old_yaw, &yaw);
    modulo2pi(c.pitch[r], pitch, &old_pitch, &pitch);
    c.yaw[r] = yaw;
    c.pitch[r] = pitch;
    c.old_yaw[r] = old_yaw;
    c.old_pitch[r] = old_pitch;
    c.roll[r] = 0.0;
    c.old_roll[r] = 0.0;
    c.roll_speed[r] = 0.0;
    c.old_roll_speed[r] = 0.0;
    c.yaw_delta[r] = 0.0;
}

static __device__ void set_velocity(const Modes& m, const Combat& c, int64_t r, D3 v) {
    // SetBallVelocity (VecCombatPark.set_velocity): the new and old velocity; the rotation snapped with dynamical
    // orientation, the yaw, pitch and roll without it
    d3_store(m.vel, r, v);
    d3_store(m.old_vel, r, v);
    if (m.dynamical) {
        xbpv::snap(m, r);
    } else if (c.exists[r]) {
#if XBPK_FAULT != 5
        ypr_snap(m, c, r);
#endif                                                   // (planted 5: the yaw, pitch and roll left unsnapped)
    }
}

// ---- missiles (MOV-724, MOV-725) ----

static __device__ __forceinline__ Q4 q_rotation_ypr(float yaw, float pitch, float roll) {
    // RotationQuaternion(yaw, pitch, roll) (vec_combat.q_rotation_ypr; DirectXMath's scalar path): half angles, cosf
    // and sinf, four sums of float products left to right (as vec_combat_cuda.cu's device function)
    const float hp = fmul(pitch, KF(SK_F_HALF)), hy = fmul(yaw, KF(SK_F_HALF)), hr = fmul(roll, KF(SK_F_HALF));
    const float cp = xbpo::g_cosf(hp), sp = xbpo::g_sinf(hp);
    const float cy = xbpo::g_cosf(hy), sy = xbpo::g_sinf(hy);
    const float cr = xbpo::g_cosf(hr), sr = xbpo::g_sinf(hr);
    return Q4{fadd(fmul(fmul(cr, sp), cy), fmul(fmul(sr, cp), sy)),
              fsub(fmul(fmul(cr, cp), sy), fmul(fmul(sr, sp), cy)),
              fsub(fmul(fmul(sr, cp), cy), fmul(fmul(cr, sp), sy)),
              fadd(fmul(fmul(cr, cp), cy), fmul(fmul(sr, sp), sy))};
}

static __device__ D3 rotated_back(const Modes& m, const Combat& c, int64_t l) {
    // Ball::GetRotatedVector of (0, 0, -1) (VecCombatPark.rotated_vector): by the rotation with dynamical orientation,
    // by RotationQuaternion of the float yaw, pitch and roll without it; the float vector as doubles
    const V3 back{KF(SK_BACK_X), KF(SK_BACK_Y), KF(SK_BACK_Z)};
#if XBPK_FAULT == 4
    return D3{tod(back.x), tod(back.y), tod(back.z)};   // planted: the launcher's orientation ignored
#else
    Q4 q;
    if (m.dynamical) {
        q = Q4{f32(m.rot[4 * l]), f32(m.rot[4 * l + 1]), f32(m.rot[4 * l + 2]), f32(m.rot[4 * l + 3])};
    } else {
        q = q_rotation_ypr(f32(c.yaw[l]), f32(c.pitch[l]), f32(c.roll[l]));
    }
    const V3 u = xbpo::transform_coord(back, xbpo::m_rotation_q(q));
    return D3{tod(u.x), tod(u.y), tod(u.z)};
#endif
}

static __device__ void missile_follow(const Modes& m, const Park& p, const Combat& c, const Srv& s, int64_t w,
                                      int64_t r, int64_t dst, int64_t owner) {
    // MissileFollow (Ballpark.cpp:3798; MOV-724; VecCombatPark.missile_follow, one row): refused for coinciding ids, a
    // missing missile or target, a moribund target; otherwise the missile is stopped, and then refused for a target in
    // another bubble or cloaked; else the follow fields (the range minus both radii, as a float), the owner and its
    // row, the launch stamp, mGoto 1e16 m along the velocity, speed fraction 1, MISSILE; a negative owner flagged
    const int64_t t = lookup(m, c, s, w, dst);
    const int64_t src = m.ids[r];
    if (!(c.exists[r] && dst >= 0 && src != dst && src != owner && dst != owner && t >= 0 && !m.moribund[t])) return;
    stop_rows(m, c, r);
    if (!(p.bubble[r] == p.bubble[t] && p.cloaked[t] == 0)) return;
    m.follow_id[r] = dst;
    m.follow_ptr[r] = t;
#if XBPK_FAULT == 7
    m.follow_range[r] = dneg(dadd(m.radius[r], m.radius[t]));    // planted: the range not rounded to float
#else
    m.follow_range[r] = dneg(tod(f32(dadd(m.radius[r], m.radius[t]))));
#endif
    m.owner_id[r] = owner;
    c.owner_ptr[r] = lookup(m, c, s, w, iabs(owner));
    m.effect_stamp[r] = m.clock[r];
    d3_store(m.goto_pt, r, d3_scale(d3_normalized(d3_load(m.vel, r)), KD(SK_MISSILE_GOTO)));
    m.speed_fraction[r] = 1.0;
    m.mode[r] = SK_MODE_MISSILE;
    if (owner < 0) c.flags[r] |= SK_FLAG_DEFENDER;
}

static __device__ void launch_missile(const Modes& m, const Park& p, const Combat& c, const Srv& s, int64_t w,
                                      int64_t r, int64_t dst, int64_t owner, int64_t aimed, int64_t massive) {
    // PyLaunchMissile (Thunkers.cpp:857; MOV-725; VecCombatPark.launch_missile, one missile row r): aimed and massive
    // are the thunk's chars. Refused for a negative target or a missing launcher (|owner|); the target's row unless it
    // is the owner; every value read before anything is written, as the torch path's batch reads them
    if (!(dst >= 0)) return;
    const int64_t l = lookup(m, c, s, w, iabs(owner));
    if (l < 0) return;
    const int64_t t = dst != owner ? lookup(m, c, s, w, dst) : -1;
    const bool has_t = t >= 0;
    const D3 ps = d3_load(m.pos, l);
    const bool aim = has_t && aimed == 1 && p.cloaked[t] == 0;
    const D3 vs = d3_load(m.vel, l);
    const double max_velocity = aim ? 0.0 : m.max_vel[l];
    D3 direction = d3_normalized(vs);
    if (!aim && d3_lensq(direction) == 0.0) direction = rotated_back(m, c, l);    // at rest: along -z, turned
    const double speed = KD(SK_LAUNCH_SPEED) < max_velocity ? max_velocity : KD(SK_LAUNCH_SPEED);   // _fmax
    const D3 v0 = aim ? d3_normalized(d3_sub(d3_load(m.pos, t), ps)) : d3_add(vs, d3_scale(direction, speed));
    if (c.exists[r]) {                                   // SetBallMassive, SetBallPosition, SetBallVelocity
        if (massive != 0) m.massive[r] = 1;
        d3_store(m.pos, r, ps);
        d3_store(m.old_pos, r, ps);
        set_velocity(m, c, r, v0);
        if (!has_t && max_velocity > 0.0) cmd_goto_direction(m, c, r, v0);
    }
    const bool cloaked_t = has_t && p.cloaked[t] != 0;
    if (has_t && !cloaked_t) missile_follow(m, p, c, s, w, r, dst, owner);
}

// ---- the adapter's cloak (CombatAdapter: _Adapter.cloak over the combat park's release) ----

static __device__ void cloak(const Modes& m, const Park& p, const Combat& c, const Srv& s, int64_t w, int64_t r,
                             int64_t n, double range) {
    // CloakBall: refused unless the (signed char) mode is positive; on the master a normal cloak adds a sensor
    // (unless float(radius) + float(range) < 0); the followers released (missiles flown off, formation followers
    // stopped); cloaked with the mode's byte, non-massive
    const int64_t mode = schar(n);
    if (!(mode > 0) || !c.exists[r]) return;
    if (s.master && mode == SV_NORMAL_CLOAK && !(f32(dadd(m.radius[r], tod(f32(range)))) < 0.0f)) m.sensor[r] = 1;
    release_followers(m, c, s, w, r);
    p.cloaked[r] = mode & 0xFF;
    m.massive[r] = 0;
}

static __device__ void cmd_set_free(const Modes& m, const Combat& c, int64_t r, bool flag) {
    // SetBallFree (Ballpark.cpp:4887; VecCombatPark: VecModesPark.set_free over this park's Stop(Ball*) and
    // SetBallVelocity): nothing where the flag is the ball's already; freed: in the free set; fixed:
    // Stop(Ball*) (a FORMATION follower's slot freed on the master), SetBallVelocity(0, 0, 0) (the rotation snapped with
    // dynamical orientation, the yaw, pitch and roll without it), out of the free set, mLastG cleared
    if (flag == (m.free_[r] != 0)) return;
    m.free_[r] = flag ? 1 : 0;
    if (flag) return;
    stop_rows(m, c, r);
    set_velocity(m, c, r, D3{0.0, 0.0, 0.0});
#if XBPK_FAULT != 12
    d3_store(m.last_g, r, D3{0.0, 0.0, 0.0});            // (planted 12: mLastG kept)
#endif
}

static __device__ void cmd_set_ball_troll(const Modes& m, const Combat& c, int64_t r, int64_t delay) {
    // PySetBallTroll ("Li") and SetBallTroll (Ballpark.cpp:6290; VecModesPark.set_ball_troll over the combat park): a delay below 1 made 1; Stop(Ball*); SetBallFree(true); SetBallInteractive(true); due at the
    // clock + the delay; TROLL (VecCombatPark's tick petrifies it: the fifth launch)
#if XBPK_FAULT == 11
    const int64_t d = delay;                             // planted: the delay's floor left out
#else
    const int64_t d = delay < SV_TROLL_MIN_DELAY ? SV_TROLL_MIN_DELAY : delay;
#endif
    stop_rows(m, c, r);
    cmd_set_free(m, c, r, true);
    m.interactive[r] = 1;
    m.effect_stamp[r] = m.clock[r] + d;
    m.mode[r] = MODE_TROLL;
}

static __device__ void apply(const Modes& m, const Park& p, const Combat& c, const Srv& s, int64_t w, int kind,
                             int64_t r, int64_t tr, int64_t t, const double* f, int64_t n, int64_t loc) {
    // one queued action on row r (tr: a FollowBall's or Orbit's target row; t: the target's id): the kinds that stop,
    // release, snap or launch here, the setters by the `apply` (they never stop or release a ball)
    const D3 xyz{f[0], f[1], f[2]};
    switch (kind) {
    case K_GOTO_POINT:
        cmd_goto_point(m, c, r, xyz);
        break;
    case K_GOTO_DIRECTION:
        cmd_goto_direction(m, c, r, xyz);
        break;
    case K_STOP:
        cmd_stop(m, c, r);
        break;
    case K_FOLLOW_BALL:
        cmd_follow(m, p, c, r, tr, f[0], MODE_FOLLOW);
        break;
    case K_ORBIT:
        cmd_follow(m, p, c, r, tr, f[0], MODE_ORBIT);
        break;
    case K_WARP_TO:
        cmd_warp_to(m, c, r, xyz, f[3], n);
        break;
    case K_ENTITY_WARP_IN:
        cmd_entity_warp_in(m, c, s, w, r, xyz, n);
        break;
    case K_SET_BALL_VELOCITY:
        set_velocity(m, c, r, xyz);
        break;
    case K_LAUNCH_MISSILE:                               // the target's id, the owner's (n), aimed and massive (chars)
        launch_missile(m, p, c, s, w, r, t, n, schar((int64_t)f[0]), schar((int64_t)f[1]));
        break;
    case K_SET_BALL_TROLL:                               // over the combat park's Stop and setters
        cmd_set_ball_troll(m, c, r, n);
        break;
    case K_SET_BALL_FREE:
        cmd_set_free(m, c, r, n != 0);
        break;
    default:
        xbpv::apply(m, p, s, w, kind, r, tr, f, n, loc);
        break;
    }
}

// ---- the phases ----

static __device__ void pass1(const Modes& m, const Park& p, const Queue& q, const Srv& s, const Combat& c, int64_t w) {
    const int64_t first = s.rows_of_world[w * s.N];
    const int64_t clock = m.clock[first];
    s.stamp[w] = s.stamp_for_system[w];
    s.clock[w] = clock;
    s.stamp_for_system[w] = clock + 1;
    bool has_last = false;
    int lk = 0;
    int64_t lb = 0, lt = 0, ln = 0, ll = 0;
    double lf[4] = {0.0, 0.0, 0.0, 0.0};
    for (int64_t j = 0; j < q.J; j++) {
        const int64_t i = j * s.W + w;
        const int kind = q.kind[j];
        const int64_t b = q.ball[i], t = q.target[i], n = q.n[i];
        const double* f = q.f + 4 * i;
        const int64_t r = lookup(m, c, s, w, b);
        const int64_t tr = (kind == K_FOLLOW_BALL || kind == K_ORBIT) ? lookup(m, c, s, w, t) : -1;
        const uint8_t mv = q.mask[i];                    // bit 0: pushed; bit 1: tested by an undo
        bool mask = (mv & 1) != 0;
        int64_t loc = q.loc_in[i];
        if (kind == K_CLOAK_BALL) {                     // its bubble (0, or -1 before its old bubble is its bubble)
            loc = (r >= 0 && s.settled[r]) ? 0 : -1;
            mask = mask && r >= 0;                       // not queued for a ball not in the park
        }
#if XBPK_FAULT == 9
        const bool tested = false;                       // planted: an undo's tested calls tested again
#else
        const bool tested = (mv & 2) != 0;               // the undo's outcome: queued as bit 0 says, no second test
#endif
        const bool dup = !tested && has_last && lk == kind && lb == b && lt == t && lf[0] == f[0] && lf[1] == f[1]
                         && lf[2] == f[2] && lf[3] == f[3] && ln == n && ll == loc;
        const bool ok = mask && !dup;
        q.ok[i] = ok;
        q.row[i] = r;
        q.trow[i] = tr;
        q.loc[i] = loc;
        if (!ok) continue;
        has_last = true;
        lk = kind;
        lb = b;
        lt = t;
        ln = n;
        ll = loc;
        for (int k = 0; k < 4; k++) lf[k] = f[k];
        if (kind != K_CLOAK_BALL && r >= 0) apply(m, p, c, s, w, kind, r, tr, t, f, n, loc);
    }
    // the bubble update: AdditionsAndDeletions and CopyBubbles once per park time, over the balls in the park
    bool any_listed = false;
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
#if XBPK_FAULT == 8
        if (x >= 0 && p.cloaked[x] == 0) any_listed = true;          // planted: empty rows listed
#else
        if (x >= 0 && c.exists[x] && p.cloaked[x] == 0) any_listed = true;
#endif
    }
    const bool again = clock == s.bubble_time[w];
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x < 0 || again) continue;
        const bool user = m.interactive[x] && c.exists[x];
        s.fresh[x] = user && !s.settled[x] && any_listed;
#if XBPK_FAULT == 8
        s.listed[x] = p.cloaked[x] == 0;
#else
        s.listed[x] = c.exists[x] && p.cloaked[x] == 0;
#endif
        s.settled[x] = s.settled[x] || user;
    }
    s.bubble_time[w] = clock;
}

static __device__ bool live_world(const Modes& m, const Srv& s, const Combat& c, int64_t w) {
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x >= 0 && m.interactive[x] && c.exists[x]) return true;
    }
    return false;
}

static __device__ void pass2(const Modes& m, const Park& p, const Queue& q, const Srv& s, const Combat& c, int64_t w,
                             int64_t j0, int64_t j1) {
    const bool live = live_world(m, s, c, w);
    for (int64_t j = j0; j < j1; j++) {
        const int64_t i = j * s.W + w;
        const int kind = q.kind[j];
        const bool ok = q.ok[i] != 0;
        const int64_t r = q.row[i];
        const bool valid = ok && r >= 0;
        int8_t route = SV_NONE;
        if (kind == K_CLOAK_BALL) {
            if (ok) {
                route = q.loc[i] == 0 ? SV_BUBBLE : SV_NONE;
                cloak(m, p, c, s, w, r, q.n[i], q.f[4 * i]);
            }
        } else if (kind == K_UNCLOAK_BALL) {
            if (valid && live && m.interactive[r] && s.pilot[r] >= 0) route = SV_CHARACTER;
            for (int64_t k = 0; k < s.N; k++) {
                const int64_t x = s.rows_of_world[w * s.N + k];
                if (x < 0) continue;
                q.notice[j * s.R + x] = valid && live && m.interactive[x] && c.exists[x] && x != r && s.pilot[x] >= 0;
            }
        } else if (kind == K_SET_BALL_POSITION) {
            if (valid && q.loc[i] == 0 && live) route = xbpv::route_of(p, s, r);
        } else if (valid && live && q.ball[i] > 0) {
            route = xbpv::route_of(p, s, r);
        }
        q.route[w * q.J + j] = route;
    }
}

static __device__ void flags(const Modes& m, const Queue& q, const Srv& s, const Combat& c, int64_t w) {
    // the post-tick's send flags (VecServer.post_tick): the park's tick changes no ball's interactive flag or presence
    bool bubble = false;
    for (int64_t j = 0; j < q.J; j++) {
        if (q.route[w * q.J + j] == SV_BUBBLE) bubble = true;
    }
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x < 0) continue;
        bool has = s.fresh[x] != 0;
        for (int64_t j = 0; j < q.J; j++) {
            const int64_t i = j * s.W + w;
            if (q.ok[i] && q.row[i] == x && q.route[w * q.J + j] == SV_CHARACTER) has = true;
            if (q.kind[j] == K_UNCLOAK_BALL && q.notice[j * s.R + x]) has = true;
        }
        const bool pilot = s.pilot[x] >= 0;
        s.has_character[x] = has && pilot;
        s.gets_bubble[x] = bubble && m.interactive[x] && c.exists[x] && pilot;
    }
}

__global__ void server_combat_kernel(Modes m, Park p, Queue q, Srv s, Combat c, int phase, int64_t j0, int64_t j1) {
    const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= s.W) return;
    if (phase == 0 || phase == 1) pass1(m, p, q, s, c, w);
    if (phase == 0) pass2(m, p, q, s, c, w, 0, q.J);
    if (phase == 2) pass2(m, p, q, s, c, w, j0, j1);
    if (phase == 0 || phase == 3) flags(m, q, s, c, w);
}

}  // namespace xbpk

// ===================================================================================================== C entry points

extern "C" {

int xbpk_abi() { return XBPK_ABI; }

int xbpk_fault() { return XBPK_FAULT; }

const char* xbpk_error_string(int rc) { return cudaGetErrorString((cudaError_t)rc); }

int xbpk_pre_tick(const xbpm::Modes* m, const xbpv::Park* p, const xbpv::Queue* q, const xbpv::Srv* s,
                  const xbpk::Combat* c, int phase, int64_t j0, int64_t j1, void* stream, int block) {
    if (q->J < 0 || q->J > XBPV_MAX_SLOTS || block <= 0 || block > 1024 || phase < 0 || phase > 3) return -1;
    if (s->W == 0) return 0;
    const int64_t grid = (s->W + block - 1) / block;
    xbpk::server_combat_kernel<<<(unsigned)grid, block, 0, (cudaStream_t)stream>>>(*m, *p, *q, *s, *c, phase, j0, j1);
    return (int)cudaGetLastError();
}

int xbpk_info(int* out) {
    // the kernel's registers, local bytes, threads a block, binary and PTX versions (loading its module now)
    cudaFuncAttributes a;
    const cudaError_t e = cudaFuncGetAttributes(&a, xbpk::server_combat_kernel);
    if (e != cudaSuccess) return (int)e;
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = a.maxThreadsPerBlock;
    out[3] = a.binaryVersion;
    out[4] = a.ptxVersion;
    return 0;
}

int xbpk_layout(int64_t* out) {
    // the compiled sizes and offsets the ctypes structures must match (no GPU needed)
    out[0] = (int64_t)sizeof(xbpk::Combat);
    out[1] = (int64_t)offsetof(xbpk::Combat, formation_id);
    out[2] = (int64_t)offsetof(xbpk::Combat, yaw);
    out[3] = (int64_t)offsetof(xbpk::Combat, yaw_delta);
    out[4] = (int64_t)offsetof(xbpk::Combat, formation_len);
    out[5] = (int64_t)offsetof(xbpk::Combat, n_formations);
    out[6] = (int64_t)offsetof(xbpk::Combat, master);
    out[7] = (int64_t)sizeof(xbpv::Queue);
    out[8] = (int64_t)offsetof(xbpv::Queue, J);
    out[9] = (int64_t)sizeof(xbpv::Srv);
    out[10] = (int64_t)offsetof(xbpv::Srv, neg_kdt);
    out[11] = (int64_t)sizeof(xbpm::Modes);
    return 0;
}

}  // extern "C"
