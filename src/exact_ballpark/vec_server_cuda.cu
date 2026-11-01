// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's server layer for VecModesPark on a CUDA device, one thread a world: a tick's queued actions
// (TIME-710's duplicate test, each ball's row), applied in queue order to the park's tensors in place by the park's own
// rules (exact_ballpark.vec_modes and vec_orient's commands and setters, the adapter's flags), the bubble's additions, the
// history (CloakBall applied there, after the bubble update; each event routed to the bubble, the ball's character or
// nowhere; the uncloaks' AddBallsToPark notices), and the send flags: what exact_ballpark.vec_server.VecServer's torch
// path does in some hundreds of launches, as one. Built and loaded by exact_ballpark.vec_server_cuda; held bit for bit to
// the torch path (and through it to exact_ballpark.server) by tests/test_movement_vec_server_cuda.py.
//
// The phases (one launch does all three, or one each; the second can run over a range of slots, so the host can read
// the park between them, as the tests do for the streams the additions and the uncloaks send):
//   1. the stamps; the queue in order: the duplicate test against the world's last queued action, the ball's (and a
//      FollowBall's or Orbit's target's) row, CloakBall's bubble, every action but CloakBall applied; then the bubble
//      update (BubbleUpdater once per park time: a new user ball gets every uncloaked ball of its bubble);
//   2. the history over slots [j0, j1): the route of each queued action, CloakBall applied, the uncloaks' notices;
//   3. the send flags of each row's pilot (a character update; the bubble's narrowcast).
//
// Exactness. The commands' arithmetic is the torch path's, operation for operation, through vec_modes_cuda.cu's and
// vec_orient_cuda.cu's device functions (included unedited): doubles with __dadd_rn ... __dsqrt_rn, floats with
// __fadd_rn ... __fsqrt_rn where the park computes in float (the setters' float arguments, the rotation, dynamical
// orientation's snap), glibc's exp for the decay factor (g_exp), the C library's float functions (g_acosf, g_atan2f,
// g_sinf, g_cosf); no function of CUDA's math library; every constant from the generated headers.

#include "vec_modes_cuda.cu"
#include "vec_server_cuda_tables.h"

#define XBPV_ABI 1
#ifndef XBPV_FAULT
#define XBPV_FAULT 0                                     // a planted fault's number (the tests' builds only)
#endif
#define XBPV_MAX_SLOTS 256

namespace xbpv {

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
using xbpm::dmul;
using xbpm::fdiv;
using xbpm::g_exp;
using xbpm::tod;
using xbpm::tof;
using xbpo::Q4;
using xbpo::V3;
using xbpo::fmax_;
using xbpo::fmin_;
using xbpo::fsqrt;
using xbpo::g_acosf;
using xbpo::g_atan2f;
using xbpo::m_rotation_q;
using xbpo::q_conj;
using xbpo::q_div;
using xbpo::q_len;
using xbpo::q_mul;
using xbpo::q_rotation;
using xbpo::transform_coord;
using xbpo::v3_cross;
using xbpo::v3_len;

// VecModesPark's tensors the commands touch beyond `Modes` (vec_modes_cuda's)
struct Park {
    int64_t *cloaked, *bubble, *refiled;
};

// the tick's pushes, J slots by W worlds (slot-major), and what the kernel writes for each
struct Queue {
    const uint8_t* mask;
    const int64_t *ball, *target;
    const double* f;                                     // [J, W, 4]
    const int64_t *n, *loc_in;
    uint8_t* ok;
    int64_t *row, *trow, *loc;
    int8_t* route;                                       // [W, J]
    uint8_t* notice;                                     // [J, R]
    int64_t J;
    int8_t kind[XBPV_MAX_SLOTS];
};

// the server layer's own tensors
struct Srv {
    const int64_t *rows_of_world, *ids_of_world, *pilot;
    int64_t *stamp_for_system, *stamp, *clock, *bubble_time;
    uint8_t *settled, *fresh, *listed, *has_character, *gets_bubble;
    int64_t W, N, R;
    double neg_kdt;                                      // -friction dt: the decay factor's numerator
    int master;
};

static __device__ __forceinline__ float f32(double x) { return tof(x); }

// ---- the rows of a world ----

static __device__ __forceinline__ int64_t lookup(const Srv& s, int64_t w, int64_t id) {
    // the world's row whose Destiny id is `id`, or -1 (Destiny's lookup: the first; ids are distinct in a world)
    for (int64_t k = 0; k < s.N; k++) {
        if (s.ids_of_world[w * s.N + k] == id && s.rows_of_world[w * s.N + k] >= 0) return s.rows_of_world[w * s.N + k];
    }
    return -1;
}

// ---- the commands (vec_modes.VecModesPark and vec_orient.VecOrientPark, one row) ----

static __device__ __forceinline__ void stop_rows(const Modes& m, int64_t r) {
    // Stop(Ball*): a FOLLOW or ORBIT ball's follow fields cleared, then STOP
    const int64_t md = m.mode[r];
    if (md == MODE_FOLLOW || md == MODE_ORBIT) {
        m.effect_stamp[r] = 0;
        m.follow_ptr[r] = -1;
        m.follow_id[r] = 0;
        m.owner_id[r] = 0;
        m.follow_range[r] = 0.0;
    }
    m.mode[r] = MODE_STOP;
}

static __device__ __forceinline__ void cmd_stop(const Modes& m, int64_t r) {
    if (m.mode[r] != MODE_STOP) stop_rows(m, r);
}

static __device__ __forceinline__ void cmd_goto_point(const Modes& m, int64_t r, D3 p) {
    if (!d3_finite(p)) return;
    stop_rows(m, r);
    d3_store(m.goto_pt, r, p);
    if (m.speed_fraction[r] == 0.0) m.speed_fraction[r] = 1.0;
    m.mode[r] = MODE_GOTO;
}

static __device__ __forceinline__ D3 unit(D3 d) {
#if XBPV_FAULT == 1
    return d;                                            // planted: GotoDirection's direction left unnormalised
#else
    return d3_normalized(d);                             // times the reciprocal of the length; zero stays zero
#endif
}

static __device__ __forceinline__ void cmd_goto_direction(const Modes& m, int64_t r, D3 d) {
    if (!d3_finite(d)) return;
    cmd_goto_point(m, r, d3_add(d3_load(m.pos, r), d3_scale(unit(d), KD(M_DIR_DIST))));
}

static __device__ __forceinline__ void cmd_follow(const Modes& m, const Park& p, int64_t r, int64_t t, double range,
                                                  int64_t mode) {
    // FollowBall and Orbit: the float range refused unless finite; the ball itself, a missing, moribund, cloaked or
    // other-bubble target refused (the target's row is in the ball's world: the lookup's)
    const double rng = tod(f32(range));
    if (!dfinite(rng) || t < 0 || t == r || m.moribund[t] || p.bubble[t] != p.bubble[r] || p.cloaked[t] != 0) return;
    stop_rows(m, r);
    m.follow_id[r] = m.ids[t];
    m.follow_ptr[r] = t;
    m.follow_range[r] = rng;
    m.mode[r] = mode;
}

static __device__ __forceinline__ void cmd_warp_to(const Modes& m, int64_t r, D3 point, double mr, int64_t wf) {
    if (!d3_finite(point)) return;
    if (mr < 0.0) mr = 0.0;
    if (wf <= 0) wf = 1;
    if (d3_lensq(d3_sub(d3_load(m.pos, r), point)) < KD(M_MIN_WARP_DISTANCE_SQ)) {
        cmd_goto_point(m, r, point);
        return;
    }
    const D3 g = d3_load(m.goto_pt, r);
    if (m.mode[r] == MODE_WARP && g.x == point.x && g.y == point.y && g.z == point.z) return;
    stop_rows(m, r);
    d3_store(m.goto_pt, r, point);
    m.effect_stamp[r] = -1;
    m.follow_range[r] = 0.0;
    m.follow_id[r] = (int64_t)dbits(mr);
    m.owner_id[r] = wf;
    m.mode[r] = MODE_WARP;
}

static __device__ __forceinline__ void release_followers(const Modes& m, const Srv& s, int64_t w, int64_t x) {
    // StopAllFollowers of row x (Ballpark.cpp:5416; VecModesPark._release_followers): its members as Destiny tests them
    // (Ballpark.cpp:5433-5450, as the member_of: follow_ptr names x and the follow id is
    // x's id: every FOLLOW or ORBIT ball, and a ball the Python SetBoid took out of one without a Stop, which a caller
    // can set on the park between server ticks): an interactive orbiter flies on by GotoDirection along its velocity
    // (refused, it stays, when that is not finite), any other member stops (Stop(Ball*): a member out of the follow
    // modes keeps its follow fields, stop_rows)
#if XBPV_FAULT == 3
    return;                                              // planted: nobody released
#endif
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t f = s.rows_of_world[w * s.N + k];
        if (f < 0) continue;
        const int64_t md = m.mode[f];
#if XBPV_FAULT == 9
        if ((md != MODE_FOLLOW && md != MODE_ORBIT) || m.follow_ptr[f] != x) continue;   // planted: the follow modes
#else
        if (m.follow_ptr[f] != x || m.follow_id[f] != m.ids[x]) continue;
#endif
        if (m.interactive[f] && md == MODE_ORBIT) {
            cmd_goto_direction(m, f, d3_load(m.vel, f));
        } else {
            stop_rows(m, f);
        }
    }
}

static __device__ __noinline__ void snap(const Modes& m, int64_t r) {
    // Ball::SnapOrientation (vec_orient's _snap_orientation): turned to the velocity and rolled upright, the rotation
    // stopped; at 1e-6 m/s or less nothing changes
    const D3 v = d3_load(m.vel, r);
    if (!(d3_lensq(v) > KD(SV_SNAP_MOVING))) return;
    const V3 d{f32(v.x), f32(v.y), f32(v.z)};
    const float len_dir = v3_len(d);
    const V3 vec = v3_cross(V3{0.0f, 0.0f, 1.0f}, d);
    const float angle = g_acosf(fmax_(-1.0f, fmin_(1.0f, fdiv(d.z, len_dir))));
    Q4 rot = q_rotation(vec, angle);
    const V3 up = transform_coord(V3{0.0f, 1.0f, 0.0f}, m_rotation_q(q_conj(rot)));
    const float up_roll = up.z < f32(KD(SV_SNAP_UPRIGHT)) ? g_atan2f(-up.x, up.y) : 0.0f;
    const V3 heading = transform_coord(V3{0.0f, 0.0f, 1.0f}, m_rotation_q(rot));
    rot = q_mul(rot, q_rotation(heading, up_roll));
    if (tod(len_dir) < KD(SV_SNAP_TINY)) rot = Q4{0.0f, 0.0f, 0.0f, 1.0f};
    const double q[4] = {tod(rot.x), tod(rot.y), tod(rot.z), tod(rot.w)};
    for (int i = 0; i < 4; i++) {
        m.rot[4 * r + i] = q[i];
        m.old_rot[4 * r + i] = q[i];
    }
    for (int i = 0; i < 3; i++) {
        m.ang_vel[3 * r + i] = 0.0;
        m.old_ang_vel[3 * r + i] = 0.0;
    }
}

static __device__ __forceinline__ void cmd_entity_warp_in(const Modes& m, const Srv& s, int64_t w, int64_t r, D3 point,
                                                          int64_t wf) {
    // EntityWarpIn: WarpTo with no minimum range, then, unless that made a GOTO, warp proper at once (RealWarp: the
    // followers released, the exit point), begun five ticks ago at warp speed along the point's direction
    cmd_warp_to(m, r, point, 0.0, wf);
    if (m.mode[r] == MODE_GOTO) return;
    release_followers(m, s, w, r);
    const D3 g = d3_load(m.goto_pt, r);
    const D3 delta = unit(d3_sub(d3_load(m.pos, r), g));
    const double mr = dfrom((uint64_t)m.follow_id[r]);
    d3_store(m.goto_pt, r, d3_add(g, d3_scale(delta, mr)));
    m.massive[r] = 0;
    m.sensor[r] = 0;
    d3_store(m.vel, r, d3_scale(unit(point), KD(M_AU)));
    if (m.dynamical) snap(m, r);
    const int64_t back = m.clock[r] - SV_WARPIN_BACK;
    m.effect_stamp[r] = back < 0 ? 0 : back;
    m.last_collision[r] = d3_len(d3_sub(d3_load(m.pos, r), d3_load(m.goto_pt, r)));
}

static __device__ __forceinline__ double decay(const Srv& s, double x) {
    // SetBallTimeFactor (MOV-704): 0 if mass x agility <= 0, else exp(-k dt / m)
    return x <= 0.0 ? 0.0 : g_exp(ddiv(s.neg_kdt, x));
}

static __device__ __forceinline__ void cloak(const Modes& m, const Park& p, const Srv& s, int64_t w, int64_t r,
                                             int64_t n, double range) {
    // CloakBall: refused unless the (signed char) mode is positive; on the master a normal cloak adds a sensor
    // (unless float(radius) + float(range) < 0); the followers released; cloaked with the mode's byte, non-massive
    const int64_t mode = ((n + 128) & 0xFF) - 128;
    if (!(mode > 0)) return;
    if (s.master && mode == SV_NORMAL_CLOAK && !(f32(dadd(m.radius[r], tod(f32(range)))) < 0.0f)) m.sensor[r] = 1;
    release_followers(m, s, w, r);
    p.cloaked[r] = mode & 0xFF;
    m.massive[r] = 0;
}

static __device__ __forceinline__ void uncloak(const Modes& m, const Park& p, const Srv& s, int64_t r) {
    // UncloakBall: a normal cloak's sensor removed (master); uncloaked; massive again unless in warp proper
    if (s.master && p.cloaked[r] == SV_NORMAL_CLOAK) m.sensor[r] = 0;
    p.cloaked[r] = 0;
    if (!(m.mode[r] == MODE_WARP && m.effect_stamp[r] >= 0)) m.massive[r] = 1;
}

static __device__ __forceinline__ void cmd_set_free(const Modes& m, int64_t r, bool flag) {
    // SetBallFree (Ballpark.cpp:4887; MOV-732; VecModesPark.set_free): nothing where the flag is the
    // ball's already; freed: in the free set (its minis and mNewTime are not kept here); fixed: Stop(Ball*),
    // SetBallVelocity(0, 0, 0) (the new and old velocity; with dynamical orientation SnapOrientation, which leaves a
    // ball at rest as it is), out of the free set, mLastG cleared
    if (flag == (m.free_[r] != 0)) return;
    m.free_[r] = flag ? 1 : 0;
    if (flag) return;
    stop_rows(m, r);
    d3_store(m.vel, r, xbpm::d3_zero());
    d3_store(m.old_vel, r, xbpm::d3_zero());
    if (m.dynamical) snap(m, r);
#if XBPV_FAULT != 11
    d3_store(m.last_g, r, xbpm::d3_zero());              // (planted 11: mLastG kept)
#endif
}

static __device__ __forceinline__ void cmd_set_ball_troll(const Modes& m, int64_t r, int64_t delay) {
    // PySetBallTroll ("Li") and SetBallTroll (Ballpark.cpp:6290; MOV-736; VecModesPark.set_ball_troll):
    // a delay below 1 made 1; Stop(Ball*); SetBallFree(true); SetBallInteractive(true); due at the clock + the delay;
    // TROLL (the park's tick petrifies it: PetrifyTroll, vec_modes_cuda's)
#if XBPV_FAULT == 10
    const int64_t d = delay;                             // planted: the delay's floor left out
#else
    const int64_t d = delay < SV_TROLL_MIN_DELAY ? SV_TROLL_MIN_DELAY : delay;
#endif
    stop_rows(m, r);
    cmd_set_free(m, r, true);
    m.interactive[r] = 1;
    m.effect_stamp[r] = m.clock[r] + d;
    m.mode[r] = MODE_TROLL;
}

static __device__ void apply(const Modes& m, const Park& p, const Srv& s, int64_t w, int kind, int64_t r, int64_t t,
                             const double* f, int64_t n, int64_t loc) {
    const D3 xyz{f[0], f[1], f[2]};
    switch (kind) {
    case K_GOTO_POINT:
        cmd_goto_point(m, r, xyz);
        break;
    case K_GOTO_DIRECTION:
        cmd_goto_direction(m, r, xyz);
        break;
    case K_STOP:
        cmd_stop(m, r);
        break;
    case K_FOLLOW_BALL:
        cmd_follow(m, p, r, t, f[0], MODE_FOLLOW);
        break;
    case K_ORBIT:
        cmd_follow(m, p, r, t, f[0], MODE_ORBIT);
        break;
    case K_WARP_TO:
        cmd_warp_to(m, r, xyz, f[3], n);
        break;
    case K_ENTITY_WARP_IN:
        cmd_entity_warp_in(m, s, w, r, xyz, n);
        break;
    case K_SET_SPEED_FRACTION: {
#if XBPV_FAULT == 4
        const double v = f[0];                           // planted: the float argument not rounded
#else
        const double v = tod(f32(f[0]));
#endif
        if (dfinite(v)) m.speed_fraction[r] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
        break;
    }
    case K_SET_MAX_SPEED: {
        const double v = tod(f32(f[0]));
        if (!(v < 0.0)) {
            m.max_vel[r] = v;
            p.refiled[r] += 1;
        }
        break;
    }
    case K_SET_BALL_MASS:
        if (!(f[0] <= 0.0)) {
            m.mass[r] = f[0];
            m.time_factor[r] = decay(s, dmul(m.mass[r], m.agility[r]));
        }
        break;
    case K_SET_BALL_AGILITY: {
        const double a = tod(f32(f[0]));
        if (!(a <= 0.0)) {
            m.agility[r] = a;
            m.time_factor[r] = decay(s, dmul(m.mass[r], m.agility[r]));
        }
        break;
    }
    case K_SET_BALL_RADIUS: {
        const double v = tod(f32(f[0]));
        if (!(v < 0.0)) {
            m.radius[r] = v;
            p.refiled[r] += 1;
        }
        break;
    }
    case K_SET_BALL_VELOCITY:
        d3_store(m.vel, r, xyz);
        d3_store(m.old_vel, r, xyz);
        if (m.dynamical) snap(m, r);
        break;
    case K_SET_BALL_MASSIVE:
        m.massive[r] = n != 0;
        break;
    case K_SET_BALL_INTERACTIVE:
#if XBPV_FAULT == 5
        break;                                           // planted: SetBallInteractive ignored
#else
        m.interactive[r] = n != 0;
        break;
#endif
    case K_SET_BALL_POSITION:
        d3_store(m.pos, r, xyz);
        d3_store(m.old_pos, r, xyz);
        break;
    case K_UNCLOAK_BALL:
        uncloak(m, p, s, r);
        break;
    case K_SET_MAX_ANGULAR_SPEED: {
        const double v = tod(f32(f[0]));
        if (!(v < 0.0)) m.max_ang_vel[r] = v;
        break;
    }
    case K_SET_BALL_ANGULAR_VELOCITY:
        for (int i = 0; i < 3; i++) {
            const double v = tod(f32(f[i]));
            m.ang_vel[3 * r + i] = v;
            m.old_ang_vel[3 * r + i] = v;
        }
        break;
    case K_SET_BALL_ROTATION: {
        const Q4 q{f32(f[0]), f32(f[1]), f32(f[2]), f32(f[3])};
        const float len = q_len(q);
        if (tod(len) > KD(SV_ROTATION_MIN)) {
            const Q4 u = q_div(q, len);
            const double qq[4] = {tod(u.x), tod(u.y), tod(u.z), tod(u.w)};
            for (int i = 0; i < 4; i++) {
                m.rot[4 * r + i] = qq[i];
                m.old_rot[4 * r + i] = qq[i];
            }
            for (int i = 0; i < 3; i++) {
                m.ang_vel[3 * r + i] = 0.0;
                m.old_ang_vel[3 * r + i] = 0.0;
            }
        }
        break;
    }
    case K_SET_BALL_ANGULAR_AGILITY: {
        const double v = tod(f32(f[0]));
        if (!(v <= 0.0)) m.rot_agility[r] = v;
        break;
    }
    case K_SET_BALL_TROLL:
        cmd_set_ball_troll(m, r, n);
        break;
    case K_SET_BALL_FREE:
        cmd_set_free(m, r, n != 0);
        break;
    default:                                             // SetMaxAngularVelocity: no park method, never applied
        break;
    }
}

// ---- the phases ----

static __device__ void pass1(const Modes& m, const Park& p, const Queue& q, const Srv& s, int64_t w) {
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
        const int64_t r = lookup(s, w, b);
        const int64_t tr = (kind == K_FOLLOW_BALL || kind == K_ORBIT) ? lookup(s, w, t) : -1;
        const uint8_t mv = q.mask[i];                    // bit 0: pushed; bit 1: tested by an undo
        bool mask = (mv & 1) != 0;
        int64_t loc = q.loc_in[i];
        if (kind == K_CLOAK_BALL) {                     // its bubble (0, or -1 before its old bubble is its bubble)
            loc = (r >= 0 && s.settled[r]) ? 0 : -1;
            mask = mask && r >= 0;                       // not queued for a ball not in the park
        }
#if XBPV_FAULT == 8
        const bool tested = false;                       // planted: an undo's tested calls tested again
#else
        const bool tested = (mv & 2) != 0;               // the undo's outcome: queued as bit 0 says, no second test
#endif
#if XBPV_FAULT == 6
        const bool dup = !tested && has_last && lk == kind && lb == b && lt == t && lf[0] == f[0] && lf[1] == f[1]
                         && lf[2] == f[2] && lf[3] == f[3] && ln == n;   // planted: the local argument not compared
#else
        const bool dup = !tested && has_last && lk == kind && lb == b && lt == t && lf[0] == f[0] && lf[1] == f[1]
                         && lf[2] == f[2] && lf[3] == f[3] && ln == n && ll == loc;
#endif
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
        for (int c = 0; c < 4; c++) lf[c] = f[c];
#if XBPV_FAULT == 2
        if (r >= 0) {                                    // planted: CloakBall applied with the others
            if (kind == K_CLOAK_BALL) cloak(m, p, s, w, r, n, f[0]);
            else apply(m, p, s, w, kind, r, tr, f, n, loc);
        }
#else
        if (kind != K_CLOAK_BALL && r >= 0) apply(m, p, s, w, kind, r, tr, f, n, loc);
#endif
    }
    // the bubble update: AdditionsAndDeletions and CopyBubbles once per park time
    bool any_listed = false;
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x >= 0 && p.cloaked[x] == 0) any_listed = true;
    }
    const bool again = clock == s.bubble_time[w];
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x < 0 || again) continue;
        const bool user = m.interactive[x] != 0;
        s.fresh[x] = user && !s.settled[x] && any_listed;
        s.listed[x] = p.cloaked[x] == 0;
        s.settled[x] = s.settled[x] || user;
    }
    s.bubble_time[w] = clock;
}

static __device__ bool live_world(const Modes& m, const Srv& s, int64_t w) {
    for (int64_t k = 0; k < s.N; k++) {
        const int64_t x = s.rows_of_world[w * s.N + k];
        if (x >= 0 && m.interactive[x]) return true;
    }
    return false;
}

static __device__ __forceinline__ int8_t route_of(const Park& p, const Srv& s, int64_t r) {
    // to the bubble; while the ball is cloaked (at this point of the history) to its character, or nowhere without one
#if XBPV_FAULT == 7
    return SV_BUBBLE;                                    // planted: a cloaked ball's events to the bubble
#else
    return p.cloaked[r] == 0 ? SV_BUBBLE : (s.pilot[r] >= 0 ? SV_CHARACTER : SV_NONE);
#endif
}

static __device__ void pass2(const Modes& m, const Park& p, const Queue& q, const Srv& s, int64_t w, int64_t j0,
                             int64_t j1) {
    const bool live = live_world(m, s, w);
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
                cloak(m, p, s, w, r, q.n[i], q.f[4 * i]);
            }
        } else if (kind == K_UNCLOAK_BALL) {
            if (valid && live && m.interactive[r] && s.pilot[r] >= 0) route = SV_CHARACTER;
            for (int64_t k = 0; k < s.N; k++) {
                const int64_t x = s.rows_of_world[w * s.N + k];
                if (x < 0) continue;
                q.notice[j * s.R + x] = valid && live && m.interactive[x] && x != r && s.pilot[x] >= 0;
            }
        } else if (kind == K_SET_BALL_POSITION) {
            if (valid && q.loc[i] == 0 && live) route = route_of(p, s, r);
        } else if (valid && live && q.ball[i] > 0) {
            route = route_of(p, s, r);
        }
        q.route[w * q.J + j] = route;
    }
}

static __device__ void flags(const Modes& m, const Queue& q, const Srv& s, int64_t w) {
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
        s.gets_bubble[x] = bubble && m.interactive[x] && pilot;
    }
}

__global__ void server_kernel(Modes m, Park p, Queue q, Srv s, int phase, int64_t j0, int64_t j1) {
    const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= s.W) return;
    if (phase == 0 || phase == 1) pass1(m, p, q, s, w);
    if (phase == 0) pass2(m, p, q, s, w, 0, q.J);
    if (phase == 2) pass2(m, p, q, s, w, j0, j1);
    if (phase == 0 || phase == 3) flags(m, q, s, w);
}

}  // namespace xbpv

// ===================================================================================================== C entry points

extern "C" {

int xbpv_abi() { return XBPV_ABI; }

int xbpv_fault() { return XBPV_FAULT; }

const char* xbpv_error_string(int rc) { return cudaGetErrorString((cudaError_t)rc); }

int xbpv_pre_tick(const xbpm::Modes* m, const xbpv::Park* p, const xbpv::Queue* q, const xbpv::Srv* s, int phase,
                  int64_t j0, int64_t j1, void* stream, int block) {
    if (q->J < 0 || q->J > XBPV_MAX_SLOTS || block <= 0 || block > 1024 || phase < 0 || phase > 3) return -1;
    if (s->W == 0) return 0;
    const int64_t grid = (s->W + block - 1) / block;
    xbpv::server_kernel<<<(unsigned)grid, block, 0, (cudaStream_t)stream>>>(*m, *p, *q, *s, phase, j0, j1);
    return (int)cudaGetLastError();
}

int xbpv_info(int* out) {
    // the kernel's registers, local bytes, threads a block, binary and PTX versions (loading its module now)
    cudaFuncAttributes a;
    const cudaError_t e = cudaFuncGetAttributes(&a, xbpv::server_kernel);
    if (e != cudaSuccess) return (int)e;
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = a.maxThreadsPerBlock;
    out[3] = a.binaryVersion;
    out[4] = a.ptxVersion;
    return 0;
}

}  // extern "C"
