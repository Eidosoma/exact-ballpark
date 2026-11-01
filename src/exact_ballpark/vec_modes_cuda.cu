// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's flight for B balls on a CUDA device, one thread a ball: GOTO and STOP with FOLLOW, ORBIT (both laws) and
// WARP, the modes' transitions (warp's alignment, activation, the release of the warper's followers, dropping out,
// commands during warp as the public code leaves them), and dynamical orientation (MOV-728) where Destiny applies it,
// in Ballpark::Evolve's order (Ballpark.cpp:421): the scalar algorithm of exact_ballpark.modes.ModeBallpark and
// exact_ballpark.full.FullBallpark run per ball, each ball taking only its own branch. Built and loaded by
// exact_ballpark.vec_modes_cuda; held to the scalar reference bit for bit by tests/test_movement_vec_modes.py.
//
// A tick is two launches. The prepass records what a ball's behaviour shows the others during phase 1 of the tick:
// its position and velocity at the start of the tick, its vertical velocity after its own turn (STOP's damping, once
// a released follower's mode is known), and whether it starts warp proper this tick (it then releases its followers,
// at its own turn: a follower with a lower id has had its turn already). The main kernel then runs each ball's tick in
// place, reading other balls only through those records: phase 1 (the behaviour, the release, the orientation's
// torques), phase 3 (the warp path or the closed-form integrator, the rotation a tick on), phase 4 (the commit), and
// the ball's world clock.
//
// (31 October 2026): a ball's follower set is the balls whose follow_ptr is its row and whose follow id
// is its id (StopAllFollowers' test, Ballpark.cpp:5433-5450), which includes a ball the Python SetBoid took out of
// FOLLOW or ORBIT without a Stop; such a member released stops and keeps its follow fields (Stop(Ball*) clears them only
// in a follow mode). A troll due this tick is petrified at the end of its tick (PetrifyTroll, Ballpark.cpp:6327: fixed,
// at rest, mLastG cleared, non-interactive, RIGID). A sine and a cosine of one argument come from glibc's sincos, as
// compiled Destiny takes them (the finding F3; its patch).
//
// Exactness. Every operation's rounding is pinned as in exact_ballpark.vec_orient_cuda (whose device code this file
// includes): double operations are __dadd_rn ... __dsqrt_rn in the reference's order, float operations __fadd_rn ...
// __fsqrt_rn where Destiny computes in float, conversions explicit. A fused multiply-add appears exactly where the C
// library's compiled code has one: the double sin, cos and log below are glibc 2.34 aarch64's (transcribed from
// exact_ballpark.dlibm, which reads them from libm.so.6's disassembly), exp is exact_ballpark.vec_exp's, the float
// functions and pow are vec_orient_cuda's. No function of CUDA's math library is called; every constant comes from the
// generated headers (exact_ballpark.vec_modes_cuda.tables_header and vec_orient_cuda.tables_header), never typed. trunc
// is one PTX conversion (cvt.rzi), exact.
//
// A ball's float fields (radius, maximum speed, agility, speed fraction, follow range, and with dynamical orientation
// the maximum angular speed, rotational agility, rotation and angular velocity) are floats held in float64 tensors, as
// the park stores them; positions, velocities, the goto point, the mass, the decay factor and the warp length are
// doubles; modes, ids, stamps, follow fields and the clock are 64-bit integers; flags are bytes.

#include "vec_orient_cuda.cu"
#include "vec_modes_cuda_tables.h"

#define XBPM_ABI 1

namespace xbpm {

using xbpo::dabs_;
using xbpo::dadd;
using xbpo::dbits;
using xbpo::ddiv;
using xbpo::dfma;
using xbpo::dfrom;
using xbpo::dinf;
using xbpo::dmul;
using xbpo::dnan;
using xbpo::dsqrt;
using xbpo::dsub;
using xbpo::dtrunc;
using xbpo::fadd;
using xbpo::fdiv;
using xbpo::fmin_;
using xbpo::fmul;
using xbpo::frinta;
using xbpo::fsub;
using xbpo::tod;
using xbpo::tof;
using xbpo::V3;
using xbpo::Q4;

#define KD(name) dfrom(name)        // a double constant from its bit pattern
#define KF(name) xbpo::ffrom(name)  // a float constant from its bit pattern

static __device__ __forceinline__ double dfmsub(double n, double m, double a) { return dfma(-n, m, a); }  // a - n m
static __device__ __forceinline__ double dfnmsub(double n, double m, double a) { return dfma(n, m, -a); } // n m - a
static __device__ __forceinline__ double dcopysign(double a, double s) {
    return dfrom((dbits(a) & 0x7fffffffffffffffULL) | (dbits(s) & 0x8000000000000000ULL));
}
static __device__ __forceinline__ double ll2d(int64_t v) { return __ll2double_rn((long long)v); }   // exact below 2^53

#include "glibc_double.cuh"   // LGPL-2.1-or-later: the C library's sin, cos, sincos, log and exp
// ============================================================================================ the balls' state

// The park's tensors, one row a ball (vectors 3 or 4 doubles a row), passed by value to both kernels.
struct Modes {
    double *pos, *vel, *old_pos, *old_vel, *goto_pt, *last_g;          // doubles
    double *mass, *time_factor, *last_collision;
    double *radius, *max_vel, *agility, *speed_fraction, *follow_range; // floats held in float64
    double *max_ang_vel, *rot_agility, *rot_roll, *rot, *old_rot, *ang_vel, *old_ang_vel, *torque;
    int64_t *ids, *mode, *effect_stamp, *follow_id, *owner_id, *follow_ptr, *clock;
    uint8_t *free_, *moribund, *massive, *interactive, *sensor;
    uint8_t* act;                                                       // the prepass's records
    double *pos0, *vel0, *vy_after;
    int64_t n;
    double friction, dt;
    float ffriction, fdt;
    int new_orbit, dynamical;
};

struct D3 { double x, y, z; };

static __device__ __forceinline__ D3 d3_load(const double* p, int64_t i) { return D3{p[3 * i], p[3 * i + 1], p[3 * i + 2]}; }
static __device__ __forceinline__ void d3_store(double* p, int64_t i, D3 v) {
    p[3 * i] = v.x;
    p[3 * i + 1] = v.y;
    p[3 * i + 2] = v.z;
}
static __device__ __forceinline__ D3 d3_add(D3 a, D3 b) { return D3{dadd(a.x, b.x), dadd(a.y, b.y), dadd(a.z, b.z)}; }
static __device__ __forceinline__ D3 d3_sub(D3 a, D3 b) { return D3{dsub(a.x, b.x), dsub(a.y, b.y), dsub(a.z, b.z)}; }
static __device__ __forceinline__ D3 d3_scale(D3 a, double c) { return D3{dmul(a.x, c), dmul(a.y, c), dmul(a.z, c)}; }
static __device__ __forceinline__ double d3_lensq(D3 v) {             // v0 v0 + v1 v1 + v2 v2, left to right
    return dadd(dadd(dmul(v.x, v.x), dmul(v.y, v.y)), dmul(v.z, v.z));
}
static __device__ __forceinline__ double d3_len(D3 v) { return dsqrt(d3_lensq(v)); }
static __device__ __forceinline__ D3 d3_normalized(D3 v) {
    // Vector3d::Normalize: times the reciprocal of the length; a zero vector stays as it is
    const double norm = d3_len(v);
    if (norm == 0.0) return v;
    return d3_scale(v, ddiv(1.0, norm));
}
static __device__ __forceinline__ D3 d3_cross(D3 u, D3 v) {
    return D3{dsub(dmul(u.y, v.z), dmul(u.z, v.y)), dsub(dmul(u.z, v.x), dmul(u.x, v.z)),
              dsub(dmul(u.x, v.y), dmul(u.y, v.x))};
}
static __device__ __forceinline__ bool dfinite(double x) { return (dbits(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
static __device__ __forceinline__ bool d3_finite(D3 v) { return dfinite(v.x) && dfinite(v.y) && dfinite(v.z); }
static __device__ __forceinline__ D3 d3_zero() { return D3{0.0, 0.0, 0.0}; }

static __device__ __forceinline__ int64_t to_int64(double x) {
    // (int64_t)x on aarch64 (fcvtzs): truncation, saturating, NaN to 0
    if (x != x) return 0;
    if (x >= 9223372036854775808.0) return INT64_MAX;
    if (x < -9223372036854775808.0) return INT64_MIN;
    return (int64_t)__double2ll_rz(x);
}

static __device__ __forceinline__ double trunc7(double x) {
    // (double)((int64_t)(x * 10000000)) / 10000000 (MOV-717): the integer conversion makes -0.3e-7 a +0
    return ddiv(ll2d(to_int64(dmul(x, KD(M_TRUNC7)))), KD(M_TRUNC7));
}

// One ball, as its thread holds it.
struct Ball {
    D3 pos, vel, gp, last_g;
    double mass, tf, last_collision, radius, max_vel, agility, sf, follow_range;
    int64_t id, mode, effect_stamp, follow_id, owner_id, follow_ptr, clock;
    bool massive, interactive, sensor;
};

// ================================================================================== the modes (Ballpark.cpp)

static __device__ __forceinline__ D3 goto_thrust(const Ball& b, D3 target, double friction, double dt) {
    // GotoThrust (MOV-705 to MOV-707): toward the target at maxThrust, tapered by (L^2 / D^2)^2 inside one tick's
    // distance; C's division where the mass may be zero
    const D3 a = d3_sub(target, b.pos);
    const double length2 = d3_lensq(a);
    const double dist = dmul(dmul(b.sf, b.max_vel), dt);
    const double max_thrust = ddiv(dmul(dmul(friction, b.sf), b.max_vel), dmul(b.mass, b.agility));
    const D3 unit = d3_normalized(a);
    const double dd = dmul(dist, dist);
    if (length2 < dd) {
        const double coff = ddiv(length2, dd);
        return d3_scale(unit, dmul(dmul(max_thrust, coff), coff));
    }
    return d3_scale(unit, max_thrust);
}

static __device__ __noinline__ D3 evolve_follow(Ball& b, D3 op, double other_radius, double friction, double dt) {
    // EvolveFollow (MOV-714): the point at centre distance followRange + both radii on the line from the target
    const D3 delta = d3_sub(b.pos, op);
    const double dist = d3_len(delta);
    const double r = dadd(dadd(b.follow_range, b.radius), other_radius);
    D3 target;
    if (dist == 0.0) {
        target = D3{dadd(op.x, dmul(1.0, r)), dadd(op.y, dmul(0.0, r)), dadd(op.z, dmul(0.0, r))};
    } else {
        const double inv = ddiv(1.0, dist);                           // Vector3d's operator/ multiplies by it
        target = D3{dadd(op.x, dmul(dmul(delta.x, r), inv)), dadd(op.y, dmul(dmul(delta.y, r), inv)),
                    dadd(op.z, dmul(dmul(delta.z, r), inv))};
    }
    b.gp = target;
    return goto_thrust(b, target, friction, dt);
}

static __device__ __forceinline__ D3 precessing_normal(const Ball& b) {
    // the precessing w vector of MOV-717 truncated to 7 decimals (DefaultOrbitalNormal: not normalised)
    const double t = ll2d(b.clock);
    const double phi1 = dmul(t, KD(M_ORBITAL_PRECESSION));
    const double phi2 = dadd(ll2d(b.id & 0x000000000000FFFFLL), dmul(t, KD(M_ORBITAL_PRECESSION)));
    double s1, c1, s2, c2;
    g_sincos(phi1, &s1, &c1);
    g_sincos(phi2, &s2, &c2);
    return D3{trunc7(dmul(c1, c2)), trunc7(s2), trunc7(dmul(s1, c2))};
}

static __device__ __noinline__ D3 evolve_old_orbit(Ball& b, D3 op, double other_radius, double friction) {
    // EvolveOldStyleOrbit (MOV-717, MOV-718)
    const double cruise = dmul(b.sf, b.max_vel);
    const double max_thrust = ddiv(dmul(friction, cruise), dmul(b.mass, b.agility));
    const double r = dadd(dadd(b.follow_range, b.radius), other_radius);
    D3 to = d3_sub(op, b.pos);
    const double dist = d3_len(to);
    to = d3_normalized(to);
    const D3 w = d3_normalized(d3_cross(precessing_normal(b), to));
    const double to_comp = dsub(dmul(dist, dist), dmul(r, r));
    if (to_comp >= 0.0) {
        const double rad_comp = ddiv(dmul(r, dsqrt(to_comp)), dist);
        const double s = ddiv(to_comp, dist);
        to = d3_normalized(D3{dadd(dmul(to.x, s), dmul(w.x, rad_comp)), dadd(dmul(to.y, s), dmul(w.y, rad_comp)),
                              dadd(dmul(to.z, s), dmul(w.z, rad_comp))});
    }
    const double dr = dsub(r, dist);
    const double rf = trunc7(g_exp(ddiv(dmul(-dr, dr), KD(M_RF_SCALE))));
    const double phi = dadd(dadd(dmul(-to.x, w.x), dmul(-to.y, w.y)), dmul(-to.z, w.z));
    double tf = dadd(1.0, dmul(dmul(rf, rf), dsub(dmul(phi, phi), 1.0)));
    if (tf > 0.0) {
        tf = dadd(dmul(rf, phi), dsqrt(tf));
    } else {
        tf = dmul(rf, phi);
    }
    tf = dmul(tf, dsub(dist, r) >= 0.0 ? 1.0 : -1.0);                // SIGNUM
    const D3 a = D3{dmul(dadd(dmul(w.x, rf), dmul(to.x, tf)), max_thrust),
                    dmul(dadd(dmul(w.y, rf), dmul(to.y, tf)), max_thrust),
                    dmul(dadd(dmul(w.z, rf), dmul(to.z, tf)), max_thrust)};
    b.gp = d3_add(b.pos, d3_scale(a, dmul(KD(M_GOTO_FAR), KD(M_AU))));
    return a;
}

static __device__ __forceinline__ D3 orbit_thrust(const Ball& b, D3 target, double friction) {
    // OrbitThrust (MOV-722): brake toward the point; zero below a tenth of the maximum acceleration
    const double fom = ddiv(friction, dmul(b.mass, b.agility));
    const double max_accel = dmul(dmul(fom, b.sf), b.max_vel);
    if (max_accel <= 0.0) return d3_zero();
    D3 a = D3{dsub(dmul(dsub(target.x, b.pos.x), fom), b.vel.x), dsub(dmul(dsub(target.y, b.pos.y), fom), b.vel.y),
              dsub(dmul(dsub(target.z, b.pos.z), fom), b.vel.z)};
    const double tacc = ddiv(d3_len(a), max_accel);
    if (tacc > 1.0) a = d3_scale(a, ddiv(1.0, tacc));
    if (tacc < KD(M_THRUST_OFF)) return d3_zero();
    return a;
}

static __device__ __noinline__ D3 goto_thrust_follow(const Ball& b, D3 target, D3 tv, double weight, int new_orbit,
                                                     double friction, double dt) {
    // GotoThrustFollow (MOV-721): reach the point and match the velocity, capped at the maximum acceleration
    const double speed = d3_len(tv);
    if (speed < dmul(dmul(KD(M_SLOW_TARGET), b.max_vel), b.sf)) {
        if (new_orbit && b.mode == MODE_ORBIT) return orbit_thrust(b, target, friction);
        return goto_thrust(b, b.gp, friction, dt);                    // unreachable: the new orbit is the only caller
    }
    const double fom = ddiv(friction, dmul(b.mass, b.agility));
    const double max_accel = dmul(dmul(fom, b.sf), b.max_vel);
    if (max_accel <= 0.0) return d3_zero();
    const double one_minus = dsub(1.0, b.tf);
    double inv = ddiv(1.0, dsub(dmul(dt, fom), one_minus));
    D3 a = D3{dmul(dmul(dsub(dmul(dsub(target.x, b.pos.x), fom), dmul(b.vel.x, one_minus)), fom), inv),
              dmul(dmul(dsub(dmul(dsub(target.y, b.pos.y), fom), dmul(b.vel.y, one_minus)), fom), inv),
              dmul(dmul(dsub(dmul(dsub(target.z, b.pos.z), fom), dmul(b.vel.z, one_minus)), fom), inv)};
    inv = ddiv(1.0, dsub(1.0, b.tf));
    const D3 avel = D3{dmul(dmul(dsub(tv.x, dmul(b.vel.x, b.tf)), fom), inv),
                       dmul(dmul(dsub(tv.y, dmul(b.vel.y, b.tf)), fom), inv),
                       dmul(dmul(dsub(tv.z, dmul(b.vel.z, b.tf)), fom), inv)};
    inv = ddiv(1.0, dadd(1.0, weight));
    a = D3{dmul(dadd(a.x, dmul(avel.x, weight)), inv), dmul(dadd(a.y, dmul(avel.y, weight)), inv),
           dmul(dadd(a.z, dmul(avel.z, weight)), inv)};
    const double accel = d3_lensq(a);
    if (accel > dmul(max_accel, max_accel)) a = d3_scale(a, ddiv(max_accel, dsqrt(accel)));
    return a;
}

static __device__ __noinline__ D3 evolve_new_orbit(Ball& b, D3 op, D3 ovel, double other_radius, const Modes& s,
                                                   Q4 rot) {
    // EvolveNewStyleOrbit (MOV-720) with GetOrbitalNormal (MOV-719): a logarithmic spiral toward the orbit radius
    const double cruise = dmul(b.sf, b.max_vel);
    const double tau = ddiv(dmul(b.mass, b.agility), s.friction);
    const double max_thrust = ddiv(cruise, tau);
    const double r = dadd(dadd(b.follow_range, b.radius), other_radius);
    D3 to = d3_sub(op, b.pos);                                        // GetOrbitalNormal: the predicted target
    to = D3{dadd(to.x, dmul(ovel.x, s.dt)), dadd(to.y, dmul(ovel.y, s.dt)), dadd(to.z, dmul(ovel.z, s.dt))};
    double dist = d3_len(to);
    if (dist > 0.0) {
        to = d3_scale(to, ddiv(1.0, dist));
    } else {
        to = D3{0.0, 0.0, 1.0};
    }
    D3 o = d3_cross(to, b.vel);
    double length2 = d3_lensq(o);
    D3 normal;
    if (length2 > KD(M_NORMAL_MIN)) {
        normal = d3_scale(o, ddiv(1.0, dsqrt(length2)));
    } else if (!s.dynamical) {
        normal = precessing_normal(b);
    } else {                                                          // the ship's heading (dynamical orientation)
        const V3 h = xbpo::transform_coord(V3{0.0f, 0.0f, 1.0f}, xbpo::m_rotation_q(rot));
        o = d3_cross(to, D3{tod(h.x), tod(h.y), tod(h.z)});
        length2 = d3_lensq(o);
        normal = length2 <= KD(M_NORMAL_MIN) ? precessing_normal(b) : d3_scale(o, ddiv(1.0, dsqrt(length2)));
    }
    const D3 radial = d3_normalized(d3_cross(normal, to));
    const double speed = d3_len(b.vel);
    const double max_speed = (cruise < speed) ? speed : cruise;      // std::max(cruise, |v|)
    double angle = ddiv(dmul(KD(M_ANGLE_SCALE), dsub(r, dist)), max_speed);
    if (angle < -1.0) angle = -1.0;
    if (angle > 1.0) angle = 1.0;
    if (dist < KD(M_ANGLE_SCALE)) dist = KD(M_ANGLE_SCALE);
    const double sq = dsqrt(dmul(dist, max_thrust));
    const double est_v = (sq < max_speed) ? sq : max_speed;          // std::min(maxSpeed, sqrt(dist maxThrust))
    const double est_d = dmul(est_v, s.dt);
    D3 gp, gv;
    if (dabs_(angle) < KD(M_ANGLE_ZERO)) {
        const double t = ddiv(est_d, dist);
        double st, ct;
        g_sincos(t, &st, &ct);
        const double cp = dmul(dist, dsub(1.0, ct)), sp = dmul(dist, st), cv = dmul(ct, est_v), sv = dmul(st, est_v);
        gp = D3{dadd(dmul(to.x, cp), dmul(radial.x, sp)), dadd(dmul(to.y, cp), dmul(radial.y, sp)),
                dadd(dmul(to.z, cp), dmul(radial.z, sp))};
        gv = D3{dadd(dmul(radial.x, cv), dmul(to.x, sv)), dadd(dmul(radial.y, cv), dmul(to.y, sv)),
                dadd(dmul(radial.z, cv), dmul(to.z, sv))};
    } else {
        const double factor = ddiv(ddiv(dmul(est_d, angle), dist), dsqrt(dadd(dmul(angle, angle), 1.0)));
        double t;
        if (factor < KD(M_FACTOR_FLOOR)) {
            t = ddiv(g_log(KD(M_LOG_FLOOR)), angle);
        } else if (dabs_(factor) > KD(M_ANGLE_ZERO)) {
            t = ddiv(g_log(dadd(1.0, factor)), angle);
        } else {
            t = ddiv(factor, angle);
        }
        const double dexp = dmul(dist, g_exp(dmul(angle, t)));
        double st, ct;
        g_sincos(t, &st, &ct);
        const double cp = dsub(dist, dmul(dexp, ct)), sp = dmul(dexp, st);
        gp = D3{dadd(dmul(to.x, cp), dmul(radial.x, sp)), dadd(dmul(to.y, cp), dmul(radial.y, sp)),
                dadd(dmul(to.z, cp), dmul(radial.z, sp))};
        const double ca = dadd(dmul(-angle, ct), st), sa = dadd(dmul(angle, st), ct);
        gv = d3_scale(d3_normalized(D3{dadd(dmul(to.x, ca), dmul(radial.x, sa)), dadd(dmul(to.y, ca), dmul(radial.y, sa)),
                                       dadd(dmul(to.z, ca), dmul(radial.z, sa))}), est_v);
    }
    b.gp = d3_add(b.pos, gp);
    return goto_thrust_follow(b, b.gp, d3_add(gv, ovel), 1.0, s.new_orbit, s.friction, s.dt);
}

// ---- warp (TRV-705 to TRV-713) ----

static __device__ __forceinline__ bool aligned_for_warp(D3 gp, D3 pos, D3 vel, double max_vel, int64_t effect_stamp) {
    // IsAlignedForWarp of an aligning ball (TRV-706): within |1 - cos| < 0.01 above 75 % of maxVel, or past 180 ticks
    const D3 d = d3_normalized(d3_sub(gp, pos));
    const D3 u = d3_normalized(vel);
    if (dabs_(dsub(1.0, dadd(dadd(dmul(u.x, d.x), dmul(u.y, d.y)), dmul(u.z, d.z)))) < KD(M_ALIGN_COS) &&
        d3_lensq(vel) > dmul(dmul(KD(M_ALIGN_SPEED2), max_vel), max_vel))
        return true;
    return (effect_stamp < 0 ? -effect_stamp : effect_stamp) > M_MAX_ALIGN_TICKS;
}

static __device__ __forceinline__ bool activates(const Modes& s, int64_t i) {
    // starts warp proper at its turn this tick: free, live, aligning, aligned (its start state: nothing changes a
    // warping ball's own state before its turn)
    if (!s.free_[i] || s.moribund[i] || s.mode[i] != MODE_WARP || s.effect_stamp[i] >= 0) return false;
    return aligned_for_warp(d3_load(s.goto_pt, i), d3_load(s.pos, i), d3_load(s.vel, i), s.max_vel[i], s.effect_stamp[i]);
}

static __device__ __forceinline__ void real_warp(Ball& b) {
    // RealWarp (TRV-707), the ball's own part: the exit point min range short of the goto point, warp proper from now
    D3 dst = b.gp;
    D3 delta = d3_normalized(d3_sub(b.pos, dst));
    const double mr = dfrom((uint64_t)b.follow_id);                   // the minimum range's bits (TRV-703)
    dst = d3_add(dst, d3_scale(delta, mr));
    delta = d3_sub(dst, b.pos);
    b.gp = dst;
    b.effect_stamp = b.clock;
    b.last_collision = d3_len(delta);
    b.massive = false;
    b.sensor = false;
}

static __device__ __noinline__ void warp_distance(Ball& b, D3* pp, D3* vp, double t) {
    // WarpDistance (TRV-708 to TRV-713) with SetupWarpConstants: the new position and velocity on the warp line, aimed
    // from the current position at the exit point; drops the ball out when slow enough, or past the deceleration
    const double wf = ll2d(b.owner_id);
    const double wd = b.last_collision;
    double s = dmul(dmul(wf, KD(M_WARP_AU_PER_SECOND)), KD(M_AU));
    const double acc = dmul(wf, KD(M_WARP_ACCELERATION));
    const double d0 = dmul(wf, KD(M_WARP_DECELERATION));
    const double dec = (2.0 < d0) ? 2.0 : d0;                         // std::min(..., 2.0)
    const double cap = ddiv(dmul(dmul(dadd(wd, 1.0), acc), dec), dadd(acc, dec));
    s = (cap < s) ? cap : s;
    const double t_a = ddiv(g_log(ddiv(s, acc)), acc);
    const double t_c = dadd(dsub(dsub(ddiv(wd, s), ddiv(1.0, acc)), ddiv(1.0, dec)), ddiv(1.0, s));
    const double t_d = ddiv(g_log(ddiv(s, dec)), dec);
    const double d_a = ddiv(s, acc), d_c = dmul(s, t_c), d_d = dsub(ddiv(s, dec), 1.0);
    const D3 p = *pp;
    const D3 d = d3_normalized(d3_sub(b.gp, p));
    const double total = dadd(dadd(d_a, d_c), d_d);
    bool drop = false;
    if (t < t_a) {
        const double e = g_exp(dmul(acc, t));
        double speed = dmul(acc, e);
        const double cur = d3_len(*vp);
        if (cur > speed) speed = cur;                                 // the reported speed is kept continuous
        *vp = d3_scale(d, speed);
        *pp = d3_sub(b.gp, d3_scale(d, dsub(total, e)));
    } else if (dsub(t, t_a) < t_c) {
        *vp = d3_scale(d, s);
        *pp = d3_sub(b.gp, d3_scale(d, dsub(total, dadd(dmul(s, dsub(t, t_a)), d_a))));
    } else if (dsub(dsub(t, t_c), t_a) < dadd(t_d, 1.0)) {
        const double e = g_exp(dmul(-dec, dsub(dsub(t, t_c), t_a)));
        const double speed = dmul(s, e);
        *vp = d3_scale(d, speed);
        const double sd = ddiv(s, dec);
        *pp = d3_sub(b.gp, d3_scale(d, dsub(total, dadd(dadd(dsub(sd, dmul(sd, e)), d_a), d_c))));
        const double half = ddiv(b.max_vel, KD(M_HALF));
        drop = speed < ((KD(M_DROP_SPEED) < half) ? KD(M_DROP_SPEED) : half);
    } else {                                                          // past the deceleration phase by more than 1 s
        *pp = b.gp;
        *vp = d3_zero();
        drop = true;
    }
    if (drop) {                                                       // massive and sensing again; Stop (no followers)
        b.massive = true;
        b.sensor = true;
        b.mode = MODE_STOP;
    }
}

// ---- the commands a tick applies: StopAllFollowers' release (MOV-730), Stop (Ballpark.cpp:4578) ----

static __device__ __forceinline__ void stop_following(Ball& b) {
    // Stop(Ball*)'s follow part (the ball in FOLLOW or ORBIT): the follow fields cleared; its target's follower set is
    // implicit here (the balls in FOLLOW or ORBIT whose follow_ptr is the target's row)
    b.effect_stamp = 0;
    b.follow_ptr = -1;
    b.follow_id = 0;
    b.owner_id = 0;
    b.follow_range = 0.0;
}

static __device__ __forceinline__ int64_t release(Ball& b, bool apply) {
    // StopAllFollowers on this follower: an interactive orbiter flies on by GotoDirection(its velocity) (refused when
    // the velocity or the point is not finite: then it stays), any other member stops: Stop(Ball*) clears the follow
    // fields only in FOLLOW or ORBIT (a member SetBoid took out of them keeps its own); returns the mode it is left in
    if (b.interactive && b.mode == MODE_ORBIT) {
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
        if (b.mode == MODE_FOLLOW || b.mode == MODE_ORBIT) stop_following(b);
        b.mode = MODE_STOP;
    }
    return MODE_STOP;
}

static __device__ __forceinline__ bool member_of(const Modes& s, int64_t t, int64_t follow_id) {
    // a ball whose follow_ptr is row t is in t's follower set as StopAllFollowers acts on it when its follow id is t's
    // id (Ballpark.cpp:5433-5450): every FOLLOW or ORBIT ball, and a ball SetBoid took out of one without a Stop
    return t >= 0 && t < s.n && s.ids[t] == follow_id;
}

// ---- the integrator (MOV-701 to MOV-703) ----

static __device__ __forceinline__ void integrate(D3* pp, D3* vp, D3 a, double m, double k, double tf, double t) {
    // Ballpark::Integrate over one tick (t = dt, so the cached decay factor): the closed form, or its Taylor branch
    if (t == 0.0) return;
    const D3 p = *pp, v = *vp;
    const double k2 = dmul(k, k), ook2 = ddiv(1.0, k2), ook = ddiv(1.0, k);
    D3 pn;
    if (k < dmul(dmul(KD(M_TAYLOR_LIMIT), m), t)) {
        const double inv_m = ddiv(1.0, m);
        const double kt = dmul(-k, t);
        pn = D3{dmul(dadd(dmul(m, dadd(dmul(a.x, kt), dmul(k, dadd(dmul(a.x, t), dmul(dmul(dmul(v.x, k), inv_m), t))))),
                          dmul(p.x, k2)), ook2),
                dmul(dadd(dmul(m, dadd(dmul(a.y, kt), dmul(k, dadd(dmul(a.y, t), dmul(dmul(dmul(v.y, k), inv_m), t))))),
                          dmul(p.y, k2)), ook2),
                dmul(dadd(dmul(m, dadd(dmul(a.z, kt), dmul(k, dadd(dmul(a.z, t), dmul(dmul(dmul(v.z, k), inv_m), t))))),
                          dmul(p.z, k2)), ook2)};
    } else {
        const double tf1 = dsub(tf, 1.0);
        pn = D3{dmul(dadd(dmul(m, dadd(dmul(dmul(m, a.x), tf1), dmul(k, dsub(dadd(dmul(a.x, t), v.x), dmul(tf, v.x))))),
                          dmul(p.x, k2)), ook2),
                dmul(dadd(dmul(m, dadd(dmul(dmul(m, a.y), tf1), dmul(k, dsub(dadd(dmul(a.y, t), v.y), dmul(tf, v.y))))),
                          dmul(p.y, k2)), ook2),
                dmul(dadd(dmul(m, dadd(dmul(dmul(m, a.z), tf1), dmul(k, dsub(dadd(dmul(a.z, t), v.z), dmul(tf, v.z))))),
                          dmul(p.z, k2)), ook2)};
    }
    *vp = D3{dmul(dsub(dmul(m, a.x), dmul(dsub(dmul(m, a.x), dmul(v.x, k)), tf)), ook),
             dmul(dsub(dmul(m, a.y), dmul(dsub(dmul(m, a.y), dmul(v.y, k)), tf)), ook),
             dmul(dsub(dmul(m, a.z), dmul(dsub(dmul(m, a.z), dmul(v.z, k)), tf)), ook)};
    *pp = pn;
}

// ---- dynamical orientation (MOV-728): vec_orient_cuda's device code ----

static __device__ __noinline__ D3 orient(const Ball& b, D3 a, const Modes& s, int64_t i, Q4 r0, V3 w0, V3* torque_out,
                                         float* roll_out, Q4* rot_out, V3* w_out) {
    // EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890): above 1 % of the maximum
    // acceleration the torque and roll are searched and the thrust laid along the heading a tick on, scaled by cos^2 of
    // its angle (zero at 90 degrees or more); below it the thrust is kept, a ball in warp proper turns toward its exit
    // point, and the roll is steered. ApplyTorque's rotation and angular velocity (the tick's end, same inputs) out.
    const float accel = tof(d3_len(a));
    const float fmass = tof(b.mass), ag = tof(b.agility), sf = tof(b.sf), mv = tof(b.max_vel);
    const float rag = tof(s.rot_agility[i]);
    const float max_accel = fdiv(fmul(fmul(s.ffriction, sf), mv), fmul(fmass, ag));
    const bool thrust = tod(accel) > dmul(KD(M_DO_THRUST), tod(max_accel));
    xbpo::Ship sh;
    sh.max_ang_vel = tof(s.max_ang_vel[i]);
    sh.t_scale = fdiv(s.ffriction, fmul(fmul(fmass, ag), rag));
    sh.tf = xbpo::g_expf(fmul(-s.fdt, sh.t_scale));
    sh.dt = s.dt;
    sh.fdt = s.fdt;
    const V3 a32 = V3{tof(a.x), tof(a.y), tof(a.z)};
    V3 torque = V3{0.0f, 0.0f, 0.0f};
    if (thrust) {
        torque = xbpo::goto_torque_breaks(sh, a32, r0, w0);
    } else if (b.mode == MODE_WARP && b.effect_stamp >= 0) {
        const D3 g = d3_sub(b.gp, b.pos);
        torque = xbpo::goto_torque_breaks(sh, V3{tof(g.x), tof(g.y), tof(g.z)}, r0, w0);
    }
    const float roll = xbpo::goto_roll_breaks(sh, torque, r0, w0);
    const float tau = fdiv(fmul(fmul(fmass, ag), rag), s.ffriction);
    xbpo::apply_torque(r0, w0, torque, roll, tau, s.fdt, rot_out, w_out);
    *torque_out = torque;
    *roll_out = roll;
    if (!thrust) return a;
    const V3 heading = xbpo::transform_coord(V3{0.0f, 0.0f, 1.0f}, xbpo::m_rotation_q(*rot_out));
    const float cosangle = fmin_(1.0f, fdiv(xbpo::v3_dot(heading, a32), accel));
    if (cosangle <= 0.0f) return d3_zero();
    const V3 h = xbpo::v3_mul(heading, fmul(fmul(cosangle, cosangle), accel));
    return D3{tod(h.x), tod(h.y), tod(h.z)};
}

// ================================================================================================== the kernels

// What each ball shows the others in phase 1: its start-of-tick position and velocity, its vertical velocity after its
// own turn (STOP's damping without dynamical orientation, a follower released before its turn counted in its new
// mode), and whether it starts warp proper at its turn.
__global__ void prepass_kernel(Modes s) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
    s.act[i] = activates(s, i) ? 1 : 0;
    const D3 vel = d3_load(s.vel, i);
    d3_store(s.pos0, i, d3_load(s.pos, i));
    d3_store(s.vel0, i, vel);
    double vy = vel.y;
    if (s.free_[i] && !s.moribund[i] && !s.dynamical) {
        int64_t m = s.mode[i];
        const int64_t t = s.follow_ptr[i];
        if (member_of(s, t, s.follow_id[i]) && s.ids[t] < s.ids[i] && activates(s, t)) {
            Ball f;
            f.mode = m;
            f.interactive = s.interactive[i] != 0;
            f.vel = vel;
            f.pos = d3_load(s.pos, i);
            m = release(f, false);
        }
        if (m == MODE_STOP) vy = dmul(dsub(vy, dmul(KD(M_STOP_DECAY), vy)), KD(M_STOP_DAMP));
    }
    s.vy_after[i] = vy;
}

// Each ball's tick in place: phase 1 (the release when its target starts warp proper, before or after its own turn
// by their ids; the mode's behaviour; dynamical orientation), phase 3 (the warp path or the integrator; the rotation a
// tick on), phase 4 (the commit), and its world's clock.
__global__ void main_kernel(Modes s) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.n) return;
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
    const bool live = s.free_[i] && !s.moribund[i];
    const bool act_self = s.act[i] != 0;
    // ---- phase 1 ----
    bool release_before = false, release_after = false;
    if (member_of(s, b.follow_ptr, b.follow_id) && s.act[b.follow_ptr]) {
        if (s.ids[b.follow_ptr] < b.id) release_before = true;
        else release_after = true;
    }
    if (release_before) release(b, true);
    V3 torque = V3{0.0f, 0.0f, 0.0f};
    float roll = 0.0f;
    Q4 r0 = Q4{0.0f, 0.0f, 0.0f, 1.0f}, rot1 = r0;
    V3 w0 = V3{0.0f, 0.0f, 0.0f}, w1 = w0;
    if (live) {
        D3 a = d3_zero();
        const int64_t t = b.follow_ptr;
        if (b.mode == MODE_WARP) {                                    // EvolveWarp (TRV-705, TRV-707)
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
        } else if ((b.mode == MODE_FOLLOW || b.mode == MODE_ORBIT) && !(t >= 0 && t < s.n)) {
            a = d3_zero();                                            // no target row: the park never leaves one so
        } else if (b.mode == MODE_FOLLOW) {
            a = evolve_follow(b, d3_load(s.pos0, t), s.radius[t], s.friction, s.dt);
        } else if (b.mode == MODE_ORBIT) {
            const D3 op = d3_load(s.pos0, t);
            if (s.new_orbit) {
                D3 ov = d3_load(s.vel0, t);
                if (s.ids[t] < b.id) ov.y = s.vy_after[t];            // the target's turn came first
                if (s.dynamical) r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
                a = evolve_new_orbit(b, op, ov, s.radius[t], s, r0);
            } else {
                a = evolve_old_orbit(b, op, s.radius[t], s.friction);
            }
        } else if (b.mode == MODE_STOP) {
            if (!s.dynamical) b.vel.y = dmul(dsub(b.vel.y, dmul(KD(M_STOP_DECAY), b.vel.y)), KD(M_STOP_DAMP));
        }                                                             // BOID, MINIBALL, FIELD, RIGID: no thrust
        if (s.dynamical) {
            r0 = Q4{tof(s.rot[4 * i]), tof(s.rot[4 * i + 1]), tof(s.rot[4 * i + 2]), tof(s.rot[4 * i + 3])};
            w0 = V3{tof(s.ang_vel[3 * i]), tof(s.ang_vel[3 * i + 1]), tof(s.ang_vel[3 * i + 2])};
            a = orient(b, a, s, i, r0, w0, &torque, &roll, &rot1, &w1);
        }
        b.last_g = a;
    }
    if (release_after) release(b, true);
    // ---- phases 3 and 4 ----
    if (live) {
        if (b.massive) b.last_collision = -1.0;                       // the gradient without neighbours
        D3 p = b.pos, v = b.vel;
        const D3 a = D3{dadd(b.last_g.x, 0.0), dadd(b.last_g.y, 0.0), dadd(b.last_g.z, 0.0)};   // mLastG + mLastC
        const double m = dmul(b.mass, b.agility);
        if (b.mode == MODE_WARP && b.effect_stamp >= 0) {
            warp_distance(b, &p, &v, dmul(ll2d(b.clock - b.effect_stamp), s.dt));
            if (b.mode == MODE_STOP) {                                // TRV-714: the drop-out point becomes the old state
                b.pos = p;
                b.vel = v;
                integrate(&p, &v, a, m, s.friction, b.tf, s.dt);
            }
        } else {
            integrate(&p, &v, a, m, s.friction, b.tf, s.dt);
        }
        d3_store(s.old_pos, i, b.pos);
        d3_store(s.old_vel, i, b.vel);
        b.pos = p;
        b.vel = v;
        if (s.dynamical) {                                            // the commit: new values where finite
            const bool rot_ok = xbpo::ffinite(xbpo::q_lensq(rot1));
            const bool w_ok = xbpo::ffinite(xbpo::v3_dot(w1, w1));
            const Q4 rn = rot_ok ? rot1 : r0, ro = rot_ok ? r0 : rot1;
            const V3 wn = w_ok ? w1 : w0, wo = w_ok ? w0 : w1;
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
            s.torque[3 * i] = tod(torque.x);
            s.torque[3 * i + 1] = tod(torque.y);
            s.torque[3 * i + 2] = tod(torque.z);
            s.rot_roll[i] = tod(roll);
        }
    }
    if (live && b.mode == MODE_TROLL && !(b.effect_stamp > b.clock)) {
        // PetrifyTroll (Ballpark.cpp:6327; MOV-736) of a troll due (TrollReady, 6315), after the commit: SetBallFree
        // false (Stop(Ball*), a troll in no follow mode; SetBallVelocity(0, 0, 0), the new and old velocity, whose
        // rotation snap leaves a ball at rest as it is; out of the free set; mLastG cleared), SetBallInteractive
        // false, RIGID. Only this ball's fields change, and no other ball reads them in this tick.
        b.vel = d3_zero();
        d3_store(s.old_vel, i, d3_zero());
        b.last_g = d3_zero();
        s.free_[i] = 0;
        s.interactive[i] = 0;
        b.mode = MODE_RIGID;
    }
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
    s.clock[i] = b.clock + 1;
    s.massive[i] = b.massive ? 1 : 0;
    s.sensor[i] = b.sensor ? 1 : 0;
}

// The double functions alone, for the tests: out = f(x)
__global__ void dlibm_kernel(int func, int64_t n, const double* __restrict__ x, double* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (func) {
        case 0: out[i] = g_sin(x[i]); break;
        case 1: out[i] = g_cos(x[i]); break;
        case 2: out[i] = g_log(x[i]); break;
        case 3: out[i] = g_exp(x[i]); break;
        case 4: {                                                     // sincos's sine
            double sn, cs;
            g_sincos(x[i], &sn, &cs);
            out[i] = sn;
            break;
        }
        case 5: {                                                     // sincos's cosine
            double sn, cs;
            g_sincos(x[i], &sn, &cs);
            out[i] = cs;
            break;
        }
        default: out[i] = dnan(); break;
    }
}

}  // namespace xbpm

// ====================================================================================== the C interface (ctypes)

extern "C" {

int xbpm_abi(void) { return XBPM_ABI; }

// One tick for the park's n balls on `stream` (torch's current stream): the prepass, then the main kernel; no
// allocation and no host sync, so it can be captured in a CUDA graph. `state` is copied into both launches.
int xbpm_tick(const xbpm::Modes* state, void* stream, int block) {
    if (state->n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    const unsigned grid = grid_for(state->n, block);
    xbpm::prepass_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(*state);
    const int rc = launch_check();
    if (rc) return rc;
    xbpm::main_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(*state);
    return launch_check();
}

// The main kernel as loaded on the current device: registers, local memory, the binary and PTX versions.
int xbpm_info(int which, int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    const cudaError_t e = which == 0 ? cudaFuncGetAttributes(&fa, xbpm::main_kernel)
                                     : cudaFuncGetAttributes(&fa, xbpm::prepass_kernel);
    if (e != cudaSuccess) return (int)e;
    *regs = fa.numRegs;
    *local_bytes = (int)fa.localSizeBytes;
    *binary_version = fa.binaryVersion;
    *ptx_version = fa.ptxVersion;
    *max_threads = fa.maxThreadsPerBlock;
    return 0;
}

int xbpm_dlibm(int func, int64_t n, const double* x, double* out, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0 || func < 0 || func > 5) return -1;
    xbpm::dlibm_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(func, n, x, out);
    return launch_check();
}

}  // extern "C"
