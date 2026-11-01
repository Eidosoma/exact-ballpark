// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's gradient collision step as one CUDA kernel: one thread a responder (a
// world's free ball in one of the park's rows), running the algorithm of exact_ballpark.vec_collide.VecCollidePark's
// _gradient, which is SpaceBallpark's Gradient and Potential (exact_ballpark.space; Ballpark.cpp:2745 and 2789, Destiny
// at 114e89fa, MIT, (c) CCP Games), with every operation's rounding pinned, so that a tick's results equal the torch
// module's bit for bit (tests/test_movement_vec_collide_cuda.py). Built and loaded by exact_ballpark.vec_collide_cuda.
//
// Exactness by construction:
// - every float64 operation is a correctly rounded IEEE operation in the torch module's order: +, -, *, / and
//   __dsqrt_rn, compiled with -fmad=false (nothing contracted), -prec-div=true, -prec-sqrt=true, -ftz=false and no fast
//   math; the build reads the PTX back and refuses an unrounded, flushed, approximate or contracted instruction;
// - a fused multiply-add only inside xbpc_exp, exactly where exact_ballpark.vec_exp emulates one (glibc 2.34 aarch64's
//   exp fuses seven of its multiply-adds and one in its special case): __fma_rn is the IEEE fused operation, rounded
//   once, which vec_exp's emulation computes exactly while no partial product overflows or underflows;
// - no function of CUDA's math library: the C library's exp is transcribed (xbpc_exp), its constants and table
//   generated from vec_exp into vec_collide_cuda_tables.h by the loader, never typed; the build scans the source;
// - the contact order is the partition's walk as vec_collide._order_keys computes it (integers only), then the other
//   ball's id, then its slot (the two stable sorts of vec_collide._sorted); a ball's contacts are processed in that
//   order, each through Potential's recursion (the deepest update first, then the light-ball rule and the early return
//   at each depth, as the scalar code meets them).
//
// The torch module computes every (ball, other) pair's Potential at once and then applies the order-dependent updates;
// this kernel walks a ball's candidates in order and computes each pair's values when it reaches them, by the same
// expressions: the values are the same, and the updates come in the same order. Its differences in form: no cap on the
// impact branch (VecCollidePark's impact_slots does not apply: every pair that reaches it is computed), and the
// collision tensors are updated in place (the torch path rebinds them to new tensors with the same values).
//
// The tick's other parts of the park, one thread a ball, by the same rules (xbpc_tick): the behaviours
// (behaviour_kernel, vec_collide._behaviour), the end of the tick (commit_kernel: Integrate with mLastG + mLastC, the
// commit and the re-filing, vec_collide.evolve's (3) and (4)) and the filing (filing_kernel, vec_collide._filing).
//
// Planted faults, for the tests only (the loader's `defines`): XBPC_FAULT=1 contracts one multiply-add (the
// integration's last sum), 2 calls CUDA's exp in place of the transcription, 3 walks a ball's contacts in reverse
// order; the loader's `perturb` flips one bit of a generated table word.
//
// The composed park's switch (the loader's `composed`): XBPC_COMPOSED=1 builds the
// same kernel for exact_ballpark.vec_composed.VecComposedPark's [world, slot] view (every slot a responder's row), with
// that park's candidates (VecComposedPark._pair_mask: a normal missile's one candidate its target, in its world and
// bubble, unfiltered and first in order; a defender missile's GetNearbyBalls with missiles kept, in the walk's order
//; every other ball's collisionCase1's less missiles, a ball dropping out of warp this
// tick a candidate of the higher ids only; Gradient's skips) and COL-711's exception (a missile's response to its own
// target not kept, the target's id appended: _GradientView._update). Every addition is inside `#if XBPC_COMPOSED`, so
// the default build (XBPC_COMPOSED=0) compiles to what it compiled to before.
//
// The master's bubbles (the default build only, every addition inside
// `#if !XBPC_COMPOSED`): xbpc_gradient takes one more pointer after overflow, the park's bubble per slot
// (vec_collide's `bubble`, written by exact_ballpark.vec_space's partition layer), or null for a client's view. Where it
// is given, a candidate in another bubble than the responder's is skipped, as the master's GetNearbyBalls skips it
// (Partition.cpp:344; SEN-708; no global balls here); with null every candidate stays, as before.
//
// The static collidables' switch (the loader's `statics`): XBPC_STATICS=1 builds the
// default kernel with Destiny's capsules and oriented boxes in the gradient (COL-712; Gradient's walk of its static
// collidables after its balls, Ballpark.cpp:2778-2783): xbpc_gradient takes two more dims (the static slots a world,
// the boxes a static collidable) and four more pointers after the bubble pointer (vec_collide's s_exists, s_geo, s_int,
// s_keys), and each responder walks its world's static collidables in the partition walk's order through
// CollideWithBall and ReactToCollision (vec_collide._gradient_statics). Every addition is inside
// `#if XBPC_STATICS` (its own library, for a park with static slots), so the default build (XBPC_STATICS=0) compiles to
// what it compiled to before; it is refused with XBPC_COMPOSED.
//
// The minis and the force fields (COL-713, MOV-737), in the statics build, which a
// park made with `minis` or `fields` runs with or without static slots (the static slots may then be 0, their pointers
// null): five more pointers after s_keys, the park's sort_id (each ball's rank in BallSortComparer's order,
// Ball.cpp:2528: local ids by their owner first; or null, the id) and its mode, harmonic, corporation and alliance (or
// null: no fields). A ball's candidates are taken in the walk's order with the rank in place of the id
// (vec_collide._sorted), and a candidate in mode FIELD is skipped when the responder passes it (its harmonic -2, or the
// field's harmonic, corporation or alliance set and the responder's: GetNearbyBalls, Partition.cpp:344;
// vec_collide._field_passed). Every addition is inside `#if XBPC_STATICS`.
//
// The global balls (SEN-708), in the statics build, which a park made with
// `global_balls` runs (with or without static slots, minis or fields): one more pointer after the alliance, the
// park's global flag per slot (vec_collide's `glob`), or null (no global balls). On the master's view (the bubble
// pointer given) a candidate in another bubble than the responder's is skipped unless it is global, as the master's
// GetNearbyBalls skips it (Partition.cpp:344; the rule reads the candidate's flag alone, so it is asymmetric:
// vec_collide._same_bubble). Every addition is inside `#if XBPC_STATICS`, so the default and composed builds compile
// to what they compiled to before.

#include <cstdint>
#include <cuda_runtime.h>

#include "vec_collide_cuda_tables.h"

#ifndef XBPC_BLOCK
#define XBPC_BLOCK 128
#endif
#ifndef XBPC_FAULT
#define XBPC_FAULT 0
#endif
#ifndef XBPC_COMPOSED
#define XBPC_COMPOSED 0
#endif
#ifndef XBPC_STATICS
#define XBPC_STATICS 0
#endif
#if XBPC_STATICS && XBPC_COMPOSED
#error "the static collidables are the default build's (XBPC_STATICS without XBPC_COMPOSED)"
#endif
#define XBPC_ABI 1

typedef double f64;
typedef long long i64;
typedef unsigned long long u64;
typedef unsigned char u8;

namespace {

// ============================================================================================== vectors (Vector3d)

struct V3 {
    f64 x, y, z;
};

__device__ __forceinline__ V3 ld3(const f64* a, i64 i) { return {a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }
__device__ __forceinline__ void st3(f64* a, i64 i, V3 v) {
    a[3 * i] = v.x;
    a[3 * i + 1] = v.y;
    a[3 * i + 2] = v.z;
}
__device__ __forceinline__ V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ V3 operator*(V3 a, f64 s) { return {a.x * s, a.y * s, a.z * s}; }

// Vector3d's dot product, x first: ((x x' + y y') + z z') (vec_collide._dot)
__device__ __forceinline__ f64 dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Vector3d::Normalize: multiply by the reciprocal of the length; a zero vector stays zero (vec_collide._normalized)
__device__ __forceinline__ V3 normalized(V3 v) {
    const f64 n = __dsqrt_rn(dot(v, v));
    if (n == 0.0) return v;
    const f64 inv = 1.0 / n;
    return v * inv;
}

#include "glibc_exp.cuh"   // LGPL-2.1-or-later: the C library's exp
// =========================================================================================== the park and the tick

struct Consts {           // the host's doubles, computed in Python as vec_collide computes them
    f64 k, dt, k2, ook2, ook, neg_k, neg_kt, inv_dt, dtk;
};

struct Park {             // the park's tensors (contiguous, on one device) and its dimensions
    i64 W, N, R, slots, limit, hits_cap, imp_cap;
    const f64 *pos, *vel, *last_g, *mass, *agility, *tf, *radius;
    const u8 *exists, *free, *massive, *junk, *cloaked, *responder;
    const i64 *id, *level, *key, *mod;
    f64 *last_c, *last_col, *imp_s, *imp_cl;
    i64 *hits, *hits_n, *imp_slot, *imp_n;
    u8* overflow;
#if !XBPC_COMPOSED
    const i64* master_bubble;      // the master's bubble per slot, or null (a client's view)
#endif
#if XBPC_COMPOSED
    const u8 *missile, *dropped;   // in MISSILE mode (and existing); dropping out of warp this tick (non-massive)
    const i64 *mode, *owner, *follow, *bubble, *tslot;   // mode, mOwnerId, the followed id, bubble, the target's slot
#endif
#if XBPC_STATICS
    i64 S, B;                      // static slots a world, boxes a static collidable
    const u8* s_exists;            // [W, S]
    const f64* s_geo;              // [W, S, XBPC_G_SIZE]: vec_collide.static_record's doubles
    const i64 *s_int, *s_keys;     // [W, S, XBPC_I_SIZE] its integers; [W, S, B, 3] its boxes' keys
    const i64* sort_id;            // BallSortComparer's rank per slot [W, N], or null (the id)
    const i64 *mode, *harmonic, *corporation, *alliance;   // [W, N] each, or null (no fields)
    const u8* glob;                // each slot's global flag [W, N], or null (no global balls)
#endif
};

struct Ball {             // a ball's start-of-tick state as Potential reads it
    V3 p, v, g;
    f64 m;                // mass x agility, or 1e34 for a fixed ball (Potential's m1, m2)
    f64 mass, tf, radius;
    bool free;
    i64 id;
};

__device__ __forceinline__ Ball load_ball(const Park& P, i64 j) {
    Ball b;
    b.p = ld3(P.pos, j);
    b.v = ld3(P.vel, j);
    b.g = ld3(P.last_g, j);
    b.mass = P.mass[j];
    b.free = P.free[j] != 0;
    b.m = b.free ? b.mass * P.agility[j] : K_FIXED_MASS;
    b.tf = P.tf[j];
    b.radius = P.radius[j];
    b.id = P.id[j];
    return b;
}

// Ballpark::Integrate over the tick with the cached decay factor, the position (vec_collide._integrate_dt)
__device__ __forceinline__ V3 end_position(const Consts& c, V3 p, V3 v, V3 a, f64 m, f64 tf) {
    const f64 k = c.k, t = c.dt;
    if (k < K_TAYLOR * m * t) {
        const f64 inv_m = 1.0 / m;
#if XBPC_FAULT == 1
        return {__fma_rn(p.x, c.k2, m * (a.x * c.neg_kt + k * (a.x * t + v.x * k * inv_m * t))) * c.ook2,
                __fma_rn(p.y, c.k2, m * (a.y * c.neg_kt + k * (a.y * t + v.y * k * inv_m * t))) * c.ook2,
                __fma_rn(p.z, c.k2, m * (a.z * c.neg_kt + k * (a.z * t + v.z * k * inv_m * t))) * c.ook2};
#else
        return {(m * (a.x * c.neg_kt + k * (a.x * t + v.x * k * inv_m * t)) + p.x * c.k2) * c.ook2,
                (m * (a.y * c.neg_kt + k * (a.y * t + v.y * k * inv_m * t)) + p.y * c.k2) * c.ook2,
                (m * (a.z * c.neg_kt + k * (a.z * t + v.z * k * inv_m * t)) + p.z * c.k2) * c.ook2};
#endif
    }
#if XBPC_FAULT == 1
    return {__fma_rn(p.x, c.k2, m * (m * a.x * (tf - 1.0) + k * (a.x * t + v.x - tf * v.x))) * c.ook2,
            __fma_rn(p.y, c.k2, m * (m * a.y * (tf - 1.0) + k * (a.y * t + v.y - tf * v.y))) * c.ook2,
            __fma_rn(p.z, c.k2, m * (m * a.z * (tf - 1.0) + k * (a.z * t + v.z - tf * v.z))) * c.ook2};
#else
    return {(m * (m * a.x * (tf - 1.0) + k * (a.x * t + v.x - tf * v.x)) + p.x * c.k2) * c.ook2,
            (m * (m * a.y * (tf - 1.0) + k * (a.y * t + v.y - tf * v.y)) + p.y * c.k2) * c.ook2,
            (m * (m * a.z * (tf - 1.0) + k * (a.z * t + v.z - tf * v.z)) + p.z * c.k2) * c.ook2};
#endif
}

// Ballpark::Integrate over a time t: nothing for t = 0, the cached factor for t = dt, else exp's value f_t
// (vec_collide._integrate_t)
__device__ __forceinline__ void integrate_t(const Consts& c, V3 p, V3 v, V3 a, f64 m, f64 tf, f64 t, f64 f_t, V3& pn,
                                            V3& vn) {
    if (t == 0.0) {
        pn = p;
        vn = v;
        return;
    }
    const f64 k = c.k;
    const f64 f = t != c.dt ? f_t : tf;
    if (k < K_TAYLOR * m * t) {
        const f64 inv_m = 1.0 / m, nkt = c.neg_k * t;
        pn = {(m * (a.x * nkt + k * (a.x * t + v.x * k * inv_m * t)) + p.x * c.k2) * c.ook2,
              (m * (a.y * nkt + k * (a.y * t + v.y * k * inv_m * t)) + p.y * c.k2) * c.ook2,
              (m * (a.z * nkt + k * (a.z * t + v.z * k * inv_m * t)) + p.z * c.k2) * c.ook2};
    } else {
        pn = {(m * (m * a.x * (f - 1.0) + k * (a.x * t + v.x - f * v.x)) + p.x * c.k2) * c.ook2,
              (m * (m * a.y * (f - 1.0) + k * (a.y * t + v.y - f * v.y)) + p.y * c.k2) * c.ook2,
              (m * (m * a.z * (f - 1.0) + k * (a.z * t + v.z - f * v.z)) + p.z * c.k2) * c.ook2};
    }
    vn = {(m * a.x - (m * a.x - v.x * k) * f) * c.ook, (m * a.y - (m * a.y - v.y * k) * f) * c.ook,
          (m * a.z - (m * a.z - v.z * k) * f) * c.ook};
}

// exp's argument for a time t that is not the tick: (-k / m) t (vec_collide._decay)
__device__ __forceinline__ f64 decay_factor(const Consts& c, f64 m, f64 t) {
    if (t == 0.0 || t == c.dt) return 0.0;                              // not read by integrate_t: exp not needed
    return exp_of((c.neg_k / m) * t);
}

// CollideTwoSpheres (Collision.cpp:108; COL-705): 0 if already touching, else the earlier root in [0, 1], else -1
// (vec_collide._collide_two_spheres)
__device__ __forceinline__ f64 collide_two_spheres(V3 p0, V3 p1, V3 q0, V3 q1, f64 cr) {
    const V3 p0q0 = p0 - q0;
    const f64 p0q0_2 = dot(p0q0, p0q0);
    const f64 cr2 = cr * cr;
    if (p0q0_2 <= cr2) return 0.0;
    const V3 dpq = (p1 - p0) - (q1 - q0);
    const f64 a = dot(dpq, dpq);
    const f64 b = 2.0 * dot(p0q0, dpq);
    const f64 cc = p0q0_2 - cr2;
    if (a == 0.0 && b == 0.0) return -1.0;
    const f64 det = b * b - 4.0 * a * cc;
    if (det < 0.0) return -1.0;
    const f64 root = __dsqrt_rn(det);
    const f64 s2 = ((-b - root) * 0.5) / a;
    if (s2 >= 0.0 && s2 <= 1.0) return s2;
    return -1.0;
}

// Potential's impact branch (s > 0; COL-707, COL-708; vec_collide._approach): the acceleration a1 and the closing
// speed along the normal at the impact (the impact record)
__device__ f64 approach(const Consts& c, const Ball& me, const Ball& ot, V3 pb, V3 pn, f64 s, V3& a1) {
    const f64 k = c.k;
    const f64 ts = s * c.dt, tr = (1.0 - s) * c.dt;
    const f64 e0 = decay_factor(c, me.m, ts), e1 = decay_factor(c, ot.m, ts), e2 = decay_factor(c, me.m, tr);
    V3 p1, vp1, q1, vq1, p2, vp2;
    integrate_t(c, pb, me.v, me.g, me.m, me.tf, ts, e0, p1, vp1);
    integrate_t(c, pn, ot.v, ot.g, ot.m, ot.tf, ts, e1, q1, vq1);
    const V3 normal = normalized(q1 - p1);
    const f64 u1 = dot(vp1, normal), u2 = dot(vq1, normal);
    if (ot.free) {                                                      // COL-708: an exchange, base masses
        const f64 u1p = ((me.mass * u1 - ot.mass * u1) + (2.0 * ot.mass) * u2) / (me.mass + ot.mass);
        vp1 = vp1 + normal * (u1p - u1);
    } else {                                                            // COL-707: a fixed ball reflects
        vp1 = vp1 - normal * (2.0 * u1);
    }
    integrate_t(c, p1, vp1, me.g, me.m, me.tf, tr, e2, p2, vp2);
    const f64 m1 = me.m, ft = me.tf;
    const f64 inv_m = 1.0 / m1, inv_t = 1.0 / (ft - 1.0);
    const V3 g = me.g, v0 = me.v;
    a1 = {(-((((g.x * -m1) + g.x * (ft * m1)) - (v0.x * ft) * k) + vp2.x * k)) * inv_m * inv_t,
          (-((((g.y * -m1) + g.y * (ft * m1)) - (v0.y * ft) * k) + vp2.y * k)) * inv_m * inv_t,
          (-((((g.z * -m1) + g.z * (ft * m1)) - (v0.z * ft) * k) + vp2.z * k)) * inv_m * inv_t};
    return u1 - u2;
}

// Potential's overlap branch (COL-709, COL-710; vec_collide._overlap): the normal (along the centres, or +-x for
// coincident centres by id), the overlap plus one metre, the two pushes and the acceleration that pushes the ball out
// within the tick against mLastG's normal part
__device__ void overlap(const Consts& c, const Ball& me, const Ball& ot, V3 pb, V3 pn, f64 cr, V3& normal, f64& dist,
                        f64& d1, f64& d2, V3& a1) {
    const V3 p0q0 = pn - pb;
    const f64 p0q0_2 = dot(p0q0, p0q0);
    if (p0q0_2 == 0.0) {
        normal = {me.id > ot.id ? 1.0 : -1.0, 0.0, 0.0};
    } else {
        normal = normalized(p0q0);
    }
    dist = (cr - __dsqrt_rn(p0q0_2)) + 1.0;
    d1 = (ot.m / (me.m + ot.m)) * dist;
    d2 = (me.m / (me.m + ot.m)) * dist;
    const f64 tmp = 1.0 / (me.m + c.dtk);
    const f64 cc = (-d1) / ((c.dt * tmp) * me.m);
    const f64 nd = dot(me.g, normal);
    a1 = {(normal.x * cc - me.v.x) * c.inv_dt - normal.x * nd, (normal.y * cc - me.v.y) * c.inv_dt - normal.y * nd,
          (normal.z * cc - me.v.z) * c.inv_dt - normal.z * nd};
}

// ============================================================================== the partition: candidates, order

struct Filing {
    i64 level, key[3], mod[3];
#if !XBPC_COMPOSED
    i64 bubble;                    // the master's bubble (0 for every ball on a client's view)
#endif
};

__device__ __forceinline__ Filing load_filing(const Park& P, i64 j) {
    Filing f;
    f.level = P.level[j];
    for (int q = 0; q < 3; ++q) {
        f.key[q] = P.key[3 * j + q];
        f.mod[q] = P.mod[3 * j + q];
    }
#if !XBPC_COMPOSED
    f.bubble = P.master_bubble != nullptr ? P.master_bubble[j] : 0;
#endif
    return f;
}

__device__ __forceinline__ i64 lo_of(const Filing& f, int q) { return f.key[q] + (f.mod[q] < 0 ? f.mod[q] : 0); }
__device__ __forceinline__ i64 hi_of(const Filing& f, int q) { return f.key[q] + (f.mod[q] > 0 ? f.mod[q] : 0); }

// whether a box of the other ball (n) is one of the ball's (b) boxes, an ancestor of one or below one, per axis
// (vec_collide._related)
__device__ bool related(const Filing& b, const Filing& n) {
    const i64 d = 2 * (b.level - n.level);
    const i64 s = d < 0 ? -d : d;
    const bool coarse = n.level <= b.level;
    for (int q = 0; q < 3; ++q) {
        const i64 lb = lo_of(b, q), hb = hi_of(b, q), ln = lo_of(n, q), hn = hi_of(n, q);
        const i64 x_lo = (coarse ? lb : ln) >> s, x_hi = (coarse ? hb : hn) >> s;
        const i64 y_lo = coarse ? ln : lb, y_hi = coarse ? hn : hb;
        if (!(x_lo == y_lo || x_lo == y_hi || x_hi == y_lo || x_hi == y_hi)) return false;
    }
    return true;
}

// the position of the other ball's first box in the ball's walk (Partition::GetNearbyBalls) as vec_collide._order_keys
// computes it: (segment x 3 + part) x CODE_SPAN + sub
__device__ i64 order_key(const Filing& b, const Filing& n) {
    const i64 Lb = b.level, Ln = n.level;
    i64 cnt_b[3], lo_b[3], hi_b[3];
    for (int q = 0; q < 3; ++q) {
        cnt_b[q] = b.mod[q] != 0 ? 2 : 1;
        lo_b[q] = lo_of(b, q);
        hi_b[q] = hi_of(b, q);
    }
    if (Ln <= Lb) {                                                     // an ancestor, or the box itself
        const i64 s_up = 2 * (Lb - Ln);
        i64 idx[3];
        for (int q = 0; q < 3; ++q) {
            const i64 v = lo_b[q] >> s_up;
            const bool a0 = v == n.key[q] || (n.mod[q] != 0 && v == n.key[q] + n.mod[q]);
            idx[q] = a0 ? 0 : 1;
        }
        const i64 seg_up = (idx[0] * cnt_b[1] + idx[1]) * cnt_b[2] + idx[2];
        if (Ln < Lb) return (seg_up * 3) * XBPC_CODE_SPAN + (Lb - Ln);
        return (seg_up * 3 + 2) * XBPC_CODE_SPAN;
    }
    // below: each of the other's boxes whose ancestor at Lb is one of the ball's, its post-order code
    const i64 s_dn = 2 * (Ln - Lb);
    i64 best = XBPC_NOT_A_CANDIDATE;
    for (int combo = 0; combo < 8; ++combo) {
        bool own = true;
        i64 box[3], j[3];
        for (int q = 0; q < 3; ++q) {
            const i64 bit = (combo >> (2 - q)) & 1;
            const bool valid = bit == 0 || n.mod[q] != 0;
            box[q] = n.key[q] + n.mod[q] * bit;
            const i64 anc = box[q] >> s_dn;
            const bool in0 = anc == lo_b[q], in1 = anc == hi_b[q] && cnt_b[q] == 2;
            own = own && valid && (in0 || in1);
            j[q] = in0 ? 0 : 1;
        }
        if (!own) continue;
        const i64 seg = (j[0] * cnt_b[1] + j[1]) * cnt_b[2] + j[2];
        i64 code = 0;
        i64 power = XBPC_CODE_SPAN;
        for (i64 lvl = 1; lvl < XBPC_NUM_LEVELS; ++lvl) {
            power /= XBPC_CODE_BASE;                                    // CODE_BASE ** (NUM_LEVELS - 1 - lvl)
            const i64 shift = Ln - lvl > 0 ? 2 * (Ln - lvl) : 0;
            i64 digit = ((box[0] >> shift) & 3) * 16 + ((box[1] >> shift) & 3) * 4 + ((box[2] >> shift) & 3);
            if (lvl > Ln) digit = XBPC_CODE_BASE - 1;
            if (lvl <= Lb) digit = 0;
            code += digit * power;
        }
        const i64 k = (seg * 3 + 1) * XBPC_CODE_SPAN + code;
        if (k < best) best = k;
    }
    return best;
}

// (key, id, slot) before (key2, id2, slot2): the two stable sorts' order (vec_collide._sorted)
__device__ __forceinline__ bool before(i64 k1, i64 id1, i64 s1, i64 k2, i64 id2, i64 s2) {
#if XBPC_FAULT == 3
    return k1 > k2 || (k1 == k2 && (id1 > id2 || (id1 == id2 && s1 > s2)));    // planted fault: the reverse
#else
    return k1 < k2 || (k1 == k2 && (id1 < id2 || (id1 == id2 && s1 < s2)));
#endif
}

// ===================================================================================== a responder's collision step

struct Responder {        // a responder's running state: Potential's updates (the torch path's rows)
    V3 last_c;
    f64 last_col;
    i64 hits_n, imp_n;
#if XBPC_COMPOSED
    bool missile;         // in MISSILE mode: its response to its own target is not kept (COL-711)
    i64 follow;           // the id it follows
#endif
};

// Potential's end (COL-711; vec_collide._update): 0.85 of the acceleration if the impact is not earlier than the one
// held (the stronger at an equal time), then the other ball on the collision list
__device__ __forceinline__ void update(const Park& P, i64 row, Responder& st, f64 s, V3 a1, i64 other) {
#if XBPC_COMPOSED
    const bool kept = !(st.missile && other == st.follow);     // _GradientView._exempt: a missile and its own target
    if (kept && s >= st.last_col) {
#else
    if (s >= st.last_col) {
#endif
        const V3 lc = a1 * K_RESPONSE_SCALE;
        if (s != st.last_col) {
            st.last_c = lc;
            st.last_col = s;
        } else if (dot(lc, lc) > dot(st.last_c, st.last_c)) {
            st.last_c = lc;
        }
    }
    const i64 at = st.hits_n < P.hits_cap - 1 ? st.hits_n : P.hits_cap - 1;
    P.hits[row * P.hits_cap + at] = other;
    st.hits_n += 1;
}

// Potential (Ballpark.cpp:2789) of one candidate with its depth-0 time of impact s0 known, through the recursion
// (vec_collide._apply_contact's order), and the impact record of an impact at depth 0
__device__ void potential(const Consts& c, const Park& P, i64 row, Responder& st, const Ball& me, const Ball& ot,
                          i64 slot, f64 s0) {
    const f64 cr = me.radius + ot.radius;
    const V3 dv = me.v - ot.v;
    const bool early = dot(dv, dv) > K_EARLY;
    const bool light = (ot.m / me.m) > K_LIGHT_BALL_RATIO;
    f64 s[3], dist[3];
    V3 a1[3];
    f64 closing = 0.0;
    int deepest = 0;
    V3 pb = me.p, pn = ot.p;
    for (int d = 0; d < 3; ++d) {
        if (d == 0) {
            s[0] = s0;
        } else {
            const V3 p1 = end_position(c, pb, me.v, me.g, me.m, me.tf);
            const V3 q1 = end_position(c, pn, ot.v, ot.g, ot.m, ot.tf);
            s[d] = collide_two_spheres(pb, p1, pn, q1, cr);
        }
        deepest = d;
        if (s[d] == -1.0) break;
        if (s[d] > 0.0) {
            const f64 cl = approach(c, me, ot, pb, pn, s[d], a1[d]);
            if (d == 0) closing = cl;
            break;
        }
        V3 normal;
        f64 d1, d2;
        overlap(c, me, ot, pb, pn, cr, normal, dist[d], d1, d2, a1[d]);
        if (!(ot.free && d < 2)) break;
        pb = pb - normal * d1;                                          // the recursion's pushed positions
        pn = pn + normal * d2;
    }
    for (int d = deepest; d >= 0; --d) {
        if (d == deepest) {
            if (s[d] != -1.0) update(P, row, st, s[d], a1[d], ot.id);
        } else {                                                        // after the recursion from depth d
            if (light) st.last_c = normalized(st.last_c) * dist[d];     // COL-710's light-ball rule
            if (!early) update(P, row, st, s[d], a1[d], ot.id);
        }
    }
    if (s0 > 0.0) {                                                     // the impact record (vec_collide._record)
        const i64 at = st.imp_n < P.imp_cap - 1 ? st.imp_n : P.imp_cap - 1;
        P.imp_slot[row * P.imp_cap + at] = slot;
        P.imp_s[row * P.imp_cap + at] = s0;
        P.imp_cl[row * P.imp_cap + at] = closing;
        st.imp_n += 1;
    }
}

// whether (ball j0, other j) call Potential (vec_collide._pairs): another massive, uncloaked ball in a related box,
// less Gradient's space-junk skip; then its depth-0 time of impact (-1: no contact)
__device__ __forceinline__ bool is_pair(const Park& P, i64 j, bool me_junk, const Filing& fb, Filing& fn) {
    if (!(P.exists[j] && P.massive[j] && !P.cloaked[j])) return false;
    if (P.junk[j] && !me_junk) return false;
    fn = load_filing(P, j);
#if !XBPC_COMPOSED
#if XBPC_STATICS
    // the master skips another bubble's ball unless it is global (GetNearbyBalls' asymmetric rule)
    if (fn.bubble != fb.bubble && !(P.glob != nullptr && P.glob[j])) return false;
#else
    if (fn.bubble != fb.bubble) return false;      // the master skips another bubble's ball
#endif
#endif
    return related(fb, fn);
}

#if XBPC_COMPOSED
// whether (ball j0, other j = slot n of world w) call Potential in the composed park (VecComposedPark._pair_mask): a
// normal missile's one candidate is its target (the slot tslot, in its world, existing, in its bubble; unfiltered); any
// other ball's are the existing, uncloaked balls that are not missiles, massive or dropping out of warp this tick with a
// lower id (_dropped_seen), in a related box, and a defender missile's (a negative owner) the same with
// missiles kept (GetNearbyBalls' exclude_missiles false, Partition.cpp:301-305); then Gradient's skips (a missile and
// its owner, the signed id, or a miniball of its owner's; a missile or mushroom and its owner; a missile and the ball
// it follows; space junk)
__device__ __forceinline__ bool is_pair_composed(const Park& P, i64 w, i64 n, i64 j0, bool me_missile, const Filing& fb,
                                                 Filing& fn) {
    const i64 j = w * P.N + n;
    fn = load_filing(P, j);
    const bool me_defender = me_missile && P.owner[j0] < 0;     // VecComposedPark._defenders
    if (me_missile && !me_defender) {
        const i64 ts = P.tslot[j0];
        if (!(ts >= 0 && ts / P.N == w && ts % P.N == n && P.exists[j] && P.bubble[j0] == P.bubble[j])) return false;
    } else {
        const bool seen = P.massive[j] || (P.dropped[j] && P.id[j] < P.id[j0]);
        if (!(P.exists[j] && seen && !P.cloaked[j] && (me_defender || !P.missile[j]))) return false;
        if (!related(fb, fn)) return false;
    }
    const i64 mn = P.mode[j], idr = P.id[j0], idn = P.id[j];
    if (me_missile && (idn == P.owner[j0] || (mn == XBPC_MINIBALL && P.owner[j] == P.owner[j0]))) return false;
    if ((mn == XBPC_MISSILE || mn == XBPC_MUSHROOM) && idr == P.owner[j]) return false;
    if (mn == XBPC_MISSILE && idr == P.follow[j]) return false;
    return !(P.junk[j] && !P.junk[j0]);
}
#endif

#if XBPC_STATICS
// ======================================= the minis' order and the force fields (COL-713, MOV-737)

// a candidate's place among a box's balls (vec_collide._sorted's idn): its rank in BallSortComparer's order (the
// park's sort_id: local ids by their owner first), or its id where the park has no minis
__device__ __forceinline__ i64 sort_of(const Park& P, i64 j, i64 id) {
    return P.sort_id != nullptr ? P.sort_id[j] : id;
}

// GetNearbyBalls' force-field filter in the collision searches (vec_collide._field_passed): the other ball (j) in mode
// FIELD, and the responder's (j0) harmonic -2 (the pass), or the other's harmonic, corporation or alliance set (not -1)
// and the responder's
__device__ __forceinline__ bool field_passed(const Park& P, i64 j0, i64 j) {
    if (P.mode == nullptr || P.mode[j] != XBPC_FIELD_MODE) return false;
    const i64 h = P.harmonic[j0];
    if (h == XBPC_HARMONIC_PASS) return true;
    if (P.harmonic[j] != XBPC_NO_GROUP && P.harmonic[j] == h) return true;
    if (P.corporation[j] != XBPC_NO_GROUP && P.corporation[j] == P.corporation[j0]) return true;
    return P.alliance[j] != XBPC_NO_GROUP && P.alliance[j] == P.alliance[j0];
}

// ================================================ the static collidables (COL-712, the gradient)
// Capsules and oriented boxes as exact_ballpark.vec_collide keeps them (static_record's doubles, integers and boxes,
// made once at the load), tested as the scalar reference tests them (exact_ballpark.full: Capsule.cpp, OrientedBox.cpp,
// BoxShape.cpp, Triangle.cpp, Collision.cpp), each expression in the torch path's order; a responder walks its world's
// candidates after its balls, in the partition walk's order (vec_collide._static_order_keys, then the parent's id, the
// id and the slot), each through CollideWithBall and ReactToCollision (vec_collide._static_values) and the update
// (vec_collide._static_update).

__device__ __forceinline__ V3 gv(const f64* g, int o) { return {g[o], g[o + 1], g[o + 2]}; }

// Vector3d::Cross, components in Destiny's order (vec_collide._cross)
__device__ __forceinline__ V3 cross(V3 u, V3 v) {
    return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
}

__device__ __forceinline__ f64 length_of(V3 v) { return __dsqrt_rn(dot(v, v)); }

// IEEE abs (the sign bit cleared, as torch.abs and Python's abs)
__device__ __forceinline__ f64 abs_of(f64 x) {
    return __longlong_as_double(__double_as_longlong(x) & 0x7fffffffffffffffLL);
}

// std::max(a, b) as full.py's _fmax: (a < b) ? b : a
__device__ __forceinline__ f64 larger_of(f64 a, f64 b) { return a < b ? b : a; }

// Ballpark::Integrate over the tick with the cached decay factor, the velocity (vec_collide._integrate_dt's second
// value): a responder's end velocity with mLastG alone, which a capsule's test reads
__device__ __forceinline__ V3 end_velocity(const Consts& c, V3 v, V3 a, f64 m, f64 tf) {
    const f64 k = c.k;
    return {(m * a.x - (m * a.x - v.x * k) * tf) * c.ook, (m * a.y - (m * a.y - v.y * k) * tf) * c.ook,
            (m * a.z - (m * a.z - v.z * k) * tf) * c.ook};
}

// the position of the static collidable j's first box in the walk of the ball filed as b (vec_collide.
// _static_order_keys): its boxes are all at one level; each relates to one of the ball's boxes as an ancestor, the box
// itself or a box below; the smallest key over its boxes, XBPC_NOT_A_CANDIDATE where none relates
__device__ i64 static_order_key(const Park& P, i64 j, const Filing& b) {
    const i64* I = P.s_int + j * XBPC_I_SIZE;
    const i64 Lb = b.level, Ln = I[XBPC_I_LEVEL], nb = I[XBPC_I_NBOX];
    i64 cnt_b[3], lo_b[3], hi_b[3];
    for (int q = 0; q < 3; ++q) {
        cnt_b[q] = b.mod[q] != 0 ? 2 : 1;
        lo_b[q] = lo_of(b, q);
        hi_b[q] = hi_of(b, q);
    }
    i64 best = XBPC_NOT_A_CANDIDATE;
    for (i64 x = 0; x < nb; ++x) {
        const i64* K = P.s_keys + (j * P.B + x) * 3;
        i64 code = 0;                                                   // below: its post-order code
        if (Ln > Lb) {
            i64 power = XBPC_CODE_SPAN;
            for (i64 lvl = 1; lvl < XBPC_NUM_LEVELS; ++lvl) {
                power /= XBPC_CODE_BASE;
                const i64 shift = Ln - lvl > 0 ? 2 * (Ln - lvl) : 0;
                i64 digit = ((K[0] >> shift) & 3) * 16 + ((K[1] >> shift) & 3) * 4 + ((K[2] >> shift) & 3);
                if (lvl > Ln) digit = XBPC_CODE_BASE - 1;
                if (lvl <= Lb) digit = 0;
                code += digit * power;
            }
        }
        for (int combo = 0; combo < 8; ++combo) {                       // the ball's boxes, by key
            i64 bit[3], box[3];
            bool valid = true;
            for (int q = 0; q < 3; ++q) {
                bit[q] = (combo >> (2 - q)) & 1;
                valid = valid && (bit[q] == 0 || cnt_b[q] == 2);
                box[q] = bit[q] ? hi_b[q] : lo_b[q];
            }
            if (!valid) continue;
            const i64 seg = (bit[0] * cnt_b[1] + bit[1]) * cnt_b[2] + bit[2];
            bool rel = true;
            i64 k;
            if (Ln < Lb) {
                const i64 s_up = 2 * (Lb - Ln);
                for (int q = 0; q < 3; ++q) rel = rel && (box[q] >> s_up) == K[q];
                k = (seg * 3) * XBPC_CODE_SPAN + (Lb - Ln);
            } else if (Ln == Lb) {
                for (int q = 0; q < 3; ++q) rel = rel && box[q] == K[q];
                k = (seg * 3 + 2) * XBPC_CODE_SPAN;
            } else {
                const i64 s_dn = 2 * (Ln - Lb);
                for (int q = 0; q < 3; ++q) rel = rel && box[q] == (K[q] >> s_dn);
                k = (seg * 3 + 1) * XBPC_CODE_SPAN + code;
            }
            if (rel && k < best) best = k;
        }
    }
    return best;
}

// (key, parent, id, slot) before (key2, parent2, id2, slot2): Box::mStaticCollidables' order within a box
// (vec_collide._static_sorted)
__device__ __forceinline__ bool static_before(i64 k1, i64 p1, i64 d1, i64 s1, i64 k2, i64 p2, i64 d2, i64 s2) {
    return k1 < k2 || (k1 == k2 && (p1 < p2 || (p1 == p2 && (d1 < d2 || (d1 == d2 && s1 < s2)))));
}

// IntersectSegmentSphere (Collision.cpp:29): found, the point and the time
__device__ bool segment_sphere(V3 p0, V3 p1, V3 sp, f64 r, V3& point, f64& t) {
    const V3 dir = p1 - p0;
    const V3 nd = normalized(dir);
    const V3 m = p0 - sp;
    const f64 b = dot(m, nd);
    const f64 cc = dot(m, m) - r * r;
    if (cc > 0.0 && b > 0.0) return false;
    const f64 disc = b * b - cc;
    if (disc < 0.0) return false;
    t = -b - __dsqrt_rn(disc);
    if (t < 0.0) {
        t = 0.0;
    } else {
        t = t / length_of(dir);
    }
    if (t > 1.0) return false;
    point = p0 + dir * t;
    return true;
}

// Triangle::ClosestPoint (Triangle.cpp), its regions in order
__device__ V3 triangle_closest(V3 a, V3 b, V3 c, V3 p) {
    const V3 ab = b - a, ac = c - a, ap = p - a;
    const f64 d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    const V3 bp = p - b;
    const f64 d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    const f64 vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
    const V3 cp = p - c;
    const f64 d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 > 0.0 && d5 <= d6) return c;
    const f64 vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
    const f64 va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const f64 denom = 1.0 / (va + vb + vc);
    return (a + ab * (vb * denom)) + ac * (vc * denom);
}

// IntersectSphereTriangle (Collision.cpp:67) with the triangle's normal n: the crossing of the plane inside the triangle,
// else the sphere swept back from the triangle's closest point
__device__ bool sphere_triangle(V3 sp, f64 r, V3 vel, V3 a, V3 b, V3 c, V3 n, V3& normal, f64& time) {
    const V3 p0 = sp - n * r;                                           // GetPointOnBallThatFirstIntersectsPlane
    const V3 p0p1 = (p0 + vel) - p0;
    const f64 ndab = dot(n, p0p1);                                      // IntersectSegmentPlane
    if (ndab == 0.0) return false;
    const f64 t = (dot(a, n) - dot(n, p0)) / ndab;
    if (!(0.0 <= t && t <= 1.0)) return false;
    const V3 point = p0 + p0p1 * t;
    const V3 v0 = c - a, v1 = b - a, v2 = point - a;                    // Triangle::Contains
    const f64 dot00 = dot(v0, v0), dot01 = dot(v0, v1), dot02 = dot(v0, v2);
    const f64 dot11 = dot(v1, v1), dot12 = dot(v1, v2);
    const f64 inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    const f64 u = (dot11 * dot02 - dot01 * dot12) * inv;
    const f64 v = (dot00 * dot12 - dot01 * dot02) * inv;
    if (u >= 0.0 && v >= 0.0 && u + v < 1.0) {
        normal = n;
        time = t;
        return true;
    }
    const V3 q = triangle_closest(a, b, c, point);
    V3 impact;
    f64 ts;
    if (!segment_sphere(q, q - vel, sp, r, impact, ts)) return false;
    normal = normalized(sp - impact);
    time = ts;
    return true;
}

// BoxShape::CollideWithSphere (BoxShape.cpp:200) of the box record g: the closest corner; the centre inside its three
// planes (the extrication normal: out along the shortest way, the ray's exit side, or the closest point's direction);
// else its three sides' triangles (RectangleShape::CollideWithSphere), the first struck
__device__ __noinline__ bool box_sphere(const f64* g, V3 sp, f64 r, V3 vel, V3& normal, f64& time) {
    int closest = 0;
    f64 min_d;
    {
        const V3 d0 = gv(g, XBPC_G_CORNERS) - sp;
        min_d = dot(d0, d0);
    }
    for (int i = 1; i < 8; ++i) {
        const V3 di = gv(g, XBPC_G_CORNERS + 3 * i) - sp;
        const f64 d = dot(di, di);
        if (d < min_d) {
            min_d = d;
            closest = i;
        }
    }
    bool inside = true;
    f64 shortest = -1.0;
    int half = 0;
    for (int i = 0; i < 3; ++i) {
        const int side = static_cast<int>(XBPC_CONNECTED_SIDES[3 * closest + i]);
        const V3 n = gv(g, XBPC_G_RNORM + 3 * side);
        const V3 point = sp - n * r;
        const f64 hs = dot(n, gv(g, XBPC_G_RECT + 12 * side) - point);
        if (hs >= 0.0) {
            if (hs < shortest || shortest == -1.0) {
                shortest = hs;
                half = side;
            }
        } else {
            inside = false;
            break;
        }
    }
    if (inside) {
        V3 q = gv(g, XBPC_G_CENTER);                                    // BoxShape::ClosestPoint
        const V3 d = sp - q;
        for (int i = 0; i < 3; ++i) {
            const V3 u = gv(g, XBPC_G_UNIT + 3 * i);
            const f64 hl = g[XBPC_G_HALF + i];
            f64 dist = dot(d, u);
            if (dist > hl) {
                dist = hl;
            } else if (dist < -hl) {
                dist = -hl;
            }
            q = q + u * dist;
        }
        V3 nrm = sp - q;
        const f64 n_len = length_of(nrm);
        if (n_len < r) {
            if (n_len < K_STATIC_TINY) {
                const f64 speed = length_of(vel);
                if (speed < K_STATIC_TINY) {
                    nrm = gv(g, XBPC_G_RNORM + 3 * half) * larger_of(shortest, K_STATIC_FLOOR);
                } else {                                                // BoxShape::RayIntersect, the slabs
                    const V3 o = sp - gv(g, XBPC_G_CENTER);
                    f64 tmin[3], tmax[3];
                    int rev[3];
                    for (int i = 0; i < 3; ++i) {
                        const V3 arm = gv(g, XBPC_G_AXES + 3 * i);
                        const f64 als = g[XBPC_G_LENSQ + i], a_o = dot(arm, o), a_r = dot(arm, vel);
                        rev[i] = 0;
                        if (a_r < 0) {
                            rev[i] = 1;
                            tmin[i] = (0.5 * als - a_o) / a_r;
                            tmax[i] = (-0.5 * als - a_o) / a_r;
                        } else if (a_r > 0) {
                            tmin[i] = (-0.5 * als - a_o) / a_r;
                            tmax[i] = (0.5 * als - a_o) / a_r;
                        } else {
                            tmin[i] = -K_DBL_MAX;
                            tmax[i] = K_DBL_MAX;
                        }
                    }
                    int i_min = 0, i_max = 0;
                    for (int i = 1; i < 3; ++i) {
                        if (tmin[i_min] < tmin[i]) i_min = i;                   // std::max_element: the first largest
                        if (tmax[i] < tmax[i_max]) i_max = i;                   // std::min_element: the first smallest
                    }
                    const f64 d1 = tmin[i_min], d2 = tmax[i_max];
                    if (d1 < d2) {
                        const int side1 = static_cast<int>(XBPC_SIDE_MAPPING[2 * i_min + rev[i_min]]);
                        const int side2 = static_cast<int>(XBPC_SIDE_MAPPING[2 * i_min + 1 - rev[i_min]]);
                        if (abs_of(d1) < abs_of(d2)) {
                            nrm = gv(g, XBPC_G_RNORM + 3 * side1) * larger_of(abs_of(d1) * speed, K_STATIC_FLOOR);
                        } else {
                            nrm = gv(g, XBPC_G_RNORM + 3 * side2) * larger_of(abs_of(d2) * speed, K_STATIC_FLOOR);
                        }
                    } else {
                        nrm = gv(g, XBPC_G_RNORM + 3 * half) * shortest;
                    }
                }
            } else {
                nrm = normalized(nrm) * abs_of(r - n_len);
            }
            normal = nrm;
            time = 0.0;
            return true;
        }
    }
    for (int i = 0; i < 3; ++i) {                                       // RectangleShape::CollideWithSphere
        const int side = static_cast<int>(XBPC_CONNECTED_SIDES[3 * closest + i]);
        const V3 a = gv(g, XBPC_G_RECT + 12 * side), b = gv(g, XBPC_G_RECT + 12 * side + 3);
        const V3 c = gv(g, XBPC_G_RECT + 12 * side + 6), d = gv(g, XBPC_G_RECT + 12 * side + 9);
        const V3 m = cross(c - sp, a - sp);
        const bool hit = dot(vel, m) <= 0.0
                             ? sphere_triangle(sp, r, vel, a, b, c, gv(g, XBPC_G_RNORM + 3 * side), normal, time)
                             : sphere_triangle(sp, r, vel, a, c, d, gv(g, XBPC_G_TNORM + 3 * side), normal, time);
        if (hit) return true;
    }
    return false;
}

// Capsule::HandleCollisionNonIteratively (Capsule.cpp:206) of the capsule record g: toi (-1 where it does not react),
// the normal before its normalisation, and whether the reaction starts from the tick's end (p1, vp1: the end velocity
// parallel to the axis, as coded)
__device__ __noinline__ f64 capsule_contact(const f64* g, V3 p0, V3 p1, V3 vp1, f64 brad, V3& normal, bool& end) {
    const V3 ca = gv(g, XBPC_G_A), cb = gv(g, XBPC_G_B), ab = gv(g, XBPC_G_AB);
    const f64 rad = brad + g[XBPC_G_RAD];
    const V3 ao = p0 - ca;
    const f64 abab = dot(ab, ab);
    const f64 m = dot(ab, vp1) / abab;
    const f64 n = dot(ab, ao) / abab;
    const V3 q = vp1 - ab * m;
    const V3 rr = ao - ab * n;
    const f64 a = dot(q, q);
    const f64 bq = 2.0 * dot(q, rr);
    const f64 cq = dot(rr, rr) - rad * rad;
    const f64 rad_sq = rad * rad;
    f64 toi = -1.0;
    normal = {0.0, 0.0, 0.0};
    end = false;
    if (a == 0.0) {                                                     // parallel: the caps alone, from the end
        end = true;
        const f64 sa = collide_two_spheres(p0, p1, ca, ca, rad);
        if (sa != -1.0 && (toi == -1.0 || sa < toi)) {
            toi = sa;
            normal = p0 - ca;
        }
        const f64 sb = collide_two_spheres(p0, p1, cb, cb, rad);
        if (sb != -1.0 && (toi == -1.0 || sb < toi)) {
            toi = sb;
            normal = p0 - cb;
        }
        return toi;
    }
    const f64 det = bq * bq - 4.0 * a * cq;                            // Quadratic (Collision.cpp:137)
    if (det < 0.0) return -1.0;
    const f64 root = __dsqrt_rn(det);
    const f64 s1 = ((-bq + root) * 0.5) / a, s2 = ((-bq - root) * 0.5) / a;
    f64 s = -1.0;
    if (0.0 <= s1 && s1 <= 1.0) s = s1;
    if (0.0 <= s2 && s2 <= 1.0 && s2 < s1) {
        s = s2;
    } else if (s == -1.0) {
        const f64 t = dot(ao, ab) / abab;
        if (0.0 <= t && t <= 1.0) {
            const V3 dist = p0 - (ca + ab * t);
            if (dot(dist, dist) < rad_sq) {                             // inside the cylinder
                normal = dist;
                return 0.0;
            }
        }
    }
    const f64 tk[2] = {(s1 * m) + n, (s2 * m) + n};
    for (int i = 0; i < 2; ++i) {
        if (tk[i] < 0.0) {
            const f64 sa = collide_two_spheres(p0, p1, ca, ca, rad);
            if (sa != -1.0 && (toi == -1.0 || sa < toi)) {
                toi = sa;
                normal = p0 - ca;
            }
        } else if (tk[i] > 1.0) {
            const f64 sb = collide_two_spheres(p0, p1, cb, cb, rad);
            if (sb != -1.0 && (toi == -1.0 || sb < toi)) {
                toi = sb;
                normal = (i == 0 ? p0 : p1) - cb;
            }
        } else if (s != -1.0 && (toi == -1.0 || s < toi)) {
            toi = s;
            normal = ao - ab * (dot(ao, ab) / abab);
            if (dot(normal, normal) < rad_sq) toi = 0.0;
        }
    }
    return toi;
}

// ReactToCollision's end (COL-711, COL-712; vec_collide._static_update): at an equal time the stronger response, else
// this one replaces mLastC and mLastCollision; the static collidable's id on the collision list
__device__ __forceinline__ void static_update(const Park& P, i64 row, Responder& st, f64 toi, V3 lc, i64 sid) {
    if (toi == st.last_col) {
        if (dot(lc, lc) > dot(st.last_c, st.last_c)) st.last_c = lc;
    } else {
        st.last_c = lc;
        st.last_col = toi;
    }
    const i64 at = st.hits_n < P.hits_cap - 1 ? st.hits_n : P.hits_cap - 1;
    P.hits[row * P.hits_cap + at] = sid;
    st.hits_n += 1;
}

// CollideWithBall and ReactToCollision of the static collidable j for the responder (vec_collide._static_values):
// whether it responds, and then toi and 0.85 of the acceleration
__device__ bool static_response(const Consts& c, const Park& P, i64 j, const Ball& me, V3 p1, V3 vp1, f64& toi,
                                V3& lc) {
    const f64* g = P.s_geo + j * XBPC_G_SIZE;
    const bool cap = P.s_int[j * XBPC_I_SIZE + XBPC_I_KIND] == XBPC_STATIC_CAPSULE;
    V3 normal, pos = me.p, vel = me.v;
    if (cap) {
        bool end;
        toi = capsule_contact(g, me.p, p1, vp1, me.radius, normal, end);
        if (end) {
            pos = p1;
            vel = vp1;
        }
    } else {
        const V3 lo = gv(g, XBPC_G_LO), hi = gv(g, XBPC_G_HI);           // AABB::CanExclude
        const f64 r = me.radius;
        if ((me.p.x - r > hi.x && p1.x - r > hi.x) || (me.p.x + r < lo.x && p1.x + r < lo.x) ||
            (me.p.y - r > hi.y && p1.y - r > hi.y) || (me.p.y + r < lo.y && p1.y + r < lo.y) ||
            (me.p.z - r > hi.z && p1.z - r > hi.z) || (me.p.z + r < lo.z && p1.z + r < lo.z)) {
            return false;
        }
        f64 time;
        if (!box_sphere(g, me.p, me.radius, p1 - me.p, normal, time)) return false;
        toi = time;
    }
    if (toi == -1.0) return false;
    const f64 cl = length_of(normal);
    const V3 nn = normalized(normal);
    const f64 nc = dot(me.g, nn);
    V3 acc;
    if (toi > 0.0) {
        if (cap && nc > 0.0) return false;                              // a capsule: mLastG points out of it
        const f64 t1 = toi * c.dt, t2 = (1.0 - toi) * c.dt;
        const f64 e1 = decay_factor(c, me.m, t1), e2 = decay_factor(c, me.m, t2);
        V3 pa, va, pb, vb;
        integrate_t(c, pos, vel, me.g, me.m, me.tf, t1, e1, pa, va);
        va = va - nn * (2.0 * dot(va, nn));
        integrate_t(c, pa, va, me.g, me.m, me.tf, t2, e2, pb, vb);
        const f64 m1 = me.m, ft = me.tf, k = c.k;
        const f64 inv_m = 1.0 / m1, inv_t = 1.0 / (ft - 1.0);
        const V3 gg = me.g, v0 = me.v;
        acc = {(-((((gg.x * -m1) + gg.x * (ft * m1)) - (v0.x * ft) * k) + vb.x * k)) * inv_m * inv_t,
               (-((((gg.y * -m1) + gg.y * (ft * m1)) - (v0.y * ft) * k) + vb.y * k)) * inv_m * inv_t,
               (-((((gg.z * -m1) + gg.z * (ft * m1)) - (v0.z * ft) * k) + vb.z * k)) * inv_m * inv_t};
    } else {
        const f64 dist = cap ? ((g[XBPC_G_RAD] + me.radius) - cl) + K_EJECT_MARGIN : cl + K_EJECT_MARGIN;
        const f64 tmp = 1.0 / (me.m + c.dtk);
        const f64 cc = dist / ((c.dt * tmp) * me.m);
        acc = {(nn.x * cc - me.v.x) * c.inv_dt - nn.x * nc, (nn.y * cc - me.v.y) * c.inv_dt - nn.y * nc,
               (nn.z * cc - me.v.z) * c.inv_dt - nn.z * nc};
    }
    lc = acc * K_RESPONSE_SCALE;
    return true;
}

// Gradient's static collidables (Ballpark.cpp:2778-2783), after the balls: the responder's world's candidates in the
// walk's order, by rescans (the smallest (key, parent, id, slot) after the last one taken)
__device__ __noinline__ void statics_step(const Consts& c, const Park& P, i64 w, i64 row, Responder& st, const Ball& me,
                                          const Filing& fb, V3 p1, V3 vp1) {
    i64 pk = 0, pp = 0, pd = 0, ps = -1;                                // the last one taken (none yet)
    for (;;) {
        i64 bk = 0, bp = 0, bd = 0, bs = -1;
        for (i64 s = 0; s < P.S; ++s) {
            const i64 j = w * P.S + s;
            if (!P.s_exists[j]) continue;
            const i64 k = static_order_key(P, j, fb);
            if (k >= XBPC_NOT_A_CANDIDATE) continue;
            const i64 par = P.s_int[j * XBPC_I_SIZE + XBPC_I_PARENT], sid = P.s_int[j * XBPC_I_SIZE + XBPC_I_ID];
            if (ps >= 0 && !static_before(pk, pp, pd, ps, k, par, sid, s)) continue;
            if (bs >= 0 && !static_before(k, par, sid, s, bk, bp, bd, bs)) continue;
            bk = k;
            bp = par;
            bd = sid;
            bs = s;
        }
        if (bs < 0) return;
        f64 toi;
        V3 lc;
        if (static_response(c, P, w * P.S + bs, me, p1, vp1, toi, lc)) static_update(P, row, st, toi, lc, bd);
        pk = bk;
        pp = bp;
        pd = bd;
        ps = bs;
    }
}
#endif

__global__ void __launch_bounds__(XBPC_BLOCK) gradient_kernel(const __grid_constant__ Park P,
                                                              const __grid_constant__ Consts c) {
    const i64 t = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= P.W * P.R) return;
    const i64 w = t / P.R, r = t % P.R, N = P.N;
    const i64 j0 = w * N + r;                                           // the ball's slot in the park
    if (!P.responder[j0]) return;
    const i64 row = w * P.R + r;                                        // its row (the collision lists)
    const Ball me = load_ball(P, j0);
    const Filing fb = load_filing(P, j0);
    const bool me_junk = P.junk[j0] != 0;
#if XBPC_COMPOSED
    const bool me_missile = P.missile[j0] != 0;     // in MISSILE mode: COL-711's exception for its own target
    const bool key0 = me_missile && P.owner[j0] >= 0;   // a normal missile: its one candidate first (_target_first)
#endif
    const V3 p1 = end_position(c, me.p, me.v, me.g, me.m, me.tf);      // depth 0's prediction, once a ball
    // the first contact in GetNearbyBalls' order, and how many there are
    i64 count = 0, bk = 0, bid = 0, bs = -1;
    f64 bs0 = -1.0;
    for (i64 n = 0; n < N; ++n) {
        if (n == r) continue;
        const i64 j = w * N + n;
        Filing fn;
#if XBPC_COMPOSED
        if (!is_pair_composed(P, w, n, j0, me_missile, fb, fn)) continue;
#else
        if (!is_pair(P, j, me_junk, fb, fn)) continue;
#endif
#if XBPC_STATICS
        if (field_passed(P, j0, j)) continue;                           // a force field passed
#endif
        const Ball ot = load_ball(P, j);
        const V3 q1 = end_position(c, ot.p, ot.v, ot.g, ot.m, ot.tf);
        const f64 s0 = collide_two_spheres(me.p, p1, ot.p, q1, me.radius + ot.radius);
        if (s0 == -1.0) continue;
        ++count;
#if XBPC_COMPOSED
        const i64 k = key0 ? 0 : order_key(fb, fn);
#else
        const i64 k = order_key(fb, fn);
#endif
#if XBPC_STATICS
        const i64 oid = sort_of(P, j, ot.id);                           // BallSortComparer's rank
        if (bs < 0 || before(k, oid, n, bk, bid, bs)) {
            bk = k;
            bid = oid;
#else
        if (bs < 0 || before(k, ot.id, n, bk, bid, bs)) {
            bk = k;
            bid = ot.id;
#endif
            bs = n;
            bs0 = s0;
        }
    }
    if (P.limit && count > P.slots) P.overflow[w] = 1;                  // a fixed contact_slots too small: an event
#if !XBPC_STATICS
    if (count == 0) return;
#endif
    Responder st;
    st.last_c = ld3(P.last_c, j0);
    st.last_col = P.last_col[j0];
    st.hits_n = P.hits_n[row];
    st.imp_n = P.imp_n[row];
#if XBPC_COMPOSED
    st.missile = me_missile;
    st.follow = P.follow[j0];
#endif
    for (i64 done = 0; done < P.slots && bs >= 0; ) {
        potential(c, P, row, st, me, load_ball(P, w * N + bs), bs, bs0);
        if (++done >= count) break;
        const i64 pk = bk, pid = bid, ps = bs;                          // the next contact in order after it
        bs = -1;
        for (i64 n = 0; n < N; ++n) {
            if (n == r) continue;
            const i64 j = w * N + n;
            Filing fn;
#if XBPC_COMPOSED
            if (!is_pair_composed(P, w, n, j0, me_missile, fb, fn)) continue;
#else
            if (!is_pair(P, j, me_junk, fb, fn)) continue;
#endif
#if XBPC_STATICS
            if (field_passed(P, j0, j)) continue;                       // a force field passed
#endif
            const Ball ot = load_ball(P, j);
#if XBPC_COMPOSED
            const i64 k = key0 ? 0 : order_key(fb, fn);
#else
            const i64 k = order_key(fb, fn);
#endif
#if XBPC_STATICS
            const i64 oid = sort_of(P, j, ot.id);                       // BallSortComparer's rank
            if (!before(pk, pid, ps, k, oid, n)) continue;
            if (bs >= 0 && !before(k, oid, n, bk, bid, bs)) continue;
#else
            if (!before(pk, pid, ps, k, ot.id, n)) continue;
            if (bs >= 0 && !before(k, ot.id, n, bk, bid, bs)) continue;
#endif
            const V3 q1 = end_position(c, ot.p, ot.v, ot.g, ot.m, ot.tf);
            const f64 s0 = collide_two_spheres(me.p, p1, ot.p, q1, me.radius + ot.radius);
            if (s0 == -1.0) continue;
            bk = k;
#if XBPC_STATICS
            bid = oid;
#else
            bid = ot.id;
#endif
            bs = n;
            bs0 = s0;
        }
    }
#if XBPC_STATICS
    statics_step(c, P, w, row, st, me, fb, p1, end_velocity(c, me.v, me.g, me.m, me.tf));
#endif
    st3(P.last_c, j0, st.last_c);
    P.last_col[j0] = st.last_col;
    P.hits_n[row] = st.hits_n;
    P.imp_n[row] = st.imp_n;
}

// ================================================== the tick's other parts: behaviours, the commit, the filing

struct Tick {             // the park's tensors for the per-ball parts of the tick (contiguous, on one device)
    i64 W, N, R, hits_cap, imp_cap, with_c;
    const u8 *live, *free, *which;
    const i64* mode;
    const f64 *mass, *agility, *max_vel, *speed_fraction, *tf, *radius, *go;
    f64 *pos, *vel, *old_pos, *old_vel, *last_g, *last_c;
    const i64 *level_in, *key_in, *mod_in;
    i64 *level, *key, *mod, *hits, *hits_n, *imp_slot, *imp_n;
};

// GotoThrust (Ballpark.cpp:1397; MOV-705 to MOV-707), as vec_collide._goto_thrust
__device__ V3 goto_thrust(const Consts& c, const Tick& T, i64 j) {
    const V3 p = ld3(T.pos, j), g = ld3(T.go, j);
    const V3 a = g - p;
    const f64 length2 = a.x * a.x + a.y * a.y + a.z * a.z;
    const f64 sf = T.speed_fraction[j], mv = T.max_vel[j];
    const f64 dist = sf * mv * c.dt;
    const f64 max_thrust = c.k * sf * mv / (T.mass[j] * T.agility[j]);
    const f64 norm = __dsqrt_rn(length2);
    const f64 inv = norm == 0.0 ? 1.0 : 1.0 / norm;
    const V3 unit = norm == 0.0 ? a : a * inv;
    const f64 dd = dist * dist;
    const f64 coff = length2 / (dd == 0.0 ? 1.0 : dd);
    const f64 scale = length2 < dd ? max_thrust * coff * coff : max_thrust;
    return unit * scale;
}

// Phase 1 (EvolveBehaviorForBall, Ballpark.cpp:789) for one live free ball, as vec_collide._behaviour: mLastG from
// GOTO's thrust (zero otherwise), STOP's vertical damping, mLastC cleared, and (a row) its collision lists cleared
__global__ void __launch_bounds__(XBPC_BLOCK) behaviour_kernel(const __grid_constant__ Tick T,
                                                               const __grid_constant__ Consts c) {
    const i64 j = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= T.W * T.N || !T.live[j]) return;
    const i64 w = j / T.N, n = j % T.N, mode = T.mode[j];
    const V3 g = mode == XBPC_GOTO ? goto_thrust(c, T, j) : V3{0.0, 0.0, 0.0};
    st3(T.last_g, j, g);
    if (mode == XBPC_STOP) {
        const f64 vy = T.vel[3 * j + 1];
        T.vel[3 * j + 1] = (vy - K_STOP_FRACTION * vy) * K_STOP_DAMP;
    }
    st3(T.last_c, j, V3{0.0, 0.0, 0.0});
    if (n < T.R) {
        const i64 row = w * T.R + n;
        for (i64 i = 0; i < T.hits_cap; ++i) T.hits[row * T.hits_cap + i] = -1;
        for (i64 i = 0; i < T.imp_cap; ++i) T.imp_slot[row * T.imp_cap + i] = -1;
        T.hits_n[row] = 0;
        T.imp_n[row] = 0;
    }
}

// int64_t(x) as the compiled code does it on aarch64 (fcvtzs): toward zero, saturating, NaN to 0
// (vec_collide._to_int64)
__device__ __forceinline__ i64 to_int64(f64 x) {
    if (x >= -9223372036854775808.0 && x < 9223372036854775808.0) return static_cast<i64>(x);
    if (x >= 9223372036854775808.0) return 9223372036854775807LL;
    if (x < -9223372036854775808.0) return -9223372036854775807LL - 1;
    return 0;
}

// Ball::InsertInBoxes's outcome for one ball (vec_collide._inflated_radius and _filing): the level of its inflated
// radius, the key of the box holding its centre, the side on each axis where the inflated sphere reaches the next box
__device__ void filing_of(const Consts& c, const Tick& T, i64 j, i64& level, i64 key[3], i64 mod[3]) {
    const V3 p = ld3(T.pos, j), v = ld3(T.vel, j);
    f64 infl = T.radius[j];
    if (T.free[j]) {
        const f64 m = T.mass[j] * T.agility[j];
        const f64 a = c.k * T.speed_fraction[j] * T.max_vel[j] / m;
        const f64 sp = __dsqrt_rn(dot(v, v));
        const f64 ma = m * a;
        const f64 mps = (ma - (ma - sp * c.k) * T.tf[j]) / c.k;
        infl = T.radius[j] + c.dt * mps;
    }
    i64 count = 0;
    for (int l = 0; l < XBPC_NUM_LEVELS; ++l) count += (3.0 * infl > K_LEVEL_WIDTH[l]) ? 0 : 1;
    level = count - 1 < 0 ? 0 : count - 1;
    const f64 half = K_LEVEL_HALF[level], fac = K_LEVEL_FAC[level], width = K_LEVEL_WIDTH[level];
    const f64 centre = K_LEVEL_CENTER[level];
    const i64 grid = K_LEVEL_GRID[level];
    const f64 pc[3] = {p.x, p.y, p.z};
    for (int q = 0; q < 3; ++q) {
        i64 k = to_int64((pc[q] + half) * fac);
        k = k < 0 ? 0 : (k >= grid ? grid - 1 : k);
        const f64 lo = static_cast<f64>(k) * width - centre;
        const f64 hi = static_cast<f64>(k + 1) * width - centre;
        const bool down = (lo + infl) - pc[q] > 0.0, up = (hi - infl) - pc[q] < 0.0;
        key[q] = k;
        mod[q] = down ? -1 : (up ? 1 : 0);
    }
}

// The end of the tick for one live free ball (vec_collide.evolve's (3) and (4) on the gradient path): Integrate over
// the tick with mLastG + mLastC (mLastG alone with collisions off), the commit, and the re-filing when it moved at
// least sqrt(0.1) m
__global__ void __launch_bounds__(XBPC_BLOCK) commit_kernel(const __grid_constant__ Tick T,
                                                            const __grid_constant__ Consts c) {
    const i64 j = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= T.W * T.N || !T.live[j]) return;
    const V3 p = ld3(T.pos, j), v = ld3(T.vel, j), g = ld3(T.last_g, j);
    const V3 a = T.with_c ? g + ld3(T.last_c, j) : g;
    const f64 m = T.mass[j] * T.agility[j], tf = T.tf[j], k = c.k;
    const V3 pn = end_position(c, p, v, a, m, tf);
    const V3 vn = {(m * a.x - (m * a.x - v.x * k) * tf) * c.ook, (m * a.y - (m * a.y - v.y * k) * tf) * c.ook,
                   (m * a.z - (m * a.z - v.z * k) * tf) * c.ook};
    st3(T.old_pos, j, p);
    st3(T.old_vel, j, v);
    st3(T.pos, j, pn);
    st3(T.vel, j, vn);
    const V3 d = pn - p;
    if (dot(d, d) < K_MOVED) return;
    i64 level, key[3], mod[3];
    filing_of(c, T, j, level, key, mod);
    T.level[j] = level;
    for (int q = 0; q < 3; ++q) {
        T.key[3 * j + q] = key[q];
        T.mod[3 * j + q] = mod[q];
    }
}

// vec_collide._filing for the balls in `which` (the others keep theirs), into new tensors
__global__ void __launch_bounds__(XBPC_BLOCK) filing_kernel(const __grid_constant__ Tick T,
                                                            const __grid_constant__ Consts c) {
    const i64 j = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= T.W * T.N) return;
    i64 level = T.level_in[j], key[3], mod[3];
    for (int q = 0; q < 3; ++q) {
        key[q] = T.key_in[3 * j + q];
        mod[q] = T.mod_in[3 * j + q];
    }
    if (T.which[j]) filing_of(c, T, j, level, key, mod);
    T.level[j] = level;
    for (int q = 0; q < 3; ++q) {
        T.key[3 * j + q] = key[q];
        T.mod[3 * j + q] = mod[q];
    }
}

// ======================================================================================= the tests' entry kernels

__global__ void __launch_bounds__(XBPC_BLOCK) exp_kernel(const f64* __restrict__ x, f64* __restrict__ out, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp_of(x[i]);
}

__global__ void __launch_bounds__(XBPC_BLOCK) fma_kernel(const f64* __restrict__ a, const f64* __restrict__ b,
                                                         const f64* __restrict__ cc, f64* __restrict__ out, i64 n) {
    const i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fma_rn(a[i], b[i], cc[i]);
}

int grid_for(i64 n, int block) { return static_cast<int>((n + block - 1) / block); }

bool block_ok(int block) { return block > 0 && block <= XBPC_BLOCK && block % 32 == 0; }

}  // namespace

extern "C" {

int xbpc_abi(void) { return XBPC_ABI; }

int xbpc_block_limit(void) { return XBPC_BLOCK; }

int xbpc_fault(void) { return XBPC_FAULT; }

const char* xbpc_error_string(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }

// One tick's gradient collision step. dims: W, N, R, slots, limit (contact_slots given), hits_cap, imp_cap. ptrs, in
// order: pos, vel, last_g, mass, agility, time_factor, radius, exists, free, massive, junk, cloaked, responder, id,
// level, key, mod, last_c, last_collision, imp_s, imp_closing, hits, hits_n, imp_slot, imp_n, overflow, and in the
// default build the master's bubble per slot or null. consts: k, dt, k2, ook2, ook, neg_k, neg_kt,
// inv_dt, dtk. Launched on `stream`; no host synchronisation, no allocation.
int xbpc_gradient(const i64* dims, void* const* ptrs, const f64* consts, void* stream, int block) {
    if (!block_ok(block)) return -1;
    Park P;
    P.W = dims[0];
    P.N = dims[1];
    P.R = dims[2];
    P.slots = dims[3];
    P.limit = dims[4];
    P.hits_cap = dims[5];
    P.imp_cap = dims[6];
    if (P.W < 0 || P.N < 1 || P.R < 1 || P.R > P.N || P.hits_cap < 1 || P.imp_cap < 1) return -1;
    int k = 0;
    P.pos = static_cast<const f64*>(ptrs[k++]);
    P.vel = static_cast<const f64*>(ptrs[k++]);
    P.last_g = static_cast<const f64*>(ptrs[k++]);
    P.mass = static_cast<const f64*>(ptrs[k++]);
    P.agility = static_cast<const f64*>(ptrs[k++]);
    P.tf = static_cast<const f64*>(ptrs[k++]);
    P.radius = static_cast<const f64*>(ptrs[k++]);
    P.exists = static_cast<const u8*>(ptrs[k++]);
    P.free = static_cast<const u8*>(ptrs[k++]);
    P.massive = static_cast<const u8*>(ptrs[k++]);
    P.junk = static_cast<const u8*>(ptrs[k++]);
    P.cloaked = static_cast<const u8*>(ptrs[k++]);
    P.responder = static_cast<const u8*>(ptrs[k++]);
    P.id = static_cast<const i64*>(ptrs[k++]);
    P.level = static_cast<const i64*>(ptrs[k++]);
    P.key = static_cast<const i64*>(ptrs[k++]);
    P.mod = static_cast<const i64*>(ptrs[k++]);
    P.last_c = static_cast<f64*>(ptrs[k++]);
    P.last_col = static_cast<f64*>(ptrs[k++]);
    P.imp_s = static_cast<f64*>(ptrs[k++]);
    P.imp_cl = static_cast<f64*>(ptrs[k++]);
    P.hits = static_cast<i64*>(ptrs[k++]);
    P.hits_n = static_cast<i64*>(ptrs[k++]);
    P.imp_slot = static_cast<i64*>(ptrs[k++]);
    P.imp_n = static_cast<i64*>(ptrs[k++]);
    P.overflow = static_cast<u8*>(ptrs[k++]);
#if !XBPC_COMPOSED
    P.master_bubble = static_cast<const i64*>(ptrs[k++]);   // null for a client's view
#endif
#if XBPC_STATICS
    // dims[7], dims[8] the static slots a world and the boxes a static collidable; after the bubble
    // pointer: s_exists, s_geo, s_int, s_keys (no static slots, 0, with null pointers, for a park
    // with minis or fields alone); then sort_id, mode, harmonic, corporation, alliance (null where
    // the park has no minis, or no fields); then glob (null where the park has no global balls)
    P.S = dims[7];
    P.B = dims[8];
    if (P.S < 0 || P.B < 1) return -1;
    P.s_exists = static_cast<const u8*>(ptrs[k++]);
    P.s_geo = static_cast<const f64*>(ptrs[k++]);
    P.s_int = static_cast<const i64*>(ptrs[k++]);
    P.s_keys = static_cast<const i64*>(ptrs[k++]);
    if (P.S > 0 && (P.s_exists == nullptr || P.s_geo == nullptr || P.s_int == nullptr || P.s_keys == nullptr)) {
        return -1;
    }
    P.sort_id = static_cast<const i64*>(ptrs[k++]);
    P.mode = static_cast<const i64*>(ptrs[k++]);
    P.harmonic = static_cast<const i64*>(ptrs[k++]);
    P.corporation = static_cast<const i64*>(ptrs[k++]);
    P.alliance = static_cast<const i64*>(ptrs[k++]);
    if (P.mode != nullptr && (P.harmonic == nullptr || P.corporation == nullptr || P.alliance == nullptr)) return -1;
    P.glob = static_cast<const u8*>(ptrs[k++]);             // the global flags, or null
#endif
#if XBPC_COMPOSED
    // the composed view's candidates, after overflow: missile, dropped, mode, owner, follow, bubble, tslot; every slot
    // a responder's row
    if (P.R != P.N) return -1;
    P.missile = static_cast<const u8*>(ptrs[k++]);
    P.dropped = static_cast<const u8*>(ptrs[k++]);
    P.mode = static_cast<const i64*>(ptrs[k++]);
    P.owner = static_cast<const i64*>(ptrs[k++]);
    P.follow = static_cast<const i64*>(ptrs[k++]);
    P.bubble = static_cast<const i64*>(ptrs[k++]);
    P.tslot = static_cast<const i64*>(ptrs[k++]);
#endif
    Consts c;
    c.k = consts[0];
    c.dt = consts[1];
    c.k2 = consts[2];
    c.ook2 = consts[3];
    c.ook = consts[4];
    c.neg_k = consts[5];
    c.neg_kt = consts[6];
    c.inv_dt = consts[7];
    c.dtk = consts[8];
    const i64 n = P.W * P.R;
    if (n == 0) return 0;
    gradient_kernel<<<grid_for(n, block), block, 0, static_cast<cudaStream_t>(stream)>>>(P, c);
    return static_cast<int>(cudaGetLastError());
}

#if XBPC_COMPOSED
int xbpc_composed(void) { return XBPC_COMPOSED; }
#endif
#if XBPC_STATICS
int xbpc_statics(void) { return XBPC_STATICS; }
#endif

// The tick's per-ball parts. part: 0 the behaviours, 1 the commit and re-filing, 2 the filing of `which` into new
// tensors. dims: W, N, R, hits_cap, imp_cap, with_c (the commit integrates mLastG + mLastC). ptrs, in order: live,
// free, which, mode, mass, agility, max_vel, speed_fraction, time_factor, radius, goto, pos, vel, old_pos, old_vel,
// last_g, last_c, level_in, key_in, mod_in, level, key, mod, hits, hits_n, imp_slot, imp_n (a part reads and writes
// only its own; the others may be null). consts as xbpc_gradient's.
int xbpc_tick(int part, const i64* dims, void* const* ptrs, const f64* consts, void* stream, int block) {
    if (!block_ok(block) || part < 0 || part > 2) return -1;
    Tick T;
    T.W = dims[0];
    T.N = dims[1];
    T.R = dims[2];
    T.hits_cap = dims[3];
    T.imp_cap = dims[4];
    T.with_c = dims[5];
    if (T.W < 0 || T.N < 1 || T.R < 1 || T.R > T.N || T.hits_cap < 1 || T.imp_cap < 1) return -1;
    int k = 0;
    T.live = static_cast<const u8*>(ptrs[k++]);
    T.free = static_cast<const u8*>(ptrs[k++]);
    T.which = static_cast<const u8*>(ptrs[k++]);
    T.mode = static_cast<const i64*>(ptrs[k++]);
    T.mass = static_cast<const f64*>(ptrs[k++]);
    T.agility = static_cast<const f64*>(ptrs[k++]);
    T.max_vel = static_cast<const f64*>(ptrs[k++]);
    T.speed_fraction = static_cast<const f64*>(ptrs[k++]);
    T.tf = static_cast<const f64*>(ptrs[k++]);
    T.radius = static_cast<const f64*>(ptrs[k++]);
    T.go = static_cast<const f64*>(ptrs[k++]);
    T.pos = static_cast<f64*>(ptrs[k++]);
    T.vel = static_cast<f64*>(ptrs[k++]);
    T.old_pos = static_cast<f64*>(ptrs[k++]);
    T.old_vel = static_cast<f64*>(ptrs[k++]);
    T.last_g = static_cast<f64*>(ptrs[k++]);
    T.last_c = static_cast<f64*>(ptrs[k++]);
    T.level_in = static_cast<const i64*>(ptrs[k++]);
    T.key_in = static_cast<const i64*>(ptrs[k++]);
    T.mod_in = static_cast<const i64*>(ptrs[k++]);
    T.level = static_cast<i64*>(ptrs[k++]);
    T.key = static_cast<i64*>(ptrs[k++]);
    T.mod = static_cast<i64*>(ptrs[k++]);
    T.hits = static_cast<i64*>(ptrs[k++]);
    T.hits_n = static_cast<i64*>(ptrs[k++]);
    T.imp_slot = static_cast<i64*>(ptrs[k++]);
    T.imp_n = static_cast<i64*>(ptrs[k++]);
    Consts c;
    c.k = consts[0];
    c.dt = consts[1];
    c.k2 = consts[2];
    c.ook2 = consts[3];
    c.ook = consts[4];
    c.neg_k = consts[5];
    c.neg_kt = consts[6];
    c.inv_dt = consts[7];
    c.dtk = consts[8];
    const i64 n = T.W * T.N;
    if (n == 0) return 0;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (part == 0) behaviour_kernel<<<grid_for(n, block), block, 0, s>>>(T, c);
    if (part == 1) commit_kernel<<<grid_for(n, block), block, 0, s>>>(T, c);
    if (part == 2) filing_kernel<<<grid_for(n, block), block, 0, s>>>(T, c);
    return static_cast<int>(cudaGetLastError());
}

int xbpc_exp(const f64* x, f64* out, i64 n, void* stream, int block) {
    if (!block_ok(block)) return -1;
    if (n == 0) return 0;
    exp_kernel<<<grid_for(n, block), block, 0, static_cast<cudaStream_t>(stream)>>>(x, out, n);
    return static_cast<int>(cudaGetLastError());
}

int xbpc_fma(const f64* a, const f64* b, const f64* c, f64* out, i64 n, void* stream, int block) {
    if (!block_ok(block)) return -1;
    if (n == 0) return 0;
    fma_kernel<<<grid_for(n, block), block, 0, static_cast<cudaStream_t>(stream)>>>(a, b, c, out, n);
    return static_cast<int>(cudaGetLastError());
}

// The gradient kernel as loaded on the current device (loading its module, so that no later launch loads it, as a
// captured one must not): registers, local bytes, the most threads a block, the binary and PTX versions, and the
// device's compute capability.
int xbpc_info(int* out) {
    cudaFuncAttributes a;
    cudaError_t e = cudaFuncGetAttributes(&a, gradient_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    cudaFuncAttributes b;
    e = cudaFuncGetAttributes(&b, exp_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    e = cudaFuncGetAttributes(&b, fma_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    e = cudaFuncGetAttributes(&b, behaviour_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    e = cudaFuncGetAttributes(&b, commit_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    e = cudaFuncGetAttributes(&b, filing_kernel);
    if (e != cudaSuccess) return static_cast<int>(e);
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    out[0] = a.numRegs;
    out[1] = static_cast<int>(a.localSizeBytes);
    out[2] = a.maxThreadsPerBlock;
    out[3] = a.binaryVersion;
    out[4] = a.ptxVersion;
    out[5] = major * 10 + minor;
    return 0;
}

}  // extern "C"
