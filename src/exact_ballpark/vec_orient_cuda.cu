// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's dynamical orientation (MOV-728) for B ships on a CUDA device, one thread a ship: the step that
// exact_ballpark.vec_orient.VecOrientPark._orient and the end of its evolve compute in torch operations, here as one
// kernel running the scalar algorithm of exact_ballpark.full.FullBallpark (GotoTorqueBreaks, GotoRollBreaks with their
// rounds, ApplyTorque, the thrust along the heading, the rotation's commit), each ship leaving its searches when its
// own rounds end. Built and loaded by exact_ballpark.vec_orient_cuda; held to the torch module and to the scalar
// reference bit for bit by tests/test_movement_vec_orient_cuda.py.
//
// Exactness. Every operation's rounding is pinned: float operations are __fadd_rn, __fsub_rn, __fmul_rn, __fdiv_rn
// and __fsqrt_rn on floats exactly where the torch module rounds a double operation of two float values to float
// (for +, -, *, / and sqrt the two are equal, 53 >= 2 x 24 + 2); double operations are __dadd_rn ... __dsqrt_rn where
// it computes in double (the sums of squares of the double thrust, the comparisons against double literals, n x dt,
// the C library's double internals); conversions are explicit. A fused multiply-add appears exactly where the C
// library's compiled code has one (glibc 2.34 aarch64, transcribed in exact_ballpark.vec_libm): __fma_rn and __fmaf_rn
// compute the IEEE fused operation that vec_exp's and vec_libm's emulations compute, so the two are equal wherever the
// emulation is exact. The build adds -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math, and
// checks the PTX (exact_ballpark.vec_orient_cuda.ptx_check). No function of CUDA's math library is called: the C
// library's expf, logf, sinf, cosf, acosf, atanf, atan2f and pow are transcribed below from vec_libm, their constants
// read from the header that exact_ballpark.vec_orient_cuda.tables_header() generates from vec_libm's and vec_exp's own
// tables (never typed). trunc and ceil are single PTX conversions (cvt.rzi, cvt.rpi), exact.
//
// A ship's float fields (agility, speed fraction, maximum speed, maximum angular speed, rotational agility, rotation,
// angular velocity) are floats held in float64 tensors, as the park stores them; the mass and the thrust are doubles.

#include <cstdint>
#include <cuda_runtime.h>

#include "vec_orient_cuda_tables.h"

#define XBPO_ABI 1

namespace xbpo {

// ======================================================================================== arithmetic, rounding pinned

static __device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
static __device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
static __device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ float fsqrt(float a) { return __fsqrt_rn(a); }
static __device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
static __device__ __forceinline__ double dadd(double a, double b) { return __dadd_rn(a, b); }
static __device__ __forceinline__ double dsub(double a, double b) { return __dsub_rn(a, b); }
static __device__ __forceinline__ double dmul(double a, double b) { return __dmul_rn(a, b); }
static __device__ __forceinline__ double ddiv(double a, double b) { return __ddiv_rn(a, b); }
static __device__ __forceinline__ double dsqrt(double a) { return __dsqrt_rn(a); }
static __device__ __forceinline__ double dfma(double a, double b, double c) { return __fma_rn(a, b, c); }

static __device__ __forceinline__ float tof(double x) { return __double2float_rn(x); }       // f32(x)
static __device__ __forceinline__ double tod(float x) { return (double)x; }                  // exact
static __device__ __forceinline__ uint32_t fbits(float x) { return __float_as_uint(x); }
static __device__ __forceinline__ float ffrom(uint32_t u) { return __uint_as_float(u); }
static __device__ __forceinline__ uint64_t dbits(double x) { return (uint64_t)__double_as_longlong(x); }
static __device__ __forceinline__ double dfrom(uint64_t u) { return __longlong_as_double((long long)u); }
static __device__ __forceinline__ float fabs_(float x) { return ffrom(fbits(x) & 0x7fffffffu); }
static __device__ __forceinline__ double dabs_(double x) { return dfrom(dbits(x) & 0x7fffffffffffffffULL); }
static __device__ __forceinline__ bool ffinite(float x) { return (fbits(x) & 0x7f800000u) != 0x7f800000u; }
static __device__ __forceinline__ float fmax_(float a, float b) { return (a < b) ? b : a; }  // std::max
static __device__ __forceinline__ float fmin_(float a, float b) { return (b < a) ? b : a; }  // std::min

#define KD(name) dfrom(name)        // a double constant from its bit pattern
#define KF(name) ffrom(name)        // a float constant from its bit pattern

static __device__ __forceinline__ double dtrunc(double x) {
    double r;
    asm("cvt.rzi.f64.f64 %0, %1;" : "=d"(r) : "d"(x));
    return r;
}

static __device__ __forceinline__ float fceil(float x) {
    float r;
    asm("cvt.rpi.f32.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

static __device__ __forceinline__ double frinta(double z) {
    // aarch64's frinta: to the nearest integer, ties away from zero (z - trunc(z) is exact)
    const double t = dtrunc(z);
    const double d = dsub(z, t);
    return d >= 0.5 ? dadd(t, 1.0) : (d <= -0.5 ? dsub(t, 1.0) : t);
}

static __device__ __forceinline__ int to_int32(double x) {
    // int(x) of a float on aarch64 (fcvtzs): truncation, saturating, NaN to 0
    if (x != x) return 0;
    if (x < -2147483648.0) return (int)(-2147483647 - 1);
    if (x > 2147483647.0) return 2147483647;
    return __double2int_rz(x);
}

static __device__ __forceinline__ float fnan() { return ffrom(0x7fc00000u); }
static __device__ __forceinline__ float finf() { return ffrom(0x7f800000u); }
static __device__ __forceinline__ double dinf() { return dfrom(0x7ff0000000000000ULL); }
static __device__ __forceinline__ double dnan() { return dfrom(0x7ff8000000000000ULL); }

#include "glibc_float.cuh"   // LGPL-2.1-or-later: the C library's float functions and pow
// ===================================================================== carbonengine's Vector3, Quaternion, Matrix

struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };
struct M3 { float m00, m01, m02, m10, m11, m12, m20, m21, m22; };   // the 4th row and column: 0 0 0 1

static __device__ __forceinline__ V3 v3_mul(V3 a, float f) { return V3{fmul(a.x, f), fmul(a.y, f), fmul(a.z, f)}; }
static __device__ __forceinline__ V3 v3_add(V3 a, V3 b) { return V3{fadd(a.x, b.x), fadd(a.y, b.y), fadd(a.z, b.z)}; }
static __device__ __forceinline__ V3 v3_sub(V3 a, V3 b) { return V3{fsub(a.x, b.x), fsub(a.y, b.y), fsub(a.z, b.z)}; }
static __device__ __forceinline__ V3 v3_div(V3 a, float f) { return v3_mul(a, fdiv(1.0f, f)); }   // times 1.0f / f
static __device__ __forceinline__ float v3_dot(V3 a, V3 b) {
    return fadd(fadd(fmul(a.x, b.x), fmul(a.y, b.y)), fmul(a.z, b.z));
}
static __device__ __forceinline__ float v3_len(V3 a) { return fsqrt(v3_dot(a, a)); }
static __device__ __forceinline__ V3 v3_cross(V3 a, V3 b) {
    return V3{fsub(fmul(a.y, b.z), fmul(a.z, b.y)), fsub(fmul(a.z, b.x), fmul(a.x, b.z)),
              fsub(fmul(a.x, b.y), fmul(a.y, b.x))};
}
static __device__ __forceinline__ V3 v3_normalize(V3 v) {
    // by the largest magnitude first, then by the length (a zero length left as it is)
    const float mx = fmax_(fmax_(fabs_(v.x), fabs_(v.y)), fabs_(v.z));
    const V3 minified = v3_div(v, mx != 0.0f ? mx : 1.0f);
    const float length = v3_len(minified);
    return v3_mul(minified, length != 0.0f ? fdiv(1.0f, length) : length);
}

static __device__ __forceinline__ Q4 q_mul(Q4 q, Q4 o) {
    return Q4{fsub(fadd(fadd(fmul(o.w, q.x), fmul(o.x, q.w)), fmul(o.y, q.z)), fmul(o.z, q.y)),
              fadd(fadd(fsub(fmul(o.w, q.y), fmul(o.x, q.z)), fmul(o.y, q.w)), fmul(o.z, q.x)),
              fadd(fsub(fadd(fmul(o.w, q.z), fmul(o.x, q.y)), fmul(o.y, q.x)), fmul(o.z, q.w)),
              fsub(fsub(fsub(fmul(o.w, q.w), fmul(o.x, q.x)), fmul(o.y, q.y)), fmul(o.z, q.z))};
}
static __device__ __forceinline__ float q_lensq(Q4 q) {
    return fadd(fadd(fadd(fmul(q.x, q.x), fmul(q.y, q.y)), fmul(q.z, q.z)), fmul(q.w, q.w));
}
static __device__ __forceinline__ float q_len(Q4 q) { return fsqrt(q_lensq(q)); }
static __device__ __forceinline__ Q4 q_div(Q4 q, float f) {
    const float d = fdiv(1.0f, f);
    return Q4{fmul(q.x, d), fmul(q.y, d), fmul(q.z, d), fmul(q.w, d)};
}
static __device__ __forceinline__ Q4 q_conj(Q4 q) { return Q4{-q.x, -q.y, -q.z, q.w}; }
static __device__ __noinline__ Q4 q_rotation(V3 axis, float angle) {
    // RotationQuaternion(axis, angle): sinf and cosf of the half angle
    const V3 t = v3_normalize(axis);
    const float h = fdiv(angle, 2.0f);
    const float s = g_sinf(h);
    return Q4{fmul(s, t.x), fmul(s, t.y), fmul(s, t.z), g_cosf(h)};
}
static __device__ __forceinline__ M3 m_rotation_q(Q4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return M3{fsub(1.0f, fmul(2.0f, fadd(fmul(y, y), fmul(z, z)))), fmul(2.0f, fadd(fmul(x, y), fmul(z, w))),
              fmul(2.0f, fsub(fmul(x, z), fmul(y, w))),
              fmul(2.0f, fsub(fmul(x, y), fmul(z, w))), fsub(1.0f, fmul(2.0f, fadd(fmul(x, x), fmul(z, z)))),
              fmul(2.0f, fadd(fmul(y, z), fmul(x, w))),
              fmul(2.0f, fadd(fmul(x, z), fmul(y, w))), fmul(2.0f, fsub(fmul(y, z), fmul(x, w))),
              fsub(1.0f, fmul(2.0f, fadd(fmul(x, x), fmul(y, y))))};
}
static __device__ __noinline__ V3 transform_coord(V3 v, M3 m) {
    // the row vector times the matrix, divided by the homogeneous norm; every sum kept (the signs of zero)
    const float norm = fadd(fadd(fadd(fmul(0.0f, v.x), fmul(0.0f, v.y)), fmul(0.0f, v.z)), 1.0f);
    if (!(norm != 0.0f)) return V3{0.0f, 0.0f, 0.0f};
    return V3{fdiv(fadd(fadd(fadd(fmul(v.x, m.m00), fmul(v.y, m.m10)), fmul(v.z, m.m20)), 0.0f), norm),
              fdiv(fadd(fadd(fadd(fmul(v.x, m.m01), fmul(v.y, m.m11)), fmul(v.z, m.m21)), 0.0f), norm),
              fdiv(fadd(fadd(fadd(fmul(v.x, m.m02), fmul(v.y, m.m12)), fmul(v.z, m.m22)), 0.0f), norm)};
}

// ======================================================================== the torque searches (Ballpark.cpp)

struct Ship {
    float max_ang_vel, t_scale, tf;     // tf = expf(-dt t_scale)
    double dt;
    float fdt;
};

static __device__ __forceinline__ float n2dt_of(int n2, double dt) { return tof(dmul((double)n2, dt)); }

// CalculateN2Alpha2, the vector version (Ballpark.cpp:1580): the braking ticks n2, kappa2 = tf^n2 (pow), the torque
static __device__ __noinline__ void n2_alpha2_v(const Ship& sh, V3 omega0, float len_omega0, int* n2, float* kappa2,
                                                V3* alpha2) {
    if (tod(len_omega0) > KD(K_D_1EM10)) {
        const float factor = fdiv(sh.max_ang_vel, len_omega0);
        *n2 = to_int32(tod(fceil(fdiv(g_logf(fdiv(factor, fadd(1.0f, factor))), fmul(-sh.fdt, sh.t_scale)))));
        *kappa2 = tof(g_pow(tod(sh.tf), (double)*n2));
        const float g = fdiv(fmul(sh.t_scale, *kappa2), fsub(1.0f, *kappa2));
        *alpha2 = v3_mul(V3{-omega0.x, -omega0.y, -omega0.z}, g);
    } else {
        *n2 = 0;
        *kappa2 = 1.0f;
        *alpha2 = V3{0.0f, 0.0f, 0.0f};
    }
}

// CalculateN2Alpha2, the float version (Ballpark.cpp:1598)
static __device__ __noinline__ void n2_alpha2_f(const Ship& sh, float omega0, int* n2, float* kappa2, float* alpha2) {
    const float len_omega0 = fabs_(omega0);
    if (tod(len_omega0) > KD(K_D_1EM10)) {
        const float factor = fdiv(sh.max_ang_vel, len_omega0);
        *n2 = to_int32(tod(fceil(fdiv(g_logf(fdiv(factor, fadd(1.0f, factor))), fmul(-sh.fdt, sh.t_scale)))));
        *kappa2 = tof(g_pow(tod(sh.tf), (double)*n2));
        const float g = fdiv(fmul(sh.t_scale, *kappa2), fsub(1.0f, *kappa2));
        *alpha2 = fmul(-omega0, g);
    } else {
        *n2 = 0;
        *kappa2 = 1.0f;
        *alpha2 = 0.0f;
    }
}

// The inner loop of GotoTorqueBreaks: the first n1 in 1 .. 99 whose torque is below the maximum, kappa1 = tf^n1 by
// Destiny's repeated float products; n1 = 100 with the 99th when none is
static __device__ __noinline__ int search_v(const Ship& sh, V3 target, V3 omega0, float tau, int n2, float kappa2,
                                            float max_alpha, V3* alpha1, float* kappa1) {
    const float n2dt = n2dt_of(n2, sh.dt);
    float k1 = 1.0f;
    V3 a1 = V3{0.0f, 0.0f, 0.0f};
    int n1 = 1;
    for (; n1 < XBPO_TORQUE_STEPS; ++n1) {
        k1 = fmul(k1, sh.tf);
        const float ndt = tof(dmul((double)n1, sh.dt));
        float num, den;
        if (n2 == 0) {
            num = fmul(tau, fsub(1.0f, k1));
            den = fmul(tau, fsub(ndt, num));
        } else {
            num = fsub(tau, fdiv(fmul(fmul(k1, kappa2), n2dt), fsub(1.0f, kappa2)));
            den = fmul(tau, fsub(ndt, fdiv(fmul(fmul(n2dt, kappa2), fsub(1.0f, k1)), fsub(1.0f, kappa2))));
        }
        a1 = v3_div(v3_sub(target, v3_mul(omega0, num)), den);
        if (v3_len(a1) < max_alpha) break;
    }
    *alpha1 = a1;
    *kappa1 = k1;
    return n1;
}

// The inner loop of GotoRollBreaks (a true division where the vector version multiplies by the reciprocal)
static __device__ __noinline__ int search_f(const Ship& sh, float target, float omega0, float tau, int n2, float kappa2,
                                            float max_alpha, float* alpha1, float* kappa1) {
    const float n2dt = n2dt_of(n2, sh.dt);
    float k1 = 1.0f;
    float a1 = 0.0f;
    int n1 = 1;
    for (; n1 < XBPO_TORQUE_STEPS; ++n1) {
        k1 = fmul(k1, sh.tf);
        const float ndt = tof(dmul((double)n1, sh.dt));
        float num, den;
        if (n2 == 0) {
            num = fmul(tau, fsub(1.0f, k1));
            den = fmul(tau, fsub(ndt, num));
        } else {
            num = fsub(tau, fdiv(fmul(fmul(k1, kappa2), n2dt), fsub(1.0f, kappa2)));
            den = fmul(tau, fsub(ndt, fdiv(fmul(fmul(n2dt, kappa2), fsub(1.0f, k1)), fsub(1.0f, kappa2))));
        }
        a1 = fdiv(fsub(target, fmul(omega0, num)), den);
        if (fabs_(a1) < max_alpha) break;
    }
    *alpha1 = a1;
    *kappa1 = k1;
    return n1;
}

// GotoTorqueBreaks (Ballpark.cpp:1616): the heading torque toward `direction`
static __device__ __noinline__ V3 goto_torque_breaks(const Ship& sh, V3 direction, Q4 rot, V3 w) {
    const V3 zero = V3{0.0f, 0.0f, 0.0f};
    const float max_alpha = fmul(sh.max_ang_vel, sh.t_scale);
    if (max_alpha <= 0.0f) return zero;
    const float len_dir = v3_len(direction);
    if (len_dir == 0.0f) return zero;
    const V3 heading = v3_normalize(transform_coord(V3{0.0f, 0.0f, 1.0f}, m_rotation_q(rot)));
    V3 target = v3_div(v3_cross(heading, direction), len_dir);
    const float rot_len = v3_len(target);
    const float rot_angle = g_acosf(fmax_(-1.0f, fmin_(1.0f, fdiv(v3_dot(direction, heading), len_dir))));
    if (tod(rot_len) > KD(K_D_1EM12)) target = v3_mul(target, fdiv(rot_angle, rot_len));
    else if (rot_angle > 1.0f) target = V3{-rot_angle, 0.0f, 0.0f};
    const V3 omega0 = v3_sub(w, v3_mul(heading, v3_dot(heading, w)));
    const float len_omega0 = v3_len(omega0);
    if (dmul(tod(len_omega0), sh.dt) < KD(K_D_1EM4) && tod(rot_angle) < KD(K_D_1EM4)) return zero;
    const float tau = fdiv(1.0f, sh.t_scale);
    int n2;
    float kappa2;
    V3 alpha2;
    n2_alpha2_v(sh, omega0, len_omega0, &n2, &kappa2, &alpha2);
    int n1 = 0;
    V3 alpha1 = zero;
    float kappa1 = 1.0f;
    for (int round = 0; round < XBPO_TORQUE_ROUNDS; ++round) {
        if (n2 > 0) {
            const float n2dt = n2dt_of(n2, sh.dt);
            const V3 theta2 = v3_add(v3_mul(v3_mul(alpha2, tau), fsub(n2dt, fmul(tau, fsub(1.0f, kappa2)))),
                                     v3_mul(v3_mul(omega0, tau), fsub(1.0f, kappa2)));
            if (tod(v3_len(v3_sub(target, theta2))) < KD(K_D_1EM3)) {
                n1 = 0;                                               // braking alone suffices
                break;
            }
        }
        n1 = search_v(sh, target, omega0, tau, n2, kappa2, max_alpha, &alpha1, &kappa1);
        const V3 omega1 = v3_add(v3_mul(alpha1, fmul(tau, fsub(1.0f, kappa1))), v3_mul(omega0, kappa1));
        const int n2_old = n2;
        n2_alpha2_v(sh, omega1, v3_len(omega1), &n2, &kappa2, &alpha2);
        if (n2 == n2_old) break;
    }
    if (n1 > 0) return n1 > 1 ? v3_mul(v3_normalize(alpha1), max_alpha) : alpha1;
    if (n2 > 0) return alpha2;
    return zero;
}

// GotoRollBreaks (Ballpark.cpp:1751): the roll torque, rolling into strong turns, otherwise upright
static __device__ __noinline__ float goto_roll_breaks(const Ship& sh, V3 heading_torque, Q4 rot, V3 w) {
    const float max_alpha = fmul(sh.t_scale, sh.max_ang_vel);
    if (max_alpha <= 0.0f) return 0.0f;
    const M3 mat = m_rotation_q(q_conj(rot));
    const V3 ship_omega = transform_coord(w, mat);
    const float omega_roll = ship_omega.z;
    const float len_torque = v3_len(heading_torque);
    float turn_roll = 0.0f, weight = 0.0f;
    if (tod(len_torque) > dmul(KD(K_D_0_05), tod(max_alpha))) {
        const V3 ship_torque = transform_coord(heading_torque, mat);
        const V3 flat = V3{ship_omega.x, ship_omega.y, 0.0f};
        if (v3_dot(flat, ship_torque) >= 0.0f || v3_len(flat) < fmul(KF(K_F_0_01), sh.max_ang_vel)) {
            turn_roll = g_atan2f(-ship_torque.y, -ship_torque.x);
            weight = fdiv(len_torque, sh.t_scale);
            if (weight > 1.0f) weight = 1.0f;
        }
    }
    const V3 up = transform_coord(V3{0.0f, 1.0f, 0.0f}, mat);
    const float up_roll = up.z < KF(K_F_0_95) ? g_atan2f(-up.x, up.y) : 0.0f;
    const float desired = weight < KF(K_F_0_05) ? up_roll : fmul(weight, turn_roll);
    if (tod(fabs_(omega_roll)) < KD(K_D_1EM4) && tod(fabs_(desired)) < KD(K_D_1EM4)) return 0.0f;
    const float tau = fdiv(1.0f, sh.t_scale);
    int n2;
    float kappa2, alpha2;
    n2_alpha2_f(sh, omega_roll, &n2, &kappa2, &alpha2);
    int n1 = 0;
    float alpha1 = 0.0f, kappa1 = 1.0f;
    for (int round = 0; round < XBPO_TORQUE_ROUNDS; ++round) {
        if (n2 > 0) {
            const float n2dt = n2dt_of(n2, sh.dt);
            const float theta2 = fadd(fmul(fmul(tau, alpha2), fsub(n2dt, fmul(tau, fsub(1.0f, kappa2)))),
                                      fmul(fmul(tau, omega_roll), fsub(1.0f, kappa2)));
            if (tod(fabs_(fsub(desired, theta2))) < KD(K_D_1EM3)) {
                n1 = 0;
                break;
            }
        }
        n1 = search_f(sh, desired, omega_roll, tau, n2, kappa2, max_alpha, &alpha1, &kappa1);
        const float omega1 = fadd(fmul(fmul(alpha1, tau), fsub(1.0f, kappa1)), fmul(omega_roll, kappa1));
        const int n2_old = n2;
        n2_alpha2_f(sh, omega1, &n2, &kappa2, &alpha2);
        if (n2 == n2_old) break;
    }
    if (n1 > 0) return n1 > 1 ? fmul(alpha1 >= 0.0f ? 1.0f : -1.0f, max_alpha) : alpha1;
    if (n2 > 0) return alpha2;
    return 0.0f;
}

// ApplyTorque (Ballpark.cpp:1936) over one tick (fraction 1): heading and roll apart
static __device__ __noinline__ void apply_torque(Q4 rotation, V3 angvel, V3 torque, float roll, float tau, float fdt,
                                                 Q4* rot_out, V3* w_out) {
    const float tf = g_expf(fdiv(-fdt, tau));
    const V3 heading = transform_coord(V3{0.0f, 0.0f, 1.0f}, m_rotation_q(rotation));
    const float omega_roll = v3_dot(angvel, heading);
    const V3 omega_heading = v3_sub(angvel, v3_mul(heading, omega_roll));
    const float span = fsub(fdt, fmul(tau, fsub(1.0f, tf)));
    const float rot_roll = fmul(tau, fadd(fmul(roll, span), fmul(omega_roll, fsub(1.0f, tf))));
    const V3 rot_heading = v3_mul(v3_add(v3_mul(torque, span), v3_mul(omega_heading, fsub(1.0f, tf))), tau);
    const float angle_heading = v3_len(rot_heading);
    const bool apply = tod(angle_heading) > KD(K_D_1EM10);
    Q4 quat_heading = Q4{0.0f, 0.0f, 0.0f, 1.0f};
    V3 heading_rotated = heading;
    if (apply) {
        quat_heading = q_rotation(rot_heading, angle_heading);
        heading_rotated = transform_coord(heading, m_rotation_q(quat_heading));
    }
    V3 w = v3_add(v3_mul(torque, fmul(tau, fsub(1.0f, tf))), v3_mul(omega_heading, tf));
    w = v3_add(w, v3_mul(heading_rotated, fsub(fmul(tau, roll), fmul(fsub(fmul(tau, roll), omega_roll), tf))));
    if (apply) rotation = q_mul(rotation, quat_heading);
    rotation = q_mul(rotation, q_rotation(heading_rotated, rot_roll));
    const float norm = q_len(rotation);
    if (norm > 0.0f) rotation = q_div(rotation, norm);
    *rot_out = rotation;
    *w_out = w;
}

// ================================================================================================== the kernels

// EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890) and, at the end of the tick, ApplyTorque's
// rotation and angular velocity committed where finite (Evolve): the torch module's _orient and its evolve's commit.
__global__ void orient_kernel(int64_t n, const double* a_in, double* a_out, const double* mass, const double* agility,
                              const double* speed_fraction, const double* max_vel, const double* max_ang_vel,
                              const double* rot_agility, const double* rot_in, const double* w_in, double* rot_out,
                              double* old_rot_out, double* w_out, double* old_w_out, double* torque_out,
                              double* rot_roll_out, double dt, float fdt, float ffriction) {
    // Each thread reads its ship's row before it writes, so an output may be its input (a_out = a_in, rot_out =
    // rot_in, w_out = w_in): nothing is declared __restrict__.
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double ax = a_in[3 * i], ay = a_in[3 * i + 1], az = a_in[3 * i + 2];
    const float accel = tof(dsqrt(dadd(dadd(dmul(ax, ax), dmul(ay, ay)), dmul(az, az))));
    const float fmass = tof(mass[i]);
    const float ag = tof(agility[i]), sf = tof(speed_fraction[i]), mv = tof(max_vel[i]);
    const float rag = tof(rot_agility[i]);
    const float max_accel = fdiv(fmul(fmul(ffriction, sf), mv), fmul(fmass, ag));
    const bool thrust = tod(accel) > dmul(KD(K_D_1EM2), tod(max_accel));
    Ship sh;
    sh.max_ang_vel = tof(max_ang_vel[i]);
    sh.t_scale = fdiv(ffriction, fmul(fmul(fmass, ag), rag));
    sh.tf = g_expf(fmul(-fdt, sh.t_scale));
    sh.dt = dt;
    sh.fdt = fdt;
    const V3 a32 = V3{tof(ax), tof(ay), tof(az)};
    const Q4 r0 = Q4{tof(rot_in[4 * i]), tof(rot_in[4 * i + 1]), tof(rot_in[4 * i + 2]), tof(rot_in[4 * i + 3])};
    const V3 w0 = V3{tof(w_in[3 * i]), tof(w_in[3 * i + 1]), tof(w_in[3 * i + 2])};
    V3 torque = V3{0.0f, 0.0f, 0.0f};
    if (thrust) torque = goto_torque_breaks(sh, a32, r0, w0);
    const float rot_roll = goto_roll_breaks(sh, torque, r0, w0);
    const float tau = fdiv(fmul(fmul(fmass, ag), rag), ffriction);
    Q4 rotation;
    V3 angvel;
    apply_torque(r0, w0, torque, rot_roll, tau, fdt, &rotation, &angvel);
    double out0 = ax, out1 = ay, out2 = az;                           // coasting: the thrust kept
    if (thrust) {
        const V3 heading = transform_coord(V3{0.0f, 0.0f, 1.0f}, m_rotation_q(rotation));
        const float cosangle = fmin_(1.0f, fdiv(v3_dot(heading, a32), accel));
        if (cosangle <= 0.0f) {
            out0 = 0.0;
            out1 = 0.0;
            out2 = 0.0;
        } else {
            const V3 h = v3_mul(heading, fmul(fmul(cosangle, cosangle), accel));
            out0 = tod(h.x);
            out1 = tod(h.y);
            out2 = tod(h.z);
        }
    }
    a_out[3 * i] = out0;
    a_out[3 * i + 1] = out1;
    a_out[3 * i + 2] = out2;
    torque_out[3 * i] = tod(torque.x);
    torque_out[3 * i + 1] = tod(torque.y);
    torque_out[3 * i + 2] = tod(torque.z);
    rot_roll_out[i] = tod(rot_roll);
    const bool rot_ok = ffinite(q_lensq(rotation));                   // the commit: new values where finite
    const bool w_ok = ffinite(v3_dot(angvel, angvel));
    const Q4 rn = rot_ok ? rotation : r0, ro = rot_ok ? r0 : rotation;
    const V3 wn = w_ok ? angvel : w0, wo = w_ok ? w0 : angvel;
    rot_out[4 * i] = tod(rn.x);
    rot_out[4 * i + 1] = tod(rn.y);
    rot_out[4 * i + 2] = tod(rn.z);
    rot_out[4 * i + 3] = tod(rn.w);
    old_rot_out[4 * i] = tod(ro.x);
    old_rot_out[4 * i + 1] = tod(ro.y);
    old_rot_out[4 * i + 2] = tod(ro.z);
    old_rot_out[4 * i + 3] = tod(ro.w);
    w_out[3 * i] = tod(wn.x);
    w_out[3 * i + 1] = tod(wn.y);
    w_out[3 * i + 2] = tod(wn.z);
    old_w_out[3 * i] = tod(wo.x);
    old_w_out[3 * i + 1] = tod(wo.y);
    old_w_out[3 * i + 2] = tod(wo.z);
}

// The float functions and pow alone, for the tests: out = f(x) or f(y, x) (atan2f), pow(x, y); floats held in doubles
__global__ void libm_kernel(int func, int64_t n, const double* __restrict__ x, const double* __restrict__ y,
                            double* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xf = tof(x[i]);
    switch (func) {
        case 0: out[i] = tod(g_expf(xf)); break;
        case 1: out[i] = tod(g_logf(xf)); break;
        case 2: out[i] = tod(g_sinf(xf)); break;
        case 3: out[i] = tod(g_cosf(xf)); break;
        case 4: out[i] = tod(g_acosf(xf)); break;
        case 5: out[i] = tod(g_atanf(xf)); break;
        case 6: out[i] = tod(g_atan2f(tof(y[i]), xf)); break;
        case 7: out[i] = g_pow(x[i], y[i]); break;
        default: out[i] = dnan(); break;
    }
}

// The native fused multiply-adds alone, for the tests: double (a b + c rounded once) or float
__global__ void fma_kernel(int is_float, int64_t n, const double* __restrict__ a, const double* __restrict__ b,
                           const double* __restrict__ c, double* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = is_float ? tod(ffma(tof(a[i]), tof(b[i]), tof(c[i]))) : dfma(a[i], b[i], c[i]);
}

}  // namespace xbpo

// ====================================================================================== the C interface (ctypes)

static int launch_check(void) {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

static unsigned grid_for(int64_t n, int block) { return (unsigned)((n + block - 1) / block); }

extern "C" {

int xbpo_abi(void) { return XBPO_ABI; }

int xbpo_block_limit(void) { return 1024; }

const char* xbpo_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

// One tick's dynamical orientation for n ships on `stream` (torch's current stream): no allocation and no host sync,
// so it can be captured in a CUDA graph. An output may be its input (a_out = a_in, rot_out = rot_in, w_out = w_in).
int xbpo_orient(int64_t n, const double* a_in, double* a_out, const double* mass, const double* agility,
                const double* speed_fraction, const double* max_vel, const double* max_ang_vel,
                const double* rot_agility, const double* rot_in, const double* w_in, double* rot_out,
                double* old_rot_out, double* w_out, double* old_w_out, double* torque, double* rot_roll, double dt,
                float fdt, float ffriction, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbpo::orient_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
        n, a_in, a_out, mass, agility, speed_fraction, max_vel, max_ang_vel, rot_agility, rot_in, w_in, rot_out,
        old_rot_out, w_out, old_w_out, torque, rot_roll, dt, fdt, ffriction);
    return launch_check();
}

int xbpo_libm(int func, int64_t n, const double* x, const double* y, double* out, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0 || func < 0 || func > 7) return -1;
    xbpo::libm_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(func, n, x, y, out);
    return launch_check();
}

int xbpo_fma(int is_float, int64_t n, const double* a, const double* b, const double* c, double* out, void* stream,
             int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbpo::fma_kernel<<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(is_float, n, a, b, c, out);
    return launch_check();
}

// The orientation kernel as loaded on the current device: registers, local memory, the binary and PTX versions.
int xbpo_info(int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    const cudaError_t e = cudaFuncGetAttributes(&fa, xbpo::orient_kernel);
    if (e != cudaSuccess) return (int)e;
    *regs = fa.numRegs;
    *local_bytes = (int)fa.localSizeBytes;
    *binary_version = fa.binaryVersion;
    *ptx_version = fa.ptxVersion;
    *max_threads = fa.maxThreadsPerBlock;
    return 0;
}

}  // extern "C"
