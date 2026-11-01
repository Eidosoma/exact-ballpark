// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's exp (Arm's optimized-routines, as glibc carries it) for the collision kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_collide_cuda.cu; a build inlines it (_cuda_build.read_source).

// ======================================================= the C library's exp (exact_ballpark.vec_exp.exp, glibc 2.34)

// IEEE roundToIntegralTowardZero (torch.trunc: the sign of a zero kept)
__device__ __forceinline__ f64 trunc_rz(f64 x) {
    f64 r;
    asm("cvt.rzi.f64.f64 %0, %1;" : "=d"(r) : "d"(x));
    return r;
}

__device__ __forceinline__ f64 dbl(u64 bits) { return __longlong_as_double(static_cast<i64>(bits)); }

// glibc 2.34 aarch64's exp, step for step as vec_exp.exp (its comments there); __fma_rn exactly where vec_exp calls its
// emulated fma; the integer steps in unsigned 64-bit arithmetic (two's complement, as torch's int64)
__device__ __noinline__ f64 xbpc_exp(f64 x) {
    const i64 bits = __double_as_longlong(x);
    const i64 abstop = (bits >> 52) & 0x7FF;
    if (abstop < 0x3C9) return x + 1.0;                                 // |x| < 2^-54: 1 + x
    if (abstop > 0x408) {                                               // |x| >= 1024, infinities and NaN
        if (abstop == 0x7FF && x != -__longlong_as_double(0x7FF0000000000000LL)) return x + 1.0;
        return bits < 0 ? 0.0 : __longlong_as_double(0x7FF0000000000000LL);
    }
    const f64 z = x * K_EXP_INVLN2N;
    const f64 kd = trunc_rz(z + (z < 0.0 ? -0.5 : 0.5));               // frinta as vec_exp computes it
    const i64 ki = static_cast<i64>(kd);
    f64 r = __fma_rn(kd, K_EXP_NEGLN2HIN, x);
    r = __fma_rn(kd, K_EXP_NEGLN2LON, r);
    const i64 idx = (ki & 0x7F) * 2;
    const f64 tail = dbl(T_EXP[idx]);
    const u64 sb = T_EXP[idx + 1];
    const u64 top = static_cast<u64>(ki) << 45;
    const f64 p23 = __fma_rn(r, K_EXP_C3, K_EXP_C2);
    const f64 p45 = __fma_rn(r, K_EXP_C5, K_EXP_C4);
    const f64 r2 = r * r;
    f64 tmp = r + tail;
    const f64 r4 = r2 * r2;
    tmp = __fma_rn(p23, r2, tmp);
    tmp = __fma_rn(r4, p45, tmp);
    if (abstop != 0x408) {
        const f64 scale = dbl(sb + top);
        return __fma_rn(tmp, scale, scale);
    }
    // the special case, 512 <= |x| < 1024
    if ((ki & 0x80000000LL) == 0) {
        const f64 sc_hi = dbl(sb + (top - (1009ULL << 52)));
        return __fma_rn(tmp, sc_hi, sc_hi) * K_TWO_1009;
    }
    const f64 sc_lo = dbl(sb + (top + (1022ULL << 52)));
    const f64 st = tmp * sc_lo;
    const f64 y = sc_lo + st;
    const f64 hi = y + 1.0;
    f64 lo = (sc_lo - y) + st;
    lo = ((1.0 - hi) + y) + lo;
    f64 y2 = (lo + hi) - 1.0;
    if (y2 == 0.0) y2 = 0.0;
    return (y < 1.0 ? y2 : y) * K_TWO_M1022;
}

__device__ __forceinline__ f64 exp_of(f64 x) {
#if XBPC_FAULT == 2
    return exp(x);                                                      // planted fault: CUDA's own exp
#else
    return xbpc_exp(x);
#endif
}

