// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's powf (Arm's optimized-routines, as glibc carries it) for the combat kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_combat_cuda.cu; a build inlines it (_cuda_build.read_source).

// ---- glibc 2.34 aarch64's powf (e_powf.c; __powf_finite at 0x411e0), transcribed from exact_ballpark.dlibm_powf
//: its operations in its order, the nine fused steps fused (dfma: __fma_rn), floats as floats; every
// constant and table from the generated header (read out of the C library by dlibm_powf's generator) ----

static __device__ __forceinline__ float pw_xflowf(uint32_t sign, uint32_t tbits) {
    // xflowf (math_errf.c, 0x4ab74): (sign ? -t : t) * t in float: __math_oflowf, __math_uflowf, __math_may_uflowf
    const float t = ffrom(tbits);
    return fmul(sign ? ffrom(tbits ^ 0x80000000u) : t, t);
}

static __device__ __forceinline__ bool pw_signaling(uint32_t u) {
    return ((u ^ (uint32_t)POWF_QNAN_BIT) & 0x7fffffffu) > (uint32_t)POWF_QNAN;   // issignalingf_inline (0x414c4)
}

static __device__ __noinline__ double pw_log2(uint32_t ix) {
    // log2_inline: log2(x) x 32 for a positive normal (or normalised) x (0x41214-0x41288)
    const uint32_t tmp = ix + POWF_NEG_OFF;                          // ix - OFF
    const uint32_t i = (tmp >> POWF_I_SHIFT) & ((1u << POWF_I_BITS) - 1u);
    const uint32_t top = tmp & POWF_TOP_MASK;
    const uint32_t iz = ix - top;
    const int32_t k = (int32_t)top >> POWF_K_SHIFT;                  // arithmetic shift (asr)
    const double invc = dfrom(T_POWF_LOG2_INVC[i]), logc = dfrom(T_POWF_LOG2_LOGC[i]);
    const double z = tod(ffrom(iz));
    const double r = dfma(invc, z, -1.0);                             // fmadd: invc z - 1
    const double y0 = dadd(__int2double_rn(k), logc);                 // scvtf (exact), fadd: k + logc
    const double p = dfma(r, dfrom(T_POWF_A[2]), dfrom(T_POWF_A[3]));  // fmadd: A2 r + A3
    double q = dfma(r, dfrom(T_POWF_A[4]), y0);                       // fmadd: A4 r + y0
    const double r2 = dmul(r, r);
    const double y = dfma(r, dfrom(T_POWF_A[0]), dfrom(T_POWF_A[1]));  // fmadd: A0 r + A1
    const double r4 = dmul(r2, r2);
    q = dfma(r2, p, q);                                               // fmadd: p r2 + q
    return dfma(y, r4, q);                                            // fmadd: y r4 + q
}

static __device__ __noinline__ float pw_exp2(double xd, uint64_t sign_bias) {
    // exp2_inline (TOINT_INTRINSICS): 2^(xd / 32) rounded to float once (0x412a0-0x412e8)
    const double kd = frinta(xd);                                     // frinta
    const int64_t ki = __double2ll_rz(kd);                            // fcvtas: the same integer
    const double r = dsub(xd, kd);
    double y = dfma(r, dfrom(T_POWF_EXP2_C[2]), 1.0);                 // fmadd: C2 r + 1
    const double z = dfma(r, dfrom(T_POWF_EXP2_C[0]), dfrom(T_POWF_EXP2_C[1]));   // fmadd: C0 r + C1
    const double r2 = dmul(r, r);
    uint64_t t = T_POWF_EXP2_TAB[(uint64_t)ki & (uint64_t)(POWF_EXP2_N - 1)];
    t += ((uint64_t)ki + sign_bias) << POWF_EXP2_SHIFT;
    y = dfma(z, r2, y);                                               // fmadd: z r2 + y
    return tof(dmul(y, dfrom(t)));                                    // fmul, fcvt
}

static __device__ __noinline__ float pw_y_special(float x, float y, uint32_t ix, uint32_t iy, bool normal_x) {
    // y zero, infinite or NaN (0x41404 for a normal x, 0x414c0 otherwise)
    const uint32_t two_iy = iy << 1;
    if (two_iy == 0u) return pw_signaling(ix) ? fadd(x, y) : 1.0f;
    if (normal_x && ix == (uint32_t)POWF_ONE_BITS) return pw_signaling(iy) ? fadd(x, y) : 1.0f;
    const uint32_t two_ix = ix << 1;
    if (two_iy != POWF_INF2 || two_ix > POWF_INF2) return fadd(x, y); // y NaN, or x NaN
    if (two_ix == (uint32_t)POWF_ONE2) return 1.0f;                   // |x| == 1
    if ((two_ix > (uint32_t)POWF_ONE2_M1 ? 1u : 0u) != ((~iy) >> 31)) return 0.0f;
    return fmul(y, y);
}

static __device__ __noinline__ float g_powf(float x, float y) {
    // powf(x, y) (0x411e0)
    uint32_t ix = fbits(x);
    const uint32_t iy = fbits(y);
    const uint32_t two_iy = iy << 1;
    const bool zinf_y = two_iy - 1u >= POWF_ZINF;                     // zeroinfnan(iy)
    uint64_t sign_bias = 0;
    if (ix - (uint32_t)POWF_X_MIN > (uint32_t)POWF_X_SPAN) {          // x < 0x1p-126, negative, infinite or NaN
        if (zinf_y) return pw_y_special(x, y, ix, iy, false);
        const uint32_t two_ix = ix << 1;
        if (two_ix - 1u >= POWF_ZINF) {                               // x zero, infinite or NaN
            float x2 = fmul(x, x);
            uint32_t sb = 0;
            if (ix >> 31) {                                           // checkint(iy) == 1 (0x4147c)
                const uint32_t e = (iy >> 23) & 0xffu;
                if (e - (uint32_t)POWF_E_ONE <= (uint32_t)POWF_E_SPAN) {
                    const uint32_t bit = 1u << ((uint32_t)POWF_INT_E - e);
                    if (!(iy & (bit - 1u)) && (iy & bit)) {
                        x2 = ffrom(fbits(x2) ^ 0x80000000u);          // fneg
                        sb = 1;
                    }
                }
            }
            if (two_ix == 0u && (iy >> 31)) return fdiv(sb ? -1.0f : 1.0f, 0.0f);   // __math_divzerof
            return (iy >> 31) ? fdiv(1.0f, x2) : x2;
        }
        if (ix >> 31) {                                               // a finite x < 0: checkint (0x41384)
            const uint32_t e = (iy >> 23) & 0xffu;
            if (e <= (uint32_t)POWF_E_HALF) return fdiv(fsub(x, x), fsub(x, x));   // __math_invalidf
            if (e <= (uint32_t)POWF_INT_E) {
                const uint32_t bit = 1u << ((uint32_t)POWF_INT_E - e);
                if (iy & (bit - 1u)) return fdiv(fsub(x, x), fsub(x, x));
                if (iy & bit) sign_bias = (uint64_t)POWF_SIGN_BIAS;
            }
            ix &= 0x7fffffffu;
        }
        if (ix <= (uint32_t)POWF_SUBNORMAL_MAX) {                      // subnormal: x 2^23, then 23 << 23 off
            ix = (fbits(fmul(x, ffrom((uint32_t)POWF_TWO23_F))) & 0x7fffffffu) - (uint32_t)POWF_NORM_SUB;
        }
    } else if (zinf_y) {
        return pw_y_special(x, y, ix, iy, true);
    }
    const double ylogx = dmul(tod(y), pw_log2(ix));                   // fcvt (exact), fmul
    if (((dbits(ylogx) >> 47) & 0xffffULL) > (uint64_t)POWF_BIG) {    // |y log2 x| >= 126 (scaled)
        if (ylogx > dfrom(POWF_OFLOW)) return pw_xflowf((uint32_t)sign_bias, (uint32_t)POWF_OFLOWF_X);
        if (ylogx > dfrom(POWF_RND)) {                                // the rounding mode's check (never, to nearest)
            const float tiny = ffrom((uint32_t)POWF_TINY_F);
            if (sign_bias ? (fsub(-1.0f, tiny) != -1.0f) : (fadd(1.0f, tiny) != 1.0f))
                return pw_xflowf((uint32_t)sign_bias, (uint32_t)POWF_OFLOWF_X);
        }
        if (ylogx <= dfrom(POWF_UFLOW)) return pw_xflowf((uint32_t)sign_bias, (uint32_t)POWF_UFLOWF_X);
        if (ylogx < dfrom(POWF_MAY_UFLOW)) return pw_xflowf((uint32_t)sign_bias, (uint32_t)POWF_MAY_UFLOWF_X);
    }
    return pw_exp2(ylogx, sign_bias);
}

