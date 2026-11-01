// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's expf, logf, sinf, cosf (Arm), acosf, atanf, atan2f (fdlibm) and pow (Arm) for the orientation kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_orient_cuda.cu; a build inlines it (_cuda_build.read_source).

// ============================================================================== the C library's float functions
// Transcribed from exact_ballpark.vec_libm (glibc 2.34 aarch64): its operations in its order, each fused step fused.

static __device__ __noinline__ float g_expf(float x) {
    // e_expf.c: k = round(x 32/ln2) ties away, r = x 32/ln2 - k, a cubic in r fused three times around 2^(i/32)
    const uint32_t ix = fbits(x);
    const uint32_t abstop = (ix >> 20) & 0x7ffu;
    if (abstop > 0x42au && !((x <= KF(K_EXPF_OFLOW)) && (x >= KF(K_EXPF_MAY_UFLOW)))) {
        if (x == ffrom(0xff800000u)) return 0.0f;
        if (abstop >= 0x7f8u) return fadd(x, x);                    // infinity or NaN
        if (x > KF(K_EXPF_OFLOW)) return finf();                      // __math_oflowf
        if (x < KF(K_EXPF_UFLOW)) return 0.0f;                        // __math_uflowf
        return KF(K_MAY_UFLOWF);                                      // __math_may_uflowf
    }
    const double z = dmul(tod(x), KD(K_EXPF_INVLN2));
    const double kd = frinta(z);
    const int64_t ki = __double2ll_rz(kd);
    const double r = dsub(z, kd);
    const double s = dfrom(T_EXP2F[ki & 31] + ((uint64_t)ki << 47));
    const double zc = dfma(r, KD(K_EXPF_C0), KD(K_EXPF_C1));          // fmadd: C0 r + C1
    double y = dfma(r, KD(K_EXPF_C2), 1.0);                           // fmadd: C2 r + 1
    const double r2 = dmul(r, r);
    y = dfma(zc, r2, y);                                              // fmadd: z r2 + y
    return tof(dmul(y, s));
}

static __device__ __noinline__ float g_logf(float x) {
    // e_logf.c: x = 2^k z, r = z/c - 1 fused, a cubic in r, log c + k ln2 fused
    uint32_t ix = fbits(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f000000u) {                            // below 2^-126, negative, infinity or NaN
        if (!(ix > 0u && ix < 0x00800000u)) {
            if ((ix & 0x7fffffffu) == 0u) return ffrom(0xff800000u);
            if (ix == 0x7f800000u) return finf();
            return fnan();
        }
        ix = fbits(fmul(x, 8388608.0f)) - (23u << 23);               // a subnormal normalised by 2^23
    }
    const uint32_t tmp = ix - 0x3f330000u;
    const uint32_t i = (tmp >> 19) & 15u;
    const int32_t k = (int32_t)tmp >> 23;
    const float z = ffrom(ix - (tmp & 0xff800000u));
    const double invc = dfrom(T_LOGF_INVC[i]), logc = dfrom(T_LOGF_LOGC[i]);
    const double r = dfma(tod(z), invc, -1.0);                        // fmadd: invc z - 1
    const double y0 = dfma((double)k, KD(K_LOGF_LN2), logc);          // fmadd: logc + k ln2
    double y = dfma(r, KD(K_LOGF_A1), KD(K_LOGF_A2));                 // fmadd: A1 r + A2
    const double r2 = dmul(r, r);
    const double ry0 = dadd(r, y0);
    y = dfma(r2, KD(K_LOGF_A0), y);                                   // fmadd: A0 r2 + y
    return tof(dfma(y, r2, ry0));                                     // fmadd: y r2 + (r + y0)
}

static __device__ __forceinline__ double sin_poly(double xv, double x2) {
    const double x3 = dmul(xv, x2);
    const double s1 = dfma(x2, KD(K_SINF_S2), KD(K_SINF_S1));
    const double x7 = dmul(x2, x3);
    const double s = dfma(x3, KD(K_SINF_S0), xv);
    return dfma(s1, x7, s);
}

static __device__ __forceinline__ double cos_poly(double x2) {
    const double c1 = dfma(x2, KD(K_COSF_C1), KD(K_COSF_C0));
    const double c2 = dfma(x2, KD(K_COSF_C4), KD(K_COSF_C3));
    const double x4 = dmul(x2, x2);
    const double x6 = dmul(x2, x4);
    const double c = dfma(x4, KD(K_COSF_C2), c1);
    return dfma(c2, x6, c);
}

static __device__ __forceinline__ void reduce_large(uint32_t xi, int64_t* np, double* xr) {
    // sincosf.h's reduce_large: x 4/pi in 62-bit fixed point from __inv_pio4 (integers: exact)
    const uint32_t idx = (xi >> 26) & 15u;
    const uint32_t shift = (xi >> 23) & 7u;
    const uint32_t m = ((xi & 0x7fffffu) | 0x800000u) << shift;
    uint64_t res0 = (uint64_t)(uint32_t)(m * T_INV_PIO4[idx]);
    const uint64_t res1 = (uint64_t)m * T_INV_PIO4[idx + 4];
    const uint64_t res2 = (uint64_t)m * T_INV_PIO4[idx + 8];
    res0 = (res2 >> 32) | (res0 << 32);
    res0 += res1;
    const uint64_t n = (res0 + (1ULL << 61)) >> 62;
    res0 -= n << 62;
    *np = (int64_t)n;
    *xr = dmul(__ll2double_rn((long long)res0), KD(K_PI63));
}

static __device__ __noinline__ float g_sincosf(float x, bool cosine) {
    // s_sinf.c, s_cosf.c: the polynomial at x below pi/4; reduce_fast below 120; else reduce_large
    const uint32_t ix = fbits(x);
    const uint32_t abstop = (ix >> 20) & 0x7ffu;
    if (abstop >= 0x7f8u) return fnan();                              // __math_invalidf
    if (abstop < 0x3f4u) {
        if (abstop < 0x398u) return cosine ? 1.0f : x;
        const double xd = tod(x);
        const double xx = dmul(xd, xd);
        return tof(cosine ? cos_poly(xx) : sin_poly(xd, xx));
    }
    int64_t n, q;
    double xr;
    if (abstop < 0x42fu) {
        const double xd = tod(x);
        const double kd = frinta(dmul(xd, KD(K_HPI_INV)));
        xr = dfma(kd, -KD(K_HPI), xd);                                // fmsub: x - round(r) hpi
        n = __double2ll_rz(kd);
        q = n;
    } else {
        reduce_large(ix, &n, &xr);
        q = n + (int64_t)(ix >> 31);
    }
    const double s = dfrom(T_SINCOSF_SIGN[q & 3]);
    const double x2 = dmul(xr, xr);
    const bool use_cos = cosine ? ((n & 1) == 0) : ((n & 1) == 1);
    const double reduced = use_cos ? dmul(cos_poly(x2), (q & 2) != 0 ? -1.0 : 1.0) : sin_poly(dmul(xr, s), x2);
    return tof(reduced);
}

static __device__ __forceinline__ float g_sinf(float x) { return g_sincosf(x, false); }
static __device__ __forceinline__ float g_cosf(float x) { return g_sincosf(x, true); }

static __device__ __forceinline__ float acosf_r(float z) {
    // e_acosf.c's r = p/q, every step fused as compiled
    const float p4 = ffma(z, KF(K_ACOSF_PS5), KF(K_ACOSF_PS4));
    const float q3 = ffma(z, KF(K_ACOSF_QS4), KF(K_ACOSF_QS3));
    const float p3 = ffma(p4, z, KF(K_ACOSF_PS3));
    const float q2 = ffma(q3, z, KF(K_ACOSF_QS2));
    const float p2 = ffma(p3, z, KF(K_ACOSF_PS2));
    const float q1 = ffma(q2, z, KF(K_ACOSF_QS1));
    const float p1 = ffma(p2, z, KF(K_ACOSF_PS1));
    const float q = ffma(q1, z, 1.0f);
    const float p0 = ffma(p1, z, KF(K_ACOSF_PS0));
    const float p = fmul(p0, z);
    return fdiv(p, q);
}

static __device__ __noinline__ float g_acosf(float x) {
    // e_acosf.c (fdlibm, in float): |x| < 0.5, x > 0.5 and x < -0.5, the exact values at +-1 and below 2^-26
    const uint32_t hx = fbits(x), ix = hx & 0x7fffffffu;
    if (ix > 0x3f800000u) return fnan();
    if (ix == 0x3f800000u) return (hx >> 31) ? fadd(KF(K_ACOSF_PI), KF(K_ACOSF_TWO_PIO2_LO)) : 0.0f;
    if (ix <= 0x32800000u) return fadd(KF(K_ACOSF_PIO2_HI), KF(K_ACOSF_PIO2_LO));
    if (ix < 0x3f000000u) {
        const float z = fmul(x, x);
        const float t = ffma(-x, acosf_r(z), KF(K_ACOSF_PIO2_LO));     // fmsub: pio2_lo - x r
        return fsub(KF(K_ACOSF_PIO2_HI), fsub(x, t));
    }
    if (hx >> 31) {
        const float z = fmul(fadd(x, 1.0f), 0.5f);
        const float r = acosf_r(z);
        const float s = fsqrt(z);
        const float w = ffma(s, r, KF(K_ACOSF_NEG_PIO2_LO));          // fmadd: -pio2_lo + s r
        return ffma(-fadd(s, w), 2.0f, KF(K_ACOSF_PI));                // fmsub: pi - (s + w) 2
    }
    const float z = fmul(fsub(1.0f, x), 0.5f);
    const float s = fsqrt(z);
    const float df = ffrom(fbits(s) & 0xfffff000u);
    const float c = fdiv(ffma(-df, df, z), fadd(s, df));               // fmsub: z - df df
    const float w = ffma(s, acosf_r(z), c);                            // fmadd: c + s r
    const float dw = fadd(df, w);
    return fadd(dw, dw);
}

static __device__ __noinline__ float g_atanf(float x) {
    // s_atanf.c (fdlibm, in float): reduced against 7/16, 11/16, 19/16 and 39/16, the polynomials fused
    const uint32_t hx = fbits(x), ix = hx & 0x7fffffffu;
    if (ix > 0x7f800000u) return fadd(x, x);
    if (ix > 0x4bffffffu)
        return (hx >> 31) ? fsub(-KF(K_ATANF_HI3), KF(K_ATANF_LO3)) : fadd(KF(K_ATANF_HI3), KF(K_ATANF_LO3));
    if (ix <= 0x30ffffffu) return x;
    const bool below = ix <= 0x3edfffffu;
    int k = 0;
    float xr = x;
    if (!below) {
        const float ax = fabs_(x);
        if (ix <= 0x3f97ffffu) {
            if (ix <= 0x3f2fffffu) {
                k = 0;
                xr = fdiv(ffma(ax, 2.0f, -1.0f), fadd(ax, 2.0f));      // fmadd: 2x - 1
            } else {
                k = 1;
                xr = fdiv(fsub(ax, 1.0f), fadd(ax, 1.0f));
            }
        } else if (ix <= 0x401bffffu) {
            k = 2;
            xr = fdiv(fsub(ax, 1.5f), ffma(ax, 1.5f, 1.0f));           // fmadd: 1 + 1.5 x
        } else {
            k = 3;
            xr = fdiv(-1.0f, ax);
        }
    }
    const float z = fmul(xr, xr);
    const float w = fmul(z, z);
    const float t8 = ffma(w, KF(K_ATANF_T10), KF(K_ATANF_T8));
    const float t7 = ffma(w, KF(K_ATANF_T9), KF(K_ATANF_T7));
    const float t6 = ffma(t8, w, KF(K_ATANF_T6));
    const float t5 = ffma(t7, w, KF(K_ATANF_T5));
    const float t4 = ffma(t6, w, KF(K_ATANF_T4));
    const float t3 = ffma(t5, w, KF(K_ATANF_T3));
    const float t2 = ffma(t4, w, KF(K_ATANF_T2));
    const float t1 = ffma(t3, w, KF(K_ATANF_T1));
    const float t0 = ffma(t2, w, KF(K_ATANF_T0));
    const float s12 = ffma(t0, z, fmul(t1, w));                        // fmadd: z (aT0 + ...) + w (aT1 + ...)
    const float xs12 = fmul(xr, s12);
    if (below) return fsub(xr, xs12);
    const float zz = fsub(ffrom(T_ATANF_HI[k]), fsub(fsub(xs12, ffrom(T_ATANF_LO[k])), xr));
    return (hx >> 31) ? -zz : zz;
}

static __device__ __noinline__ float g_atan2f(float y, float x) {
    // e_atan2f.c (fdlibm, in float), its special cases in their order
    const uint32_t hy = fbits(y), hx = fbits(x);
    const uint32_t iy = hy & 0x7fffffffu, ix = hx & 0x7fffffffu;
    if (ix > 0x7f800000u || iy > 0x7f800000u) return fadd(x, y);
    if (hx == 0x3f800000u) return g_atanf(y);
    const uint32_t m = ((hx >> 30) & 2u) | (hy >> 31);
    const float pi_t = fadd(KF(K_ATAN2F_PI), KF(K_ATAN2F_TINY));
    const float mpi_t = fsub(KF(K_ATAN2F_NEG_PI), KF(K_ATAN2F_TINY));
    if (iy == 0u) return m <= 1u ? y : (m == 2u ? pi_t : mpi_t);
    const float pio2_t = (hy >> 31) ? fsub(KF(K_ATAN2F_NEG_PIO2), KF(K_ATAN2F_TINY))
                                    : fadd(KF(K_ATAN2F_PIO2), KF(K_ATAN2F_TINY));
    if (ix == 0u) return pio2_t;
    if (ix == 0x7f800000u) {
        if (iy == 0x7f800000u) {
            if (m == 0u) return fadd(KF(K_ATAN2F_PIO4), KF(K_ATAN2F_TINY));
            if (m == 1u) return fsub(KF(K_ATAN2F_NEG_PIO4), KF(K_ATAN2F_TINY));
            if (m == 2u) return ffma(3.0f, KF(K_ATAN2F_PIO4), KF(K_ATAN2F_TINY));
            return ffma(-3.0f, KF(K_ATAN2F_PIO4), KF(K_ATAN2F_NEG_TINY));
        }
        if (m == 0u) return 0.0f;
        if (m == 1u) return -0.0f;
        return m == 2u ? pi_t : mpi_t;
    }
    if (iy == 0x7f800000u) return pio2_t;
    const int32_t k = ((int32_t)iy - (int32_t)ix) >> 23;
    float z;
    if (k > 60) z = fsub(KF(K_ATAN2F_PIO2), KF(K_ATAN2F_NEG_HALF_PI_LO));
    else if ((hx >> 31) && k < -60) z = 0.0f;
    else z = g_atanf(fabs_(fdiv(y, x)));
    const float zl = fadd(z, KF(K_ATAN2F_NEG_PI_LO));                  // z - pi_lo
    if (m == 0u) return z;
    if (m == 1u) return -z;
    if (m == 2u) return fsub(KF(K_ATAN2F_PI), zl);
    return fsub(zl, KF(K_ATAN2F_PI));
}

// ===================================================================================================== pow (double)

static __device__ __forceinline__ int checkint(uint64_t iy) {
    const int e = (int)((iy >> 52) & 0x7ff);
    if (e < 0x3ff) return 0;
    if (e > 0x3ff + 52) return 2;
    if (iy & ((1ULL << (0x3ff + 52 - e)) - 1)) return 0;
    if (iy & (1ULL << (0x3ff + 52 - e))) return 1;
    return 2;
}

static __device__ __forceinline__ double log_inline(uint64_t ix, double* tail) {
    // pow's log_inline: log x as y + tail, every step as compiled
    const uint64_t tmp = ix - K_POW_OFF;
    const int i = (int)((tmp >> 45) & 127);
    const double kd = __ll2double_rn((long long)((int64_t)tmp >> 52));
    const double z = dfrom(ix - (tmp & 0xfff0000000000000ULL));
    const double invc = dfrom(T_POW_INVC[i]), logc = dfrom(T_POW_LOGC[i]), logctail = dfrom(T_POW_LOGCTAIL[i]);
    const double t1 = dfma(kd, KD(K_POW_LN2HI), logc);                // logc + kd ln2hi
    const double r = dfma(z, invc, -1.0);                             // z invc - 1
    const double lo1 = dfma(kd, KD(K_POW_LN2LO), logctail);           // logctail + kd ln2lo
    const double ar = dmul(r, KD(K_POW_A0));
    const double a56 = dfma(r, KD(K_POW_A6), KD(K_POW_A5));
    const double t2 = dadd(r, t1);
    const double a34 = dfma(r, KD(K_POW_A4), KD(K_POW_A3));
    const double a12 = dfma(r, KD(K_POW_A2), KD(K_POW_A1));
    const double ar2 = dmul(r, ar);
    const double d = dsub(t1, t2);
    const double lo3 = dfma(ar, r, -ar2);                             // fnmsub: ar r - ar2
    const double a3456 = dfma(ar2, a56, a34);
    const double hi = dadd(t2, ar2);
    const double lo2 = dadd(d, r);
    const double ar3 = dmul(r, ar2);
    const double e = dsub(t2, hi);
    double s = dadd(lo1, lo2);
    const double poly = dfma(ar2, a3456, a12);
    const double lo4 = dadd(e, ar2);
    s = dadd(s, lo3);
    s = dadd(s, lo4);
    const double lo = dfma(ar3, poly, s);
    const double y = dadd(hi, lo);
    *tail = dadd(dsub(hi, y), lo);
    return y;
}

static __device__ __forceinline__ double exp_inline(double ehi, double elo, bool negative) {
    // pow's exp_inline: sign exp(ehi + elo), the sign through the scale's sign bit, glibc's exp otherwise
    const uint32_t abstop = (uint32_t)((dbits(ehi) >> 52) & 0x7ff);
    if (abstop < 0x3c9u) return negative ? -dadd(ehi, 1.0) : dadd(ehi, 1.0);
    if (abstop > 0x408u) {
        const double flow = ehi < 0.0 ? 0.0 : dinf();                 // __math_uflow, __math_oflow
        return negative ? -flow : flow;
    }
    const double kd = frinta(dmul(ehi, KD(K_EXP_INVLN2N)));
    const int64_t ki = __double2ll_rz(kd);
    double r = dfma(kd, KD(K_EXP_NEGLN2HIN), ehi);
    r = dfma(kd, KD(K_EXP_NEGLN2LON), r);
    const int idx = (int)((ki & 0x7f) * 2);
    const double tail = dfrom(T_EXP[idx]);
    const uint64_t base = T_EXP[idx + 1] + ((uint64_t)ki << 45);
    const uint64_t sgn = negative ? 0x8000000000000000ULL : 0ULL;
    r = dadd(elo, r);
    const double p23 = dfma(r, KD(K_EXP_C3), KD(K_EXP_C2));
    const double p45 = dfma(r, KD(K_EXP_C5), KD(K_EXP_C4));
    const double r2 = dmul(r, r);
    const double t = dadd(r, tail);
    const double r4 = dmul(r2, r2);
    double tmp = dfma(r2, p23, t);
    tmp = dfma(r4, p45, tmp);
    if (abstop != 0x408u) {
        const double scale = dfrom(base ^ sgn);
        return dfma(tmp, scale, scale);                               // scale + tmp scale
    }
    if ((ki & 0x80000000LL) == 0) {                                   // specialcase, k > 0: may overflow
        const double sc_hi = dfrom((base - (1009ULL << 52)) ^ sgn);
        return dmul(dfma(tmp, sc_hi, sc_hi), KD(K_TWO_1009));
    }
    const uint64_t sb_lo = (base + (1022ULL << 52)) ^ sgn;            // k < 0: may be subnormal
    const double sc_lo = dfrom(sb_lo);
    const double st = dmul(tmp, sc_lo);                               // not fused
    const double y = dadd(sc_lo, st);
    if (!(dabs_(y) < 1.0)) return dmul(y, KD(K_TWO_M1022));
    const double one = y < 0.0 ? -1.0 : 1.0;
    double lo = dadd(dsub(sc_lo, y), st);
    const double hi = dadd(y, one);
    lo = dadd(dadd(dsub(one, hi), y), lo);
    double y2 = dsub(dadd(lo, hi), one);
    if (y2 == 0.0) y2 = dfrom(sb_lo & 0x8000000000000000ULL);
    return dmul(y2, KD(K_TWO_M1022));
}

static __device__ __noinline__ double g_pow(double x, double y) {
    // e_pow.c: log_inline, y log x in two parts (fused), exp_inline; the special cases in glibc's order
    const uint64_t SIGN = 0x8000000000000000ULL, ONE = 0x3ff0000000000000ULL, INF = 0x7ff0000000000000ULL;
    uint64_t ix = dbits(x);
    const uint64_t iy = dbits(y);
    uint32_t topx = (uint32_t)(ix >> 52);
    const uint32_t topy = (uint32_t)(iy >> 52);
    bool negative = false;
    if (topx - 0x001u >= 0x7ffu - 0x001u || (topy & 0x7ffu) - 0x3beu >= 0x43eu - 0x3beu) {
        if ((topy & 0x7ffu) == 0x7ffu || (iy & ~SIGN) == 0) {         // y zero, infinite or NaN
            if ((iy & ~SIGN) == 0) return 1.0;
            if (ix == ONE) return 1.0;
            if ((ix & ~SIGN) > INF || (iy & ~SIGN) > INF) return dadd(x, y);
            if ((ix & ~SIGN) == ONE) return 1.0;
            if (((ix & ~SIGN) < ONE) == !(iy >> 63)) return 0.0;
            return dmul(y, y);
        }
        if ((topx & 0x7ffu) == 0x7ffu || (ix & ~SIGN) == 0) {         // x zero, infinite or NaN
            double x2 = dmul(x, x);
            if ((ix >> 63) && checkint(iy) == 1) x2 = -x2;
            return (iy >> 63) ? ddiv(1.0, x2) : x2;
        }
        if (ix >> 63) {                                               // x negative, finite
            const int yint = checkint(iy);
            if (yint == 0) return dnan();                             // __math_invalid
            if (yint == 1) negative = true;
            ix &= ~SIGN;
            topx &= 0x7ffu;
        }
        if ((topy & 0x7ffu) - 0x3beu >= 0x43eu - 0x3beu) {            // |y| below 2^-65 or at 2^63 and beyond
            if (ix == ONE) return 1.0;
            if ((topy & 0x7ffu) < 0x3beu) return ix > ONE ? dadd(1.0, y) : dsub(1.0, y);
            return (ix > ONE) == (topy < 0x800u) ? dinf() : 0.0;
        }
        if (topx == 0u) {                                             // a subnormal base normalised by 2^52
            ix = dbits(dmul(dfrom(ix), KD(K_TWO_52))) & ~SIGN;
            ix -= 52ULL << 52;
        }
    }
    double lo;
    const double hi = log_inline(ix, &lo);
    const double ehi = dmul(y, hi);
    const double elo = dfma(y, lo, dfma(y, hi, -ehi));                // fnmsub: y hi - ehi; fmadd: + y lo
    return exp_inline(ehi, elo, negative);
}

