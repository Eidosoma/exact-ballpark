// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's sin, cos, asin and atan2 (IBM Accurate Mathematical Library) for the yaw, pitch and roll kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_ypr_cuda.cu; a build inlines it (_cuda_build.read_source).

// ======================================================================== the C library's double sin and cos
// Copied from exact_ballpark.vec_modes_cuda (its transcription of exact_ballpark.dlibm: glibc 2.34 aarch64's __sin, __cos
// and __branred): their operations in their order, each fused step fused. Only the names of the constants differ.


static __device__ __forceinline__ double taylor_sin(double a, double da) {
    // TAYLOR_SIN(a a, a, da): a + ((POLY(a a) a - da / 2) a a + da)
    const double xx = dmul(a, a);
    const double poly = dfma(xx, dfma(xx, dfma(xx, dfma(xx, YKD(YL_S5), YKD(YL_S4)), YKD(YL_S3)), YKD(YL_S2)),
                             YKD(YL_S1));
    const double t = dfnmsub(poly, a, dmul(da, 0.5));
    return dadd(dfma(xx, t, da), a);
}

static __device__ __forceinline__ double sincos_table(double x, double* sn, double* ssn, double* cs, double* ccs) {
    // u = big + |x|; the table row of u's low word; returns |x| - (u - big)
    const double ax = dabs_(x);
    const double u = dadd(YKD(YL_BIG), ax);
    const int k = (int)((uint32_t)(dbits(u) & 0xffffffffULL) << 2);
    *sn = dfrom(YL_SINCOSTAB[k]);
    *ssn = dfrom(YL_SINCOSTAB[k + 1]);
    *cs = dfrom(YL_SINCOSTAB[k + 2]);
    *ccs = dfrom(YL_SINCOSTAB[k + 3]);
    return dsub(ax, dsub(u, YKD(YL_BIG)));
}

static __device__ __noinline__ double do_sin(double x, double dx) {
    // do_sin: the Taylor polynomial below 0.126, else the table combined with short polynomials; the sign of x
    if (dabs_(x) < YKD(YL_TAYLOR_LIMIT)) return taylor_sin(x, dx);
    if (x <= 0.0) dx = -dx;
    double sn, ssn, cs, ccs;
    const double r = sincos_table(x, &sn, &ssn, &cs, &ccs);
    const double xx = dmul(r, r);
    const double pc = dfma(xx, YKD(YL_CS6), YKD(YL_CS4));
    const double ps = dfma(xx, YKD(YL_SN5), YKD(YL_SN3));
    const double r3 = dmul(r, xx);
    const double c1 = dfma(xx, pc, YKD(YL_CS2));
    const double s = dadd(r, dfma(r3, ps, dx));
    const double c = dfma(r, dx, dmul(xx, c1));
    const double cor = dfma(s, cs, dfmsub(c, sn, dfma(s, ccs, ssn)));
    return dcopysign(dadd(sn, cor), x);
}

static __device__ __noinline__ double do_cos(double x, double dx) {
    // do_cos: the table's cos and sin combined with short polynomials in the rest
    if (x < 0.0) dx = -dx;
    double sn, ssn, cs, ccs;
    const double r = dadd(sincos_table(x, &sn, &ssn, &cs, &ccs), dx);
    const double xx = dmul(r, r);
    const double pc = dfma(xx, YKD(YL_CS6), YKD(YL_CS4));
    const double ps = dfma(xx, YKD(YL_SN5), YKD(YL_SN3));
    const double r3 = dmul(r, xx);
    const double c1 = dfma(xx, pc, YKD(YL_CS2));
    const double s = dfma(r3, ps, r);
    const double c = dmul(xx, c1);
    const double cor = dfmsub(s, sn, dfmsub(c, cs, dfmsub(s, ssn, ccs)));
    return dadd(cs, cor);
}

static __device__ __forceinline__ int reduce_sincos(double x, double* a, double* da) {
    // reduce_sincos, fused as compiled: x = n pi/2 + a + da
    const double t = dfma(x, YKD(YL_HPINV), YKD(YL_TOINT));
    const double xn = dsub(t, YKD(YL_TOINT));
    const double y = dfmsub(xn, YKD(YL_MP2), dfmsub(xn, YKD(YL_MP1), x));
    const double t2 = dfmsub(xn, YKD(YL_PP3), y);
    const double b = dfmsub(xn, YKD(YL_PP4), t2);
    double db = dfmsub(xn, YKD(YL_PP3), dsub(y, t2));
    db = dadd(db, dfmsub(xn, YKD(YL_PP4), dsub(t2, b)));
    *a = b;
    *da = db;
    return (int)(dbits(t) & 3ULL);
}

static __device__ __forceinline__ void branred_half(double xh, double* sum_out, double* b_out, double* bb_out) {
    // one half of branred: xh 2/pi from six 24-bit pieces of toverp, the integer part split off (nothing fused)
    const int e = (int)((dbits(xh) >> 52) & 0x7ffULL);
    int k = 0;
    double first = YKD(YL_BR_TOVERP0);
    if (e > YL_BR_EXP_LIMIT) {
        k = (e - YL_BR_EXP_BIAS) / 24;                              // C's division: toward zero
        first = dfrom(YL_BR_TOVERP[k]);
    }
    double gor = dfrom((uint64_t)((uint32_t)YL_BR_T576_HI - ((uint32_t)(3 * k) << 23)) << 32);
    double r[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        r[i] = dmul(dmul(xh, i == 0 ? first : dfrom(YL_BR_TOVERP[k + i])), gor);
        gor = dmul(gor, YKD(YL_BR_TM24));
    }
    double total = 0.0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const double s = dsub(dadd(r[i], YKD(YL_BR_BIG)), YKD(YL_BR_BIG));
        total = dadd(total, s);
        r[i] = dsub(r[i], s);
    }
    double t = 0.0;
#pragma unroll
    for (int i = 0; i < 6; ++i) t = dadd(t, r[5 - i]);
    double bb = dadd(dadd(dadd(dadd(dadd(dsub(r[0], t), r[1]), r[2]), r[3]), r[4]), r[5]);
    double s = dsub(dadd(t, YKD(YL_BR_BIG)), YKD(YL_BR_BIG));
    total = dadd(s, total);
    t = dsub(t, s);
    s = dsub(dadd(total, YKD(YL_BR_BIG1)), YKD(YL_BR_BIG1));
    total = dsub(total, s);
    const double b = dadd(bb, t);
    bb = dadd(dsub(t, b), bb);
    *sum_out = total;
    *b_out = b;
    *bb_out = bb;
}

static __device__ __noinline__ int branred(double x, double* a, double* aa) {
    // __branred: x = n pi/2 + a + aa for huge x, the argument scaled by 2^-600 and split Dekker's way
    x = dmul(x, YKD(YL_BR_TM600));
    const double t0 = dmul(x, YKD(YL_BR_SPLIT));
    const double x1 = dsub(t0, dsub(t0, x));
    const double x2 = dsub(x, x1);
    double sum1, b1, bb1, sum2, b2, bb2;
    branred_half(x1, &sum1, &b1, &bb1);
    branred_half(x2, &sum2, &b2, &bb2);
    double total = dadd(sum1, sum2);
    double b = dadd(b1, b2);
    const double bb = dabs_(b1) > dabs_(b2) ? dadd(dsub(b1, b), b2) : dadd(dsub(b2, b), b1);
    if (b > 0.5) {
        b = dsub(b, 1.0);
        total = dadd(total, 1.0);
    } else if (b < -0.5) {
        b = dadd(b, 1.0);
        total = dsub(total, 1.0);
    }
    double s = dadd(dadd(dadd(bb, bb1), bb2), b);
    const double t = dadd(dadd(dsub(b, s), bb), dadd(bb1, bb2));
    const double sp = dmul(s, YKD(YL_BR_SPLIT));
    const double t1 = dsub(sp, dsub(sp, s));
    const double t2 = dsub(s, t1);
    b = dmul(s, YKD(YL_BR_HP0));
    const double bb2x = dadd(dadd(dadd(dsub(dmul(t1, YKD(YL_BR_MP1)), b), dmul(t1, YKD(YL_BR_MP2))),
                                  dmul(t2, YKD(YL_BR_MP1))),
                             dadd(dadd(dmul(t2, YKD(YL_BR_MP2)), dmul(s, YKD(YL_BR_HP1))), dmul(t, YKD(YL_BR_HP0))));
    s = dadd(b, bb2x);
    *a = s;
    *aa = dadd(dsub(b, s), bb2x);
    return __double2int_rz(total) & 3;
}

static __device__ __forceinline__ double sincos_n(double a, double da, int n) {
    // do_sincos: do_cos for odd n, else do_sin; negated when n & 2
    const double v = (n & 1) ? do_cos(a, da) : do_sin(a, da);
    return (n & 2) ? -v : v;
}

static __device__ __noinline__ double g_sin(double x) {
    // s_sin.c's __sin (glibc 2.34 aarch64, 0x362b0)
    const uint32_t k = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    if (k < YL_K_TINY_SIN) return x;
    if (k < YL_K_DO_SIN) return do_sin(x, 0.0);
    if (k < YL_K_MID) return dcopysign(do_cos(dsub(YKD(YL_HP0), dabs_(x)), YKD(YL_HP1)), x);
    double a, da;
    if (k < YL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        return sincos_n(a, da, n);
    }
    if (k < YL_K_HUGE) {
        const int n = branred(x, &a, &da);
        return sincos_n(a, da, n);
    }
    return dnan();
}

static __device__ __noinline__ double g_cos(double x) {
    // s_sin.c's __cos (glibc 2.34 aarch64, 0x369f0)
    const uint32_t k = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    if (k < YL_K_TINY_COS) return 1.0;
    if (k < YL_K_DO_SIN) return do_cos(x, 0.0);
    if (k < YL_K_MID) {
        const double y = dsub(YKD(YL_HP0), dabs_(x));
        const double a = dadd(y, YKD(YL_HP1));
        return do_sin(a, dadd(dsub(y, a), YKD(YL_HP1)));
    }
    double a, da;
    if (k < YL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        return sincos_n(a, da, n + 1);
    }
    if (k < YL_K_HUGE) {
        const int n = branred(x, &a, &da);
        return sincos_n(a, da, n + 1);
    }
    return dnan();
}

static __device__ __noinline__ void g_sincos(double x, double* s, double* c) {
    // s_sincos.c's __sincos (glibc 2.34 aarch64, 0x37ca0; exact_ballpark.dlibm.sincos): __sin's and __cos's branches,
    // but between 0.855469 and 2.426265 the sine is do_cos after the two-sum (as __cos's cosine there is do_sin)
    const uint32_t k = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    if (k < YL_K_TINY_COS) {
        *s = x;
        *c = 1.0;
        return;
    }
    if (k < YL_K_DO_SIN) {
        *s = do_sin(x, 0.0);
        *c = do_cos(x, 0.0);
        return;
    }
    if (k < YL_K_MID) {
        const double y = dsub(YKD(YL_HP0), dabs_(x));
        const double a = dadd(y, YKD(YL_HP1));
        const double da = dadd(dsub(y, a), YKD(YL_HP1));
        *s = dcopysign(do_cos(a, da), x);
        *c = do_sin(a, da);
        return;
    }
    double a, da;
    if (k < YL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        *s = sincos_n(a, da, n);
        *c = sincos_n(a, da, n + 1);
        return;
    }
    if (k < YL_K_HUGE) {
        const int n = branred(x, &a, &da);
        *s = sincos_n(a, da, n);
        *c = sincos_n(a, da, n + 1);
        return;
    }
    *s = dnan();
    *c = dnan();
}

// ============================================================================== the C library's double asin
// Transcribed from exact_ballpark.dlibm_inv (glibc 2.34 aarch64's __ieee754_asin, 0x2f760; e_asin.c).

#define YASNCS(i) dfrom(YI_ASIN_ASNCS[i])

static __device__ __forceinline__ double asin_table(double xa, int n, int top) {
    // xx = |x| - asncs[n]; a2 .. a_top in xx (Horner, fused); p = xx xx poly + a_(top+1); t = a1 xx + p; a_(top+2) + t
    const double xx = dsub(xa, YASNCS(n));
    double poly = dfma(YASNCS(n + top), xx, YASNCS(n + top - 1));
    for (int j = top - 2; j > 1; --j) poly = dfma(poly, xx, YASNCS(n + j));
    const double p = dfma(dmul(xx, xx), poly, YASNCS(n + top + 1));
    const double t = dfma(YASNCS(n + 1), xx, p);
    return dadd(t, YASNCS(n + top + 2));
}

static __device__ __noinline__ double g_asin(double x) {
    // e_asin.c's __ieee754_asin; the wrapper asin adds only errno for |x| > 1
    const uint64_t u = dbits(x);
    const int32_t m = (int32_t)(uint32_t)(u >> 32);                // the high word, signed
    const uint32_t k = (uint32_t)(u >> 32) & 0x7fffffffu;
    const uint32_t lo = (uint32_t)u;
    if (k < YI_ASIN_K_TINY) return x;                                // |x| < 2^-26
    if (k < YI_ASIN_K_TAYLOR) {                                      // |x| < 0.125
        const double x2 = dmul(x, x);
        double p = dfma(x2, YKD(YI_ASIN_F6), YKD(YI_ASIN_F5));
        const double t = dmul(x, x2);
        p = dfma(p, x2, YKD(YI_ASIN_F4));
        p = dfma(p, x2, YKD(YI_ASIN_F3));
        p = dfma(p, x2, YKD(YI_ASIN_F2));
        p = dfma(p, x2, YKD(YI_ASIN_F1));
        return dfma(p, t, x);
    }
    const double xa = m > 0 ? x : -x;
    double res;
    if (k < YI_ASIN_K_HALF) {                                        // 0.125 <= |x| < 0.5
        const int n = k < YI_ASIN_K_QUARTER ? YI_ASIN_N_Q1_MUL * (int)((k >> 15) & 0x1fu)
                                            : YI_ASIN_N_Q1_MUL * (int)((k >> 14) & 0x3fu) + YI_ASIN_N_Q2_OFF;
        res = asin_table(xa, n, 6);
    } else if (k < YI_ASIN_K_075) {                                  // 0.5 <= |x| < 0.75
        res = asin_table(xa, YI_ASIN_N_H_OFF + 3 * (int)((k >> 11) & 0x1fcu), 7);
    } else if (k < YI_ASIN_K_0921875) {                              // 0.75 <= |x| < 0.921875
        res = asin_table(xa, YI_ASIN_N_A_OFF + YI_ASIN_N_A_MUL * (int)((k >> 13) & 0x7fu), 8);
    } else if (k < YI_ASIN_K_0953125) {                              // 0.921875 <= |x| < 0.953125
        res = asin_table(xa, YI_ASIN_N_B_OFF + 14 * (int)((k >> 13) & 0x7fu), 9);
    } else if (k < YI_ASIN_K_096875) {                               // 0.953125 <= |x| < 0.96875
        res = asin_table(xa, YI_ASIN_N_C_OFF + 15 * (int)((k >> 13) & 0x7fu), 10);
    } else if (k < YI_ASIN_K_NEAR_ONE) {                             // 0.96875 <= |x| < 1
        const double z = dmul(m > 0 ? dsub(YKD(YI_ASIN_ONE), x) : dadd(x, YKD(YI_ASIN_ONE)), YKD(YI_ASIN_HALF));
        const uint64_t zb = dbits(z);
        double t = dmul(dfrom(YI_ASIN_INROOT[(zb >> 46) & 0x7fULL]),
                        dfrom(YI_ASIN_POWTWO[YI_ASIN_POWTWO_TOP - (int)(zb >> 53)]));
        double p = dfma(z, YKD(YI_ASIN_F6), YKD(YI_ASIN_F5));
        p = dfma(p, z, YKD(YI_ASIN_F4));
        const double r = dfmsub(dmul(t, t), z, YKD(YI_ASIN_ONE));           // 1 - t t z
        p = dfma(p, z, YKD(YI_ASIN_F3));
        double q = dfma(r, YKD(YI_ASIN_RT3), YKD(YI_ASIN_RT2));
        p = dfma(p, z, YKD(YI_ASIN_F2));
        q = dfma(q, r, YKD(YI_ASIN_RT1));
        p = dfma(p, z, YKD(YI_ASIN_F1));
        q = dfma(q, r, YKD(YI_ASIN_RT0));
        p = dmul(p, z);
        t = dmul(q, t);
        const double c = dmul(z, t);
        const double w = dfmsub(dmul(t, YKD(YI_ASIN_HALF)), c, YKD(YI_ASIN_THREE_HALVES));   // 1.5 - 0.5 t c
        const double y = dsub(dadd(c, YKD(YI_ASIN_T24)), YKD(YI_ASIN_T24));
        const double ty = dfma(w, c, y);                                    // t + y, t = c (1.5 - 0.5 t c)
        const double e = dfmsub(y, y, z);                                   // z - y y
        const double res1 = dfmsub(y, YKD(YI_ASIN_TWO), YKD(YI_ASIN_HP0)); // hp0 - 2 y
        const double cc = ddiv(e, ty);
        double s = dadd(y, cc);
        double cor = dfmsub(cc, YKD(YI_ASIN_TWO), YKD(YI_ASIN_HP1));       // hp1 - 2 cc
        s = dadd(s, s);
        cor = dfmsub(s, p, cor);                                            // (hp1 - 2 cc) - 2 (y + cc) p
        res = dadd(cor, res1);
    } else if (k == YI_ASIN_K_ONE && lo == 0u) {                     // |x| = 1
        return m > 0 ? YKD(YI_ASIN_HP0) : YKD(YI_ASIN_NEG_HP0);
    } else if (k > YI_ASIN_K_INF || (k == YI_ASIN_K_INF && lo != 0u)) {
        return dadd(x, x);                                           // NaN
    } else {
        return dnan();                                               // |x| > 1 and the infinities
    }
    return m > 0 ? res : -res;
}

#undef YASNCS

// ============================================================================= the C library's double atan2
// Transcribed from exact_ballpark.dlibm_inv (glibc 2.34 aarch64's __ieee754_atan2, 0x30410; e_atan2.c).

#define YCIJ(i, j) dfrom(YI_ATAN2_CIJ[7 * (i) + (j)])

static __device__ __forceinline__ double signed_(double z, double y) {
    // signArctan2(y, z): z's magnitude with y's sign
    return dfrom((dbits(y) & 0x8000000000000000ULL) | (dbits(z) & 0x7fffffffffffffffULL));
}

static __device__ __forceinline__ int cij_index(double u) {
    // (int) ((TWO52 + TWO8 u) - TWO52) - 16, the sum fused (an integer below 2^52: the truncation is exact)
    const double t = dsub(dfma(u, YKD(YI_ATAN2_TWO8), YKD(YI_ATAN2_TWO52)), YKD(YI_ATAN2_TWO52));
    return __double2int_rz(t) - YI_ATAN2_I_OFF;
}

static __device__ __forceinline__ double small_poly(double v) {
    // d3 + v (d5 + v (d7 + v (d9 + v (d11 + v d13)))), every step fused
    double p = dfma(v, YKD(YI_ATAN2_D13), YKD(YI_ATAN2_D11));
    p = dfma(p, v, YKD(YI_ATAN2_D9));
    p = dfma(p, v, YKD(YI_ATAN2_D7));
    p = dfma(p, v, YKD(YI_ATAN2_D5));
    return dfma(p, v, YKD(YI_ATAN2_D3));
}

static __device__ __forceinline__ double table_poly(int i, double v) {
    // cij[i][2] + v (cij[i][3] + v (cij[i][4] + v (cij[i][5] + v cij[i][6]))), every step fused
    double p = dfma(YCIJ(i, 6), v, YCIJ(i, 5));
    p = dfma(p, v, YCIJ(i, 4));
    p = dfma(p, v, YCIJ(i, 3));
    return dfma(p, v, YCIJ(i, 2));
}

static __device__ __noinline__ double g_atan2(double y, double x) {
    // e_atan2.c's __ieee754_atan2 (round to nearest); the wrapper atan2 adds only errno
    const uint64_t bx = dbits(x), by = dbits(y);
    const int32_t ux = (int32_t)(uint32_t)(bx >> 32), uy = (int32_t)(uint32_t)(by >> 32);
    const uint32_t dx = (uint32_t)bx, dy = (uint32_t)by;
    const int32_t inf_hi = 0x7ff00000, minf_hi = (int32_t)0xfff00000u, mzero_hi = (int32_t)0x80000000u;
    if ((ux & inf_hi) == inf_hi && (((uint32_t)ux & 0xfffffu) | dx) != 0u) return dadd(x, y);   // x NaN
    const bool y_inf = (uy & inf_hi) == inf_hi;
    if (y_inf && (((uint32_t)uy & 0xfffffu) | dy) != 0u) return dadd(y, y);                     // y NaN
    if (!y_inf && uy == 0) {
        if (dy == 0u) return ux < 0 ? YKD(YI_ATAN2_OPI) : 0.0;                                   // y = +0
        if (x == 0.0) return YKD(YI_ATAN2_HPI);
    } else {
        if (dy == 0u && uy == mzero_hi) return ux < 0 ? YKD(YI_ATAN2_MOPI) : -0.0;               // y = -0
        if (x == 0.0) return uy < 0 ? YKD(YI_ATAN2_MHPI) : YKD(YI_ATAN2_HPI);                    // x = +-0
    }
    if (ux == inf_hi && dx == 0u) {                                                              // x = +inf
        if (uy == inf_hi && dy == 0u) return YKD(YI_ATAN2_QPI);
        if (uy == minf_hi && dy == 0u) return YKD(YI_ATAN2_MQPI);
        return uy < 0 ? -0.0 : 0.0;
    }
    if (ux == minf_hi && dx == 0u) {                                                             // x = -inf
        if (uy == inf_hi && dy == 0u) return YKD(YI_ATAN2_TQPI);
        if (uy == minf_hi && dy == 0u) return YKD(YI_ATAN2_MTQPI);
        return uy < 0 ? YKD(YI_ATAN2_MOPI) : YKD(YI_ATAN2_OPI);
    }
    if (uy == inf_hi) return YKD(YI_ATAN2_HPI);                                                  // y = +inf
    if (uy == minf_hi && dy == 0u) return YKD(YI_ATAN2_MHPI);                                    // y = -inf
    double ax = x < 0.0 ? -x : x;
    double ay = y < 0.0 ? -y : y;
    const int32_t de = (uy & inf_hi) - (ux & inf_hi);
    if (de >= YI_ATAN2_EP) return y > 0.0 ? YKD(YI_ATAN2_HPI) : YKD(YI_ATAN2_MHPI);
    if (de <= YI_ATAN2_EM) {
        if (x > 0.0) return signed_(ddiv(ay, ax), y);
        return y > 0.0 ? YKD(YI_ATAN2_OPI) : YKD(YI_ATAN2_MOPI);
    }
    if (ax < YKD(YI_ATAN2_TWOM500) || ay < YKD(YI_ATAN2_TWOM500)) {
        ax = dmul(ax, YKD(YI_ATAN2_TWO500));
        ay = dmul(ay, YKD(YI_ATAN2_TWO500));
    }
    if (ax > YKD(YI_ATAN2_TWO500) || ay > YKD(YI_ATAN2_TWO500)) {
        ax = dmul(ax, YKD(YI_ATAN2_TWOM500));
        ay = dmul(ay, YKD(YI_ATAN2_TWOM500));
    }
    double u, du;
    if (ay < ax) {                                                   // u = ay / ax, its error by EMULV (fused)
        u = ddiv(ay, ax);
        const double v = dmul(ax, u);
        const double vv = dfnmsub(ax, u, v);
        du = ddiv(dsub(dsub(ay, v), vv), ax);
    } else {                                                         // u = ax / ay
        u = ddiv(ax, ay);
        const double v = dmul(ay, u);
        const double vv = dfnmsub(ay, u, v);
        du = ddiv(dsub(dsub(ax, v), vv), ay);
    }
    if (x > 0.0) {
        if (ay < ax) {                                               // (i) atan(ay / ax)
            if (u < YKD(YI_ATAN2_INV16)) {
                const double v = dmul(u, u);
                const double zz = dfma(dmul(u, v), small_poly(v), du);
                return signed_(dadd(u, zz), y);
            }
            const int i = cij_index(u);
            const double t3 = dsub(u, YCIJ(i, 0));
            const double v = dadd(du, t3);                           // EADD(t3, du, v, dv)
            const double dv = dabs_(t3) > dabs_(du) ? dadd(dsub(t3, v), du) : dadd(dsub(du, v), t3);
            double p = dfma(YCIJ(i, 6), v, YCIJ(i, 5));
            p = dfma(p, v, YCIJ(i, 4));
            p = dfma(p, v, YCIJ(i, 3));
            double zz = dmul(dmul(v, v), p);
            zz = dfma(dv, YCIJ(i, 2), zz);
            zz = dfma(v, YCIJ(i, 2), zz);
            return signed_(dadd(zz, YCIJ(i, 1)), y);
        }
        if (u < YKD(YI_ATAN2_INV16)) {                               // (ii) pi/2 - atan(ax / ay)
            const double v = dmul(u, u);
            const double zz = dmul(dmul(u, v), small_poly(v));
            const double t2 = dsub(YKD(YI_ATAN2_HPI), u);            // ESUB(hpi, u, t2, cor)
            const double cor = dabs_(u) < YKD(YI_ATAN2_HPI) ? dsub(dsub(YKD(YI_ATAN2_HPI), t2), u)
                                                            : dsub(YKD(YI_ATAN2_HPI), dadd(u, t2));
            return signed_(dadd(dsub(dsub(dadd(cor, YKD(YI_ATAN2_HPI1)), du), zz), t2), y);
        }
        const int i = cij_index(u);
        const double v = dadd(dsub(u, YCIJ(i, 0)), du);
        return signed_(dadd(dsub(YKD(YI_ATAN2_HPI), YCIJ(i, 1)), dfmsub(table_poly(i, v), v, YKD(YI_ATAN2_HPI1))), y);
    }
    if (ax < ay) {                                                   // (iii) pi/2 + atan(ax / ay)
        if (u < YKD(YI_ATAN2_INV16)) {
            const double v = dmul(u, u);
            const double zz = dmul(dmul(v, u), small_poly(v));
            const double t2 = dadd(u, YKD(YI_ATAN2_HPI));            // EADD(hpi, u, t2, cor)
            const double cor = dabs_(u) < YKD(YI_ATAN2_HPI) ? dadd(dsub(YKD(YI_ATAN2_HPI), t2), u)
                                                            : dadd(dsub(u, t2), YKD(YI_ATAN2_HPI));
            return signed_(dadd(dadd(dadd(dadd(cor, YKD(YI_ATAN2_HPI1)), du), zz), t2), y);
        }
        const int i = cij_index(u);
        const double v = dadd(dsub(u, YCIJ(i, 0)), du);
        return signed_(dadd(dadd(YCIJ(i, 1), YKD(YI_ATAN2_HPI)), dfma(table_poly(i, v), v, YKD(YI_ATAN2_HPI1))), y);
    }
    if (u < YKD(YI_ATAN2_INV16)) {                                   // (iv) pi - atan(ay / ax)
        const double v = dmul(u, u);
        const double zz = dmul(dmul(v, u), small_poly(v));
        const double t2 = dsub(YKD(YI_ATAN2_OPI), u);                // ESUB(opi, u, t2, cor)
        const double cor = dabs_(u) < YKD(YI_ATAN2_OPI) ? dsub(dsub(YKD(YI_ATAN2_OPI), t2), u)
                                                        : dsub(YKD(YI_ATAN2_OPI), dadd(t2, u));
        return signed_(dadd(dsub(dsub(dadd(cor, YKD(YI_ATAN2_OPI1)), du), zz), t2), y);
    }
    const int i = cij_index(u);
    const double v = dadd(dsub(u, YCIJ(i, 0)), du);
    return signed_(dadd(dsub(YKD(YI_ATAN2_OPI), YCIJ(i, 1)), dfmsub(table_poly(i, v), v, YKD(YI_ATAN2_OPI1))), y);
}

#undef YCIJ

