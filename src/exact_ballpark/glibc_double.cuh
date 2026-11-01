// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's sin, cos, sincos (IBM Accurate Mathematical Library), log and exp (Arm) for the modes kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_modes_cuda.cu; a build inlines it (_cuda_build.read_source).

// ======================================================================== the C library's double functions
// Transcribed from exact_ballpark.dlibm (glibc 2.34 aarch64's __sin, __cos, __branred, __log) and exact_ballpark.vec_exp
// (__exp): their operations in their order, each fused step fused.

static __device__ __forceinline__ double taylor_sin(double a, double da) {
    // TAYLOR_SIN(a a, a, da): a + ((POLY(a a) a - da / 2) a a + da)
    const double xx = dmul(a, a);
    const double poly = dfma(xx, dfma(xx, dfma(xx, dfma(xx, KD(DL_S5), KD(DL_S4)), KD(DL_S3)), KD(DL_S2)), KD(DL_S1));
    const double t = dfnmsub(poly, a, dmul(da, 0.5));
    return dadd(dfma(xx, t, da), a);
}

static __device__ __forceinline__ double sincos_table(double x, double* sn, double* ssn, double* cs, double* ccs) {
    // u = big + |x|; the table row of u's low word; returns |x| - (u - big)
    const double ax = dabs_(x);
    const double u = dadd(KD(DL_BIG), ax);
    const int k = (int)((uint32_t)(dbits(u) & 0xffffffffULL) << 2);
    *sn = dfrom(DL_SINCOSTAB[k]);
    *ssn = dfrom(DL_SINCOSTAB[k + 1]);
    *cs = dfrom(DL_SINCOSTAB[k + 2]);
    *ccs = dfrom(DL_SINCOSTAB[k + 3]);
    return dsub(ax, dsub(u, KD(DL_BIG)));
}

static __device__ __noinline__ double do_sin(double x, double dx) {
    // do_sin: the Taylor polynomial below 0.126, else the table combined with short polynomials; the sign of x
    if (dabs_(x) < KD(DL_TAYLOR_LIMIT)) return taylor_sin(x, dx);
    if (x <= 0.0) dx = -dx;
    double sn, ssn, cs, ccs;
    const double r = sincos_table(x, &sn, &ssn, &cs, &ccs);
    const double xx = dmul(r, r);
    const double pc = dfma(xx, KD(DL_CS6), KD(DL_CS4));
    const double ps = dfma(xx, KD(DL_SN5), KD(DL_SN3));
    const double r3 = dmul(r, xx);
    const double c1 = dfma(xx, pc, KD(DL_CS2));
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
    const double pc = dfma(xx, KD(DL_CS6), KD(DL_CS4));
    const double ps = dfma(xx, KD(DL_SN5), KD(DL_SN3));
    const double r3 = dmul(r, xx);
    const double c1 = dfma(xx, pc, KD(DL_CS2));
    const double s = dfma(r3, ps, r);
    const double c = dmul(xx, c1);
    const double cor = dfmsub(s, sn, dfmsub(c, cs, dfmsub(s, ssn, ccs)));
    return dadd(cs, cor);
}

static __device__ __forceinline__ int reduce_sincos(double x, double* a, double* da) {
    // reduce_sincos, fused as compiled: x = n pi/2 + a + da
    const double t = dfma(x, KD(DL_HPINV), KD(DL_TOINT));
    const double xn = dsub(t, KD(DL_TOINT));
    const double y = dfmsub(xn, KD(DL_MP2), dfmsub(xn, KD(DL_MP1), x));
    const double t2 = dfmsub(xn, KD(DL_PP3), y);
    const double b = dfmsub(xn, KD(DL_PP4), t2);
    double db = dfmsub(xn, KD(DL_PP3), dsub(y, t2));
    db = dadd(db, dfmsub(xn, KD(DL_PP4), dsub(t2, b)));
    *a = b;
    *da = db;
    return (int)(dbits(t) & 3ULL);
}

static __device__ __forceinline__ void branred_half(double xh, double* sum_out, double* b_out, double* bb_out) {
    // one half of branred: xh 2/pi from six 24-bit pieces of toverp, the integer part split off (nothing fused)
    const int e = (int)((dbits(xh) >> 52) & 0x7ffULL);
    int k = 0;
    double first = KD(DL_BR_TOVERP0);
    if (e > DL_BR_EXP_LIMIT) {
        k = (e - DL_BR_EXP_BIAS) / 24;                              // C's division: toward zero
        first = dfrom(DL_BR_TOVERP[k]);
    }
    double gor = dfrom((uint64_t)((uint32_t)DL_BR_T576_HI - ((uint32_t)(3 * k) << 23)) << 32);
    double r[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        r[i] = dmul(dmul(xh, i == 0 ? first : dfrom(DL_BR_TOVERP[k + i])), gor);
        gor = dmul(gor, KD(DL_BR_TM24));
    }
    double total = 0.0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const double s = dsub(dadd(r[i], KD(DL_BR_BIG)), KD(DL_BR_BIG));
        total = dadd(total, s);
        r[i] = dsub(r[i], s);
    }
    double t = 0.0;
#pragma unroll
    for (int i = 0; i < 6; ++i) t = dadd(t, r[5 - i]);
    double bb = dadd(dadd(dadd(dadd(dadd(dsub(r[0], t), r[1]), r[2]), r[3]), r[4]), r[5]);
    double s = dsub(dadd(t, KD(DL_BR_BIG)), KD(DL_BR_BIG));
    total = dadd(s, total);
    t = dsub(t, s);
    s = dsub(dadd(total, KD(DL_BR_BIG1)), KD(DL_BR_BIG1));
    total = dsub(total, s);
    const double b = dadd(bb, t);
    bb = dadd(dsub(t, b), bb);
    *sum_out = total;
    *b_out = b;
    *bb_out = bb;
}

static __device__ __noinline__ int branred(double x, double* a, double* aa) {
    // __branred: x = n pi/2 + a + aa for huge x, the argument scaled by 2^-600 and split Dekker's way
    x = dmul(x, KD(DL_BR_TM600));
    const double t0 = dmul(x, KD(DL_BR_SPLIT));
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
    const double sp = dmul(s, KD(DL_BR_SPLIT));
    const double t1 = dsub(sp, dsub(sp, s));
    const double t2 = dsub(s, t1);
    b = dmul(s, KD(DL_BR_HP0));
    const double bb2x = dadd(dadd(dadd(dsub(dmul(t1, KD(DL_BR_MP1)), b), dmul(t1, KD(DL_BR_MP2))), dmul(t2, KD(DL_BR_MP1))),
                             dadd(dadd(dmul(t2, KD(DL_BR_MP2)), dmul(s, KD(DL_BR_HP1))), dmul(t, KD(DL_BR_HP0))));
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
    if (k < DL_K_TINY_SIN) return x;
    if (k < DL_K_DO_SIN) return do_sin(x, 0.0);
    if (k < DL_K_MID) return dcopysign(do_cos(dsub(KD(DL_HP0), dabs_(x)), KD(DL_HP1)), x);
    double a, da;
    if (k < DL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        return sincos_n(a, da, n);
    }
    if (k < DL_K_HUGE) {
        const int n = branred(x, &a, &da);
        return sincos_n(a, da, n);
    }
    return dnan();
}

static __device__ __noinline__ double g_cos(double x) {
    // s_sin.c's __cos (glibc 2.34 aarch64, 0x369f0)
    const uint32_t k = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    if (k < DL_K_TINY_COS) return 1.0;
    if (k < DL_K_DO_SIN) return do_cos(x, 0.0);
    if (k < DL_K_MID) {
        const double y = dsub(KD(DL_HP0), dabs_(x));
        const double a = dadd(y, KD(DL_HP1));
        return do_sin(a, dadd(dsub(y, a), KD(DL_HP1)));
    }
    double a, da;
    if (k < DL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        return sincos_n(a, da, n + 1);
    }
    if (k < DL_K_HUGE) {
        const int n = branred(x, &a, &da);
        return sincos_n(a, da, n + 1);
    }
    return dnan();
}

static __device__ __noinline__ void g_sincos(double x, double* s, double* c) {
    // s_sincos.c's __sincos (glibc 2.34 aarch64, 0x37ca0; exact_ballpark.dlibm.sincos): __sin's and __cos's branches, but
    // between 0.855469 and 2.426265 the sine is do_cos after the two-sum, as __cos's cosine there is do_sin (it differs from __sin at +-0x1.ad1fb54442d18p+0 only); compiled Destiny takes a sine and a cosine of one
    // argument from sincos
    const uint32_t k = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    if (k < DL_K_TINY_COS) {
        *s = x;
        *c = 1.0;
        return;
    }
    if (k < DL_K_DO_SIN) {
        *s = do_sin(x, 0.0);
        *c = do_cos(x, 0.0);
        return;
    }
    if (k < DL_K_MID) {
        const double y = dsub(KD(DL_HP0), dabs_(x));
        const double a = dadd(y, KD(DL_HP1));
        const double da = dadd(dsub(y, a), KD(DL_HP1));
        *s = dcopysign(do_cos(a, da), x);
        *c = do_sin(a, da);
        return;
    }
    double a, da;
    if (k < DL_K_REDUCE) {
        const int n = reduce_sincos(x, &a, &da);
        *s = sincos_n(a, da, n);
        *c = sincos_n(a, da, n + 1);
        return;
    }
    if (k < DL_K_HUGE) {
        const int n = branred(x, &a, &da);
        *s = sincos_n(a, da, n);
        *c = sincos_n(a, da, n + 1);
        return;
    }
    *s = dnan();
    *c = dnan();
}

static __device__ __noinline__ double g_log(double x) {
    // e_log.c's __log (glibc 2.34 aarch64, 0x33610): near 1 the degree-11 polynomial, else the table and degree 5
    uint64_t ix = dbits(x);
    if (ix + DL_LOG_NEG_LO <= DL_LOG_RANGE) {                         // 1 - 2^-4 <= x < 1 + 0x1.09p-4
        if (ix == 0x3ff0000000000000ULL) return 0.0;
        const double r = dsub(x, 1.0);
        double rhi = dfma(r, KD(DL_LOG_TWO27), r);                    // r + w, w = r 2^27
        const double r2 = dmul(r, r);
        const double p78 = dfma(dfrom(DL_LOG_B[8]), r, dfrom(DL_LOG_B[7]));
        rhi = dfmsub(r, KD(DL_LOG_TWO27), rhi);                       // (r + w) - w
        const double r3 = dmul(r, r2);
        const double p789 = dfma(dfrom(DL_LOG_B[9]), r2, p78);
        const double p45 = dfma(dfrom(DL_LOG_B[5]), r, dfrom(DL_LOG_B[4]));
        const double p710 = dfma(dfrom(DL_LOG_B[10]), r3, p789);
        const double rhi2 = dmul(rhi, rhi);
        const double p456 = dfma(dfrom(DL_LOG_B[6]), r2, p45);
        const double b0 = dfrom(DL_LOG_B[0]);
        const double hi = dfma(rhi2, b0, r);                          // r + rhi rhi B0
        const double p12 = dfma(dfrom(DL_LOG_B[2]), r, dfrom(DL_LOG_B[1]));
        const double p410 = dfma(p710, r3, p456);
        const double rlo = dsub(r, rhi);
        const double rr = dadd(r, rhi);
        const double p123 = dfma(dfrom(DL_LOG_B[3]), r2, p12);
        const double b0rlo = dmul(b0, rlo);
        double lo = dsub(r, hi);
        lo = dfma(rhi2, b0, lo);
        const double p = dfma(p410, r3, p123);
        lo = dfma(b0rlo, rr, lo);
        const double y = dfma(p, r3, lo);
        return dadd(hi, y);
    }
    const uint32_t top = (uint32_t)(ix >> 48);
    if (top - 0x0010u >= 0x7fe0u) {                                   // below 2^-1022, negative, infinite or NaN
        if ((ix << 1) == 0) return -dinf();
        if (ix == 0x7ff0000000000000ULL) return x;
        if ((top & 0x8000u) || (top & 0x7ff0u) == 0x7ff0u) return dnan();
        ix = dbits(dmul(x, KD(DL_LOG_TWO52))) + DL_LOG_NEG_52;        // a subnormal x normalised by 2^52
    }
    const uint64_t tmp = ix + DL_LOG_NEG_OFF;
    const int i = (int)((tmp >> 45) & 127ULL);
    const int k = (int)((int64_t)tmp >> 52);
    const uint64_t iz = ix - (tmp & 0xfff0000000000000ULL);
    const double kd = (double)k;                                      // exact
    const double invc = dfrom(DL_LOG_INVC[i]), logc = dfrom(DL_LOG_LOGC[i]);
    const double w = dfma(KD(DL_LOG_LN2HI), kd, logc);
    const double r = dfma(dfrom(iz), invc, -1.0);
    const double hi = dadd(r, w);
    const double p34 = dfma(dfrom(DL_LOG_A[4]), r, dfrom(DL_LOG_A[3]));
    const double p12 = dfma(dfrom(DL_LOG_A[2]), r, dfrom(DL_LOG_A[1]));
    const double r2 = dmul(r, r);
    double lo = dsub(w, hi);
    const double r3 = dmul(r, r2);
    lo = dadd(lo, r);
    const double p = dfma(p34, r2, p12);
    lo = dfma(KD(DL_LOG_LN2LO), kd, lo);
    lo = dfma(dfrom(DL_LOG_A[0]), r2, lo);
    lo = dfma(r3, p, lo);
    return dadd(lo, hi);
}

static __device__ __noinline__ double g_exp(double x) {
    // e_exp.c's __exp (glibc 2.34 aarch64, 0x30e70) as exact_ballpark.vec_exp transcribes it
    const uint64_t ix = dbits(x);
    const uint32_t abstop = (uint32_t)((ix >> 52) & 0x7ffULL);
    if (abstop < 0x3c9u) return dadd(x, 1.0);                         // |x| < 2^-54
    if (abstop > 0x408u) {                                            // |x| >= 1024, infinities, NaN
        if (ix == 0xfff0000000000000ULL) return 0.0;
        if (abstop == 0x7ffu) return dadd(x, 1.0);
        return (ix >> 63) ? 0.0 : dinf();                             // __math_uflow, __math_oflow
    }
    const double z = dmul(x, KD(K_EXP_INVLN2N));
    const double kd = frinta(z);
    const int64_t ki = __double2ll_rz(kd);
    double r = dfma(kd, KD(K_EXP_NEGLN2HIN), x);
    r = dfma(kd, KD(K_EXP_NEGLN2LON), r);
    const int idx = (int)((ki & 0x7f) * 2);
    const double tail = dfrom(T_EXP[idx]);
    const uint64_t sbits = T_EXP[idx + 1] + ((uint64_t)ki << 45);
    const double p23 = dfma(r, KD(K_EXP_C3), KD(K_EXP_C2));
    const double p45 = dfma(r, KD(K_EXP_C5), KD(K_EXP_C4));
    const double r2 = dmul(r, r);
    double tmp = dadd(r, tail);
    const double r4 = dmul(r2, r2);
    tmp = dfma(p23, r2, tmp);
    tmp = dfma(r4, p45, tmp);
    if (abstop != 0x408u) {
        const double scale = dfrom(sbits);
        return dfma(tmp, scale, scale);                               // scale + tmp scale
    }
    if ((ki & 0x80000000LL) == 0) {                                   // specialcase, k > 0: may overflow
        const double sc_hi = dfrom(sbits - (1009ULL << 52));
        return dmul(dfma(tmp, sc_hi, sc_hi), KD(K_TWO_1009));
    }
    const double sc_lo = dfrom(sbits + (1022ULL << 52));              // k < 0: may be subnormal
    const double st = dmul(tmp, sc_lo);                               // not fused
    const double y = dadd(sc_lo, st);
    if (!(y < 1.0)) return dmul(y, KD(K_TWO_M1022));
    double lo = dadd(dsub(sc_lo, y), st);
    const double hi = dadd(y, 1.0);
    lo = dadd(dadd(dsub(1.0, hi), y), lo);
    double y2 = dsub(dadd(lo, hi), 1.0);
    if (y2 == 0.0) y2 = 0.0;
    return dmul(y2, KD(K_TWO_M1022));
}

