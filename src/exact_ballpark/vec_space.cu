// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's space partition on the master, its bubbles, proximity sensors and the visibility and range queries, for
// many independent worlds: one sequential algorithm a world, the transcription of exact_ballpark.space.SpaceBallpark's
// code (Carbon's destiny at 114e89fa, MIT, (c) CCP Games), compiled twice from this one source: as host functions (a
// loop over the worlds: the CPU path) and as CUDA kernels (one thread a world). Built and loaded by
// exact_ballpark.vec_space; held bit for bit to the scalar reference (exact_ballpark.full.FullBallpark, master) by
// tests/test_movement_vec_space.py on the CPU and the GPU.
//
// Exactness by construction. Every double operation is __dadd_rn ... __dsqrt_rn on the device and the plain IEEE
// operator on the host (nvcc -fmad=false; the host compiler -ffp-contract=off -fno-fast-math, so nothing is fused);
// float sums are binary32 operations (__fadd_rn on the device, float + float on the host: aarch64 evaluates float
// in float). Divisions are IEEE divisions (no reciprocal). No function of CUDA's math library is called; no
// transcendental is needed (sqrt only). Every constant comes from the generated header vec_space_tables.h
// (exact_ballpark.vec_space.tables_header: the partition's level widths and grids from exact_ballpark.space, the modes'
// codes, the thresholds), never typed here.
//
// What the state holds (exact_ballpark.vec_space for the layout): per slot of a world the ball's filing (level, key of
// the box holding its centre, the side on each axis its inflated sphere reaches: -1, 0, +1), its bubble and old
// bubble, whether it is in its bubble's interactive and keep-alive sets, its sensor, tracking and notification
// state and its sensor's list (libstdc++'s unordered_map order, with the map's bucket count and the rehash policy's
// next resize); per world the top boxes that exist (a top exists
// while some ball's filing reaches it: a reference count) with their bubble ids, the bubble dictionaries' entries
// (bubble, ball, +1 / 0 / -1, insertion order), which dictionaries exist, the bubble id counter, the proximity
// balls (unordered_map order, its bucket count and next resize), the tick's events. Boxes below the top level are not stored: a box exists while a
// ball is filed in it or below it, so GetNearbyBalls' walk follows from the filings (exact_ballpark.vec_collide's
// order key, here per pair).

#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "vec_space_tables.h"

#define XBPS_ABI 4
#define HD __host__ __device__

namespace xbps {

// ======================================================================================== arithmetic, rounding pinned

static HD inline double dadd(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
static HD inline double dsub(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dsub_rn(a, b);
#else
    return a - b;
#endif
}
static HD inline double dmul(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}
static HD inline double ddiv(double a, double b) {
#ifdef __CUDA_ARCH__
    return __ddiv_rn(a, b);
#else
    return a / b;
#endif
}
static HD inline double dsqrt(double a) {
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(a);
#else
    return __builtin_sqrt(a);
#endif
}
static HD inline float tof(double x) {             // f32(x): to the nearest float
#ifdef __CUDA_ARCH__
    return __double2float_rn(x);
#else
    return (float)x;
#endif
}
static HD inline double tod(float x) { return (double)x; }
static HD inline double f32(double x) { return tod(tof(x)); }
static HD inline double fsum(double a, double b) {  // float + float (two float values held as doubles)
#ifdef __CUDA_ARCH__
    return tod(__fadd_rn(tof(a), tof(b)));
#else
    const float r = tof(a) + tof(b);
    return tod(r);
#endif
}
static HD inline uint64_t dbits(double x) {
#ifdef __CUDA_ARCH__
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t u;
    __builtin_memcpy(&u, &x, 8);
    return u;
#endif
}
static HD inline double dfrom(uint64_t u) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double((long long)u);
#else
    double x;
    __builtin_memcpy(&x, &u, 8);
    return x;
#endif
}
#define KD(name) dfrom(name)

static HD inline double lensq3(double x, double y, double z) {   // _lensq: x*x + y*y + z*z, left to right
    return dadd(dadd(dmul(x, x), dmul(y, y)), dmul(z, z));
}

static HD inline int64_t to_int64(double x) {       // _to_int64: toward zero, saturating, NaN to 0 (aarch64 fcvtzs)
    if (x != x) return 0;
    if (x >= KD(K_TWO63)) return INT64_MAX;
    if (x < -KD(K_TWO63)) return INT64_MIN;
    return (int64_t)x;
}

static HD inline double level_width(int64_t i) {
    switch (i) {
        case 0: return KD(K_LW0); case 1: return KD(K_LW1); case 2: return KD(K_LW2); case 3: return KD(K_LW3);
        case 4: return KD(K_LW4); case 5: return KD(K_LW5); case 6: return KD(K_LW6); default: return KD(K_LW7);
    }
}
static HD inline int64_t level_grid(int64_t i) {
    switch (i) {
        case 0: return K_LG0; case 1: return K_LG1; case 2: return K_LG2; case 3: return K_LG3;
        case 4: return K_LG4; case 5: return K_LG5; case 6: return K_LG6; default: return K_LG7;
    }
}

// ============================================================================================ the park and the state

struct Park {                    // the batched park's rows (row-indexed), read and in part written
    int64_t R;
    const int64_t* ids;
    double* pos;                 // [R,3]
    double* vel;                 // [R,3]
    const double* old_pos;       // [R,3] the position before the commit (the warp's drop-out point)
    const double* old_vel;       // [R,3]
    double* radius;              // [R] a float value
    const double* max_vel;       // [R] float values
    const double* agility;
    const double* sf;
    const double* mass;          // [R] doubles
    const double* tf;
    const uint8_t* free_;        // [R] bool
    const uint8_t* massive;
    uint8_t* interactive;
    const int64_t* cloaked;
    uint8_t* moribund;
    const int64_t* mode;
    const int64_t* follow_id;    // [R] the followed ball's Destiny id
    const double* follow_range;  // [R] a float value
    const double* goto_;         // [R,3]
    uint8_t* sensor;             // [R] mSensor.active
    const int64_t* effect_stamp; // [R] (warp proper: WARP with a stamp >= 0)
    int64_t* bubble_out;         // [R] the park's own bubble field (null: none)
    const uint8_t* global_;      // [R] the global flag (null: no global balls)
    double friction, dt;
};

struct Space {
    int64_t W, N, T, E, D, M, S, I;
    const int64_t* rows;         // [W,N] the row of each slot, -1 for an empty slot
    const int64_t* by_id;        // [W,N] the slots in ascending id, -1 padded
    int64_t* lvl;                // [W,N] the filing's level, -1: no box
    int64_t* key;                // [W,N,3]
    int64_t* mod;                // [W,N,3]
    int64_t* nbox;               // [W,N] boxes the filing holds, its ancestors counted
    int64_t* bubble;             // [W,N] Partitionable's mNewBubble
    int64_t* old_bubble;         // [W,N] mOldBubble
    uint8_t* in_park;            // [W,N] in the park's ball dictionary
    uint8_t* ka_ball;            // [W,N] in bubbleKeepAliveBalls
    int64_t* bseq;               // [W,N] its place in the ball dictionary
    int64_t* tkey;               // [W,T,3] the top boxes that exist
    int64_t* tbub;               // [W,T]
    int64_t* tref;               // [W,T] filings reaching it
    int64_t* tn;                 // [W]
    int64_t* ebub;               // [W,E] the bubble dictionaries' entries
    int64_t* eslot;
    int64_t* eact;
    int64_t* eseq;
    int64_t* en;                 // [W]
    int64_t* eseq_next;          // [W]
    int64_t* dbub;               // [W,D] the dictionaries that exist
    int64_t* dn;                 // [W]
    int64_t* iab;                // [W,I] bubbleInteractives as (bubble, slot) pairs (a ball may be in several:
    int64_t* ias;                //       UpdateBallBubble moves its bubble without the sets)
    int64_t* ian;                // [W]
    int64_t* kab;                // [W,I] bubbleKeepAlives as pairs
    int64_t* kas;
    int64_t* kan;                // [W]
    int64_t* bubble_id;          // [W] mBubbleId
    int64_t* box_sum;            // [W] the sum of nbox
    int64_t* bseq_next;          // [W]
    double* srange;              // [W,N] the sensor's range (a float value)
    double* speriod;
    double* selapsed;
    uint8_t* sonly;
    uint8_t* has_prox;
    int64_t* slist;              // [W,N,S] the sensor's balls (slots) in unordered_map order
    int64_t* slen;               // [W,N]
    int64_t* trk;                // [W,N] the tracked ball's slot, -1: none
    double* trk_range;
    uint8_t* trk_in;
    double* ntf_range;
    uint8_t* ntf_in;
    int64_t* pmap;               // [W,S] the proximity balls (slots) in unordered_map order
    int64_t* pn;                 // [W]
    int64_t* ev;                 // [W,M,4] (kind, ball id, a, b)
    int64_t* evn;                // [W]
    int64_t* flags;              // [W] FLAG_* bits
    uint8_t* dead;               // [W,N] in a dead bubble at the evolve's start
    const uint8_t* dropped;      // [W,N] dropped out of warp proper this tick (for the commit pass)
    int64_t* hits;               // [W,H] branch counts (tests' coverage), or null
    int64_t* ad_kind;            // [W,N] AdditionsAndDeletions per user ball (AD_*), or null
    int64_t* ad_add;             // [W,N,N] its additions (slots), in order
    int64_t* ad_addn;            // [W,N]
    int64_t* ad_del;             // [W,N,N] its deletions (slots, a removed ball's too), in order
    int64_t* ad_deln;            // [W,N]
    int64_t* pbkt;               // [W] the proximity balls' bucket count (1: nothing inserted yet)
    int64_t* pnr;                // [W] their rehash policy's _M_next_resize
    int64_t* sbkt;               // [W,N] each sensor list's bucket count
    int64_t* snr;                // [W,N] its policy's _M_next_resize
    int64_t* rnd;                // [W,K_RAND_WORDS] glibc's random() state a world: the TYPE_3
                                 //   table r[0..30] (int32 values), then fptr and rptr as indices; srand(seed) by
                                 //   exact_ballpark.vec_space.glibc_srand; floatrand's source for shuffled sensors
};

struct Tape {                    // per world, K operations applied in order (code 0: none)
    int64_t K;
    const int64_t* code;         // [W,K]
    const int64_t* a;            // [W,K]
    const int64_t* b;            // [W,K]
    const double* f;             // [W,K,4]
};

struct Query {
    int64_t kind;
    const int64_t* a;            // [W]
    const int64_t* b;            // [W]
    const double* f;             // [W,5]
    int64_t* out;                // [W, 1 + N]
    double* outf;                // [W, N] (distances), or null
};

// ======================================================================================================== access

struct Ctx {
    const Park* p;
    const Space* s;
    int64_t w;
};

static HD inline int64_t row_of(const Ctx& c, int64_t slot) {
    if (slot < 0 || slot >= c.s->N) return -1;
    return c.s->rows[c.w * c.s->N + slot];
}
static HD inline int64_t ix(const Ctx& c, int64_t slot) { return c.w * c.s->N + slot; }
static HD inline void hit(const Ctx& c, int64_t which) {
    if (c.s->hits) c.s->hits[c.w * K_HITS + which] += 1;
}
static HD inline void flag(const Ctx& c, int64_t bit) { c.s->flags[c.w] |= bit; }

static HD inline void event(const Ctx& c, int64_t kind, int64_t ball, int64_t a, int64_t b) {
    const int64_t n = c.s->evn[c.w];
    if (n >= c.s->M) {
        flag(c, FLAG_EVENTS);
        return;
    }
    int64_t* e = c.s->ev + (c.w * c.s->M + n) * 4;
    e[0] = kind;
    e[1] = ball;
    e[2] = a;
    e[3] = b;
    c.s->evn[c.w] = n + 1;
}

static HD inline int64_t slot_of_id(const Ctx& c, int64_t id) {   // the slot of a ball in the park, or -1
    for (int64_t j = 0; j < c.s->N; ++j) {
        const int64_t r = row_of(c, j);
        if (r >= 0 && c.s->in_park[ix(c, j)] && c.p->ids[r] == id) return j;
    }
    return -1;
}

// =========================================================================================================== filing

// Ball::GetInflatedRadius (Ball.cpp:838; SEN-706): a free ball's radius plus a tick at full thrust.
static HD inline double inflated_radius(const Ctx& c, int64_t r, const double* vel) {
    const Park* p = c.p;
    const double radius = p->radius[r];
    if (!p->free_[r]) return radius;
    const double m = dmul(p->mass[r], p->agility[r]);
    const double a = ddiv(dmul(dmul(p->friction, p->sf[r]), p->max_vel[r]), m);
    const double v = dsqrt(lensq3(vel[0], vel[1], vel[2]));
    const double k = p->friction;
    const double ma = dmul(m, a);
    const double mps = ddiv(dsub(ma, dmul(dsub(ma, dmul(v, k)), p->tf[r])), k);
    return dadd(radius, dmul(p->dt, mps));
}

static HD inline int64_t level_for(double radius) {   // GetBox(radius, pos)'s level (SEN-706)
    int64_t level = 0;
    for (int64_t i = 0; i < K_NUM_LEVELS; ++i) {
        if (dmul(3.0, radius) > level_width(i)) break;
        level = i;
    }
    return level;
}

static HD inline int64_t clamp_key(int64_t k, int64_t n) { return k < 0 ? 0 : (k >= n ? n - 1 : k); }

struct Filing {
    int64_t lvl;
    int64_t key[3];
    int64_t mod[3];
};

// Partition::GetBox(radius, pos) and Ball::InsertInBoxes's sides: the level, the clamped key, the sides.
static HD inline Filing filing_for(double inflated, const double* pos, bool* clamped) {
    Filing f;
    f.lvl = level_for(inflated);
    const double w = level_width(f.lvl);
    const int64_t n = level_grid(f.lvl);
    const double fac = ddiv(1.0, w);
    const double half = dmul(dmul(0.5, w), (double)n);
    const double center = dmul(dmul(w, (double)n), 0.5);
    *clamped = false;
    for (int q = 0; q < 3; ++q) {
        const int64_t raw = to_int64(dmul(dadd(pos[q], half), fac));
        f.key[q] = clamp_key(raw, n);
        if (f.key[q] != raw) *clamped = true;
        const double lo = dsub(dmul((double)f.key[q], w), center);
        const double hi = dsub(dmul((double)(f.key[q] + 1), w), center);
        if (dsub(dadd(lo, inflated), pos[q]) > 0.0) {
            f.mod[q] = -1;
        } else if (dsub(dsub(hi, inflated), pos[q]) < 0.0) {
            f.mod[q] = 1;
        } else {
            f.mod[q] = 0;
        }
    }
    return f;
}

static HD inline int64_t shr(int64_t k, int64_t s) { return k >> s; }   // arithmetic shift (keys are >= 0 here)

// The distinct top boxes a filing reaches (up to 8), and whether every one keeps its 26 neighbours in the grid.
static HD inline int tops_of(const Filing& f, int64_t out[8][3], bool* edge) {
    const int64_t s = 2 * f.lvl;
    int64_t opt[3][2];
    int cnt[3];
    *edge = false;
    for (int q = 0; q < 3; ++q) {
        const int64_t a = shr(f.key[q], s);
        const int64_t b = shr(f.key[q] + f.mod[q], s);
        opt[q][0] = a < b ? a : b;
        opt[q][1] = a < b ? b : a;
        cnt[q] = (a == b) ? 1 : 2;
        const int64_t side = f.key[q] + f.mod[q];
        if (side < 0 || side >= level_grid(f.lvl)) *edge = true;
        if (opt[q][0] < 1 || opt[q][1] > K_LG0 - 2) *edge = true;
    }
    int n = 0;
    for (int i = 0; i < cnt[0]; ++i)
        for (int j = 0; j < cnt[1]; ++j)
            for (int k = 0; k < cnt[2]; ++k) {
                out[n][0] = opt[0][i];
                out[n][1] = opt[1][j];
                out[n][2] = opt[2][k];
                ++n;
            }
    return n;
}

// The boxes a filing holds with their ancestors (each box once): the bound on the live boxes.
static HD inline int64_t boxes_of(const Filing& f) {
    int64_t total = 0;
    for (int64_t l = 0; l <= f.lvl; ++l) {
        const int64_t s = 2 * (f.lvl - l);
        int64_t prod = 1;
        for (int q = 0; q < 3; ++q) prod *= (shr(f.key[q], s) == shr(f.key[q] + f.mod[q], s)) ? 1 : 2;
        total += prod;
    }
    return total;
}

static HD inline Filing filing_of(const Ctx& c, int64_t slot) {
    Filing f;
    const int64_t i = ix(c, slot);
    f.lvl = c.s->lvl[i];
    for (int q = 0; q < 3; ++q) {
        f.key[q] = c.s->key[i * 3 + q];
        f.mod[q] = c.s->mod[i * 3 + q];
    }
    return f;
}

// ============================================================================================== the top boxes

static HD inline int64_t top_find(const Ctx& c, const int64_t* k) {
    const int64_t n = c.s->tn[c.w];
    const int64_t* t = c.s->tkey + c.w * c.s->T * 3;
    for (int64_t i = 0; i < n; ++i)
        if (t[i * 3] == k[0] && t[i * 3 + 1] == k[1] && t[i * 3 + 2] == k[2]) return i;
    return -1;
}

static HD inline int64_t top_bubble(const Ctx& c, const int64_t* k) {   // -1 also for a top that does not exist
    const int64_t i = top_find(c, k);
    return i < 0 ? -1 : c.s->tbub[c.w * c.s->T + i];
}

static HD inline void top_ref(const Ctx& c, const int64_t* k) {
    int64_t i = top_find(c, k);
    if (i < 0) {
        i = c.s->tn[c.w];
        if (i >= c.s->T) {
            flag(c, FLAG_TOPS);
            return;
        }
        int64_t* t = c.s->tkey + (c.w * c.s->T + i) * 3;
        t[0] = k[0];
        t[1] = k[1];
        t[2] = k[2];
        c.s->tbub[c.w * c.s->T + i] = -1;          // a new top box: no bubble
        c.s->tref[c.w * c.s->T + i] = 0;
        c.s->tn[c.w] = i + 1;
    }
    c.s->tref[c.w * c.s->T + i] += 1;
}

static HD inline void top_unref(const Ctx& c, const int64_t* k) {
    const int64_t i = top_find(c, k);
    if (i < 0) return;
    int64_t* ref = c.s->tref + c.w * c.s->T;
    ref[i] -= 1;
    if (ref[i] > 0 || XBPS_FAULT == 2) return;
    const int64_t last = c.s->tn[c.w] - 1;          // released: its bubble is lost
    int64_t* t = c.s->tkey + c.w * c.s->T * 3;
    int64_t* b = c.s->tbub + c.w * c.s->T;
    t[i * 3] = t[last * 3];
    t[i * 3 + 1] = t[last * 3 + 1];
    t[i * 3 + 2] = t[last * 3 + 2];
    b[i] = b[last];
    ref[i] = ref[last];
    c.s->tn[c.w] = last;
}

static HD inline void top_set_bubble(const Ctx& c, const int64_t* k, int64_t bubble) {
    const int64_t i = top_find(c, k);
    if (i >= 0) c.s->tbub[c.w * c.s->T + i] = bubble;
}

// Ball::DeleteFromBoxes: the ball leaves every box (a top no filing reaches any more is released).
static HD inline void unfile(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    if (c.s->lvl[i] < 0) return;
    const Filing f = filing_of(c, slot);
    int64_t tops[8][3];
    bool edge;
    const int n = tops_of(f, tops, &edge);
    for (int t = 0; t < n; ++t) top_unref(c, tops[t]);
    c.s->box_sum[c.w] -= c.s->nbox[i];
    c.s->nbox[i] = 0;
    c.s->lvl[i] = -1;
}

// Ball::InsertInBoxes's filing (box1 and the adjacent boxes its inflated sphere reaches).
static HD inline void file(const Ctx& c, int64_t slot, const Filing& f) {
    const int64_t i = ix(c, slot);
    c.s->lvl[i] = f.lvl;
    for (int q = 0; q < 3; ++q) {
        c.s->key[i * 3 + q] = f.key[q];
        c.s->mod[i * 3 + q] = f.mod[q];
    }
    int64_t tops[8][3];
    bool edge;
    const int n = tops_of(f, tops, &edge);
    if (edge) flag(c, FLAG_EDGE);
    for (int t = 0; t < n; ++t) top_ref(c, tops[t]);
    c.s->nbox[i] = boxes_of(f);
    c.s->box_sum[c.w] += c.s->nbox[i];
    if (c.s->box_sum[c.w] + K_NUM_LEVELS >= K_BOX_CHUNK) flag(c, FLAG_BOXES);
}

// ======================================================================================== the bubble dictionaries

static HD inline int64_t dict_find(const Ctx& c, int64_t bubble) {
    const int64_t n = c.s->dn[c.w];
    const int64_t* d = c.s->dbub + c.w * c.s->D;
    for (int64_t i = 0; i < n; ++i)
        if (d[i] == bubble) return i;
    return -1;
}

static HD inline void dict_add(const Ctx& c, int64_t bubble) {
    if (dict_find(c, bubble) >= 0) return;
    const int64_t n = c.s->dn[c.w];
    if (n >= c.s->D) {
        flag(c, FLAG_DICTS);
        return;
    }
    c.s->dbub[c.w * c.s->D + n] = bubble;
    c.s->dn[c.w] = n + 1;
}

static HD inline void dict_remove_at(const Ctx& c, int64_t i) {
    const int64_t last = c.s->dn[c.w] - 1;
    int64_t* d = c.s->dbub + c.w * c.s->D;
    d[i] = d[last];
    c.s->dn[c.w] = last;
}

static HD inline int64_t entry_find(const Ctx& c, int64_t bubble, int64_t slot) {
    const int64_t n = c.s->en[c.w];
    const int64_t base = c.w * c.s->E;
    for (int64_t i = 0; i < n; ++i)
        if (c.s->ebub[base + i] == bubble && c.s->eslot[base + i] == slot) return i;
    return -1;
}

static HD inline void entry_add(const Ctx& c, int64_t bubble, int64_t slot, int64_t act) {
    const int64_t n = c.s->en[c.w];
    if (n >= c.s->E) {
        flag(c, FLAG_ENTRIES);
        return;
    }
    const int64_t base = c.w * c.s->E;
    c.s->ebub[base + n] = bubble;
    c.s->eslot[base + n] = slot;
    c.s->eact[base + n] = act;
    c.s->eseq[base + n] = c.s->eseq_next[c.w]++;
    c.s->en[c.w] = n + 1;
}

static HD inline void entry_remove_at(const Ctx& c, int64_t i) {   // the others keep their order (by eseq)
    const int64_t last = c.s->en[c.w] - 1;
    const int64_t base = c.w * c.s->E;
    c.s->ebub[base + i] = c.s->ebub[base + last];
    c.s->eslot[base + i] = c.s->eslot[base + last];
    c.s->eact[base + i] = c.s->eact[base + last];
    c.s->eseq[base + i] = c.s->eseq[base + last];
    c.s->en[c.w] = last;
}

static HD inline bool dict_empty(const Ctx& c, int64_t bubble) {
    const int64_t n = c.s->en[c.w];
    const int64_t base = c.w * c.s->E;
    for (int64_t i = 0; i < n; ++i)
        if (c.s->ebub[base + i] == bubble) return false;
    return true;
}

// a set of (bubble, slot) pairs: bubbleInteractives or bubbleKeepAlives (a bubble's key exists while it has one)
static HD inline bool pairs_any(const Ctx& c, const int64_t* pb, const int64_t* n, int64_t bubble) {
    if (bubble == -1) return false;
    const int64_t base = c.w * c.s->I;
    for (int64_t i = 0; i < n[c.w]; ++i)
        if (pb[base + i] == bubble) return true;
    return false;
}

static HD inline void pairs_add(const Ctx& c, int64_t* pb, int64_t* ps, int64_t* n, int64_t bubble, int64_t slot) {
    const int64_t base = c.w * c.s->I;
    for (int64_t i = 0; i < n[c.w]; ++i)
        if (pb[base + i] == bubble && ps[base + i] == slot) return;
    if (n[c.w] >= c.s->I) {
        flag(c, FLAG_SETS);
        return;
    }
    pb[base + n[c.w]] = bubble;
    ps[base + n[c.w]] = slot;
    n[c.w] += 1;
}

static HD inline void pairs_remove(const Ctx& c, int64_t* pb, int64_t* ps, int64_t* n, int64_t bubble, int64_t slot) {
    const int64_t base = c.w * c.s->I;
    for (int64_t i = 0; i < n[c.w]; ++i) {
        if (pb[base + i] == bubble && ps[base + i] == slot) {
            const int64_t last = n[c.w] - 1;
            pb[base + i] = pb[base + last];
            ps[base + i] = ps[base + last];
            n[c.w] = last;
            return;
        }
    }
}

// Ball::mIsGlobal of a row: the view's flag, or false where the view has none.
static HD inline bool is_global(const Ctx& c, int64_t r) { return c.p->global_ != nullptr && c.p->global_[r] != 0; }

static HD inline bool any_ia(const Ctx& c, int64_t bubble) {     // bubble in bubbleInteractives
    return pairs_any(c, c.s->iab, c.s->ian, bubble);
}

static HD inline bool any_ka(const Ctx& c, int64_t bubble) {     // bubble in bubbleKeepAlives
    return pairs_any(c, c.s->kab, c.s->kan, bubble);
}

static HD inline void increase_interactive(const Ctx& c, int64_t slot, int64_t bubble) {
    if (c.p->interactive[row_of(c, slot)] && bubble != -1) pairs_add(c, c.s->iab, c.s->ias, c.s->ian, bubble, slot);
}

static HD inline void decrease_interactive(const Ctx& c, int64_t slot, int64_t bubble) {
    if (c.p->interactive[row_of(c, slot)] && bubble != -1) pairs_remove(c, c.s->iab, c.s->ias, c.s->ian, bubble, slot);
}

// Ballpark::AddToBubble (Ballpark.cpp:6429): +1 when added, 0 when kept, -1 when removed (a just-added ball is
// dropped instead); then the interactive and keep-alive sets (the ball in its new bubble's, out of its old one's).
static HD inline void add_to_bubble(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    const int64_t nb = c.s->bubble[i], ob = c.s->old_bubble[i];
    if (nb != -1) {
        dict_add(c, nb);
        const int64_t e = entry_find(c, nb, slot);
        if (e < 0) {
            entry_add(c, nb, slot, 1);
        } else if (c.s->eact[c.w * c.s->E + e] == -1) {
            if (XBPS_FAULT == 3) {
                entry_remove_at(c, e);
                entry_add(c, nb, slot, 0);
            } else {
                c.s->eact[c.w * c.s->E + e] = 0;
            }
        }
    }
    if (XBPS_FAULT == 4 && ob == nb) return;
    if (ob != -1 && dict_find(c, ob) >= 0) {
        const int64_t e = entry_find(c, ob, slot);
        if (e >= 0) {
            const int64_t v = c.s->eact[c.w * c.s->E + e];
            if (v == 0) {
                c.s->eact[c.w * c.s->E + e] = -1;
            } else if (v == 1) {
                entry_remove_at(c, e);
            }
        }
    }
    increase_interactive(c, slot, nb);                // UpdateInteractiveCnt: in the new bubble's set, out of the old's
    decrease_interactive(c, slot, ob);
    if (c.s->ka_ball[i]) {                            // UpdateKeepAliveBalls
        if (nb != -1) pairs_add(c, c.s->kab, c.s->kas, c.s->kan, nb, slot);
        if (ob != -1) pairs_remove(c, c.s->kab, c.s->kas, c.s->kan, ob, slot);
    }
}

// Box::NearbyBubbles: the first of the 26 neighbours (i, j, k order) that exists and has a bubble.
static HD inline int64_t nearby_bubbles(const Ctx& c, const int64_t* t) {
    for (int64_t i = -1; i <= 1; ++i)
        for (int64_t j = -1; j <= 1; ++j)
            for (int64_t k = -1; k <= 1; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                const int64_t nk[3] = {t[0] + i, t[1] + j, t[2] + k};
                const int64_t b = top_bubble(c, nk);
                if (b != -1) return b;
            }
    return -1;
}

// ============================================================================================ InsertInBoxes

// Ballpark::InsertInBoxes (Ballpark.cpp:3115; SEN-706, SEN-707) on the master, then Ball::InsertInBoxes
// (Ball.cpp:2354). `pos` and `vel` are the ball's (the drop-out point and its velocity in warp's drop-out insert).
static HD inline void insert_in_boxes(const Ctx& c, int64_t slot, const double* pos, const double* vel) {
    const int64_t r = row_of(c, slot);
    const int64_t i = ix(c, slot);
    const double inflated = inflated_radius(c, r, vel);
    bool clamped;
    const Filing box1 = filing_for(inflated, pos, &clamped);
    if (clamped) flag(c, FLAG_EDGE);
    if (box1.lvl == 0 && dmul(c.p->radius[r], 2.0) > level_width(0)) flag(c, FLAG_HUGE);
    int64_t top[3];
    for (int q = 0; q < 3; ++q) top[q] = shr(box1.key[q], 2 * box1.lvl);
    int64_t new_bubble = top_bubble(c, top);             // box1 (and its top) created if missing: a new top has -1
    if (new_bubble == -1) {
        hit(c, H_NO_BUBBLE);
        unfile(c, slot);                                 // DeleteFromBoxes: may release box1's top (stale: defined)
        new_bubble = XBPS_FAULT == 1 ? -1 : nearby_bubbles(c, top);
        if (new_bubble == -1) {
            hit(c, H_NEW);
            new_bubble = c.s->bubble_id[c.w]++;
        } else {
            hit(c, H_NEIGHBOUR);
        }
        top_set_bubble(c, top, new_bubble);              // the top's bubble (rewritten after the insertion)
    }
    if (c.s->bubble[i] != new_bubble) {
        hit(c, H_CHANGED);
        c.s->old_bubble[i] = c.s->bubble[i];
        c.s->bubble[i] = new_bubble;
        add_to_bubble(c, slot);
    }
    // Ball::InsertInBoxes: unchanged when box1 is its box and the sides are the same
    const int64_t lv = c.s->lvl[i];
    if (lv >= 0 && lv == box1.lvl) {
        bool same = true;
        for (int q = 0; q < 3; ++q)
            if (c.s->key[i * 3 + q] != box1.key[q] || c.s->mod[i * 3 + q] != box1.mod[q]) same = false;
        if (same) {
            hit(c, H_UNCHANGED);
            return;
        }
    }
    unfile(c, slot);
    file(c, slot, box1);
    top_set_bubble(c, top, new_bubble);
    if (box1.mod[0] || box1.mod[1] || box1.mod[2]) hit(c, H_ADJACENT);
}

// Ballpark::InsertBallInBoxes (Ballpark.cpp:3189): the filing, then the ball registered (outside Evolve).
static HD inline void insert_ball_in_boxes(const Ctx& c, int64_t slot, bool in_evolve) {
    const int64_t r = row_of(c, slot);
    insert_in_boxes(c, slot, c.p->pos + r * 3, c.p->vel + r * 3);
    if (!in_evolve && !c.s->in_park[ix(c, slot)]) {
        c.s->in_park[ix(c, slot)] = 1;
        c.s->bseq[ix(c, slot)] = c.s->bseq_next[c.w]++;
    }
}

static HD inline void sync_bubble(const Ctx& c, int64_t slot) {
    if (c.p->bubble_out == nullptr) return;
    const int64_t r = row_of(c, slot);
    if (r >= 0) c.p->bubble_out[r] = c.s->bubble[ix(c, slot)];
}

// ===================================================================================== InitializeBubbles et al.

// Ballpark::InitializeBubbles (Ballpark.cpp:6565; SEN-707): every top's bubble cleared, the dictionaries and sets
// cleared, then from each ball's top in the ball dictionary's order a seed fill (Box::Traverse) of the connected
// top boxes; a ball whose bubble does not change ends in no dictionary and no set (DESTINY_FINDINGS 1.4).
static HD inline void initialize_bubbles(const Ctx& c) {
    const int64_t T = c.s->T, n = c.s->tn[c.w];
    for (int64_t t = 0; t < n; ++t) c.s->tbub[c.w * T + t] = -1;
    c.s->bubble_id[c.w] = 0;
    c.s->en[c.w] = 0;
    c.s->dn[c.w] = 0;
    c.s->ian[c.w] = 0;
    c.s->kan[c.w] = 0;
    for (int64_t j = 0; j < c.s->N; ++j) c.s->ka_ball[ix(c, j)] = 0;
    // the balls in the dictionary's order
    int64_t prev = -1;
    for (;;) {
        int64_t best = -1, best_seq = 0;
        for (int64_t j = 0; j < c.s->N; ++j) {
            const int64_t i = ix(c, j);
            if (row_of(c, j) < 0 || !c.s->in_park[i]) continue;
            const int64_t q = c.s->bseq[i];
            if (q > prev && (best < 0 || q < best_seq)) {
                best = j;
                best_seq = q;
            }
        }
        if (best < 0) break;
        prev = best_seq;
        const int64_t i = ix(c, best);
        if (c.s->lvl[i] < 0) {
            c.s->old_bubble[i] = c.s->bubble[i];
            c.s->bubble[i] = -1;
            continue;
        }
        int64_t top[3];
        for (int q = 0; q < 3; ++q) top[q] = shr(c.s->key[i * 3 + q], 2 * c.s->lvl[i]);
        const int64_t tb = top_bubble(c, top);
        if (tb != -1) {
            hit(c, H_INIT_TRAVERSED);
            c.s->old_bubble[i] = c.s->bubble[i];
            c.s->bubble[i] = tb;
            add_to_bubble(c, best);
            continue;
        }
        // Box::Traverse: every top connected to this one through tops without a bubble (all of them here)
        const int64_t seed = c.s->bubble_id[c.w];
        top_set_bubble(c, top, seed);
        bool grew = true;
        while (grew) {
            grew = false;
            for (int64_t t = 0; t < n; ++t) {
                if (c.s->tbub[c.w * T + t] != -1) continue;
                const int64_t* k = c.s->tkey + (c.w * T + t) * 3;
                for (int64_t u = 0; u < n && c.s->tbub[c.w * T + t] == -1; ++u) {
                    if (c.s->tbub[c.w * T + u] != seed) continue;
                    const int64_t* h = c.s->tkey + (c.w * T + u) * 3;
                    const int64_t dx = k[0] - h[0], dy = k[1] - h[1], dz = k[2] - h[2];
                    if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1 && dz >= -1 && dz <= 1) {
                        c.s->tbub[c.w * T + t] = seed;
                        grew = true;
                    }
                }
            }
        }
        c.s->old_bubble[i] = c.s->bubble[i];
        c.s->bubble[i] = seed;
        add_to_bubble(c, best);
        c.s->bubble_id[c.w] = seed + 1;
    }
}

// Ballpark::UpdateBallBubble (Ballpark.cpp:5959): PyInitializeBubbles(src): the first ball within a top width (in
// the dictionary's order) gives its bubble, else one above the highest bubble seen; no dictionary changes.
static HD inline void update_ball_bubble(const Ctx& c, int64_t src) {
    if (src < 0 || !c.s->in_park[ix(c, src)]) return;
    const int64_t rs = row_of(c, src);
    const double max_width = dmul(level_width(0), level_width(0));
    int64_t hi_id = -1, prev = -1;
    for (;;) {
        int64_t best = -1, best_seq = 0;
        for (int64_t j = 0; j < c.s->N; ++j) {
            const int64_t i = ix(c, j);
            if (row_of(c, j) < 0 || !c.s->in_park[i]) continue;
            const int64_t q = c.s->bseq[i];
            if (q > prev && (best < 0 || q < best_seq)) {
                best = j;
                best_seq = q;
            }
        }
        if (best < 0) break;
        prev = best_seq;
        const int64_t b = c.s->bubble[ix(c, best)];
        if (b > hi_id) hi_id = b;
        if (best == src || b == -1) continue;
        const int64_t rb = row_of(c, best);
        const double* ps = c.p->pos + rs * 3;
        const double* pb = c.p->pos + rb * 3;
        if (lensq3(dsub(ps[0], pb[0]), dsub(ps[1], pb[1]), dsub(ps[2], pb[2])) < max_width) {
            c.s->bubble[ix(c, src)] = b;
            break;
        }
    }
    if (c.s->bubble[ix(c, src)] == -1) c.s->bubble[ix(c, src)] = hi_id + 1;
}

// Ballpark::CopyBubbles (Ballpark.cpp:6154): additions become kept, removals go, an empty dictionary goes.
static HD inline void copy_bubbles(const Ctx& c) {
    const int64_t base = c.w * c.s->E;
    for (int64_t e = 0; e < c.s->en[c.w];) {
        const int64_t v = c.s->eact[base + e];
        if (v == 1) {
            c.s->eact[base + e] = 0;
        } else if (v == -1) {
            entry_remove_at(c, e);
            continue;
        }
        ++e;
    }
    for (int64_t d = 0; d < c.s->dn[c.w];) {
        if (XBPS_FAULT != 9 && dict_empty(c, c.s->dbub[c.w * c.s->D + d])) {
            dict_remove_at(c, d);
            continue;
        }
        ++d;
    }
}

static HD inline bool in_dead_bubble(const Ctx& c, int64_t slot) {   // Ballpark::InDeadBubble (TIME-714)
    const int64_t r = row_of(c, slot);
    if (c.p->mode[r] == K_MODE_WARP) return false;
    const int64_t b = c.s->bubble[ix(c, slot)];
    if (any_ia(c, b)) return false;
    if (any_ka(c, b)) return false;
    return true;
}

// ========================================================================================= unordered_map order
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// A clean-room implementation, written from a behavioural specification (the rules in words, the data tables) without
// the C++ library's code. The proximity balls and each sensor's list in libstdc++'s unordered_map order, as
// exact_ballpark.space.StdHashMap keeps it, for maps of at most K_UMAP_CAP keys. A map is a list of slots in iteration
// order (a slot's key: its ball's id, c.p->ids[rows[slot]]), its length, its bucket count and its rehash policy's next
// resize; a fresh map has no keys, 1 bucket and next resize 0. Every count here is far below 2**53 and the maximum
// load factor is 1 (exact_ballpark.vec_space refuses another), so the policy's doubles are these integers exactly.
// The tables come from the generated header: the prime list's entries through the bucket count K_UMAP_CAP keys reach,
// and the small table of the bucket counts for fewer than K_NFAST_BKT buckets; device and host copies.

static __device__ const int64_t xbps_primes_dev[K_NPRIMES] = K_PRIME_LIST;   // in both passes (a device variable)
static __device__ const int64_t xbps_fast_bkt_dev[K_NFAST_BKT] = K_FAST_BKT;
#ifdef __CUDA_ARCH__
#define XBPS_PRIMES xbps_primes_dev
#define XBPS_FAST_BKT xbps_fast_bkt_dev
#else
static const int64_t xbps_primes_host[K_NPRIMES] = K_PRIME_LIST;
static const int64_t xbps_fast_bkt_host[K_NFAST_BKT] = K_FAST_BKT;
#define XBPS_PRIMES xbps_primes_host
#define XBPS_FAST_BKT xbps_fast_bkt_host
#endif

// A key's bucket: the id's 64-bit two's-complement pattern as an unsigned integer, modulo the bucket count.
static HD inline int64_t bucket_of(int64_t id, int64_t nbkt) {
    return (int64_t)((uint64_t)id % (uint64_t)nbkt);
}

// Whether the slot is among the map's n keys.
static HD inline bool umap_has(const int64_t* list, int64_t n, int64_t slot) {
    for (int64_t k = 0; k < n; ++k) {
        if (list[k] == slot) return true;
    }
    return false;
}

// The bucket count for at least n buckets, *nr its next resize (the count x 1.0, floored: the count). Below
// K_NFAST_BKT the small table's entry (n = 0: 1 bucket, *nr left as it is); otherwise the prime list's first entry
// not less than n, searched from K_PRIME_FROM. 0, *nr left as it is, when the header's list ends first.
static HD inline int64_t umap_next_bkt(int64_t n, int64_t* nr) {
    if (n < K_NFAST_BKT) {
        if (n == 0) return 1;
        *nr = XBPS_FAST_BKT[n];
        return XBPS_FAST_BKT[n];
    }
    for (int64_t j = K_PRIME_FROM; j < K_NPRIMES; ++j) {
        if (XBPS_PRIMES[j] >= n) {
            *nr = XBPS_PRIMES[j];
            return XBPS_PRIMES[j];
        }
    }
    return 0;
}

// The rehash policy with nbkt buckets, nelt keys present and nins to insert; *nr, its next resize, read and updated.
// 1 and *grow_to the new bucket count to grow; 0 to keep the buckets; -1 when the growth passes the header's list.
static HD inline int umap_need_rehash(int64_t nbkt, int64_t nelt, int64_t nins, int64_t* nr, int64_t* grow_to) {
    if (XBPS_FAULT == 12 && nbkt == K_UMAP_BUCKETS) *nr = nelt + 1;      // (fault 12: K_UMAP_BUCKETS for ever)
    const int64_t total = nelt + nins;
    if (total <= *nr) return 0;
    const int64_t least = *nr == 0 ? K_UMAP_FIRST_MIN : 0;              // a map that never grew asks for more
    const int64_t m = total > least ? total : least;                     // the keys / 1.0
    if (m < nbkt) {
        *nr = nbkt;                                                      // floor(nbkt x 1.0)
        return 0;
    }
    int64_t want = m + 1;
    const int64_t grown = nbkt * K_UMAP_GROWTH;
    if (XBPS_FAULT != 15 && grown > want) want = grown;                  // (fault 15: without the growth factor)
    const int64_t b = umap_next_bkt(want, nr);
    if (b == 0) return -1;
    *grow_to = b;
    return 1;
}

// The linking rule: the slot goes immediately before the first of the list's n keys in its bucket, or at the front
// when none is; the list holds one more (counted by the caller).
static HD inline void umap_link(const Ctx& c, int64_t* list, int64_t n, int64_t nbkt, int64_t slot) {
    const int64_t b = bucket_of(c.p->ids[row_of(c, slot)], nbkt);
    int64_t at = 0;
    for (int64_t k = 0; k < n; ++k) {
        if (bucket_of(c.p->ids[row_of(c, list[k])], nbkt) == b) {
            at = k;
            break;
        }
    }
    for (int64_t k = n; k > at; --k) list[k] = list[k - 1];
    list[at] = slot;
}

// A growth to nbkt buckets: from an empty order, the n keys taken in their old order and each placed by the linking
// rule. In place: before step k the first k entries are the new order of the old first k keys, entry k the next one.
static HD inline void umap_rehash(const Ctx& c, int64_t* list, int64_t n, int64_t nbkt) {
    if (XBPS_FAULT == 13) return;                                        // (fault 13: the keys keep their order)
    for (int64_t k = 1; k < n; ++k) umap_link(c, list, k, nbkt, list[k]);
}

// insert(): a present slot changes nothing (true); an absent one is linked after the policy's growth, if any.
// False, and the map unchanged, when it holds the layer's capacity c.s->S or the growth passes the header's list.
static HD inline bool umap_insert(const Ctx& c, int64_t* list, int64_t* n, int64_t* bkt, int64_t* nr, int64_t slot) {
    const int64_t len = *n;
    if (umap_has(list, len, slot)) return true;
    if (len >= c.s->S) return false;
    int64_t b = *bkt, r = *nr, grow_to = 0;
    const int g = umap_need_rehash(b, len, 1, &r, &grow_to);
    if (g < 0) return false;
    if (g > 0) {
        b = grow_to;
        umap_rehash(c, list, len, b);
    }
    if (XBPS_FAULT == 7) list[len] = slot;                               // (fault 7: appended, not linked)
    else umap_link(c, list, len, b, slot);
    *n = len + 1;
    *bkt = b;
    *nr = r;
    return true;
}

// erase(key): the other keys keep their order; the bucket count and next resize are kept. An absent slot: nothing.
static HD inline void umap_erase(int64_t* list, int64_t* n, int64_t slot) {
    const int64_t len = *n;
    for (int64_t k = 0; k < len; ++k) {
        if (list[k] != slot) continue;
        for (int64_t q = k + 1; q < len; ++q) list[q - 1] = list[q];
        *n = len - 1;
        return;
    }
}

// clear(): no keys; the bucket count and next resize are kept (a cleared map is not a fresh one).
static HD inline void umap_clear(int64_t* n, int64_t* bkt, int64_t* nr) {
    *n = 0;
    if (XBPS_FAULT == 14) {                                              // (fault 14: a fresh map)
        *bkt = 1;
        *nr = 0;
    }
}

// ================================================================================================== the sensors

static HD inline int64_t* slist_of(const Ctx& c, int64_t slot) { return c.s->slist + ix(c, slot) * c.s->S; }

// Ball::UpdateProximityStatus.
static HD inline void update_proximity_status(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    const int64_t r = row_of(c, slot);
    int64_t* pm = c.s->pmap + c.w * c.s->S;
    if (c.s->trk_range[i] != 0.0 || c.s->ntf_range[i] != 0.0 || c.p->sensor[r]) {
        c.s->has_prox[i] = 1;
        if (!umap_insert(c, pm, c.s->pn + c.w, c.s->pbkt + c.w, c.s->pnr + c.w, slot)) flag(c, FLAG_PROXIMITY);
    } else {
        c.s->has_prox[i] = 0;
        umap_erase(pm, c.s->pn + c.w, slot);
    }
}

#include "glibc_rand.cuh"   // LGPL-2.1-or-later: the C library's rand()
// AddProximitySensor (Ball.cpp:244; SEN-701): refused when float(radius) + float(range) < 0; a shuffled sensor
// (shuffle nonzero) starts its period at floatrand(0, period): 0 + rand() / RAND_MAX x (period - 0), in doubles, the
// world's rand() drawn once (Ball.cpp:19, 262-263; space.py's add_proximity_sensor).
static HD inline void add_proximity_sensor(const Ctx& c, int64_t slot, double range, double period, bool only,
                                           bool shuffle) {
    const int64_t r = row_of(c, slot);
    const int64_t i = ix(c, slot);
    const double rng = f32(range);
    if (fsum(c.p->radius[r], rng) < 0.0) return;
    umap_clear(c.s->slen + i, c.s->sbkt + i, c.s->snr + i);
    c.s->srange[i] = rng;
    c.s->speriod[i] = period;
    c.p->sensor[r] = 1;
    c.s->sonly[i] = only ? 1 : 0;
    if (shuffle && XBPS_FAULT != 16) {                  // (fault 16: the shuffle ignored)
        hit(c, H_SENSOR_SHUFFLE);
        const double u = ddiv((double)glibc_rand(c), KD(K_RAND_MAX));
        c.s->selapsed[i] = dadd(0.0, dmul(u, dsub(c.s->speriod[i], 0.0)));
    }
    update_proximity_status(c, slot);
}

static HD inline void remove_proximity_sensor(const Ctx& c, int64_t slot) {
    c.p->sensor[row_of(c, slot)] = 0;
    update_proximity_status(c, slot);
}

static HD inline void set_notification_range(const Ctx& c, int64_t slot, double n) {   // SEN-704
    if (n < 0.0) return;
    c.s->ntf_range[ix(c, slot)] = n;
    update_proximity_status(c, slot);
    c.s->ntf_in[ix(c, slot)] = 0;
}

static HD inline void set_target_tracking(const Ctx& c, int64_t slot, int64_t target_slot, int64_t target_id,
                                          double range) {                                // SEN-703
    const int64_t i = ix(c, slot);
    if (range <= 0.0 || target_id == 0) {
        c.s->trk[i] = -1;
        c.s->trk_range[i] = 0.0;
        return;
    }
    if (target_slot < 0) return;
    c.s->trk[i] = target_slot;
    c.s->trk_range[i] = range;
    update_proximity_status(c, slot);
    c.s->trk_in[i] = 0;
}

static HD inline bool is_follow_mode(int64_t m) {
    return m == K_MODE_FOLLOW || m == K_MODE_ORBIT || m == K_MODE_MISSILE || m == K_MODE_FORMATION;
}

// The walk position of ball o (filing fn) in Partition::GetNearbyBalls from ball b (filing fb), or NOT_A_CANDIDATE:
// b's boxes by key; for each its ancestors bottom-up, its descendants in post-order (children by key), then the box
// itself; each ball at its first box (exact_ballpark.vec_collide's _related and _order_keys).
static HD inline int64_t walk_key(const Filing& fb, const Filing& fn) {
    const int64_t Lb = fb.lvl, Ln = fn.lvl;
    int64_t lo_b[3], hi_b[3], lo_n[3], hi_n[3];
    int64_t cnt_b[3];
    for (int q = 0; q < 3; ++q) {
        lo_b[q] = fb.key[q] + (fb.mod[q] < 0 ? fb.mod[q] : 0);
        hi_b[q] = fb.key[q] + (fb.mod[q] > 0 ? fb.mod[q] : 0);
        lo_n[q] = fn.key[q] + (fn.mod[q] < 0 ? fn.mod[q] : 0);
        hi_n[q] = fn.key[q] + (fn.mod[q] > 0 ? fn.mod[q] : 0);
        cnt_b[q] = fb.mod[q] != 0 ? 2 : 1;
    }
    // related: on each axis an option of the finer, shifted to the coarser level, equals an option of the coarser
    const int64_t sh = 2 * (Lb > Ln ? Lb - Ln : Ln - Lb);
    for (int q = 0; q < 3; ++q) {
        int64_t xl, xh, yl, yh;
        if (Ln <= Lb) {
            xl = shr(lo_b[q], sh); xh = shr(hi_b[q], sh); yl = lo_n[q]; yh = hi_n[q];
        } else {
            xl = shr(lo_n[q], sh); xh = shr(hi_n[q], sh); yl = lo_b[q]; yh = hi_b[q];
        }
        if (!(xl == yl || xl == yh || xh == yl || xh == yh)) return K_NOT_A_CANDIDATE;
    }
    if (XBPS_FAULT == 6) return 0;
    if (Ln <= Lb) {                                       // an ancestor (or the same level): the first box under it
        const int64_t s_up = 2 * (Lb - Ln);
        int64_t idx[3];
        for (int q = 0; q < 3; ++q) {
            const int64_t v = shr(lo_b[q], s_up);
            const bool holds = (v == fn.key[q]) || (fn.mod[q] != 0 && v == fn.key[q] + fn.mod[q]);
            idx[q] = holds ? 0 : 1;
        }
        const int64_t seg = (idx[0] * cnt_b[1] + idx[1]) * cnt_b[2] + idx[2];
        if (Ln < Lb) return (seg * 3) * K_CODE_SPAN + (Lb - Ln);
        return (seg * 3 + 2) * K_CODE_SPAN;
    }
    // below: each of o's boxes whose ancestor at Lb is one of b's, by its post-order code under that box
    const int64_t s_dn = 2 * (Ln - Lb);
    int64_t best = K_NOT_A_CANDIDATE;
    int cnt_n[3];
    for (int q = 0; q < 3; ++q) cnt_n[q] = fn.mod[q] != 0 ? 2 : 1;
    for (int bx = 0; bx < cnt_n[0]; ++bx)
        for (int by = 0; by < cnt_n[1]; ++by)
            for (int bz = 0; bz < cnt_n[2]; ++bz) {
                const int bits[3] = {bx, by, bz};
                int64_t box[3], j[3];
                bool own = true;
                for (int q = 0; q < 3; ++q) {
                    box[q] = fn.key[q] + (bits[q] ? fn.mod[q] : 0);
                    const int64_t anc = shr(box[q], s_dn);
                    if (anc == lo_b[q]) {
                        j[q] = 0;
                    } else if (anc == hi_b[q] && cnt_b[q] == 2) {
                        j[q] = 1;
                    } else {
                        own = false;
                        j[q] = 0;
                    }
                }
                if (!own) continue;
                const int64_t seg = (j[0] * cnt_b[1] + j[1]) * cnt_b[2] + j[2];
                int64_t code = 0;
                for (int64_t li = 1; li < K_NUM_LEVELS; ++li) {   // the path digits below Lb, 64 past Ln
                    int64_t digit;
                    if (li <= Lb) {
                        digit = 0;
                    } else if (li > Ln) {
                        digit = K_CODE_BASE - 1;
                    } else {
                        const int64_t s = 2 * (Ln - li);
                        digit = (shr(box[0], s) & 3) * 16 + (shr(box[1], s) & 3) * 4 + (shr(box[2], s) & 3);
                    }
                    int64_t power = 1;
                    for (int64_t e = li; e < K_NUM_LEVELS - 1; ++e) power *= K_CODE_BASE;
                    code += digit * power;
                }
                const int64_t key = (seg * 3 + 1) * K_CODE_SPAN + code;
                if (key < best) best = key;
            }
    return best;
}

// Partition::GetNearbyBalls (Partition.cpp:344) as the sensor calls it (exclude_missiles, exclude_cloaked false;
// exclude_harmonics true): the accepted balls' slots in the walk's order. Returns the count.
static HD inline int64_t nearby_for_sensor(const Ctx& c, int64_t slot, bool exclude_fixed, bool exclude_inert,
                                           int64_t* out, int64_t* keys) {
    const int64_t i = ix(c, slot);
    const int64_t b_bubble = c.s->bubble[i];
    if (exclude_inert && !any_ia(c, b_bubble)) {
        hit(c, H_NEARBY_NO_INTERACTIVES);
        return 0;
    }
    const Filing fb = filing_of(c, slot);
    int64_t n = 0;
    for (int64_t j = 0; j < c.s->N; ++j) {
        if (j == slot) continue;
        const int64_t r = row_of(c, j);
        const int64_t k = ix(c, j);
        if (r < 0 || !c.s->in_park[k] || c.s->lvl[k] < 0) continue;
        const int64_t key = walk_key(fb, filing_of(c, j));
        if (key == K_NOT_A_CANDIDATE) continue;
        if (c.p->moribund[r]) continue;
        if (exclude_inert) {
            if (!c.p->interactive[r]) continue;
        } else {
            if (c.s->bubble[k] != b_bubble && !is_global(c, r)) {   // another bubble's ball, unless global
                hit(c, H_NEARBY_OTHER_BUBBLE);
                continue;
            }
            if (exclude_fixed && !c.p->free_[r]) continue;
        }
        // insertion by (key, id)
        const int64_t id = c.p->ids[r];
        int64_t at = n;
        while (at > 0) {
            const int64_t pk = keys[at - 1];
            const int64_t pid = c.p->ids[row_of(c, out[at - 1])];
            if (pk < key || (pk == key && pid < id)) break;
            out[at] = out[at - 1];
            keys[at] = keys[at - 1];
            --at;
        }
        out[at] = j;
        keys[at] = key;
        ++n;
    }
    return n;
}

static HD inline void set_ball_radius(const Ctx& c, int64_t slot, double r) {   // SetBallRadius(Ball*, float)
    if (r < 0.0) return;
    c.p->radius[row_of(c, slot)] = r;
    insert_ball_in_boxes(c, slot, true);
}

// CheckForProximity_Sensor (Ball.cpp:642; SEN-701, SEN-702).
static HD inline void check_sensor(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    const int64_t r = row_of(c, slot);
    c.s->selapsed[i] = dadd(c.s->selapsed[i], c.p->dt);
    if (c.s->selapsed[i] < c.s->speriod[i]) {
        hit(c, H_SENSOR_WAIT);
        return;
    }
    hit(c, H_SENSOR_CHECK);
    c.s->selapsed[i] = 0.0;
    const double radius = c.p->radius[r];
    double* vel = c.p->vel + r * 3;
    const double v0 = vel[0], v1 = vel[1], v2 = vel[2];
    vel[0] = 0.0;
    vel[1] = 0.0;
    vel[2] = 0.0;
    set_ball_radius(c, slot, fsum(radius, c.s->srange[i]));
    const bool exclude_fixed = c.p->cloaked[r] == 0;
    int64_t uni[K_MAX_SLOTS], keys[K_MAX_SLOTS];
    const int64_t nu = nearby_for_sensor(c, slot, exclude_fixed, c.s->sonly[i] != 0, uni, keys);
    vel[0] = v0;
    vel[1] = v1;
    vel[2] = v2;
    set_ball_radius(c, slot, radius);
    int64_t* list = slist_of(c, slot);
    const int64_t id = c.p->ids[r];
    // The walk over the live map, as Destiny's (store = i++, then erase(store)): an erasure keeps the others' order,
    // so the walk stays where it is.
    for (int64_t k = 0; k < c.s->slen[i];) {
        const int64_t o = list[k];
        bool found = o == slot;
        for (int64_t u = 0; u < nu && !found; ++u) found = uni[u] == o;
        if (found) {
            ++k;
            continue;
        }
        hit(c, H_SENSOR_VANISHED);
        umap_erase(list, c.s->slen + i, o);
        event(c, EV_PROXIMITY, id, c.p->ids[row_of(c, o)], 0);
        if (!c.p->sensor[r]) break;
    }
    for (int64_t u = 0; u < nu; ++u) {
        if (c.p->moribund[r]) break;
        const int64_t o = uni[u];
        const int64_t ro = row_of(c, o);
        if (o == slot || c.p->moribund[ro]) continue;
        const bool in_sensor = umap_has(list, c.s->slen[i], o);
        const double* ps = c.p->pos + r * 3;
        const double* po = c.p->pos + ro * 3;
        const double d2 = lensq3(dsub(ps[0], po[0]), dsub(ps[1], po[1]), dsub(ps[2], po[2]));
        const double ck = fsum(fsum(c.s->srange[i], c.p->radius[ro]), c.p->radius[r]);
        if (d2 > dmul(ck, ck)) {
            if (in_sensor) {
                hit(c, H_SENSOR_LEFT);
                umap_erase(list, c.s->slen + i, o);
                event(c, EV_PROXIMITY, id, c.p->ids[ro], 0);
            }
        } else if (!in_sensor) {
            hit(c, H_SENSOR_ENTERED);
            if (!umap_insert(c, list, c.s->slen + i, c.s->sbkt + i, c.s->snr + i, o)) flag(c, FLAG_SENSOR_LIST);
            event(c, EV_PROXIMITY, id, c.p->ids[ro], 1);
        }
        if (!c.p->sensor[r]) break;
    }
}

// CheckForProximity_NotificationRange (Ball.cpp:520; SEN-704).
static HD inline void check_notification_range(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    const int64_t r = row_of(c, slot);
    bool send = false, inside = false;
    if (c.s->ntf_range[i] != 0.0) {
        const double n = c.s->ntf_range[i];
        const int64_t m = c.p->mode[r];
        const double* p = c.p->pos + r * 3;
        if (m == K_MODE_GOTO) {
            const double* g = c.p->goto_ + r * 3;
            const double lim = dadd(n, c.p->radius[r]);
            inside = lensq3(dsub(p[0], g[0]), dsub(p[1], g[1]), dsub(p[2], g[2])) < dmul(lim, lim);
        } else if (m == K_MODE_FOLLOW || m == K_MODE_MISSILE || m == K_MODE_ORBIT) {
            const int64_t o = slot_of_id(c, c.p->follow_id[r]);
            const int64_t ro = row_of(c, o);
            const double* q = c.p->pos + ro * 3;
            const double len2 = lensq3(dsub(p[0], q[0]), dsub(p[1], q[1]), dsub(p[2], q[2]));
            const double hi = dadd(dadd(dadd(c.p->radius[ro], c.p->follow_range[r]), n), c.p->radius[r]);
            double lo = dsub(hi, dmul(2.0, n));
            if (lo < 0.0) lo = 0.0;
            inside = len2 < dmul(hi, hi) && len2 > dmul(lo, lo);
        }
        if (inside) {
            if (!c.s->ntf_in[i]) {
                send = true;
                c.s->ntf_in[i] = 1;
            }
        } else if (c.s->ntf_in[i]) {
            send = true;
            c.s->ntf_in[i] = 0;
        }
    }
    if (send) {
        hit(c, H_RANGE_EVENT);
        event(c, EV_RANGE, c.p->ids[r], c.s->ntf_in[i], 0);
    }
}

// CheckForProximity_TargetTracking (Ball.cpp:595; SEN-703).
static HD inline void check_target_tracking(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    const int64_t r = row_of(c, slot);
    bool send = false;
    const int64_t t = c.s->trk[i];
    if (c.s->trk_range[i] >= KD(K_TRACKING_MIN) && t >= 0) {
        const int64_t rt = row_of(c, t);
        const double* p = c.p->pos + r * 3;
        const double* q = c.p->pos + rt * 3;
        const double d2 = lensq3(dsub(p[0], q[0]), dsub(p[1], q[1]), dsub(p[2], q[2]));
        const double lim = dadd(c.s->trk_range[i], c.p->radius[r]);
        if (d2 < dmul(lim, lim)) {
            if (!c.s->trk_in[i]) {
                send = true;
                c.s->trk_in[i] = 1;
            }
        } else if (c.s->trk_in[i]) {
            send = true;
            c.s->trk_in[i] = 0;
        }
    }
    if (send) {
        hit(c, H_TRACKING_EVENT);
        event(c, EV_TRACKING, c.p->ids[r], c.s->trk_in[i], 0);
    }
}

static HD inline void check_for_proximity(const Ctx& c, int64_t slot) {   // Ball::CheckForProximity
    const int64_t r = row_of(c, slot);
    if (c.p->moribund[r]) return;
    check_notification_range(c, slot);
    check_target_tracking(c, slot);
    if (!c.p->sensor[r]) return;
    check_sensor(c, slot);
}

// Ballpark::HandleProximities (Ballpark.cpp:5822): every proximity ball in the map's order; one in a dead bubble
// gets one forced sensor check if its sensor holds balls, and its list is emptied (TIME-714). Destiny walks a copy of
// the map (a DoProximity handler may change it); here nothing in the walk writes the map (the events are recorded,
// not dispatched), so the walk reads it in place.
static HD inline void handle_proximities(const Ctx& c) {
    const int64_t n = c.s->pn[c.w];
    const int64_t* order = c.s->pmap + c.w * c.s->S;
    for (int64_t k = 0; k < n; ++k) {
        const int64_t slot = order[k];
        const int64_t i = ix(c, slot);
        const int64_t r = row_of(c, slot);
        if (c.p->moribund[r]) continue;
        if (in_dead_bubble(c, slot)) {
            if (c.s->slen[i] > 0) {
                hit(c, H_SENSOR_DEAD_BUBBLE_CHECK);
                c.s->selapsed[i] = c.s->speriod[i];
                check_for_proximity(c, slot);
            }
            umap_clear(c.s->slen + i, c.s->sbkt + i, c.s->snr + i);
            continue;
        }
        if (!c.s->has_prox[i]) continue;
        check_for_proximity(c, slot);
    }
}

// ============================================================================================ removal, interactive

// Ballpark::RemoveBall outside Evolve without delay (Ballpark.cpp:5471; TIME-711): the park's side (the Stop, the
// followers released, the ball flagged) is the caller's; here it leaves its boxes, its bubble (-1, AddToBubble), the
// proximity balls and the ball dictionary.
static HD inline void remove_ball(const Ctx& c, int64_t slot) {
    const int64_t i = ix(c, slot);
    if (!c.s->in_park[i]) return;
    const int64_t r = row_of(c, slot);
    c.p->moribund[r] = 1;
    event(c, EV_FINAL_CLEANUP, c.p->ids[r], 0, 0);
    unfile(c, slot);
    c.s->old_bubble[i] = c.s->bubble[i];
    c.s->bubble[i] = -1;
    add_to_bubble(c, slot);
    if (c.s->has_prox[i]) umap_erase(c.s->pmap + c.w * c.s->S, c.s->pn + c.w, slot);
    c.s->in_park[i] = 0;
}

// SetBallInteractive: the bubble's interactive set follows.
static HD inline void set_interactive(const Ctx& c, int64_t slot, bool flag_) {
    const int64_t i = ix(c, slot);
    const int64_t r = row_of(c, slot);
    if (flag_ && !c.p->interactive[r]) {
        c.p->interactive[r] = 1;
        increase_interactive(c, slot, c.s->bubble[i]);
    } else if (!flag_ && c.p->interactive[r]) {
        decrease_interactive(c, slot, c.s->bubble[i]);
        c.p->interactive[r] = 0;
    }
}

// ============================================================================================== the evolve's parts

static HD inline void mark_dead(const Ctx& c) {            // phase 1's mInDeadBubble, before the park's tick
    for (int64_t j = 0; j < c.s->N; ++j) {
        const int64_t r = row_of(c, j);
        const int64_t i = ix(c, j);
        c.s->dead[i] = 0;
        if (r < 0 || !c.s->in_park[i] || !c.p->free_[r] || c.p->moribund[r]) continue;
        if (in_dead_bubble(c, j) && XBPS_FAULT != 5) {
            hit(c, H_TICK_DEAD_BUBBLE);
            c.s->dead[i] = 1;
        }
    }
}

// After the park's tick: phase 3's drop-out inserts, phase 4's re-filing of each live ball that moved at least
// sqrt(0.1) m, in ascending id; then HandleProximities.
static HD inline void after_evolve(const Ctx& c) {
    for (int64_t k = 0; k < c.s->N; ++k) {
        const int64_t j = c.s->by_id[c.w * c.s->N + k];
        if (j < 0) continue;
        const int64_t i = ix(c, j);
        const int64_t r = row_of(c, j);
        if (!c.s->in_park[i] || !c.s->dropped[i]) continue;
        const double* ov = c.p->old_vel + r * 3;
        if (lensq3(ov[0], ov[1], ov[2]) < KD(K_MIN_MOVE_SQ) && XBPS_FAULT != 11) {
            hit(c, H_WARP_DROP_OUT_INSERT);
            insert_in_boxes(c, j, c.p->old_pos + r * 3, ov);
        }
    }
    for (int64_t k = 0; k < c.s->N; ++k) {
        const int64_t j = XBPS_FAULT == 10 ? (row_of(c, k) >= 0 ? k : -1) : c.s->by_id[c.w * c.s->N + k];
        if (j < 0) continue;
        const int64_t i = ix(c, j);
        const int64_t r = row_of(c, j);
        if (!c.s->in_park[i] || !c.p->free_[r] || c.p->moribund[r] || c.s->dead[i]) continue;
        const double* p = c.p->pos + r * 3;
        const double* o = c.p->old_pos + r * 3;
        if (lensq3(dsub(p[0], o[0]), dsub(p[1], o[1]), dsub(p[2], o[2])) < KD(K_MIN_MOVE_SQ)) continue;
        insert_ball_in_boxes(c, j, true);
    }
    handle_proximities(c);
    for (int64_t j = 0; j < c.s->N; ++j) c.s->dead[ix(c, j)] = 0;
}

// =============================================================================== the server layer's bubble update

// CloakBall's sensor on the master (Ballpark.cpp:5243): a normal cloak (mode 1, a signed char) adds a 2 s proximity
// sensor of the cloak's range (AddProximitySensor's refusal kept); the rest of CloakBall is the park's.
static HD inline void cloak_sensor(const Ctx& c, int64_t slot, int64_t mode, double range) {
    const int64_t m = ((mode + 128) & 0xFF) - 128;
    if (m == K_NORMAL_CLOAK) add_proximity_sensor(c, slot, range, KD(K_CLOAK_SENSOR_PERIOD), false, false);
}

// UncloakBall's part on the sensors (Ballpark.cpp:5257): a normal cloak's sensor removed; unless in warp proper the
// ball dropped from the sensor lists of the proximity balls of its bubble (in the map's order); the rest (the flags) is
// the park's, after this.
static HD inline void uncloak_sensors(const Ctx& c, int64_t slot) {
    const int64_t r = row_of(c, slot);
    if (c.p->cloaked[r] == K_NORMAL_CLOAK) remove_proximity_sensor(c, slot);
    if (c.p->mode[r] == K_MODE_WARP && c.p->effect_stamp[r] >= 0) return;
    const int64_t b = c.s->bubble[ix(c, slot)];
    for (int64_t k = 0; k < c.s->pn[c.w]; ++k) {
        const int64_t q = c.s->pmap[c.w * c.s->S + k];
        const int64_t rq = row_of(c, q);
        if (c.s->bubble[ix(c, q)] != b || c.p->moribund[rq] || !c.s->has_prox[ix(c, q)]) continue;
        umap_erase(slist_of(c, q), c.s->slen + ix(c, q), slot);
    }
}

// Ballpark::PopulateAddDelForBubble (Thunkers.cpp:2509): the bubble's additions (+1) and removals (-1) in its
// dictionary's order; no global or cloaked additions (the global flag read), no removals of a cloaked
// ball still in the park.
static HD inline void populate(const Ctx& c, int64_t bubble, int64_t* add, int64_t* addn, int64_t* del,
                               int64_t* deln) {
    const int64_t base = c.w * c.s->E;
    int64_t prev = -1;
    for (;;) {
        int64_t best = -1, best_seq = 0;
        for (int64_t e = 0; e < c.s->en[c.w]; ++e) {
            if (c.s->ebub[base + e] != bubble) continue;
            const int64_t q = c.s->eseq[base + e];
            if (q > prev && (best < 0 || q < best_seq)) {
                best = e;
                best_seq = q;
            }
        }
        if (best < 0) break;
        prev = best_seq;
        const int64_t j = c.s->eslot[base + best];
        const int64_t act = c.s->eact[base + best];
        const bool here = c.s->in_park[ix(c, j)] != 0;
        const int64_t rj = row_of(c, j);
        if (act != -1) {
            if (!here || c.p->cloaked[rj] || is_global(c, rj)) continue;
        } else if (here && c.p->cloaked[rj]) {
            continue;
        }
        if (act == 1) add[(*addn)++] = j;
        else if (act == -1) del[(*deln)++] = j;
    }
}

// Ballpark::PyAdditionsAndDeletions (Thunkers.cpp:2570-2895) and CopyBubbles, as BubbleUpdater runs them once per park
// time: per user (interactive) ball its additions and deletions, its bubble's when it stayed (the bubble's lists, or
// its own when the bubble has no interactive ball), else its own across its old and new bubbles; then its old bubble
// becomes its bubble; then CopyBubbles. A ball in two interactive sets (after UpdateBallBubble) is refused.
static HD inline void additions_and_deletions(const Ctx& c) {
    const int64_t N = c.s->N;
    for (int64_t j = 0; j < N; ++j) {
        c.s->ad_kind[ix(c, j)] = AD_NONE;
        c.s->ad_addn[ix(c, j)] = 0;
        c.s->ad_deln[ix(c, j)] = 0;
    }
    const int64_t base = c.w * c.s->I;
    for (int64_t k = 0; k < c.s->ian[c.w]; ++k) {
        const int64_t u = c.s->ias[base + k];
        for (int64_t k2 = 0; k2 < k; ++k2)
            if (c.s->ias[base + k2] == u) flag(c, FLAG_SETS);
        const int64_t i = ix(c, u);
        if (!c.s->in_park[i]) continue;
        const int64_t nb = c.s->bubble[i], ob = c.s->old_bubble[i];
        const bool dict_b = nb != -1 && dict_find(c, nb) >= 0;
        const bool dict_o = ob != -1 && dict_find(c, ob) >= 0;
        int64_t* add = c.s->ad_add + i * N;
        int64_t* del = c.s->ad_del + i * N;
        int64_t* addn = c.s->ad_addn + i;
        int64_t* deln = c.s->ad_deln + i;
        if (nb == ob && nb != -1 && dict_b) {
            c.s->ad_kind[i] = any_ia(c, nb) ? AD_SAME_BUBBLE : AD_SAME_OWN;
            populate(c, nb, add, addn, del, deln);
        } else {
            c.s->ad_kind[i] = AD_CHANGED;
            const int64_t eb = c.w * c.s->E;
            int64_t prev = -1;
            for (int pass = 0; pass < 2; ++pass) {          // the new bubble's entries, then the old bubble's
                const int64_t bub = pass == 0 ? nb : ob;
                if (pass == 0 ? !dict_b : !dict_o) continue;
                prev = -1;
                for (;;) {
                    int64_t best = -1, best_seq = 0;
                    for (int64_t e = 0; e < c.s->en[c.w]; ++e) {
                        if (c.s->ebub[eb + e] != bub) continue;
                        const int64_t q = c.s->eseq[eb + e];
                        if (q > prev && (best < 0 || q < best_seq)) {
                            best = e;
                            best_seq = q;
                        }
                    }
                    if (best < 0) break;
                    prev = best_seq;
                    const int64_t j = c.s->eslot[eb + best];
                    const int64_t act = c.s->eact[eb + best];
                    const bool here = c.s->in_park[ix(c, j)] != 0;
                    const bool cl = here && c.p->cloaked[row_of(c, j)] != 0;
                    const bool gl = here && is_global(c, row_of(c, j));
                    if (pass == 0) {
                        if (cl || gl) continue;             // no global or cloaked ball, either way
                        const bool was_in = dict_o && entry_find(c, ob, j) >= 0;
                        if (act >= 0 && !was_in) add[(*addn)++] = j;
                        else if (act == -1 && (was_in || !dict_o)) del[(*deln)++] = j;
                    } else {
                        if (act != -1) {
                            if (!here || cl || gl) continue;
                        } else if (cl) {
                            continue;
                        }
                        const bool is_in = dict_b && entry_find(c, nb, j) >= 0;
                        if (!is_in && act <= 0) del[(*deln)++] = j;
                    }
                }
            }
        }
        c.s->old_bubble[i] = nb;                            // Thunkers.cpp:2887
    }
    copy_bubbles(c);
}

// ======================================================================================================== the tape

static HD inline void apply(const Ctx& c, int64_t code, int64_t a, int64_t b, const double* f) {
    if (code == OP_NONE) return;
    const bool has_slot = a >= 0 && a < c.s->N && row_of(c, a) >= 0;
    switch (code) {
        case OP_ADD:                                      // AddBall of a new ball (its fields set by the caller)
            if (has_slot) insert_ball_in_boxes(c, a, false);
            break;
        case OP_REFILE:                                   // SetBallRadius, SetMaxSpeed, SetBallPosition (accepted)
            if (has_slot && c.s->in_park[ix(c, a)]) insert_ball_in_boxes(c, a, false);
            break;
        case OP_INIT_BUBBLES:
            if (a < 0) initialize_bubbles(c);
            else update_ball_bubble(c, a);
            break;
        case OP_COPY_BUBBLES: copy_bubbles(c); break;
        case OP_KEEP_ALIVE:
            if (has_slot) c.s->ka_ball[ix(c, a)] = 1;
            break;
        case OP_SET_INTERACTIVE:
            if (has_slot && c.s->in_park[ix(c, a)]) set_interactive(c, a, b != 0);
            break;
        case OP_ADD_SENSOR:
            if (has_slot && c.s->in_park[ix(c, a)]) add_proximity_sensor(c, a, f[0], f[1], b != 0, f[2] != 0.0);
            break;
        case OP_REMOVE_SENSOR:
            if (has_slot && c.s->in_park[ix(c, a)]) remove_proximity_sensor(c, a);
            break;
        case OP_NOTIFY_RANGE:
            if (has_slot && c.s->in_park[ix(c, a)]) set_notification_range(c, a, f[0]);
            break;
        case OP_TRACK: {                                  // b: the target's Destiny id (0: off)
            if (!(has_slot && c.s->in_park[ix(c, a)])) break;
            set_target_tracking(c, a, slot_of_id(c, b), b, f[0]);
            break;
        }
        case OP_REMOVE_BALL:
            if (has_slot) remove_ball(c, a);
            break;
        case OP_DEAD: mark_dead(c); break;
        case OP_AFTER_EVOLVE: after_evolve(c); break;
        case OP_CLEAR_EVENTS: c.s->evn[c.w] = 0; break;
        case OP_CLOAK_SENSOR:
            if (has_slot && c.s->in_park[ix(c, a)]) cloak_sensor(c, a, b, f[0]);
            break;
        case OP_UNCLOAK:
            if (has_slot && c.s->in_park[ix(c, a)]) uncloak_sensors(c, a);
            break;
        case OP_ADDITIONS_AND_DELETIONS:
            if (c.s->ad_kind) additions_and_deletions(c);
            else flag(c, FLAG_BAD_OP);
            break;
        default: flag(c, FLAG_BAD_OP); break;
    }
}

static HD inline void run_tape(const Park& p, const Space& s, const Tape& t, int64_t w) {
    const Ctx c{&p, &s, w};
    for (int64_t k = 0; k < t.K; ++k) {
        const int64_t q = w * t.K + k;
        apply(c, t.code[q], t.a[q], t.b[q], t.f + q * 4);
    }
    for (int64_t j = 0; j < s.N; ++j) sync_bubble(c, j);     // the park's own bubble field follows
}

// ===================================================================================================== the queries

// Ballpark::CheckVisibility (Ballpark.cpp:5299; SEN-709): -2 for a missing ball or another bubble; the first
// massive, uncloaked ball of the bubble (the ball dictionary's order) entered on the segment; else 0.
static HD inline int64_t check_visibility(const Ctx& c, int64_t src, int64_t dst) {
    if (src < 0 || src >= c.s->N || dst < 0 || dst >= c.s->N) return -2;
    if (row_of(c, src) < 0 || row_of(c, dst) < 0 || !c.s->in_park[ix(c, src)] || !c.s->in_park[ix(c, dst)]) return -2;
    const int64_t bs = c.s->bubble[ix(c, src)];
    if (bs != c.s->bubble[ix(c, dst)]) {
        hit(c, H_VISIBILITY_OTHER_BUBBLE);
        return -2;
    }
    const double* ps = c.p->pos + row_of(c, src) * 3;
    const double* pd = c.p->pos + row_of(c, dst) * 3;
    double ray[3] = {dsub(pd[0], ps[0]), dsub(pd[1], ps[1]), dsub(pd[2], ps[2])};
    const double distance = dsqrt(lensq3(ray[0], ray[1], ray[2]));
    const double norm = dsqrt(dadd(dadd(dmul(ray[0], ray[0]), dmul(ray[1], ray[1])), dmul(ray[2], ray[2])));
    if (norm != 0.0) {
        const double inv = ddiv(1.0, norm);
        ray[0] = dmul(ray[0], inv);
        ray[1] = dmul(ray[1], inv);
        ray[2] = dmul(ray[2], inv);
    }
    int64_t prev = -1;
    for (;;) {                                            // the ball dictionary's order
        int64_t best = -1, best_seq = 0;
        for (int64_t j = 0; j < c.s->N; ++j) {
            const int64_t i = ix(c, j);
            if (row_of(c, j) < 0 || !c.s->in_park[i]) continue;
            const int64_t q = c.s->bseq[i];
            if (q > prev && (best < 0 || q < best_seq)) {
                best = j;
                best_seq = q;
            }
        }
        if (best < 0) break;
        prev = best_seq;
        const int64_t ro = row_of(c, best);
        if (c.s->bubble[ix(c, best)] != bs || best == src || best == dst || !c.p->massive[ro] || c.p->cloaked[ro])
            continue;
        const double* po = c.p->pos + ro * 3;
        const double tc[3] = {dsub(po[0], ps[0]), dsub(po[1], ps[1]), dsub(po[2], ps[2])};
        const double v = dadd(dadd(dmul(ray[0], tc[0]), dmul(ray[1], tc[1])), dmul(ray[2], tc[2]));
        const double rr = c.p->radius[ro];
        const double dsq = dsub(dmul(rr, rr), dsub(dadd(dadd(dmul(tc[0], tc[0]), dmul(tc[1], tc[1])),
                                                         dmul(tc[2], tc[2])), dmul(v, v)));
        if (dsq < 0.0) continue;
        const double d = dsqrt(dsq);
        if (dsub(v, d) > 0.0 && dsub(v, d) < distance) {
            hit(c, H_VISIBILITY_OCCLUDED);
            return c.p->ids[ro];
        }
    }
    return 0;
}

// The existing top box at a point (GetBox(1e7, pos, false): level 0, the key clamped): its bubble, or NO_TOP.
static HD inline int64_t bubble_at(const Ctx& c, const double* pos) {
    const double w = level_width(0);
    const int64_t n = level_grid(0);
    const double fac = ddiv(1.0, w);
    const double half = dmul(dmul(0.5, w), (double)n);
    int64_t k[3];
    for (int q = 0; q < 3; ++q) k[q] = clamp_key(to_int64(dmul(dadd(pos[q], half), fac)), n);
    const int64_t t = top_find(c, k);
    return t < 0 ? K_NO_TOP : c.s->tbub[c.w * c.s->T + t];
}

// GetBallIdsInRange / GetBallIdsAndDistInRange (Thunkers.cpp:1344; SEN-710): the bubble's dictionary in its order,
// balls that left it included until CopyBubbles; out[0] the count (-1: the reference ball is missing).
static HD inline void ids_in_range(const Ctx& c, int64_t ref, const double* center_in, double rng, bool cloaked,
                                   int64_t* out, double* outf) {
    double center[3];
    int64_t bubble;
    if (ref >= 0) {
        if (ref >= c.s->N || row_of(c, ref) < 0 || !c.s->in_park[ix(c, ref)]) {
            out[0] = -1;
            return;
        }
        const double* p = c.p->pos + row_of(c, ref) * 3;
        center[0] = p[0];
        center[1] = p[1];
        center[2] = p[2];
        bubble = c.s->bubble[ix(c, ref)];
    } else {
        center[0] = center_in[0];
        center[1] = center_in[1];
        center[2] = center_in[2];
        bubble = bubble_at(c, center);
        if (bubble == K_NO_TOP) bubble = -1;
    }
    int64_t n = 0;
    out[0] = 0;
    if (bubble == -1 || dict_find(c, bubble) < 0) return;
    const int64_t base = c.w * c.s->E;
    int64_t prev = -1;
    for (;;) {                                            // the dictionary's entries in insertion order
        int64_t best = -1, best_seq = 0;
        for (int64_t e = 0; e < c.s->en[c.w]; ++e) {
            if (c.s->ebub[base + e] != bubble) continue;
            const int64_t q = c.s->eseq[base + e];
            if (q > prev && (best < 0 || q < best_seq)) {
                best = e;
                best_seq = q;
            }
        }
        if (best < 0) break;
        prev = best_seq;
        const int64_t j = c.s->eslot[base + best];
        if (!c.s->in_park[ix(c, j)] || j == ref) continue;
        if (XBPS_FAULT == 8 && c.s->eact[base + best] == -1) continue;
        const int64_t ro = row_of(c, j);
        if (c.p->moribund[ro]) {
            hit(c, H_RANGE_MORIBUND);
            continue;
        }
        if (c.p->cloaked[ro] && !cloaked) continue;
        const double* po = c.p->pos + ro * 3;
        const double d2 = lensq3(dsub(po[0], center[0]), dsub(po[1], center[1]), dsub(po[2], center[2]));
        const double lim = dadd(rng, c.p->radius[ro]);
        if (d2 <= dmul(lim, lim)) {
            out[1 + n] = j;
            if (outf) outf[n] = d2;
            ++n;
        }
    }
    out[0] = n;
}

// PyGetRemoteFollowers (Thunkers.cpp:1193; MOV-731): free balls, ascending id, following a ball missing or in
// another bubble.
static HD inline void remote_followers(const Ctx& c, int64_t* out) {
    int64_t n = 0;
    for (int64_t k = 0; k < c.s->N; ++k) {
        const int64_t j = c.s->by_id[c.w * c.s->N + k];
        if (j < 0) continue;
        const int64_t r = row_of(c, j);
        if (!c.s->in_park[ix(c, j)] || !c.p->free_[r]) continue;
        if (!is_follow_mode(c.p->mode[r])) continue;
        const int64_t d = slot_of_id(c, c.p->follow_id[r]);
        if (d >= 0 && c.s->bubble[ix(c, j)] == c.s->bubble[ix(c, d)]) continue;
        out[1 + n] = j;
        ++n;
    }
    out[0] = n;
}

static HD inline void run_query(const Park& p, const Space& s, const Query& q, int64_t w) {
    const Ctx c{&p, &s, w};
    int64_t* out = q.out + w * (1 + s.N);
    double* outf = q.outf ? q.outf + w * s.N : nullptr;
    const double* f = q.f + w * 5;
    switch (q.kind) {
        case Q_VISIBILITY: out[0] = check_visibility(c, q.a[w], q.b[w]); break;
        case Q_RANGE_BALL: ids_in_range(c, q.a[w] < 0 ? s.N : q.a[w], f, f[3], q.b[w] != 0, out, outf); break;
        case Q_RANGE_POINT: ids_in_range(c, -1, f, f[3], q.b[w] != 0, out, outf); break;
        case Q_BUBBLE_AT: out[0] = bubble_at(c, f); break;
        case Q_REMOTE_FOLLOWERS: remote_followers(c, out); break;
        default: out[0] = K_NO_TOP; break;
    }
}

#ifdef __CUDACC__
__global__ void tape_kernel(Park p, Space s, Tape t) {
    const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= s.W) return;
    run_tape(p, s, t, w);
}

__global__ void query_kernel(Park p, Space s, Query q) {
    const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= s.W) return;
    run_query(p, s, q, w);
}
#endif

}  // namespace xbps

// ====================================================================================== the C interface (ctypes)

extern "C" {

int xbps_abi(void) { return XBPS_ABI; }

// The structures' layout as compiled (the ctypes mirrors are checked against it): sizes, then each pointer field's
// offset in order.
int xbps_layout(int64_t* out) {
    out[0] = (int64_t)sizeof(xbps::Park);
    out[1] = (int64_t)sizeof(xbps::Space);
    out[2] = (int64_t)sizeof(xbps::Tape);
    out[3] = (int64_t)sizeof(xbps::Query);
    out[4] = (int64_t)offsetof(xbps::Park, effect_stamp);
    out[5] = (int64_t)offsetof(xbps::Park, bubble_out);
    out[6] = (int64_t)offsetof(xbps::Park, dt);
    out[7] = (int64_t)offsetof(xbps::Space, iab);
    out[8] = (int64_t)offsetof(xbps::Space, hits);
    out[9] = (int64_t)offsetof(xbps::Space, ad_deln);
    out[10] = (int64_t)offsetof(xbps::Space, snr);
    out[11] = (int64_t)offsetof(xbps::Space, rnd);
    out[12] = (int64_t)offsetof(xbps::Park, global_);
    return 13;
}

static int xbps_valid(const xbps::Space* s) {
    if (s->N > K_MAX_SLOTS || s->S > K_UMAP_CAP || s->S < 1 || s->N <= 0 || s->W < 0) return -1;
    return 0;
}

int xbps_tape_host(const xbps::Park* p, const xbps::Space* s, const xbps::Tape* t) {
    if (xbps_valid(s)) return -1;
    for (int64_t w = 0; w < s->W; ++w) xbps::run_tape(*p, *s, *t, w);
    return 0;
}

int xbps_query_host(const xbps::Park* p, const xbps::Space* s, const xbps::Query* q) {
    if (xbps_valid(s)) return -1;
    for (int64_t w = 0; w < s->W; ++w) xbps::run_query(*p, *s, *q, w);
    return 0;
}

#ifdef __CUDACC__
static unsigned grid_for(int64_t n, int block) { return (unsigned)((n + block - 1) / block); }

static int launch_check(void) {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

int xbps_tape_cuda(const xbps::Park* p, const xbps::Space* s, const xbps::Tape* t, void* stream, int block) {
    if (xbps_valid(s)) return -1;
    if (s->W == 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbps::tape_kernel<<<grid_for(s->W, block), block, 0, (cudaStream_t)stream>>>(*p, *s, *t);
    return launch_check();
}

int xbps_query_cuda(const xbps::Park* p, const xbps::Space* s, const xbps::Query* q, void* stream, int block) {
    if (xbps_valid(s)) return -1;
    if (s->W == 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0) return -1;
    xbps::query_kernel<<<grid_for(s->W, block), block, 0, (cudaStream_t)stream>>>(*p, *s, *q);
    return launch_check();
}

int xbps_info(int which, int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    const cudaError_t e = which == 0 ? cudaFuncGetAttributes(&fa, xbps::tape_kernel)
                                     : cudaFuncGetAttributes(&fa, xbps::query_kernel);
    if (e != cudaSuccess) return (int)e;
    *regs = fa.numRegs;
    *local_bytes = (int)fa.localSizeBytes;
    *binary_version = fa.binaryVersion;
    *ptx_version = fa.ptxVersion;
    *max_threads = fa.maxThreadsPerBlock;
    return 0;
}

const char* xbps_error_string(int e) { return cudaGetErrorString((cudaError_t)e); }
#endif

}  // extern "C"
