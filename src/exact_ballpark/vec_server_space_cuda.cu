// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// Destiny's server layer over the bubble partition for VecModesPark on a CUDA device, one thread a world: what exact_ballpark.vec_server_space.SpaceServer's torch path does in some hundreds of launches a tick, as
// one launch for the pre-tick and one for the post-tick's send flags. The pre-tick: the follower clean-up's Stops
// (GetRemoteFollowers, queued behind the pushed calls); TIME-710's duplicate test against the world's last queued
// action; each ball's (and a FollowBall's or Orbit's target's) row; CloakBall's bubble captured from the partition as
// it stood when the call was queued (its old bubble while the two differ); every action but CloakBall applied in queue
// order by the park's own rules, each followed by the partition's part (the re-filing of SetBallPosition, and of
// SetBallRadius and SetMaxSpeed when accepted; SetBallInteractive's sets; UncloakBall's sensors); the bubble update
// (AdditionsAndDeletions and CopyBubbles once per park time) with its lists per row; the history in queue order with
// the routes per bubble (an event to its ball's bubble when that bubble has a dictionary and an interactive ball, to
// its ball's character while the ball is cloaked; CloakBall to the bubble it was queued in, its sensor added and then
// applied; an uncloak's notices to the other interactive balls of its bubble). The post-tick (after Evolve, which is
// exact_ballpark.vec_space's tick, unchanged): each row's pilot's flags (a character update; the narrowcast of the bubble
// whose interactive set holds its ship). Built and loaded by exact_ballpark.vec_server_space_cuda; held bit for bit to
// the torch path (and through it to exact_ballpark.server) by tests/test_movement_vec_server_space_cuda.py.
//
// It includes vec_server_cuda.cu (the kernel: the commands and setters of VecModesPark, through
// vec_modes_cuda.cu and vec_orient_cuda.cu) and vec_space.cu (the partition: its operations are run here
// as its tape runs them, one operation then the park's bubble field synced), all unedited, for their functions.
//
// The phases (one launch does 1 and 2, or one each; 2 can run over a range of slots, so the host can read the park
// between them, as the tests do for the streams the bubble update and the uncloaks send):
//   1. the pushed calls' duplicate test and lookups, CloakBall's bubble; the clean-up's Stops; the stamps; every action
//      but CloakBall applied with the partition's part; the bubble update and its lists per row;
//   2. the history over slots [j0, j1): the routes and their bubbles, CloakBall's sensor and the cloak, the notices;
//   3. (after Evolve) the send flags of each row's pilot.
//
// Exactness. The commands' arithmetic is the kernel's (the torch path's, operation for operation); the
// partition's is the (doubles with __dadd_rn ... __dsqrt_rn, float sums in binary32); this file adds index work
// and integer steps only, no floating-point arithmetic but the setters' acceptance test (a float argument rounded to
// float and compared with zero, as the torch path's L.f32). No function of CUDA's math library; every constant from
// the generated headers (the kinds' codes and routes, the partition's operations and lists' kinds).

#include "vec_server_cuda.cu"
// vec_space.cu's two launch helpers have the names of vec_orient_cuda.cu's (both file-local, at global scope): renamed
// for this translation unit only, the sources unedited
#define grid_for xbps_grid_for_
#define launch_check xbps_launch_check_
#include "vec_space.cu"
#undef grid_for
#undef launch_check

#define XBPQ_ABI 1
#ifndef XBPQ_FAULT
#define XBPQ_FAULT 0                                     // a planted fault's number (the tests' builds only)
#endif

namespace xbpq {

using xbpm::Modes;
using xbpv::Queue;
using xbpv::Srv;

// the server layer's state and outputs beyond the Queue and Srv
struct Extra {
    int64_t* ad_time;                // [W] the park time of the last bubble update (SpaceServer._ad_time)
    int64_t* cleanup;                // [W, 1 + N] GetRemoteFollowers at this pre-tick: the count, then the slots
    int64_t* cleanup_total;          // [W] the clean-up's Stops, summed over the pre-ticks
    int64_t* rbubble;                // [W, J] the bubble a BUBBLE route goes to
    int64_t* ad_kind;                // [R] the bubble update's kind per row (AD_*)
    int64_t* ad_adds;                // [R, N] its additions as rows
    int64_t* ad_addn;                // [R]
    int64_t* ad_dels;                // [R, N] its deletions as rows
    int64_t* ad_deln;                // [R]
    int64_t* narrow;                 // [R] the bubble whose narrowcast the row's pilot gets, or -1
    const int64_t* slot_of_row;      // [R] the row's slot in its world (the partition's)
    int64_t J_user;                  // the pushed calls' slots; the clean-up's Stops take slots J_user .. J - 1
};

static __device__ __forceinline__ float f32(double x) { return xbpm::tof(x); }

// ---- the partition ----

static __device__ void op(const xbps::Ctx& c, int64_t code, int64_t slot, int64_t n, double f0) {
    // one operation of the partition layer, as exact_ballpark.vec_space's tape runs it (VecSpace.tape with one
    // operation): applied (a slot below zero: none), then the park's bubble field synced from the partition
    const double f[4] = {f0, 0.0, 0.0, 0.0};
    if (slot >= 0 || code == OP_ADDITIONS_AND_DELETIONS) xbps::apply(c, code, slot, n, f);
    for (int64_t j = 0; j < c.s->N; ++j) xbps::sync_bubble(c, j);
}

static __device__ __forceinline__ int64_t bubble_of(const xbps::Ctx& c, const Extra& x, int64_t r) {
    return c.s->bubble[xbps::ix(c, x.slot_of_row[r])];   // newBubbleId
}

static __device__ __forceinline__ bool dict_exists(const xbps::Ctx& c, int64_t bubble) {   // park.bubbles
    return bubble != -1 && xbps::dict_find(c, bubble) >= 0;
}

static __device__ __forceinline__ bool in_ia(const xbps::Ctx& c, int64_t slot, int64_t bubble) {
    // whether the slot is in the interactive set of `bubble` (park.bubbleInteractives[bubble])
    if (bubble == -1) return false;
    const int64_t base = c.w * c.s->I;
    for (int64_t k = 0; k < c.s->ian[c.w]; ++k) {
        if (c.s->iab[base + k] == bubble && c.s->ias[base + k] == slot) return true;
    }
    return false;
}

static __device__ __forceinline__ int64_t ia_bubble_of(const xbps::Ctx& c, int64_t slot) {
    // the bubble whose interactive set holds the slot, or -1 (the last pair: a ball in two sets is flagged)
    int64_t b = -1;
    const int64_t base = c.w * c.s->I;
    for (int64_t k = 0; k < c.s->ian[c.w]; ++k) {
        if (c.s->ias[base + k] == slot) b = c.s->iab[base + k];
    }
    return b;
}

// ---- the adapter's cloak and uncloak (SpaceModesAdapter: the sensor is the partition's) ----

static __device__ void cloak_space(const Modes& m, const xbpv::Park& p, const Srv& s, int64_t w, int64_t r, int64_t n) {
    // _Adapter.cloak: refused unless the (signed char) mode is positive; the followers released; cloaked with the
    // mode's byte, non-massive (the normal cloak's sensor was added by the partition's CLOAK_SENSOR before)
    const int64_t mode = ((n + 128) & 0xFF) - 128;
    if (!(mode > 0)) return;
    xbpv::release_followers(m, s, w, r);
    p.cloaked[r] = mode & 0xFF;
    m.massive[r] = 0;
}

static __device__ void uncloak_space(const Modes& m, const xbpv::Park& p, int64_t r) {
    // _Adapter.uncloak: uncloaked; massive again unless in warp proper (the sensor removed by the partition's UNCLOAK)
    p.cloaked[r] = 0;
    if (!(m.mode[r] == MODE_WARP && m.effect_stamp[r] >= 0)) m.massive[r] = 1;
}

// ---- the phases ----

static __device__ void pass1(const Modes& m, const xbpv::Park& p, const Queue& q, const Srv& s, const xbps::Park& sp,
                             const xbps::Space& ss, const Extra& x, int64_t w) {
    const xbps::Ctx c{&sp, &ss, w};
    const int64_t N = s.N;
    // (1) the pushed calls, as SpaceServer.push queued them: the duplicate test, the rows, CloakBall's bubble from the
    //     partition before anything of this tick is applied, not queued for a ball not in the park
    bool has_last = false;
    int lk = 0;
    int64_t lb = 0, lt = 0, ln = 0, ll = 0;
    double lf[4] = {0.0, 0.0, 0.0, 0.0};
    for (int64_t j = 0; j < x.J_user; j++) {
        const int64_t i = j * s.W + w;
        const int kind = q.kind[j];
        const int64_t b = q.ball[i], t = q.target[i], n = q.n[i];
        const double* f = q.f + 4 * i;
        const int64_t r = xbpv::lookup(s, w, b);
        const int64_t tr = (kind == K_FOLLOW_BALL || kind == K_ORBIT) ? xbpv::lookup(s, w, t) : -1;
        const uint8_t mv = q.mask[i];                    // bit 0: pushed; bit 1: tested by an undo
        bool mask = (mv & 1) != 0;
        int64_t loc = q.loc_in[i];
        if (kind == K_CLOAK_BALL) {
            if (r >= 0) {
                const int64_t k = xbps::ix(c, x.slot_of_row[r]);
                const int64_t nb = ss.bubble[k], ob = ss.old_bubble[k];
#if XBPQ_FAULT == 2
                loc = nb;                                // planted: the ball's current bubble, the old one ignored
#else
                loc = nb != ob ? ob : nb;
#endif
            }
            mask = mask && r >= 0;
        }
#if XBPQ_FAULT == 8
        const bool tested = false;                       // planted: an undo's tested calls tested again
#else
        const bool tested = (mv & 2) != 0;               // the undo's outcome: queued as bit 0 says, no second test
#endif
        const bool dup = !tested && has_last && lk == kind && lb == b && lt == t && lf[0] == f[0] && lf[1] == f[1]
                         && lf[2] == f[2] && lf[3] == f[3] && ln == n && ll == loc;
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
        for (int k = 0; k < 4; k++) lf[k] = f[k];
    }
    // (2) the follower clean-up (Actions.clean_up_followers, MOV-731): a Stop for each free ball following a ball
    //     missing or in another bubble, ascending id, queued behind the pushed calls under the same duplicate test
    int64_t* out = x.cleanup + w * (1 + N);
    xbps::remote_followers(c, out);
#if XBPQ_FAULT == 1
    out[0] = 0;                                          // planted: the follower clean-up skipped
#endif
    x.cleanup_total[w] += out[0];
    int64_t* qball = const_cast<int64_t*>(q.ball);       // the clean-up's slots: their balls are written here
    for (int64_t j = x.J_user; j < q.J; j++) {
        const int64_t k = j - x.J_user;
        const int64_t i = j * s.W + w;
        const bool has = k < out[0];
        const int64_t b = has ? s.ids_of_world[w * N + out[1 + k]] : 0;
        qball[i] = b;
        const int64_t r = xbpv::lookup(s, w, b);
        const bool dup = has_last && lk == K_STOP && lb == b && lt == 0 && lf[0] == 0.0 && lf[1] == 0.0
                         && lf[2] == 0.0 && lf[3] == 0.0 && ln == 0 && ll == 0;
        const bool ok = has && !dup;
        q.ok[i] = ok;
        q.row[i] = r;
        q.trow[i] = -1;
        q.loc[i] = 0;
        if (!ok) continue;
        has_last = true;
        lk = K_STOP;
        lb = b;
        lt = 0;
        ln = 0;
        ll = 0;
        for (int e = 0; e < 4; e++) lf[e] = 0.0;
    }
    // (3) the stamps (_increment_timestamp after the clean-up)
    const int64_t first = s.rows_of_world[w * N];
    const int64_t clock = m.clock[first];
    s.stamp[w] = s.stamp_for_system[w];
    s.clock[w] = clock;
    s.stamp_for_system[w] = clock + 1;
    // (4) every action but CloakBall in queue order, the partition's part after each
    for (int64_t j = 0; j < q.J; j++) {
        const int64_t i = j * s.W + w;
        const int kind = q.kind[j];
        const int64_t r = q.row[i];
        if (!q.ok[i] || r < 0 || kind == K_CLOAK_BALL) continue;
        const int64_t slot = x.slot_of_row[r];
        if (kind == K_SET_BALL_INTERACTIVE) {            // the partition sets the flag and the bubble's set
            op(c, OP_SET_INTERACTIVE, slot, q.n[i], 0.0);
            continue;
        }
        if (kind == K_UNCLOAK_BALL) {                    // the sensors (they read the cloak's mode), then the flags
            op(c, OP_UNCLOAK, slot, 0, 0.0);
            uncloak_space(m, p, r);
            continue;
        }
        const double* f = q.f + 4 * i;
        bool refile = kind == K_SET_BALL_POSITION || kind == K_SET_BALL_RADIUS || kind == K_SET_MAX_SPEED;
        if (kind == K_SET_BALL_RADIUS || kind == K_SET_MAX_SPEED) refile = !(xbpm::tod(f32(f[0])) < 0.0);
#if XBPQ_FAULT != 9
        if (kind == K_SET_BALL_FREE) refile = q.n[i] == 0 && m.free_[r];    // fixing a free ball
#endif                                                   // (planted 9: SetBallFree's re-filing left out)
        xbpv::apply(m, p, s, w, kind, r, q.trow[i], f, q.n[i], q.loc[i]);
#if XBPQ_FAULT == 3
        refile = false;                                  // planted: the setters do not re-file their ball
#endif
        if (refile) op(c, OP_REFILE, slot, 0, 0.0);
    }
    // (5) BubbleUpdater: AdditionsAndDeletions and CopyBubbles once per park time
    if (clock != x.ad_time[w]) op(c, OP_ADDITIONS_AND_DELETIONS, -1, 0, 0.0);
    x.ad_time[w] = clock;
    // the bubble update's lists per row (SpaceServer._ad_rows: every entry mapped, slots made rows)
    for (int64_t k = 0; k < N; k++) {
        const int64_t r = s.rows_of_world[w * N + k];
        if (r < 0) continue;
        const int64_t i = w * N + k;
        x.ad_kind[r] = ss.ad_kind[i];
        x.ad_addn[r] = ss.ad_addn[i];
        x.ad_deln[r] = ss.ad_deln[i];
        for (int64_t e = 0; e < N; e++) {
            const int64_t a = ss.ad_add[i * N + e], d = ss.ad_del[i * N + e];
            x.ad_adds[r * N + e] = s.rows_of_world[w * N + (a < 0 ? 0 : a)];
            x.ad_dels[r * N + e] = s.rows_of_world[w * N + (d < 0 ? 0 : d)];
        }
    }
}

static __device__ void pass2(const Modes& m, const xbpv::Park& p, const Queue& q, const Srv& s, const xbps::Park& sp,
                             const xbps::Space& ss, const Extra& x, int64_t w, int64_t j0, int64_t j1) {
    const xbps::Ctx c{&sp, &ss, w};
    const int64_t N = s.N;
    for (int64_t j = j0; j < j1; j++) {
        const int64_t i = j * s.W + w;
        const int kind = q.kind[j];
        const bool ok = q.ok[i] != 0;
        const int64_t r = q.row[i];
        const bool valid = ok && r >= 0;
        int8_t route = SV_NONE;
        int64_t rb = -1;
        if (kind == K_CLOAK_BALL) {                      // to the bubble queued in; its sensor; then applied
            if (ok) {
                route = SV_BUBBLE;
                rb = q.loc[i];
            }
            if (valid) {
                op(c, OP_CLOAK_SENSOR, x.slot_of_row[r], q.n[i], q.f[4 * i]);
                cloak_space(m, p, s, w, r, q.n[i]);
            }
        } else if (kind == K_UNCLOAK_BALL) {             // its own character; AddBallsToPark to its bubble's others
            const int64_t bub = valid ? bubble_of(c, x, r) : -1;
            if (valid && in_ia(c, x.slot_of_row[r], bub) && s.pilot[r] >= 0) route = SV_CHARACTER;
            for (int64_t k = 0; k < N; k++) {
                const int64_t y = s.rows_of_world[w * N + k];
                if (y < 0) continue;
#if XBPQ_FAULT == 6
                const bool rec = valid && ia_bubble_of(c, k) != -1;   // planted: every interactive ball of the world
#else
                const bool rec = in_ia(c, k, bub);
#endif
                q.notice[j * s.R + y] = rec && y != r && s.pilot[y] >= 0;
            }
        } else if (valid) {                              // its bubble's, or its character's while cloaked
            const int64_t bub = bubble_of(c, x, r);
#if XBPQ_FAULT == 4
            bool live = q.ball[i] > 0 && bub != -1;      // planted: a dead bubble's events sent all the same
#else
            bool live = q.ball[i] > 0 && dict_exists(c, bub) && xbps::any_ia(c, bub);
#endif
            if (kind == K_SET_BALL_POSITION) live = live && q.loc[i] == 0;
#if XBPQ_FAULT == 5
            const bool cl = false;                       // planted: a cloaked ball's events to the bubble
#else
            const bool cl = p.cloaked[r] != 0;
#endif
            if (live) route = !cl ? SV_BUBBLE : (s.pilot[r] >= 0 ? SV_CHARACTER : SV_NONE);
            if (live && !cl) rb = bub;
        }
        q.route[w * q.J + j] = route;
        x.rbubble[w * q.J + j] = rb;
    }
}

static __device__ void flags(const Modes& m, const Queue& q, const Srv& s, const xbps::Park& sp,
                             const xbps::Space& ss, const Extra& x, int64_t w) {
    // send_batch after Evolve: per row, the bubble whose narrowcast its pilot gets (its ship in that bubble's
    // interactive set, the bubble with history), and whether it has a character update
    const xbps::Ctx c{&sp, &ss, w};
    const int64_t N = s.N;
    for (int64_t k = 0; k < N; k++) {
        const int64_t y = s.rows_of_world[w * N + k];
        if (y < 0) continue;
#if XBPQ_FAULT == 7
        const int64_t iab = ss.bubble[xbps::ix(c, k)];   // planted: the ship's bubble, not its interactive set's
#else
        const int64_t iab = ia_bubble_of(c, k);
#endif
        bool hist = false, has = false;
        for (int64_t j = 0; j < q.J; j++) {
            const int64_t i = j * s.W + w;
            const int8_t route = q.route[w * q.J + j];
            const int64_t rb = x.rbubble[w * q.J + j];
            if (route == SV_BUBBLE && rb != -1 && iab == rb) hist = true;
            if (q.ok[i] && q.row[i] == y && route == SV_CHARACTER) has = true;
            if (q.kind[j] == K_UNCLOAK_BALL && q.notice[j * s.R + y]) has = true;
        }
        if (x.ad_kind[y] != AD_NONE && (x.ad_addn[y] > 0 || x.ad_deln[y] > 0)) has = true;
        const bool pilot = s.pilot[y] >= 0;
        s.has_character[y] = has && pilot;
        s.gets_bubble[y] = hist && pilot;
        x.narrow[y] = hist && pilot ? iab : -1;
    }
}

__global__ void server_space_kernel(Modes m, xbpv::Park p, Queue q, Srv s, xbps::Park sp, xbps::Space ss, Extra x,
                                    int phase, int64_t j0, int64_t j1) {
    const int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= s.W) return;
    if (phase == 0 || phase == 1) pass1(m, p, q, s, sp, ss, x, w);
    if (phase == 0) pass2(m, p, q, s, sp, ss, x, w, 0, q.J);
    if (phase == 2) pass2(m, p, q, s, sp, ss, x, w, j0, j1);
    if (phase == 3) flags(m, q, s, sp, ss, x, w);
}

}  // namespace xbpq

// ===================================================================================================== C entry points

extern "C" {

int xbpq_abi() { return XBPQ_ABI; }

int xbpq_fault() { return XBPQ_FAULT; }

// Extra's layout as compiled (its ctypes mirror is checked against it): the size, then each field's offset in order.
int xbpq_layout(int64_t* out) {
    out[0] = (int64_t)sizeof(xbpq::Extra);
    out[1] = (int64_t)offsetof(xbpq::Extra, ad_time);
    out[2] = (int64_t)offsetof(xbpq::Extra, cleanup);
    out[3] = (int64_t)offsetof(xbpq::Extra, cleanup_total);
    out[4] = (int64_t)offsetof(xbpq::Extra, rbubble);
    out[5] = (int64_t)offsetof(xbpq::Extra, ad_kind);
    out[6] = (int64_t)offsetof(xbpq::Extra, ad_adds);
    out[7] = (int64_t)offsetof(xbpq::Extra, ad_addn);
    out[8] = (int64_t)offsetof(xbpq::Extra, ad_dels);
    out[9] = (int64_t)offsetof(xbpq::Extra, ad_deln);
    out[10] = (int64_t)offsetof(xbpq::Extra, narrow);
    out[11] = (int64_t)offsetof(xbpq::Extra, slot_of_row);
    out[12] = (int64_t)offsetof(xbpq::Extra, J_user);
    out[13] = (int64_t)sizeof(xbpv::Queue);
    out[14] = (int64_t)offsetof(xbpv::Queue, J);
    out[15] = (int64_t)sizeof(xbpv::Srv);
    out[16] = (int64_t)offsetof(xbpv::Srv, neg_kdt);
    return 17;
}

const char* xbpq_error_string(int rc) { return cudaGetErrorString((cudaError_t)rc); }

int xbpq_pre_tick(const xbpm::Modes* m, const xbpv::Park* p, const xbpv::Queue* q, const xbpv::Srv* s,
                  const xbps::Park* sp, const xbps::Space* ss, const xbpq::Extra* x, int phase, int64_t j0,
                  int64_t j1, void* stream, int block) {
    // phases 0 to 3 (see the file's head); -1: refused (the slots, the block, the phase, the sizes)
    if (q->J < 0 || q->J > XBPV_MAX_SLOTS || x->J_user < 0 || x->J_user > q->J || block <= 0 || block > 1024
        || phase < 0 || phase > 3 || ss->W != s->W || ss->N != s->N || ss->N > K_MAX_SLOTS || ss->S > K_UMAP_CAP
        || j0 < 0 || j1 > q->J || j0 > j1)
        return -1;
    if (s->W == 0) return 0;
    const int64_t grid = (s->W + block - 1) / block;
    xbpq::server_space_kernel<<<(unsigned)grid, block, 0, (cudaStream_t)stream>>>(*m, *p, *q, *s, *sp, *ss, *x, phase,
                                                                                  j0, j1);
    return (int)cudaGetLastError();
}

int xbpq_info(int* out) {
    // the kernel's registers, local bytes, threads a block, binary and PTX versions (loading its module now)
    cudaFuncAttributes a;
    const cudaError_t e = cudaFuncGetAttributes(&a, xbpq::server_space_kernel);
    if (e != cudaSuccess) return (int)e;
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = a.maxThreadsPerBlock;
    out[3] = a.binaryVersion;
    out[4] = a.ptxVersion;
    return 0;
}

}  // extern "C"
