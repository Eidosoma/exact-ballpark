// SPDX-License-Identifier: LGPL-2.1-or-later
// The GNU C Library 2.34's rand() (random_r, TYPE_3) for the space kernel.
// Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md. Included where it stands by vec_space.cu; a build inlines it (_cuda_build.read_source).

// glibc's rand() (random_r, TYPE_3: degree 31, separation 3), on the world's own state: the sum of
// the table's words at fptr and rptr, modulo 2^32, replaces the word at fptr (as a signed 32-bit value); the result is
// the sum shifted right by one; fptr and rptr advance, wrapping at the table's end (rptr wraps only when fptr does
// not). exact_ballpark.space.GlibcRandom.next, transcribed; integer operations only.
static HD inline int64_t glibc_rand(const Ctx& c) {
    int64_t* s = c.s->rnd + c.w * K_RAND_WORDS;
    int64_t f = s[K_RAND_DEG], r = s[K_RAND_DEG + 1];
    const uint32_t val = (uint32_t)s[f] + (uint32_t)s[r];
    const int64_t word = val >= 0x80000000u ? (int64_t)val - 0x100000000LL : (int64_t)val;
    if (XBPS_FAULT != 18) s[f] = word;                  // (fault 18: the table never fed back)
    f += 1;
    if (f >= K_RAND_DEG) {
        f = 0;
        r += 1;
    } else {
        r += 1;
        if (r >= K_RAND_DEG) r = 0;
    }
    s[K_RAND_DEG] = f;
    s[K_RAND_DEG + 1] = r;
    return XBPS_FAULT == 17 ? (int64_t)val : (int64_t)(val >> 1);    // (fault 17: not shifted)
}

