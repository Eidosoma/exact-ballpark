# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (c) 2026 Eidosoma
# Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md.
"""The GNU C Library's rand() (random_r, TYPE_3: degree 31, separation 3) after srand(seed), as destiny's floatrand
(Ball.cpp:19) draws it for shuffled sensors. Transcribed from the GNU C Library 2.34: LGPL-2.1-or-later."""


def _cdiv(a: int, b: int) -> int:
    """C's integer division (toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class GlibcRandom:
    """glibc's rand() (random(), TYPE_3: degree 31, separation 3) after srand(seed): floatrand's source (Ball.cpp:19)
    for shuffled sensors. The driver seeds it per case."""

    def __init__(self, seed: int):
        seed &= 0xFFFFFFFF
        if seed == 0:
            seed = 1
        word = seed - (1 << 32) if seed >= (1 << 31) else seed
        s = [word]
        for _ in range(1, 31):
            hi, lo = _cdiv(word, 127773), word - 127773 * _cdiv(word, 127773)
            word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            s.append(word)
        self.s, self.f, self.r = s, 3, 0
        for _ in range(310):
            self.next()

    def next(self) -> int:
        val = (self.s[self.f] + self.s[self.r]) & 0xFFFFFFFF
        self.s[self.f] = val - (1 << 32) if val >= (1 << 31) else val
        self.f += 1
        if self.f >= 31:
            self.f = 0
            self.r += 1
        else:
            self.r += 1
            if self.r >= 31:
                self.r = 0
        return val >> 1
