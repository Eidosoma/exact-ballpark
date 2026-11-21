# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Small CPU checks of the arithmetic operations and domains in the Lean register."""
from fractions import Fraction
import math
from pathlib import Path
import random
import re
import struct
from unittest.mock import patch

import torch

from exact_ballpark import vec_accuracy, vec_exp
from exact_ballpark.full import FullBallpark
import contact_replay
from replay_discrepancies import radial_intermediates


def word(value):
    return struct.pack(">d", value).hex()


def grid(value):
    """Spacing on the side of a binary binade containing a nonzero dyadic."""
    return Fraction(2) ** (abs(value.numerator).bit_length() - value.denominator.bit_length() - 52)


def width(value):
    """Number of significand bits after removing trailing binary zeros."""
    n = abs(value.numerator)
    return (n // (n & -n)).bit_length() if n else 0


def check_decay():
    path = Path(__file__).resolve().parents[1] / "ExactBallparkProofs/Collision/ReplayTick.lean"
    pairs = re.findall(r"\| 0x(\w{16}) => 0x(\w{16})", path.read_text(encoding="utf-8"))
    if len(pairs) != 9 or len(dict(pairs)) != 9:
        raise SystemExit("Unexpected four-ball decay table")
    for argument, result in pairs:
        (x,) = struct.unpack(">d", bytes.fromhex(argument))
        if word(math.exp(x)) != result:
            raise SystemExit(f"Host exp differs for the four-ball replay at {argument}")


def check_witnesses(run_lake):
    engine, control = contact_replay.rebound()
    observed = {"engine-endpoint": engine, "control-endpoint": control,
                "return-times": contact_replay.free_repeat(), "radial-inputs": radial_intermediates()}
    rows = run_lake("env", "lean", "--run", "scripts/ArithmeticWitnessProbe.lean", capture=True).splitlines()
    for row in rows:
        fields = row.split()
        if len(fields) != 3 or fields[0] not in observed or not all(n.isdecimal() for n in fields[1:]):
            raise SystemExit(f"Unexpected Lean witness row {row!r}")
        label, *digits = fields
        certified = [f"{int(n):016x}" for n in digits]
        cpu = [word(value) for value in observed.pop(label)]
        if cpu != certified:
            raise SystemExit(f"{label} differs: Lean {certified}, scalar replay {cpu}")
    if observed:
        raise SystemExit(f"Missing Lean witness rows: {', '.join(observed)}")


def check_accuracy():
    park = FullBallpark()
    park.add_ball(1, 1e6, 1, 100, is_free=True)
    park.add_ball(2, 1e6, 1, 100, is_free=True, x=3, y=4, z=12)
    park.set_velocity(2, 0.03, 0.04, 0.12)
    p, v = park.balls[2].pos, park.balls[2].vel
    (nan,) = struct.unpack(">d", bytes.fromhex("7ff8000000000042"))
    # dst, optimal, falloff, tracking, signature. Each intended early return
    # must preserve the original optimal and tracking words.
    early = [
        (1, 37.0, 1, 0.75, 1), (9, -0.0, 1, nan, 1), (2, -1, 1, 1, 1),
        (2, -0.0, 1, -0.0, 1), (2, -0.0, 1, 1, 0), (2, nan, -0.0, 1, 1),
        (3, -0.0, 1, 1, 1),
    ]
    zero = (0.0, 0.0, 0.0)
    park.add_ball(3, 1e6, 1, 100, is_free=True)
    calls = early + [(2, 1e30, 1, 1, 1)]
    # A real ball queried against itself also has coincident centers. Check that the
    # scalar same-ID return happens before the first ball lookup.
    with patch.object(park, "balls", wraps=park.balls) as registry:
        park.get_accuracy(1, *early[0], 1e30)
    if registry.get.called:
        raise SystemExit("Same-ball scalar accuracy reached the registry lookup")

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    targets, optimal, falloff, tracking, signature = zip(*calls)
    # Use separated tensor coordinates for the same-ID lane so only the same mask can stop it.
    out = vec_accuracy.get_accuracy(
        tensor([zero] * len(calls)), tensor([zero] * len(calls)), 1.0,
        tensor([zero if dst == 3 else p for dst in targets]),
        tensor([zero if dst == 3 else v for dst in targets]), 1.0,
        tensor(optimal), tensor(falloff), tensor(tracking), tensor(signature), 1e30,
        same=torch.tensor([dst == 1 for dst in targets]),
        present=torch.tensor([dst != 9 for dst in targets]),
    )
    for i, (dst, opt, fall, track, sig) in enumerate(calls):
        scalar = park.get_accuracy(1, dst, opt, fall, track, sig, 1e30)
        lane = tuple(vec[i].item() for vec in out)
        if i < len(early):
            expected = list(map(word, (0.0, opt, track)))
            if list(map(word, scalar)) != expected or list(map(word, lane)) != expected:
                raise SystemExit(f"Early accuracy words changed in lane {i}")
        elif not all(a == math.inf and d == 11 and math.isnan(s) for a, d, s in (scalar, lane)):
            raise SystemExit("Radial accuracy fixture changed")
    return len(calls)


def check_fma():
    rng = random.Random(17017)
    # Include both signs, binade boundaries, cancellation and an inexact product.
    triples = [
        (1.0 + 2**-27, 1.0 - 2**-27, -1.0), (1.0, 1.0, 2**-53),
        (1.0 + 2**-52, 1.0 + 2**-52, -1.0), (-1.0, 1.0, -2**-53),
    ]

    def sample():
        significand = 2**52 + rng.randrange(2**52)
        return math.ldexp(rng.choice((-1, 1)) * significand, rng.randrange(-10, 11) - 52)

    triples.extend(tuple(sample() for _ in range(3)) for _ in range(64))
    a, b, c = (torch.tensor(col, dtype=torch.float64) for col in zip(*triples))
    ah, al = vec_exp.split(a)
    bh, bl = vec_exp.split(b)
    uh, ul = vec_exp.two_product(a, b)
    th, tl = vec_exp.two_sum(c, uh)
    odd = vec_exp.add_odd(tl, ul)
    result = vec_exp.fma(a, b, c)
    certified = 0
    inexact_odd = 0
    for i, (x, y, z) in enumerate(triples):
        product = Fraction(x) * Fraction(y)
        high_x, low_x, high_y, low_y = (Fraction(v[i].item()) for v in (ah, al, bh, bl))
        product_hi, product_lo, sum_hi, sum_lo = (Fraction(v[i].item()) for v in (uh, ul, th, tl))
        hh, lh, hl, ll = high_x * high_y, low_x * high_y, high_x * low_y, low_x * low_y
        parts = (hh, lh, hl, ll, product_hi - hh, product_hi - hh - lh, product_hi - hh - lh - hl)
        if (high_x + low_x != x or high_y + low_y != y or max(map(width, parts)) > 53
                or product_hi + product_lo != product or product_hi != float(product)):
            raise SystemExit(f"Product expansion or partial width failed for {triples[i]}")
        head_sum = Fraction(z) + product_hi
        if sum_hi != float(head_sum) or sum_hi + sum_lo != head_sum:
            raise SystemExit(f"TwoSum expansion failed for {triples[i]}")
        exact = product + Fraction(z)
        low = product_lo + sum_lo
        expected_odd = float(low)
        rounded = Fraction(expected_odd)
        if low != rounded and int(word(expected_odd), 16) % 2 == 0:
            expected_odd = math.nextafter(expected_odd, math.inf if low > rounded else -math.inf)
        if expected_odd != odd[i].item() or word(result[i]) != word(math.fma(x, y, z)):
            raise SystemExit(f"Odd rounding or single rounding failed for {triples[i]}")
        assembled = sum_hi + Fraction(expected_odd)
        if low and exact and assembled:
            fine, coarse = grid(low), grid(exact)
            if coarse >= 4 * fine and grid(assembled) == coarse and (sum_hi / fine) % 2 == 0:
                certified += 1
                inexact_odd += low != rounded
    if not inexact_odd:
        raise SystemExit("FMA fixtures missed an inexact odd step inside the theorem's domain")
    return len(triples), certified, inexact_odd


def compare(run_lake):
    check_decay()
    check_witnesses(run_lake)
    lanes = check_accuracy()
    cases, admitted, odd = check_fma()
    return lanes, f"{cases} software FMAs ({admitted} inside the grid conditions, {odd} with inexact low sums)"
