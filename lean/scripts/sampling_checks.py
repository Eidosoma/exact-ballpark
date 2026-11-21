# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Small CPU fixtures for exponential reduction, warp samples and ordered range results."""
import math
from decimal import Decimal, localcontext
from unittest.mock import patch

import torch

from exact_ballpark import vec_exp
from exact_ballpark.modes import ModeBallpark
from exact_ballpark.space import SpaceBallpark


def exponential():
    if vec_exp.INVLN2N.as_integer_ratio() != (3248660424278399, 17592186044416):
        raise SystemExit("The stored reduction multiplier differs from the proof")
    tiny = [0.0, math.ulp(0.0), 2.0**-55, math.nextafter(2.0**-54, 0)]
    lanes = torch.tensor(tiny + [-x for x in tiny], dtype=torch.float64)
    if not torch.equal(vec_exp.exp(lanes).view(torch.int64), torch.ones_like(lanes).view(torch.int64)):
        raise SystemExit("Tiny exponential output differs from the proved result")

    # Choose the centre of each reduction cell independently of the engine's multiplier.
    # The negative special arm has fixtures with both normal and subnormal final values.
    quotients = (-1023, -900, -2, 3, 900)
    ks = [128*q+i for q in quotients for i in range(128)]
    with localcontext() as ctx:
        ctx.prec = 100
        step = Decimal(2).ln() / 128
        xs = [float(k*step) for k in ks]
        expected = [float(Decimal.from_float(x).exp()) for x in xs]
    inputs = torch.tensor(xs, dtype=torch.float64)
    table = vec_exp._table(inputs.device)
    with patch.object(vec_exp, "_table") as get_table:
        reads = get_table.return_value.__getitem__
        reads.side_effect = table.__getitem__  # Forward every engine lookup to its original tensor.
        try:
            result = vec_exp.exp(inputs)
        except IndexError as error:
            raise SystemExit(f"vec_exp.exp accessed an invalid table index: {error}") from None
    indices = [call.args[0].tolist() for call in reads.call_args_list]
    even = list(range(0, 256, 2))*len(quotients)
    if indices != [even, [i+1 for i in even]]:
        raise SystemExit("vec_exp.exp read unexpected tail or scale slots on the signed reduction fixtures")
    if result.shape != inputs.shape or result.dtype != torch.float64:
        raise SystemExit("vec_exp.exp returned an unexpected shape or dtype on the signed reduction fixtures")
    for k, actual, oracle in zip(ks, result.tolist(), expected, strict=True):
        if actual.hex() != oracle.hex():
            raise SystemExit(f"vec_exp.exp at reduction index {k}: got {actual.hex()}, expected {oracle.hex()}")
    return len(ks)


def warp():
    park = ModeBallpark()
    origin = [0.0, 0.0, 0.0]
    ball = park.add_ball(1, 1.0, 1.0, 0.0, origin, origin, 1.0, 0.0)
    ball.owner_id, ball.last_collision = 1000, 1000000.0
    ball.goto = [ball.last_collision, 0.0, 0.0]
    ta, tc, td, _, _, _, a, b, _ = park.warp_constants(ball.owner_id, ball.last_collision)
    for incoming in (0.0, 3000.0):
        pos, vel = park._warp_distance(ball, origin, [0.0, incoming, 0.0], 0.0)
        if abs(pos[0]-1) > 1e-8 or vel != [max(a, incoming), 0.0, 0.0]:
            raise SystemExit("Warp acceleration did not keep the sampled speed along the sightline")
    for delta in (0.25, 0.75):
        t = ta + tc + td + delta
        pos, vel = park._warp_distance(ball, origin, origin, t)
        expected = math.exp(-b*delta)
        if (abs((pos[0]-ball.goto[0])-(1-expected)) > 1e-8
                or not math.isclose(vel[0], b*expected, rel_tol=1e-12)):
            raise SystemExit("Warp deceleration remainder or velocity changed")
        second, reversed_vel = park._warp_distance(ball, pos, vel, t)
        if not second[0] < ball.goto[0] < pos[0] or reversed_vel[0] >= 0:
            raise SystemExit("Recomputed warp direction did not reverse beyond the exit")


def ranges():
    park = SpaceBallpark()
    # The reference has a large radius. Only the candidate's radius enters the source predicate.
    for bid, offset, radius in [(5, 0, 25), (9, 6, 1), (2, 2, 0), (3, 7, 1), (8, 1, 0)]:
        park.add_ball(bid, 1, radius, 0, x=100+offset, y=100, z=100)
    park.initialize_bubbles()
    ref = park.balls[5]
    park.balls[2].is_cloaked = 1
    park.balls[8].is_moribund = True
    # A stale member exercises the registry miss. Use deliberately non-distance-sorted dictionary order.
    park.bubbles[ref.bubble] = dict.fromkeys([99, 9, 2, 5, 3, 8], 1)
    for cloak in (False, True):
        expected = [(36.0, 9)] + ([(4.0, 2)] if cloak else [])
        for probe, rows in [((5, 5.0, cloak), expected),
                            ((*ref.pos, 5.0, cloak), expected+[(0.0, 5)])]:
            if (park.ball_ids_in_range(*probe, include_dist=True) != rows
                    or park.ball_ids_in_range(*probe) != [bid for _, bid in rows]):
                raise SystemExit(f"Range query changed order, boundary or eligibility at {probe!r}")
