# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare one-impact iterative ticks through scalar Evolve, Lean and the pinned C++ replay."""
from collections import Counter
import json
from pathlib import Path
import struct

from exact_ballpark.space import SpaceBallpark


CASES = (("equal", 1., 2.), ("unequal", 2., 2.), ("limited", 1., .5))
VECTORS = ("thrust", "incomingPos", "incomingVel", "raw", "applied", "contactPos", "outgoingVel", "pos", "vel")


def word(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def vector(values):
    return [word(value) for value in values]


class ObservedPair(SpaceBallpark):
    """Observe the two impulse methods called by the normal iterative tick."""

    def __init__(self):
        super().__init__(friction=1., tick_interval_ms=1000, iterative=True)
        self.trace = Counter()
        self.incoming = {}
        self.raw = {}
        self.applied = {}

    def _impact_free(self, prop, other_prop, item):
        for p in (prop, other_prop):
            if p.ball.id in self.incoming:
                raise SystemExit("The head-on tick calculated another free-pair response")
            self.incoming[p.ball.id] = p.position(item.time)
        other = super()._impact_free(prop, other_prop, item)
        self.raw[prop.ball.id] = tuple(item.impact.dv)
        self.raw[other_prop.ball.id] = tuple(other.impact.dv)
        return other

    def _add_impact(self, impact, ball):
        super()._add_impact(impact, ball)
        self.applied[ball.id] = tuple(impact.dv)


def scalar_rows():
    rows = {}
    for name, right_mass, left_max in CASES:
        park = ObservedPair()
        for bid, x, vx, mass, maximum in ((1, 0., 2., 1., left_max), (2, 3., -2., right_mass, 2.)):
            park.add_ball(bid, mass, .5, maximum, is_free=True, is_massive=True, x=x, vx=vx,
                          agility=1/mass, speed_fraction=2/maximum)
            park.goto_point(bid, x+(64 if bid == 1 else -64), 0., 0.)
        park.evolve()
        if (park.trace["iterative.free"] != 1 or park.trace["iterative.ignored"] < 1
                or park.trace["iterative.limited"] != int(name == "limited") or park.trace["iterative.cap"]):
            raise SystemExit(f"Unexpected iteration, repeat or limiter branch in the {name} tick: {park.trace}")
        for bid in (1, 2):
            ball = park.balls[bid]
            history = park.collision_locations[bid].locations
            if [(loc.time, loc.collider) for loc in history] != [(0., -1), (.5, 3-bid), (1., -1)]:
                raise SystemExit(f"Unexpected one-impact history for {name} ball {bid}")
            if vector(history[-1].p + history[-1].v) != vector(ball.pos + ball.vel):
                raise SystemExit(f"The history endpoint differs from the committed state for {name} ball {bid}")
            before_pos, before_vel = park.incoming[bid]
            samples = (ball.last_g, before_pos, before_vel, park.raw[bid], park.applied[bid], history[1].p,
                       history[1].v, ball.pos, ball.vel)
            rows[name, bid] = dict(cache=word(ball.time_factor), time=word(history[1].time),
                                   **{field: vector(value) for field, value in zip(VECTORS, samples, strict=True)})
    return rows


def check_rows(label, actual, expected):
    if actual.keys() != expected.keys():
        raise SystemExit(f"{label} did not return the six expected head-on ball records")
    for key, row in expected.items():
        if actual[key] != row:
            field = next((key2 for key2 in row if actual[key].get(key2) != row[key2]), "record shape")
            raise SystemExit(f"{label} differs from scalar in {key}, {field}: "
                             f"{actual[key].get(field)} vs {row.get(field)}")


def compare(run_lake):
    expected = scalar_rows()
    output = run_lake("env", "lean", "--run", "scripts/ReboundTick.lean", capture=True)
    actual = {}
    for line in output.splitlines():
        row = json.loads(line)
        key = (row.pop("case"), row.pop("id"))
        if key in actual:
            raise SystemExit(f"Duplicate Lean head-on record for {key}")
        actual[key] = row
    check_rows("Lean one-impact replay", actual, expected)
    # Keep the public endpoint examples tied to fresh committed x and vx words.
    page = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for name, _, _ in CASES:
        values = [struct.unpack(">d", expected[name, bid][field][0].to_bytes(8, "big"))[0]
                  for bid in (1, 2) for field in ("pos", "vel")]
        row = f"| {name.capitalize()} | " + " | ".join(f"{x:.6f}" for x in values) + " |"
        if row not in page:
            raise SystemExit(f"Update the committed {name} endpoint row in lean/README.md")
    return len(CASES)


def compare_cpp(lines):
    actual = {}
    for line in lines:
        if not line.startswith("REBOUND "):
            continue
        _, name, bid, *hex_values = line.split()
        words = [word(float.fromhex(x)) for x in hex_values]
        key = (name, int(bid))
        if key in actual or len(words) != 2 + 3*len(VECTORS):
            raise SystemExit(f"Malformed pinned head-on record for {key}")
        groups = [words[i:i+3] for i in range(2, len(words), 3)]
        actual[key] = dict(cache=words[0], time=words[1], **dict(zip(VECTORS, groups, strict=True)))
    check_rows("Pinned Destiny one-impact replay", actual, scalar_rows())
