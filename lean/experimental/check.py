# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Audit the optional proofs and replay the published observations and engine controls."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LEAN = HERE.parent
ROOT = LEAN.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(LEAN / "scripts"), str(ROOT / "src"), str(ROOT / "conformance")]
from check import check_toolchain, load_registry, run_lake
import integrator_reference
from build import BALLPARK, PINNED, PRELUDE, extract
from exact_ballpark.full import FullBallpark
from exact_ballpark.ref import Ballpark
from exact_ballpark.space import CollisionItem


def bits(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def require(condition, label):
    if not condition:
        raise SystemExit(label)


def collisions(cpp_inputs, cpp_expected, lean_rows):
    cases = [(1., 9, True, True), (2., 9, True, False), (.5, 16, False, True),
             (1., 11, True, True), (1., 12, False, False)]
    table = []
    for duration, centre, raw, timed in cases:
        park = FullBallpark(friction=2048, iterative=True, tick_interval_ms=int(duration * 1000))
        speed = 100000 / duration
        ship = park.add_ball(1, 1, .5, speed, True, is_massive=True, agility=1, vx=speed)
        ship.last_g = [park.friction * speed, 0., 0.]
        path = park._properties(ship)
        path.locations[0].collider = 7
        fixed = park.add_ball(7, 1, .5, 0, x=centre, is_massive=True)
        start, end = (path.position(t)[0] for t in (0., duration))
        require(end == [100000., 0., 0.], "The fixed-obstacle sweep length changed")
        u = park.collide_two_spheres(start, end, fixed.pos, fixed.pos, 1.)
        item = CollisionItem()
        park._next_collision_static(path, item, [fixed])
        actual = item.collider is None
        ignore = path.ignore_collision
        with patch.object(path, "ignore_collision", lambda collider, frac: ignore(collider, frac * duration)):
            control = CollisionItem()
            park._next_collision_static(path, control, [fixed])
        require((actual, control.collider is None) == (raw, timed), f"Static sweep changed at {duration} s, x={centre}")
        for candidate in (item, control):
            if candidate.collider is not None:
                require(candidate.collider.id == 7 and candidate.time == u * duration, "Wrong contact or timestamp")
        lean_rows.append((f"C,{bits(duration)},{bits(u)}", f"{str(raw).lower()} {str(timed).lower()}"))
        cpp_inputs.append(f"C {duration} {centre}")
        cpp_expected.append((bits(u), int(raw), int(timed)))
        table.append(f"| {duration:g} | {u:g} | {u * duration * 1e6:.0f} | "
                     f"{'ignored' if raw else 'retained'} | {'ignored' if timed else 'retained'} |")
    return table


def drag_and_spheres(data, cpp_inputs, cpp_expected, lean_rows):
    steps, words = [], []
    def step(p, v, a, m, k, cached, t):
        # The fixture park has dt=1. Its actual integrator recomputes exp on partial ticks.
        park = Ballpark(friction=k)
        pos, vel = park.integrate([p, 0., 0.], [v, 0., 0.], [a, 0., 0.], m, k, cached, t)
        row = (p, v, a, m, k, cached, t, park.dt)
        steps.append(",".join(str(bits(x)) for x in (*row, math.exp(-k / m * t))))
        words.append((bits(pos[0]), bits(vel[0])))
        cpp_inputs.append("I " + " ".join(map(str, row)))
        cpp_expected.append(words[-1])
        return pos[0], vel[0]

    # Dyadic equilibria reproduce the real identity exactly in these two binary64 cases.
    for m, k, a in ((2., 1., .5), (2.**40, 1., 2.**-40)):
        require(step(2., 1., a, m, k, .25, 1.) == (3., 1.), "Balanced full tick changed")

    drag_table = []
    for dt in (Decimal("0.5"), Decimal(2)):
        t, k = float(dt), 1e-10
        pos, _ = step(0., 0., 1., 1., k, math.exp(-k), t)
        p_word, v_word = words[-1]
        lean_rows.append((f"D,{t:g}", f"{bits(math.exp(-k * t))} {p_word} {v_word}"))
        rate = Decimal.from_float(k)
        ideal = (dt - (1 - (-rate * dt).exp()) / rate) / rate
        special, exponent = k < 1e-10 * t, k * t < 1e-10
        require(special != exponent, "Drag guard controls no longer disagree")
        drag_table.append(f"| {t:g} | {'special' if special else 'regular'} | "
                          f"{'special' if exponent else 'regular'} | {pos:.6f} | {ideal:.6f} |")

    lab = data["sheffield"]
    rho = Decimal(lab["solid_density_kg_m3"])
    fluid = Decimal(lab["fluid_density_kg_m3"])
    eta = Decimal(lab["dynamic_viscosity_Pa_s"])
    g = Decimal(lab["gravity_m_s2"])
    resolution = Decimal(lab["printed_increment_m_s"])
    lab_table, radii, observed = [], {}, {}
    for sphere in lab["spheres"]:
        name = sphere["id"]
        radius = radii[name] = Decimal(sphere["diameter_mm"]) / 2000
        measured = observed[name] = Decimal(sphere["observed_m_s"])
        values = list(map(Fraction, (radius, g, rho, fluid, eta)))
        r, gravity, solid_density, fluid_density, viscosity = values
        exact = 2 * r**2 * gravity * (solid_density - fluid_density) / (9 * viscosity)
        inputs = ",".join(f"{x.numerator}/{x.denominator}" for x in values)
        lean_rows.append(("S," + inputs, f"{exact.numerator}/{exact.denominator}"))
        terminal = Decimal(exact.numerator) / Decimal(exact.denominator)
        tau = 2 * radius**2 * rho / (9 * eta)
        reynolds = 2 * radius * measured * fluid / eta
        # Independent scalar inputs retain pi in both the mass and the drag coefficient.
        m = 4 * math.pi / 3 * float(radius)**3 * float(rho)
        k = 6 * math.pi * float(eta) * float(radius)
        a = float(g * (rho - fluid) / rho)
        for t in (Decimal("0.01"), Decimal(1)):
            pos, vel = step(0., 0., a, m, k, math.exp(-k / m), float(t))
            gain = 1 - (-t / tau).exp()
            require(abs(vel - float(terminal * gain)) < 1e-14 and
                    abs(pos - float(terminal * (t - tau * gain))) < 1e-14,
                    f"Scalar settling trajectory differs from the independent SI calculation for {name}")
        match = "inside" if terminal.quantize(resolution) == measured else "outside"
        lab_table.append(f"| {name} | {2 * radius * 1000:.1f} | {reynolds:.2f} | {terminal:.6f} | "
                         f"{measured:.2f} | {terminal - measured:+.6f} | {match} |")
    scaled = observed["S3"] * (radii["S4"] / radii["S3"])**2
    scaled_half_bin = resolution / 2 * (radii["S4"] / radii["S3"])**2
    scaling = (f"Anchoring the radius-squared law to S3 gives **{scaled:.4f} m/s** for S4. "
               f"At the published increment of {resolution} m/s this gives {scaled.quantize(resolution)} m/s. "
               f"The reported S4 speed is {observed['S4']} m/s. Propagating just the half-unit of "
               f"rounding in the S3 reading gives {scaled - scaled_half_bin:.4f}–{scaled + scaled_half_bin:.4f} m/s.")
    raw = data["georgia_tech"]
    distance = (Decimal(raw["start_mL"]) - Decimal(raw["end_mL"])) * Decimal(raw["metres_per_mL"])
    seconds = (Decimal(raw["end_centiseconds"]) - Decimal(raw["start_centiseconds"])) / 100
    held = (f"Georgia Tech trial 1 covers {distance:g} m in {seconds:g} s, giving **{distance/seconds:.6f} m/s**. "
            "The authors report that liquid temperature was unrecorded. Absolute calibration for this run is held open.")
    output = run_lake("env", "lean", "--run", "scripts/IntegratorProbe.lean", *steps, capture=True)
    require([tuple(map(int, line.split())) for line in output.splitlines()] == words,
            "Scalar movement output differs from the translated Lean binary64 program")
    return drag_table, lab_table, scaling, held


def upstream(repo, inputs, expected):
    def source(name):
        return subprocess.check_output(["git", "-C", repo, "show", f"{PINNED}:src/{name}"], encoding="utf-8")
    collision = source("Collision.cpp")
    history = source("CollisionBallProperties.cpp")
    methods = [extract(collision, "bool Quadratic("), extract(collision, "double CollideTwoSpheres("),
               extract(history, "CollisionBallProperties::CollisionBallProperties()"),
               extract(history, "bool CollisionBallProperties::IgnoreCollision("),
               extract(source("Ballpark.cpp"), "void Ballpark::Integrate(")]
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "Vector3d.h").write_text(source("Vector3d.h"), encoding="utf-8")
        (folder / "movement.hpp").write_text(PRELUDE + '\n#include "Vector3d.h"\n' + BALLPARK, encoding="utf-8")
        (folder / "methods.hpp").write_text("\n".join(methods), encoding="utf-8")
        binary = folder / "probe"
        subprocess.run(["g++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                        "-I", folder, HERE / "GuardProbe.cpp", "-o", binary], check=True)
        out = subprocess.check_output([binary], input="\n".join(inputs) + "\n", encoding="utf-8")
    actual = []
    for row, line in zip(inputs, out.splitlines(), strict=True):
        fields = line.split()
        if row.startswith("C "):
            actual.append((bits(float.fromhex(fields[0])), *map(int, fields[1:])))
        else:
            actual.append(tuple(bits(float.fromhex(x)) for x in fields))
    require(actual == expected, "Pinned C++ diverged from the scalar guard or integrator replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate RESULTS.md after the checks")
    parser.add_argument("--destiny-repo", type=Path, help="also replay the pinned C++ methods")
    args = parser.parse_args()
    require(sys.version_info >= (3, 13), "Use CPython 3.13 or later for the engine replay")
    check_toolchain()
    load_registry()
    require(integrator_reference.DEST.read_text(encoding="utf-8") == integrator_reference.extract(),
            "The scalar integrator translation needs review")
    run_lake("build", "RealPhysics", "ProofAudit")
    data = json.loads((HERE / "falling_spheres.json").read_text(encoding="utf-8"))
    cpp_inputs, expected, lean_rows = [], [], []
    times = collisions(cpp_inputs, expected, lean_rows)
    with localcontext() as ctx:
        ctx.prec = 70
        drag, lab, scaling, held = drag_and_spheres(data, cpp_inputs, expected, lean_rows)
    lean_inputs, lean_expected = zip(*lean_rows, strict=True)
    output = run_lake("env", "lean", "-DwarningAsError=true", "--run", "experimental/CheckPhysics.lean",
                      *lean_inputs, capture=True)
    count_line, *outputs = output.splitlines()
    label, count = count_line.split()
    require(label == "registered" and count.isdecimal(), "Missing experimental proof count")
    require(outputs == list(lean_expected), "Lean guard outputs or exact laboratory predictions differ")
    if args.destiny_repo:
        upstream(args.destiny_repo, cpp_inputs, expected)
        print(f"Matched {len(expected)} cases against compiled Destiny {PINNED}.")
    lines = ["# Physical comparison results", "",
             f"The optional library has **{count} registered experimental theorems**. [Experimental declarations](claims.json)",
             "", "## Terminal speed", "",
             "| Sphere | Diameter (mm) | Re from observation | Predicted (m/s) | Observed (m/s) | Prediction − observed (m/s) | Rounding interval of the printed speed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |", *lab, "", scaling, "", held, "",
             "Inputs, measurement conditions and interpretation are in the [experiment register](README.md#falling-spheres).", "",
             "## Repeat-collision decisions", "",
             "| Sweep (s) | Hit fraction | Elapsed (µs) | Engine | Elapsed-time control |",
             "| ---: | ---: | ---: | --- | --- |", *times, "",
             "## MOV-703 branch decisions", "",
             "| Step (s) | Engine | Fixed exponent cutoff | Scalar position (m) | Continuous drag position (m) |",
             "| ---: | --- | --- | ---: | ---: |", *drag, "",
             "Both timing comparisons have reproducible witnesses in the [physical discrepancy log](DISCREPANCIES.md).", ""]
    page = "\n".join(lines)
    path = HERE / "RESULTS.md"
    if args.write:
        path.write_text(page, encoding="utf-8")
    else:
        require(path.is_file() and path.read_text(encoding="utf-8") == page,
                "Experimental results changed. Run experimental/check.py --write and review the measurements.")
    print("Verified experimental proof dependencies, scalar/Lean replays and the physical comparison results.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
