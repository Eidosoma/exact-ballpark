# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Replay logged inputs against Exact Ballpark and the pinned Destiny C++ and client Python."""
import argparse
from collections import Counter
from fractions import Fraction
from unittest.mock import patch
import math
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "conformance"))
sys.path.insert(0, str(ROOT / "src"))
import contact_replay
from boundary_checks import limited_contact
import curve_checks
import discrepancy_sources
import mixed_history_checks
from build import BALLPARK, PINNED, PRELUDE, extract
from build_modes import LINES, extract_line
from exact_ballpark import full
from exact_ballpark.full import Capsule, FullBallpark
from exact_ballpark.modes import AU, ModeBallpark, WARP
from exact_ballpark.ref import Ballpark, f32
from exact_ballpark.space import CollisionImpact, CollisionItem, FUZZY_DELTA, SpaceBallpark


def stored_limiter_velocity():
    """Replay the COL-718 case through lookup, limiting and impact storage."""
    park = SpaceBallpark(friction=1.0, iterative=True)
    ball = park.add_ball(1, 1.0, 1.0, 4097.0, is_free=True, is_massive=True, agility=1.0)
    path = park._properties(ball)
    impact = CollisionImpact(0.0)
    impact.dv = [20485.0 + 2.0**-13, 0.0, 0.0]
    park._add_impact(impact, ball)
    return path.locations[-1].v[0]


def history_path():
    """Create the scalar one-second path used by both history probes."""
    park = SpaceBallpark(friction=1.0, iterative=True)
    ball = park.add_ball(1, 1.0, 1.0, 1.0, is_free=True, is_massive=True, agility=1.0)
    path = park._properties(ball)
    impact = CollisionImpact(0.0625)
    impact.collider = 2
    impact.dv = [1.0, 0.0, 0.0]
    park._add_impact(impact, ball)
    path.calculate_end()
    return path


def history_lookup_witness():
    path = history_path()
    query = float.fromhex("0x1.0068db8bac711p-4")
    following = math.nextafter(query, math.inf)
    return path.find_index(query), path.find_index(following), path.position(query)[1][0]


def history_second_contact():
    """Inject a static contact at the witness time. Vary only its lookup predicate."""
    def response(exact_cutoff):
        path = history_path()
        park, ship = path.park, path.ball
        item = CollisionItem()
        item.time = float.fromhex("0x1.0068db8bac711p-4")
        item.normal = [-1., 0., 0.]
        item.collider = park.add_ball(3, 1., 1., 0., is_free=False)

        def exact_index(t):
            passed = 0
            for location in path.locations:
                if Fraction(location.time) + Fraction(FUZZY_DELTA) >= Fraction(t):
                    break
                passed += 1
            return max(0, passed - 1)

        if exact_cutoff:
            with patch.object(path, "find_index", exact_index):
                park._impact_static(path, item)
        else:
            park._impact_static(path, item)
        park._add_impact(item.impact, ship)
        path.calculate_end()
        p, v = park._position_velocity(ship, ship.pos, ship.vel)
        return item.impact.dv[0], p[0], v[0]

    ordinary = response(False)
    control = response(True)
    if not (ordinary[0] == 0 and control[0] < -1.99 and ordinary[2] > 0 > control[2]):
        raise SystemExit(f"Injected contact changed: {ordinary}, exact cutoff {control}")
    return ordinary, control


def warp_impact_sample():
    """Compare whole-second scalar phase decisions with the two cruise fields zeroed."""
    class ZeroCruisePark(ModeBallpark):
        @staticmethod
        def warp_constants(factor, distance):
            ta, _, td, da, _, dd, a, b, s = ModeBallpark.warp_constants(factor, distance)
            return ta, 0., td, da, 0., dd, a, b, s

    def trajectory(park, distance):
        origin = [0., 0., 0.]
        ship = park.add_ball(1, 1., 1., 200., origin, origin, 1., 1.)
        ship.mode, ship.owner_id, ship.last_collision = WARP, 1000, distance
        ship.goto = [distance, 0., 0.]
        position, velocity = ship.pos, ship.vel
        samples = []
        for second in range(100):
            park.trace = Counter()
            position, velocity = park._warp_distance(ship, position, velocity, second)
            samples.append((set(park.trace), position[0]))
            if ship.mode != WARP:
                return samples
        raise SystemExit("Warp sample did not drop out within 100 s")

    times = [ModeBallpark.warp_constants(1000, d)[1] for d in range(100000, 101000)]
    counts = Counter((t > 0) - (t < 0) for t in times)
    samples = []
    for distance in (100000., 100001.):
        measured = trajectory(ModeBallpark(), distance)
        control = trajectory(ZeroCruisePark(), distance)
        if [phase for phase, _ in measured] != [phase for phase, _ in control]:
            raise SystemExit(f"Warp phase comparison changed at {distance} m")
        delta = max(abs(actual - other) for (_, actual), (_, other) in zip(measured, control, strict=True))
        samples.append((len(measured)-1, delta))
    return counts, samples


def special_goto_positions():
    """Two full scalar GOTO ticks differing only by a translation of the world."""
    positions = []
    for x in (0., AU):
        park = Ballpark()
        ship = park.add_ball(1, 1.1e16, 1., 10000., (x, 0., 0.), (100., 0., 0.), 1., 1.)
        park.goto_point(1, x + 65536., 0., 0.)
        park.evolve()
        positions.append(ship.pos[0])
    return positions


def special_error_scales():
    """Absolute forward error against the real special branch, for round-to-nearest normal operations.

    Expand the source expression by input term, retaining its shared rounded k².
    Each elementary roundoff factor is in [lo, hi]. The two signed ma*t/k
    paths have 3 and 2 operation factors before they meet. Their difference
    is then scaled by 5 common factors and the inverse k² rounding factor.
    The v*t path has 11 factors over one. On the p path k² cancels, leaving 4.
    All interval endpoints below are exact rationals.
    """
    u = Fraction(1, 2**53)
    lo, hi = 1-u, 1+u
    thrust = hi**5 / lo * max(hi**3 - lo**2, hi**2 - lo**3)
    drift = max(hi**11 / lo - 1, 1 - lo**11 / hi)
    origin = max(hi**4 - 1, 1 - lo**4)
    if not (thrust < 6*u and drift < 13*u and origin < 5*u):
        raise SystemExit("MOV-703 forward-error coefficients exceed the documented bounds")
    # p=0, |v|=t=1. Below this terminal speed the bound excludes a metre of error.
    if 6*u*1_500_000_000_000_000 + 13*u >= 1:
        raise SystemExit("MOV-703 metre-scale threshold changed")
    # Alternating Taylor terms for the regular ODE displacement at t=1.
    z = Fraction(1, 2**40)
    net_accel = 2**20 - z
    lower = 1 + net_accel * (Fraction(1, 2) - z/6)
    upper = 1 + net_accel / 2
    if not (524289 - Fraction(1, 1_000_000) < lower < upper < 524289):
        raise SystemExit("MOV-703 regular drag displacement outside the documented interval")
    friction = Ballpark().friction
    if not (1e-10 * 1e16 == friction < 1e-10 * math.nextafter(1e16, math.inf)):
        raise SystemExit("MOV-703 default-friction branch boundary changed")
    return round(lower)


def radial_accuracy():
    """CMB-702 at the fixed binary64 inputs used by Combat.Execution."""
    park = FullBallpark()
    park.add_ball(1, 1e6, 1, 100, is_free=True)
    park.add_ball(2, 1e6, 1, 100, is_free=True, x=3, y=4, z=12)
    park.set_velocity(2, 0.03, 0.04, 0.12)
    return park.get_accuracy(1, 2, 1e30, 1.0, 1.0, 1.0, 1e30)


def radial_intermediates():
    """Record the arguments actually passed to sqrt and pow in the scalar radial witness."""
    with patch.object(full, "_sqrt", wraps=full._sqrt) as sqrt, patch.object(math, "pow", wraps=math.pow) as power:
        radial_accuracy()
    if sqrt.call_count != 1 or power.call_count != 1 or power.call_args.args[0] != 2.0:
        raise SystemExit("CMB-702 radial witness did not make the expected square-root and power calls")
    return sqrt.call_args.args[0], power.call_args.args[1]


CAPSULE_OFFSETS = (2.0**-20, 2.0**-16)


def capsule_contacts():
    park = FullBallpark()
    capsule = Capsule()
    capsule.initialize(1, 0, (0, 0, -1), (0, 0, 1), 0.5)
    for offset in CAPSULE_OFFSETS:
        found, normal, toi = park._capsule_check(capsule, (1+offset, 0, 0), (1-offset, 0, 0), 0.5)
        if found and normal != [1.0, 0.0, 0.0]:
            raise SystemExit(f"Unexpected barrel contact normal {normal}")
        yield toi if found else -1.0


def scalar_cases():
    origin = (0.0, 0.0, 0.0)
    cache = float.fromhex("0x1.fffffffffep-1")
    pos, _ = Ballpark(friction=1).integrate(
        origin, (1.0, 0.0, 0.0), (2.0**20, 0.0, 0.0), 2.0**40, 1.0, cache, 1.0
    )
    sweep = SpaceBallpark.collide_two_spheres
    extent = 2.0**28
    end = float.fromhex("0x1.0000000000001p0")
    index, following, velocity = history_lookup_witness()
    if (index, following) != (0, 1):
        raise SystemExit(f"Fuzzy lookup changed: {(index, following)}")
    cases = [
        ("MOV-703 · final position", pos[0], 0.0),
        ("COL-705 · long sweep", sweep((-extent, 2.0, 0.0), (extent, 2.0, 0.0), origin, origin, 1.0), 0.5),
        ("COL-705 · tick endpoint", sweep((4.0, 0.0, 0.0), (end, 0.0, 0.0), origin, origin, 1.0), 1.0),
        ("COL-705 · exact tangency", sweep((-3.0, extent, 0.0), (3.0, extent, 0.0), origin, origin, extent), -1.0),
        ("COL-718 · impact velocity limiter", stored_limiter_velocity(), 20485.0 + 2.0**-13),
        ("COL-715 · history lookup and x velocity", velocity, 0.0),
        ("TRV-709 · cruise duration at 100,000 and 100,001 m", ModeBallpark.warp_constants(1000, 100000)[1],
         float.fromhex("0x1.f75c000000000p-53")),
        ("TRV-709 · cruise duration at 100,000 and 100,001 m", ModeBallpark.warp_constants(1000, 100001)[1],
         float.fromhex("-0x1.a6dc000000000p-53")),
    ]
    accuracy, distance, angular = radial_accuracy()
    if not math.isnan(angular):
        raise SystemExit("CMB-702 radial angular speed changed")
    return cases + [
        ("MOV-703 · GOTO positions", stored, float.fromhex(expected))
        for stored, expected in zip(special_goto_positions(),
                                    ("0x1.8ffffffffffd7p+6", "0x1.16a5d2d67ffffp+37"), strict=True)
    ] + [("CMB-702 · radial accuracy and surface distance", accuracy, math.inf),
         ("CMB-702 · radial accuracy and surface distance", distance, 11.0)] + [
        ("COL-712 · shallow capsule crossing and control", value, expected)
        for value, expected in zip(capsule_contacts(), (-1.0, 0.5), strict=True)
    ]


def check_rankings(log):
    """Both public summaries and the two fix tables use the log's severity order."""
    levels = ("High", "Moderate", "Low", "Negligible")
    row = r"^\| (\w+) \| \[([A-Z]+-\d+) · [^\]]+\]\([^)]*#([\w-]+)\)"
    primary = re.findall(row, log, re.MULTILINE)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if not primary or primary != re.findall(row, readme, re.MULTILINE):
        raise SystemExit("Discrepancy rankings differ between README.md and the log")
    if f"**{len(primary)} engine behaviors**" not in log:
        raise SystemExit("The discrepancy headline differs from the severity table")
    if not re.search(rf"^\[!\[{len(primary)} replayed cases\b", log, re.MULTILINE):
        raise SystemExit("Update the discrepancy infographic and its caption for the ranked cases")
    tiers = [tier for tier, _, _ in primary]
    if any(tier not in levels for tier in tiers) or tiers != sorted(tiers, key=levels.index):
        raise SystemExit("Discrepancies must be in descending gameplay severity")
    expected = [(tier, rule) for tier, rule, _ in primary]
    fixes = re.findall(r"^\| \*\*(\w+) · ([A-Z]+-\d+) · ", log, re.MULTILINE)
    if fixes != expected + expected:
        raise SystemExit("The two candidate-fix tables must follow the gameplay ranking")


def check_log(destiny_repo=None):
    """Compare the published words and rounded gameplay measurements with a fresh scalar replay."""
    log = (ROOT / "lean/DISCREPANCIES.md").read_text(encoding="utf-8")
    discrepancy_sources.check(log, ROOT, PINNED, destiny_repo)
    check_rankings(log)
    mixed_history_checks.check_log(log, destiny_repo, PINNED)
    cases = scalar_cases()
    regular_metres = special_error_scales()
    words = {}
    for name, actual, expected in cases:
        word = struct.pack(">d", actual).hex()
        if word != struct.pack(">d", expected).hex():
            raise SystemExit(f"{name} produced {word}, outside the expected regression result")
        words.setdefault(name, []).append(f"`{word}`")
    ordinary, control = history_second_contact()
    speed2, exponent = map(contact_replay.word, radial_intermediates())
    rendered = [f"negative squared angular speed, word `{speed2}`", f"exponent, yielding binary64 word `{exponent}`",
                f"about {CAPSULE_OFFSETS[0]*1e6:.3f} µm inside",
                f"cutoff for 10⁻¹⁰ has word `{struct.pack('>d', 1e-10).hex()}`",
                f"approximately **{regular_metres:,} m**", "above **1.5 × 10¹⁵ m/s**",
                "u(6|V·t| + 13|v·t| + 5|p|)"]
    near, translated = [Fraction(actual) for name, actual, _ in cases if name == "MOV-703 · GOTO positions"]
    near_pm = abs(near - 100) * 10**12
    translated_um = abs(translated - Fraction(AU) - 100) * 10**6
    if abs(translated - Fraction(AU) - 100) != Fraction(math.ulp(AU)):
        raise SystemExit("Translated MOV-703 result is no longer one spacing from the exact branch")
    rendered += [f"{float(near_pm):.3f} pm", f"{float(translated_um):.3f} µm",
                 f"{2**60 / 1e18:.2f} × 10¹⁸ m/s"]
    for label, (dv, x, v) in [("Engine binary64 cutoff", ordinary), ("Exact cutoff in that lookup only", control)]:
        impulse = f"{dv:.5f}" if dv else f"{dv:.0f}"
        rendered.append(f"| {label} | {impulse} m/s | {x:+.6f} m | {v:+.6f} m/s |".replace("-", "−"))
    pairs = contact_replay.fixed_pair()
    for label, (collider, bits) in zip(("near first", "far first"), pairs, strict=True):
        rendered.append(f"| Fixed balls, {label} | {collider} | `{bits}` |")
    engine, earliest = contact_replay.rebound()
    for label, (x, v) in [("Engine comparison", engine), ("Earliest-time control", earliest)]:
        rendered.append(f"| {label} | {x:.6f} m | {v:+.6f} m/s |".replace(" | -", " | −"))
    rendered.append(f"{abs(engine[0]-earliest[0]):.6f} m apart")
    def decode(word):
        return struct.unpack(">d", bytes.fromhex(word))[0]
    positions = []
    for label, use_control in [("Engine lookup", False), ("Exact-cutoff control", True)]:
        contacts, (xword, vword), _ = contact_replay.detected_rebound(use_control)
        x, v = decode(xword), decode(vword)
        positions.append(x)
        values = f"{x:+.6f} m, {v:+.6f} m/s".replace("-", "−")
        rendered.append(f"| {label} | {len(contacts)} | {values} | `{xword}`, `{vword}` |")
        if not use_control:
            a, b = (decode(hit[1]) for hit in contacts[:2])
            if Fraction(b)-Fraction(a)-Fraction(FUZZY_DELTA) != Fraction(211, 2**66):
                raise SystemExit("The two detected contacts no longer straddle the documented exact cutoff")
            rendered.extend([f"first contact at {a:.17f} s", f"second at {b:.17f} s"])
    rendered.append(f"positions differ by **{abs(positions[1]-positions[0]):.6f} m**")
    whole, leg = contact_replay.free_repeat()
    for label, value in [("Both free-ball subintervals", whole), ("Final subinterval alone", leg)]:
        rendered.append(f"| {label} | `{contact_replay.word(value)}` |")
    words["COL-715 · fixed obstacle order"] = [f"`{bits}`" for _, bits in pairs]
    words["COL-715 · free-ball second contact"] = [f"`{contact_replay.word(t)}`" for t in (whole, leg)]
    for name, bits in words.items():
        rows = [line for line in log.splitlines() if line.startswith(f"| {name} |")]
        if len(rows) != 1 or rows[0].split("|")[-2].strip() != ", ".join(bits):
            raise SystemExit(f"Discrepancy log has stale output bits for {name}")
    rendered.append(f"{leg:.9f} s")
    counts, samples = warp_impact_sample()
    (exit_a, dx_a), (exit_b, dx_b) = samples
    def magnitude(x):
        mantissa, exponent = f"{x:.2e}".split("e")
        return f"{mantissa} × 10{str(int(exponent)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
    first = f32(25 * 4097)
    exact_square = 25 * 4097**2
    if first != 25 * 4097 or (25 * (2**24-1)**2).bit_length() > 53:
        raise SystemExit("COL-718 numerator precision changed")
    rendered += [f"25 × 4097 = {first:.0f}", f"rounds {exact_square} up to {f32(first * 4097):.0f}",
                 f"{abs(ordinary[1]-control[1]):.2f} m of separation",
                 f"rounded positive {counts[1]} times and negative {counts[-1]} times",
                 f"All four runs left warp at second {exit_a}",
                 f"{magnitude(dx_a)} m at 100,000 m and {magnitude(dx_b)} m at 100,001 m",
                 f"exceeds five times the configured maximum speed by {stored_limiter_velocity()-5*4097}"]
    if exit_a != exit_b or counts[0] or sum(counts.values()) != 1000:
        raise SystemExit("Short-warp grid or exit comparison changed")
    u, v, momentum_change, energy_change = limited_contact()
    signed = lambda x: f"{float(x):+g}".replace("-", "−")
    rendered += [f"applied changes of {signed(u+100)} and {signed(v+90)} m/s",
                 f"loses {float(-momentum_change):g} kg·m/s of momentum and gains {float(energy_change):g} J"]
    for statement in rendered + curve_checks.log_measurements():
        if statement not in log:
            raise SystemExit(f"Discrepancy measurement differs from the log: {statement}")
    return cases


def replay(repo):
    if sys.version_info < (3, 13):
        raise SystemExit("Python 3.13 or later is required for engine correspondence.")
    def source(name):
        return subprocess.check_output(["git", "-C", repo, "show", f"{PINNED}:src/{name}"], encoding="utf-8")

    collision = source("Collision.cpp")
    capsule = source("Capsule.cpp")
    ballpark = source("Ballpark.cpp")
    add_impact = extract(ballpark, "void Ballpark::AddImpact(")
    # Only the velocity limiter is replayed here. Keep the expression and field types upstream.
    limiter = add_impact[add_impact.index("const auto speed2"):add_impact.index("if(g_useDynamicalOrientation)")]
    history = source("CollisionBallProperties.cpp")
    setup = extract(ballpark, "void Ballpark::SetupWarpConstants(")
    declaration = setup.split("{", 1)[0].replace("Ballpark::", "").strip() + ";"
    warp_sources = {"src/Ballpark.cpp": ballpark, "src/Ballpark.h": source("Ballpark.h")}
    constants = [extract_line(warp_sources[path], prefix, name) for path, name, prefix in LINES
                 if name == "AU" or name.startswith("WARP_FACTOR_TO_")]
    accuracy = extract(ballpark, "double Ballpark::GetAccuracy(")
    accuracy_decl = accuracy.split("{", 1)[0].replace("Ballpark::", "").strip() + ";"
    unit = "\n".join([
        PRELUDE, '#include "Vector3d.h"', "#include <vector>", "#include <algorithm>",
        *constants, BALLPARK.replace("struct Ballpark {", "struct Ballpark {\n" + declaration),
        "namespace CombatProbe { using ID = long long;",
        "struct Ball { Vector3d mNewPos, mNewVel; float mRadius; };",
        "struct Ballpark { Ball* mBalls[3]; " + accuracy_decl + " };",
        # GetAccuracy calls these two functions unqualified. Observe their inputs and forward to libm.
        "double squaredAngularSpeed, powerExponent;",
        "double sqrt(double x) { squaredAngularSpeed = x; return std::sqrt(x); }",
        "double pow(double x, double y) { powerExponent = y; return std::pow(x, y); }",
        "#define CCP_LOGWARN_CH(...) ((void)0)", accuracy, "#undef CCP_LOGWARN_CH", "}",
        setup,
        extract(ballpark, "void Ballpark::Integrate("),
        extract(ballpark, "Vector3d Ballpark::GotoThrust("),
        extract(ballpark, "void Ballpark::SetBallTimeFactor("),
        extract(collision, "bool Quadratic("),
        extract(collision, "double CollideTwoSpheres("),
        """namespace CapsuleProbe { using ID = long long;
struct Capsule {
    ID mId, mParentBallId;
    Vector3d mHemisphereA, mHemisphereB, mAB, mNewPos, mOldPos;
    float mRadius;
    void Initialize(ID, ID, double, double, double, double, double, double, float);
    float GetBoundingRadius();
    Vector3d GetCenter();
    bool CheckCollision(const Vector3d&, const Vector3d&, float, Vector3d&, double&);
};""",
        *(extract(capsule, signature) for signature in (
            "void Capsule::Initialize(", "float Capsule::GetBoundingRadius(",
            "Vector3d Capsule::GetCenter(", "bool Capsule::CheckCollision(")),
        "}",
        "struct ProbeImpact { Vector3d velocityChange; };",
        "void limitVelocity(ProbeImpact& impact, Ball* ball) {", limiter, "}",
        """struct HistoryBall : Ball { Ballpark* mPark; Vector3d mLastG, mLastC; };
struct Location { double time; Vector3d p, v; };
struct CollisionBallProperties {
    HistoryBall* m_ball;
    const double m_fuzzyDelta;
    std::vector<Location> m_locations;
    size_t FindLocationsIndex(double time);
    void CalculatePosition(Vector3d& position, Vector3d& velocity, double time);
};""",
        extract(history, "size_t CollisionBallProperties::FindLocationsIndex("),
        extract(history, "void CollisionBallProperties::CalculatePosition("),
    ])
    unit += r"""
int main() {
    const Vector3d origin(0, 0, 0);
    Ballpark park;
    Vector3d pos = origin, vel(1, 0, 0);
    park.Integrate(pos, vel, Vector3d(0x1p20, 0, 0), 0x1p40, 1, 0x1.fffffffffep-1, 1);
    auto sweep = [&](double x0, double x1, double y, double radius = 1) {
        return CollideTwoSpheres(Vector3d(x0, y, 0), Vector3d(x1, y, 0), origin, origin, radius);
    };
    Ball ball;
    ball.mMaxVel = 4097;
    ProbeImpact impact{Vector3d(20485.0 + 0x1p-13, 0, 0)};
    limitVelocity(impact, &ball);
    HistoryBall body;
    body.mPark = &park;
    body.mMass = 1;
    park.mFriction = 1;
    CollisionBallProperties path{&body, 1e-4, {{0, origin, origin}, {0.0625, origin, Vector3d(1,0,0)}}};
    auto end = path.m_locations.back();
    end.time = 1;
    park.Integrate(end.p, end.v, origin, 1, 1, 0, 1 - 0.0625);
    path.m_locations.push_back(end);
    Vector3d queryPos, queryVel;
    path.CalculatePosition(queryPos, queryVel, 0x1.0068db8bac711p-4);
    std::printf("%a\n%a\n%a\n%a\n%a\n%a\n", pos.x, sweep(-0x1p28, 0x1p28, 2),
                sweep(4, 0x1.0000000000001p0, 0), sweep(-3, 3, 0x1p28, 0x1p28), impact.velocityChange.x, queryVel.x);
    for (double distance : {100000.0, 100001.0}) {
        double ta, tc, td, da, dc, dd, a, b, speed;
        park.SetupWarpConstants(1000, distance, ta, tc, td, da, dc, dd, a, b, speed);
        std::printf("%a\n", tc);
    }
    for (double x : {0.0, AU}) {
        Ball mover;
        mover.mNewPos = Vector3d(x, 0, 0);
        mover.mNewVel = Vector3d(100, 0, 0);
        mover.mMass = 1.1e16;
        mover.mMaxVel = 10000;
        mover.mSpeedFraction = 1;
        park.mFriction = 1e6;
        park.SetBallTimeFactor(&mover);
        auto thrust = park.GotoThrust(&mover, Vector3d(x + 65536, 0, 0), false);
        park.Integrate(mover.mNewPos, mover.mNewVel, thrust, mover.mMass, park.mFriction, mover.mTimeFactor, park.dt);
        std::printf("%a\n", mover.mNewPos.x);
    }
    CombatProbe::Ball src{origin, origin, 1.0f}, dst{Vector3d(3,4,12), Vector3d(0.03,0.04,0.12), 1.0f};
    CombatProbe::Ballpark combat{{nullptr, &src, &dst}};
    double optimal = 1e30, tracking = 1.0;
    double chance = combat.GetAccuracy(1, 2, optimal, 1.0, tracking, 1.0, 1e30);
    if (!std::isnan(tracking)) {
        std::fprintf(stderr, "CMB-702 radial angular speed: expected NaN, got %a\n", tracking);
        return 1;
    }
    std::printf("%a\n%a\n", chance, optimal);
    CapsuleProbe::Capsule capsule;
    capsule.Initialize(1, 0, 0, 0, -1, 0, 0, 1, 0.5f);
    for (double offset : {0x1p-20, 0x1p-16}) {
        Vector3d normal;
        double toi;
        const bool found = capsule.CheckCollision(Vector3d(1+offset,0,0), Vector3d(1-offset,0,0), 0.5f, normal, toi);
        if (found && (normal.x != 1 || normal.y != 0 || normal.z != 0)) {
            std::fprintf(stderr, "COL-712 capsule at offset %a: expected normal (1, 0, 0), got (%a, %a, %a)\n",
                         offset, normal.x, normal.y, normal.z);
            return 1;
        }
        std::printf("%a\n", found ? toi : -1.0);
    }
    std::printf("%a %a\n", CombatProbe::squaredAngularSpeed, CombatProbe::powerExponent);
    std::printf("%zu %zu\n", path.FindLocationsIndex(0x1.0068db8bac711p-4),
                path.FindLocationsIndex(0x1.0068db8bac712p-4));
}
"""
    with TemporaryDirectory() as temp:
        cpp, binary = Path(temp) / "witness.cpp", Path(temp) / "witness"
        (cpp.parent / "Vector3d.h").write_text(source("Vector3d.h"), encoding="utf-8")
        cpp.write_text(unit, encoding="utf-8")
        subprocess.run(["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
                        str(cpp), "-o", str(binary)], check=True)
        result = subprocess.run([binary], capture_output=True, encoding="utf-8")
        if result.returncode:
            raise SystemExit(result.stderr.strip() or f"Destiny witness exited {result.returncode}")
        upstream = result.stdout.splitlines()

    cpp_indices = tuple(map(int, upstream.pop().split()))
    if cpp_indices != (0, 1):
        raise SystemExit(f"Destiny fuzzy lookup changed: {cpp_indices}")
    cpp_radial = [contact_replay.word(float.fromhex(x)) for x in upstream.pop().split()]
    if cpp_radial != list(map(contact_replay.word, radial_intermediates())):
        raise SystemExit(f"CMB-702 Destiny square-root and power inputs differ from the scalar replay: {cpp_radial}")
    cases = check_log(repo)
    gcc = subprocess.check_output(["g++", "-dumpfullversion"], encoding="utf-8").strip()
    print(f"Destiny {PINNED}, {platform.python_implementation()} {platform.python_version()}, "
          f"{platform.system()} {platform.machine()}, GCC {gcc}, -O2 -ffp-contract=off -fno-fast-math")
    print("Pinned Destiny Python and scalar client matched both mixed-stamp packet traces in the log.")
    print(f"CMB-702 squared angular speed and power exponent matched pinned C++: {' '.join(cpp_radial)}")
    for (name, scalar, expected), raw in zip(cases, upstream, strict=True):
        words = [struct.pack(">d", v).hex() for v in (scalar, float.fromhex(raw), expected)]
        print(f"{name}: scalar {words[0]}, Destiny {words[1]}, logged {words[2]}")
        if len(set(words)) != 1:
            raise SystemExit("Witness output changed")
    pairs, tick, free = contact_replay.upstream(repo)
    print("Equal-mass, unequal-mass and limited one-impact ticks matched scalar Python on both balls' impulse and endpoint words.")
    print("Turning GOTO matched pinned C++ on the cache, thrust, positions at half a tick and both window edges, "
          "full-tick endpoints and all three sweep results.")
    print(f"Fixed-ball selection (ID, time bits) {pairs}, end-of-tick x and vx {tick}")
    print("Three-ball cutoff replay matched pinned C++ at every stored contact and at the tick endpoint.")
    print(f"Free-ball repeat, complete search and final subinterval {free}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destiny_repo", type=Path, help="Git clone containing the pinned Destiny commit")
    try:
        replay(parser.parse_args().destiny_repo)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Destiny replay: {Path(error.cmd[0]).name} exited {error.returncode}") from None
