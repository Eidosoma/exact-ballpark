# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Build the proofs, audit their axioms, check source models and compare scalar witnesses."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
import delivery_checks
import integrator_reference
import scalar_reference
import server_reference
import warp_reference
import visibility_checks

LEAN = Path(__file__).resolve().parents[1]
ROOT = LEAN.parent


def run_lake(*args, capture=False):
    bundled = LEAN / ".lake/toolchain/bin/lake"
    lake = bundled if bundled.is_file() else shutil.which("lake")
    if lake is None:
        raise SystemExit("Lake is required. See lean/README.md for the pinned toolchain.")
    result = subprocess.run([lake, *args], cwd=LEAN, encoding="utf-8", stdout=subprocess.PIPE if capture else None)
    if result.returncode:
        if capture:
            print(result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def load_registry():
    registry = json.loads((LEAN / "claims.json").read_text(encoding="utf-8"))
    catalogue = {}
    for line in (ROOT / "docs/rules.md").read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| [A-Z]+-\d+ \|", line):
            _, rule, description, _, status, _, _ = (cell.strip() for cell in line.split("|"))
            if rule in catalogue:
                raise SystemExit(f"Duplicate catalogue rule {rule}")
            catalogue[rule] = (description, status)
    for path, digest in registry["sources"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Source changed: {path}. Review its claims before updating the digest.")
    names = set()
    if not registry["claims"]:
        raise SystemExit("Empty claim registry")
    for claim in registry["claims"]:
        sources = [claim["source"], *claim.get("relatedSources", [])]
        if (any(path not in registry["sources"] for path in sources)
                or not claim["theorems"] or not claim["rules"]):
            raise SystemExit(f"Incomplete claim: {claim['title']}")
        if "ruleTheorems" in claim:
            mapping = claim["ruleTheorems"]
            short = {name.rsplit(".", 1)[-1] for name in claim["theorems"]}
            if (set(mapping) != set(claim["rules"]) or not all(mapping.values())
                    or {name for group in mapping.values() for name in group} != short):
                raise SystemExit(f"Inconsistent per-rule theorem mapping: {claim['title']}")
        for rule in claim["rules"]:
            if rule not in catalogue:
                raise SystemExit(f"Unknown rule: {rule}")
        for name in claim["theorems"]:
            if name in names or not name.startswith("ExactBallpark."):
                raise SystemExit(f"Duplicate or invalid theorem: {name}")
            names.add(name)
    return registry, catalogue, names


def compare_setters():
    from exact_ballpark.ref import Ballpark

    # Include both binary32 subnormal and ordinary halfway cases and their adjacent encodings.
    magnitudes = (0, 1, 0x368fffffffffffff, 0x3690000000000000, 0x3690000000000001,
                  0x3ff0000000000000, 0x3ff000000fffffff, 0x3ff0000010000000, 0x3ff0000010000001,
                  0x7ff0000000000000, 0x7ff8000000000001, 0x7ff8000000001234)
    inputs = [sign | bits for sign in (0, 1 << 63) for bits in magnitudes]
    output = run_lake("env", "lean", "--run", "scripts/SetterProbe.lean", *map(str, inputs), capture=True)
    park = Ballpark()
    for raw, row in zip(inputs, output.splitlines(), strict=True):
        mass_bits, agility_bits = row.split()
        value, = struct.unpack(">d", raw.to_bytes(8, "big"))
        ball = park.add_ball(1, value, 1, 1, (0, 0, 0), (0, 0, 0), 1, 1)
        expected = int(mass_bits).to_bytes(8, "big")
        if not (math.isnan(ball.mass) and math.isnan(struct.unpack(">d", expected)[0])):
            if struct.pack(">d", ball.mass) != expected:
                raise SystemExit(f"Mass clamp mismatch for input {raw:#018x}")

        ball = park.add_ball(2, 1, 1, 1, (0, 0, 0), (0, 0, 0), 3, 1)
        ball.time_factor = -1.0  # Every successful refresh replaces this sentinel, including for NaN.
        park.set_agility(2, value)
        if (agility_bits == "-") != (ball.time_factor == -1.0):
            raise SystemExit(f"Agility refresh mismatch for input {raw:#018x}")
        expected = struct.pack(">f", 3) if agility_bits == "-" else int(agility_bits).to_bytes(4, "big")
        if not (math.isnan(ball.agility) and math.isnan(struct.unpack(">f", expected)[0])):
            if struct.pack(">f", ball.agility) != expected:
                raise SystemExit(f"Agility write mismatch for input {raw:#018x}")
    return len(inputs)


def compare_integrator():
    import torch
    from exact_ballpark.ref import Ballpark
    from exact_ballpark.vec_collide import VecCollidePark

    park = Ballpark(friction=1)
    # Each row exercises the actual scalar entry point on three coordinates.
    # In partial ticks the probe receives Python's exp bits so this comparison isolates arithmetic.
    cases = [(1, 1, 0.0), (1, 0.25, -7), (1, 0.5, -7), (1, -0.0, 0.25),
             (2**40, 1, math.exp(-2**-40)), (2**40, 0.5, -7),
             (10**10, 1, 0.75), (math.nextafter(10**10, math.inf), 1, 0.75)]
    rows, expected = [], []
    p, v, a = [-0.0, 0.0, -8.5], [-0.0, 1.0, -3.0], [0.0, 2**20, 0.5]
    def bits(x):
        return int.from_bytes(struct.pack(">d", x), "big")
    for m, t, cache in cases:
        pos, vel = park.integrate(p, v, a, m, 1.0, cache, t)
        exp_val = math.exp((-1.0 / m)*t)
        for i in range(3):
            values = p[i], v[i], a[i], m, 1.0, cache, t, park.dt, exp_val
            rows.append(",".join(str(bits(x)) for x in values))
            expected.append((bits(pos[i]), bits(vel[i])))
    output = run_lake("env", "lean", "--run", "scripts/IntegratorProbe.lean", *rows, capture=True)
    for inputs, actual, reference in zip(rows, output.splitlines(), expected, strict=True):
        if tuple(map(int, actual.split())) != reference:
            raise SystemExit(f"Integrator mismatch for binary64 inputs {inputs}")
    batched = VecCollidePark(1, len(cases), friction=1.0, device="cpu")
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)
    pos, vel, acc = [tensor([values]*len(cases)) for values in (p, v, a)]
    mass, time, cached = map(tensor, zip(*cases))
    partial = tensor([math.exp((-1.0/m)*t) for m, t, _ in cases])
    pn, vn = batched._integrate_t(pos, vel, acc, mass, cached, time, partial)
    full = time == park.dt
    fp, fv = batched._integrate_dt(pos[full], vel[full], acc[full], mass[full], cached[full])
    if list(zip(map(bits, pn.flatten().tolist()), map(bits, vn.flatten().tolist()))) != expected:
        raise SystemExit("The PyTorch CPU partial-tick lanes differ from the scalar integrator")
    if not (torch.equal(fp.view(torch.int64), pn[full].view(torch.int64)) and
            torch.equal(fv.view(torch.int64), vn[full].view(torch.int64))):
        raise SystemExit("The full-tick tensor integrator disagrees on the cached lanes")
    return len(rows)


def compare_sweep_examples():
    from exact_ballpark.space import SpaceBallpark

    origin = (0.0, 0.0, 0.0)
    extent = 2.0**28
    cases = [
        ((-extent, 2.0, 0.0), (extent, 2.0, 0.0), 1.0, 0.5),
        ((-extent, 5.0, 0.0), (extent, 5.0, 0.0), 1.0, -1.0),
        ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), 1.0, -1.0),
        ((2.0, 0.0, 0.0), (-1048574.0, 0.0, 0.0), 1.0, 2.0**-20),
        ((2.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, 1.0),
        ((4.0, 0.0, 0.0), (math.nextafter(1.0, math.inf), 0.0, 0.0), 1.0, 1.0),
        ((-3.0, extent, 0.0), (3.0, extent, 0.0), extent, -1.0),
        ((-3.0, 1.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0),
        ((-3.0, 1.0, 0.0), (-1.0, 1.0, 0.0), 1.0, -1.0),
    ]
    for start, end, radius, expected in cases:
        actual = SpaceBallpark.collide_two_spheres(start, end, origin, origin, radius)
        if struct.pack(">d", actual) != struct.pack(">d", expected):
            raise SystemExit(f"Sweep {start} to {end} returned {actual}, expected {expected}")
    return len(cases)


def compare_face_and_clock():
    from exact_ballpark.full import FullBallpark, Triangle, _point_on_ball, _segment_plane

    x = (1., 0., 0.)
    origin = (0., 0., 0.)
    if _point_on_ball(x, 1., x) != list(origin):
        raise SystemExit("Sphere support point changed")
    for level, expected in [(0., (True, list(origin), 0.)), (1., (True, list(x), 1.)),
                            (2., (False, None, 2.))]:
        if _segment_plane(origin, x, level, x) != expected:
            raise SystemExit(f"Unexpected plane crossing at offset {level}")
    for level in (0., 1.):
        if _segment_plane(origin, x, level, (0., 1., 0.)) != (False, None, None):
            raise SystemExit("Parallel sweep returned a plane crossing")
    face = Triangle(origin, x, (0., 1., 0.))
    for point, expected in [(origin, True), ((0., .5, 0.), True), ((.25, .25, 0.), True),
                            ((.5, .5, 0.), False), (x, False)]:
        if face.contains(point) != expected:
            raise SystemExit(f"Unexpected triangle membership for {point}")

    interval = 10000  # 1 ms in engine clock units
    for master, whole, steps, carried in [(True, 5, 5, 7), (True, 6, 1, 0), (False, 6, 6, 7)]:
        park = FullBallpark(tick_interval_ms=1, is_master=master)
        park.on_tick(interval)
        previous_step = park.current_time
        poll = interval + whole * interval + 7
        park.on_tick(poll)
        stored = poll - carried
        if park.current_time != previous_step + steps or park.time != stored:
            raise SystemExit(f"Unexpected clock after {whole} intervals on master={master}")
        park.on_tick(stored + interval - 1)
        if park.current_time != previous_step + steps or park.time != stored:
            raise SystemExit("Early repeat poll advanced the clock")
        park.on_tick(stored + interval)
        if park.current_time != previous_step + steps + 1 or park.time != stored + interval:
            raise SystemExit("Next tick missed its exact deadline")


def compare_history():
    from exact_ballpark.space import CollisionImpact, FUZZY_DELTA, SpaceBallpark
    from replay_discrepancies import history_lookup_witness
    from contact_replay import fixed_pair

    def event(time, collider):
        impact = CollisionImpact(time)
        impact.collider = collider
        return impact

    def fixture():
        park = SpaceBallpark(friction=1.0, iterative=True)
        ball = park.add_ball(1, 1.0, 1.0, 1.0, is_free=True, is_massive=True, agility=1.0)
        path = park._properties(ball)
        for time, collider in [(0.25, 10), (0.5, 11), (0.75, 12)]:
            path.add_impact(event(time, collider))
            path.calculate_end()
        return path

    def bits(t):
        return str(int.from_bytes(struct.pack(">d", t), "big"))

    def snapshot(path, removed):
        locations = ",".join(f"{bits(x.time)}/{x.collider}" for x in path.locations)
        events = ",".join(f"{bits(t)}/{collider}" for t, collider in removed)
        return "|".join([bits(path.t_start), bits(path.t_end), locations, events])

    path = fixture()
    rows = []
    for collider in [13, 14]:
        removed = path.add_impact(event(0.25, collider))
        rows.append(snapshot(path, removed))
    for collider in [10, 99]:
        path = fixture()
        removed = path.remove_impact(0.25, collider)
        rows.append(snapshot(path, removed))
    path = fixture()
    rows.append(" ".join(str(path.ignore_collision(cid, FUZZY_DELTA)).lower() for cid in [12, 11]))
    for compare_seconds in (False, True):
        rows.extend(f"{i} {int(word, 16)}" for i, word in fixed_pair(compare_seconds=compare_seconds))
    observed = run_lake("env", "lean", "--run", "scripts/HistoryProbe.lean", capture=True).splitlines()
    if observed != rows:
        raise SystemExit(f"Collision history or fixed-ball selection differs between Lean {observed} and scalar Python {rows}")
    index, following, vx = history_lookup_witness()
    if (index, following, struct.pack(">d", vx)) != (0, 1, bytes(8)):
        raise SystemExit("COL-715 history lookup differs from the logged witness")


def claim_page(registry, catalogue, audit, probes, coordinates, candidate_rows, combat_lanes, fma, mixed_rows,
               exp_lanes, movement_prefixes, rejected_suffixes):
    claims = registry["claims"]
    linked = {rule for claim in claims for rule in claim["rules"]}
    families = defaultdict(list)
    for rule in catalogue:
        families[rule.split("-")[0]].append(rule)
    lines = [
        "# Lean claim register", "",
        f"{len(audit)} kernel-checked theorems have claims linked to {len(linked)} of the "
        f"{len(catalogue)} catalogue rules. A link covers the scope stated below.", "",
        f"Engine revision `{registry['revision']}`. {registry['correspondence']}", "",
        registry["deviceQualification"] + " [Platform](../docs/platform.md) · [Evidence](../docs/exactness.md)", "",
        "## Catalogue coverage", "",
        "| Family | Rules | With a scoped claim | Awaiting a first claim |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, rules in families.items():
        count = len(linked.intersection(rules))
        lines.append(f"| {family} | {len(rules)} | {count} | {len(rules) - count} |")
    lines += ["", "The rule descriptions, upstream locations and per-rule test files are in the "
              "[engine catalogue](../docs/rules.md).", ""]
    lines += ["| Rule | Engine law | Scope and supporting theorems |", "| --- | --- | --- |"]
    for rule, (description, _) in catalogue.items():
        links = []
        for index, claim in enumerate(claims, 1):
            if rule in claim["rules"]:
                names = claim.get("ruleTheorems", {}).get(rule, [t.rsplit(".", 1)[-1] for t in claim["theorems"]])
                links.append(f"[{', '.join(names)}](#claim-{index})")
        lines.append(f"| {rule} | {description} | {', '.join(links) or 'Open'} |")
    lines.append("")
    for index, claim in enumerate(claims, 1):
        path = claim["moduleName"].replace(".", "/") + ".lean"
        status = ", ".join(f"{rule} {catalogue[rule][1]}" for rule in claim["rules"])
        source_links = [f"[Source](../{claim['source']})"]
        source_links.extend(f"[{Path(path).name}](../{path})" for path in claim.get("relatedSources", []))
        lines += [f'<a id="claim-{index}"></a>', "", f"## {claim['title']}", "", f"**{claim['model']}.** {claim['scope']}", "",
                  f"Engine symbols `{claim['symbols']}`. " + " · ".join(source_links), "",
                  f"Recorded upstream evidence — {status}.", ""]
        if "hostEvidence" in claim:
            lines += [claim["hostEvidence"], ""]
        if "ExactBallpark.Movement.Scalar.trace_refines" in claim["theorems"]:
            lines += [f"CPython and Lean agree on the clock, IDs, modes and every stored field after "
                      f"{movement_prefixes} movement trace prefixes. The probe compares non-NaN words exactly "
                      f"and checks {rejected_suffixes} rejected suffixes against scalar exceptions. "
                      "[Trace replay](scripts/scalar_trace_checks.py)", ""]
        if any(name in claim["theorems"] for name in ("ExactBallpark.Numerics.clamp_other",
                                                      "ExactBallpark.Numerics.agility_accepts_iff")):
            lines += [f"Host correspondence check — Lean and Python agree on {probes} binary64 inputs.", ""]
        if "ExactBallpark.Movement.full_tick_velocity_bits" in claim["theorems"]:
            lines += [f"Host correspondence check — {coordinates} coordinate updates match Python for position and velocity bits.", ""]
        if "ExactBallpark.Collision.long_chord_rounded_hit" in claim["theorems"]:
            lines += ["Host correspondence check — the Python scalar reference returns the same binary64 midpoint.", ""]
        if "ExactBallpark.Collision.certified_chord_miss" in claim["theorems"]:
            lines += ["Host correspondence check — the scalar solver returns −1 for the five-metre offset and for the sweep whose roots lie outside the tick.", ""]
        if "ExactBallpark.Space.CandidateOrder.ordered_candidates" in claim["theorems"]:
            lines += [f"CPU correspondence — {candidate_rows} ordered candidate lists and their active packed keys agree with scalar Python traversal and PyTorch on the CPU across all eight tree levels.", ""]
        if "ExactBallpark.Time.MixedHistory.merged_payload" in claim["theorems"]:
            lines += [f"Lean and the scalar client agree on the complete ordered payloads and wait flags for {mixed_rows} mixed-stamp histories. [CPU probe](scripts/mixed_history_checks.py)", ""]
        if "ExactBallpark.Time.MixedHistory.waiting_tail" in claim["theorems"]:
            lines += ["Both public client updates report one fatal desync before the admission decision on a mixed-stamp packet. "
                      "With a returning handler the replays reproduce the extra wait and the reset-order queues. "
                      "[Packet replays](DISCREPANCIES.md#time-715-extra-wait)", ""]
        if "ExactBallpark.Collision.Capsule.shallow_crossing_missed" in claim["theorems"]:
            lines += ["Scalar Python and pinned Destiny C++ miss the shallow crossing and return half a tick on "
                      "the faster control. At 9.9 µm of travel they miss, at 10.1 µm they hit, and starting inside "
                      "at the missed endpoint returns contact at time zero. "
                      "[Capsule replay](DISCREPANCIES.md#col-712-shallow-crossing)", ""]
        if "ExactBallpark.Collision.ReplayTick.four_ball_endpoint" in claim["theorems"]:
            lines += ["The scalar replay matches the certified x and vx words for the engine and the earliest-hit control, and both words for the free-ball return. [Witness comparison](scripts/arithmetic_checks.py)", ""]
        if "ExactBallpark.Numerics.ExpReduction.table_address" in claim["theorems"]:
            lines += [f"The CPU exponential reads every tail and scale slot on {exp_lanes} non-tiny inputs. Each returned word matches a 100-digit decimal reference rounded to binary64. [Sampling checks](scripts/sampling_checks.py)", ""]
        if "ExactBallpark.Numerics.Compensated.fma_single_round" in claim["theorems"]:
            lines += [f"CPU comparison on the registered source — {fma}. "
                      "The FMA oracle is CPython math.fma. Exact dyadic arithmetic checks the product "
                      "widths and both rounding grids for each sampled call. [Probe](scripts/arithmetic_checks.py)", ""]
        if "ExactBallpark.Combat.accuracy_overflow_input" in claim["theorems"]:
            lines += [f"The scalar and PyTorch CPU checks cover {combat_lanes} combat lanes, including the same-ball guard. "
                      "For the radial case, the scalar and pinned C++ square-root and power inputs match the certified words. "
                      "At tracking ratio 200 the chance is 1.0000000000000007. At 10³⁰ it is infinite and the "
                      "exact chance on the stored inputs rounds to zero. Both queries return NaN angular speed. "
                      "[Lane checks](scripts/arithmetic_checks.py) · [Pinned C++ replay](DISCREPANCIES.md#cmb-702-tracking)", ""]
        if "ExactBallpark.Collision.SearchWindow.partial_tick_selection" in claim["theorems"]:
            lines += ["CPU replay — the scalar and pinned C++ selectors choose different obstacles when the same pair is enumerated in the opposite order. [Inputs and full-tick control](DISCREPANCIES.md#col-715-fixed-window)", ""]
        lines += [f"| [Theorems]({path}) | Transitive axioms |", "| --- | --- |"]
        for name in claim["theorems"]:
            axioms = ", ".join(audit[name]) or "None"
            lines.append(f"| `{name}` | {axioms} |")
        lines.append("")
    return "\n".join(lines)


def main():
    if sys.version_info < (3, 13):
        raise SystemExit("Python 3.13 or later is required for engine correspondence.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate CLAIMS.md after all checks pass")
    parser.add_argument("--destiny-repo", type=Path, help="also replay flight, collisions and routing using a local clone containing the pinned Destiny commit")
    args = parser.parse_args()
    pin = (LEAN / "lean-toolchain").read_text(encoding="utf-8").strip().removeprefix("leanprover/lean4:v")
    version = run_lake("env", "lean", "--version", capture=True)
    if not re.search(r"Lean \(version " + re.escape(pin) + r"(?:,|\))", version):
        raise SystemExit(f"Expected Lean {pin}, got {version.strip()}")
    registry, catalogue, expected = load_registry()
    for reference in (integrator_reference, scalar_reference, server_reference, warp_reference):
        if reference.DEST.read_text(encoding="utf-8") != reference.extract():
            raise SystemExit(f"Generated source differs from Python. Review scripts/{reference.__name__}.py.")
    run_lake("build")
    output = run_lake("env", "lean", "-DwarningAsError=true", "scripts/CheckClaims.lean", capture=True)
    rows = [json.loads(line) for line in output.splitlines()]
    audit = {row["theorem"]: row["axioms"] for row in rows}
    if set(audit) != expected or len(rows) != len(expected):
        raise SystemExit("The audit did not return exactly one result per registered theorem")
    sys.path.insert(0, str(ROOT / "src"))
    import arithmetic_checks
    import boundary_checks
    import candidate_checks
    import curve_checks
    import contact_replay
    import formation_checks
    import geometry_checks
    import mixed_history_checks
    import law_trace_checks
    import rebound_checks
    import sampling_checks
    import scalar_trace_checks
    law_trace_checks.compare(run_lake, args.destiny_repo.resolve() if args.destiny_repo else None)
    boundary_checks.compare(run_lake)
    movement_prefixes, rejected_suffixes = scalar_trace_checks.compare(run_lake)
    formation_cases = formation_checks.compare(run_lake)
    probes = compare_setters()
    coordinates = compare_integrator()
    sweeps = compare_sweep_examples()
    compare_history()
    compare_face_and_clock()
    geometry_checks.compare(run_lake)
    curve_cases = curve_checks.compare(run_lake)
    contact_cases = curve_checks.compare_contacts(run_lake)
    rebound_cases = rebound_checks.compare(run_lake)
    contact_replay.static_order()
    candidate_rows = candidate_checks.compare(run_lake)
    combat_lanes, fma, (fma_inputs, indices) = arithmetic_checks.compare(run_lake)
    mixed_rows = mixed_history_checks.compare(run_lake)
    exp_lanes = sampling_checks.exponential()
    sampling_checks.warp()
    sampling_checks.ranges()
    visibility = visibility_checks.compare(run_lake)
    batches, snapshot_steps = delivery_checks.compare(run_lake)
    from exact_ballpark.modes import ModeBallpark
    for distance, word in [(100000, "3caf75c000000000"), (100001, "bcaa6dc000000000")]:
        cruise = ModeBallpark.warp_constants(1000, distance)[1]
        if struct.pack(">d", cruise).hex() != word:
            raise SystemExit(f"Short warp at {distance} m differs from the Lean witness")
    from replay_discrepancies import check_log
    check_log(args.destiny_repo)
    if args.destiny_repo:
        contact_replay.upstream(args.destiny_repo.resolve())
    overview = (LEAN / "README.md").read_text(encoding="utf-8")
    linked = {rule for claim in registry["claims"] for rule in claim["rules"]}
    headline = f"**{len(linked)} engine laws** to **{len(audit)} kernel-checked theorems**"
    if headline not in overview or not all(re.search(rf"(?<![0-9]){count} {noun}\b", overview) for count, noun in [
            (formation_cases["reserve"], "reservations"), (formation_cases["free"], "releases"),
            (formation_cases["join"], "joins"), (formation_cases["resize"], "resize sequences"),
            (batches, "outgoing batches"), (snapshot_steps, "snapshot operations"),
            (candidate_rows, "ordered candidate lists"), (mixed_rows, "mixed-stamp histories"),
            (curve_cases, "curve clearance checks"),
            (contact_cases, "first-contact checks"),
            (rebound_cases, "one-impact ticks"),
            (movement_prefixes, "movement trace prefixes"), (rejected_suffixes, "rejected suffixes"),
            (exp_lanes, "non-tiny inputs"), (fma_inputs, "first-reduction inputs"), (indices, "reachable indices")]):
        raise SystemExit("Update the theorem and CPU probe counts in lean/README.md")
    coverage = f"all {len(catalogue)}" if linked == catalogue.keys() else f"{len(linked)} of {len(catalogue)}"
    if not re.search(rf"(?<![0-9]){len(audit)} kernel-checked theorems across {coverage} engine rules\b",
                     (ROOT / "README.md").read_text(encoding="utf-8")):
        raise SystemExit("Update the theorem and law counts in README.md")
    page = claim_page(registry, catalogue, audit, probes, coordinates, candidate_rows, combat_lanes, fma,
                      mixed_rows, exp_lanes, movement_prefixes, rejected_suffixes)
    path = LEAN / "CLAIMS.md"
    if args.write:
        path.write_text(page, encoding="utf-8")
    elif not path.exists() or path.read_text(encoding="utf-8") != page:
        raise SystemExit("Stale CLAIMS.md. Run python3 scripts/check.py --write and review the diff.")
    print("Matched proximity callbacks and recipients, warp handoff, formation slots, field access, "
          "queued and rewound GOTO, tracking signs, mixed CPU worlds and the three-interval work budget.")
    if args.destiny_repo:
        print("Matched all 11 warp commits against compiled Destiny C++ and the recipient fixtures "
              "against five pinned Destiny Python server methods.")
        print("Matched the stored velocities of the independently limited pair against compiled Destiny C++.")
    print("Matched STOP and orbit trajectories, capped impulses, solid boxes, range callbacks, cleanup polls "
          "and tracking and falloff controls on the CPU.")
    print(f"Passed {contact_cases} first-contact checks on the CPU.")
    print(f"Matched {rebound_cases} one-impact ticks on both balls through the committed endpoints.")
    print(f"Matched {movement_prefixes} scalar movement trace prefixes, {rejected_suffixes} rejected suffixes "
          "and the five-second coast against the continuous drag solution.")
    print(f"Verified {len(audit)} theorems, source hashes, {probes} inputs each for mass and agility, {coordinates} integrator updates, {sweeps} sweep examples, {curve_cases} curve clearance checks, the impact limiter, impact history, triangle faces, tick deadlines, short warps, {visibility['sensor']} sensor scans, 8 uncloaking cases, {visibility['pilot']} pilot transitions, {visibility['route']} event routes, {batches} outgoing batches, {snapshot_steps} snapshot operations, {mixed_rows} mixed-stamp histories, {sum(formation_cases.values())} formation cases, {candidate_rows} ordered candidate lists on PyTorch CPU, geometric certificates and detected rebound traces, warp and range sampling, {exp_lanes} non-tiny exponentials, {fma_inputs} first-reduction inputs across {indices} indices, tiny-input checks, {combat_lanes} combat lanes, {fma} and CLAIMS.md.")


if __name__ == "__main__":
    main()
