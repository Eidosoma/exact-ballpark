# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""CPU correspondence for the sensor, routing, flight, combat and world-mask claims."""
import ast
from collections import Counter, defaultdict
from contextlib import ExitStack, nullcontext
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from exact_ballpark import client, full, server
from exact_ballpark.modes import ModeBallpark
from exact_ballpark.ref import Ballpark


def word(x):
    return int.from_bytes(struct.pack(">d", x), "big")


def floating(bits):
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def equal(label, actual, expected):
    if actual != expected:
        raise SystemExit(f"{label}: got {actual!r}, expected {expected!r}")


def session(ships, pilots, up=0):
    park = server.DestinyPark(is_master=True)
    for bid, x, radius in ships:
        park.AddBall(bid, 2e6, radius, 300., 1, 0, 0, int(bid in pilots), 0,
                     x, 0., 0., 0., 0., 0., 1., 1.)
    run = client.Session(park, pilots, up=up)
    run.start()
    return run


def sensor_scene(rows, forward):
    run = session([(3, 0, 1), (4, 13, 1), (5, 2**32, 1)], {3: 0, 4: 1, 5: 2})
    p = run.server.lane
    owner = p.balls[3]
    p.add_proximity_sensor(3, 10, 2)
    if owner.bubble != p.balls[4].bubble or owner.bubble == p.balls[5].bubble:
        raise SystemExit("Sensor replay requires two local pilots and one distant pilot")
    messages = []
    pending = []
    for stamp, x, member, elapsed, flag in rows:
        p.set_position(4, floating(x), 0, 0)
        p.events.clear()
        run.tick()
        callbacks = [event[4] for event in p.events if event[1:4] == ("DoProximity", 3, 4)]
        equal(f"sensor poll {stamp}", (int(4 in owner.sensor_balls), word(owner.sensor_elapsed), callbacks),
              (member, elapsed, [] if flag == 2 else [flag]))
        if forward and callbacks:
            radius = 1 if callbacks == [1] else 2
            run.actions.set_ball_radius(3, radius)
            pending.append((stamp + 1, radius))
        messages.extend((stamp, packet) for packet in run.sent_log[-1][1])
    run.tick()  # Deliver the action queued on the last leave.
    messages.extend((rows[-1][0] + 1, packet) for packet in run.sent_log[-1][1])
    sent = []
    for stamp, (kind, to, method, payload, wait, count) in messages:
        for _, (action, args) in payload:
            if action == "DoProximity":
                raise SystemExit("Stock server forwarded a raw DoProximity callback")
            if action == "SetBallRadius" and args[0] == 3:
                sent.append((stamp, args[1], kind, sorted(to), method, wait, int(count)))
    equal("sensor response recipients", sent,
          [(s, r, "narrow", [10000, 10001], "DoDestinyUpdate", False, 1) for s, r in pending])


def route_scene(known, live, forward):
    info = client.PilotInterests({3: 0, 4: 1, 5: 2})
    park = SimpleNamespace(currentTime=7, balls={3: SimpleNamespace(newBubbleId=7, isCloaked=False)},
                           bubbles={7: {}} if known else {}, bubbleInteractives={7: [3, 4]} if live else {})
    net = client.RecordingNetwork()
    batcher = server.ParkUpdateBatcher(park, net, info, info, None)
    actions = server.Actions(SimpleNamespace(currentTime=0))
    actions.set_stamp_for_system(7)
    if forward:
        actions.set_ball_radius(3, 1)
    ticker = server.Ticker(park, info, actions, batcher)
    ticker.current_system_history = actions.flush_history()
    ticker._apply_history()
    batcher.send_batch()
    sent = Counter()
    for kind, ids, method, payload, waits, count in net.take():
        equal("routed action payload", (kind, method, payload, waits, int(count)),
              ("narrow", "DoDestinyUpdate", [(7, ("SetBallRadius", (3, 1)))], False, 1))
        sent.update(ids)
    return sent


def upstream_routing(repo, pin):
    """Run the named, unmodified pinned Python methods against the same network and park adapters."""
    scope = dict(Ticker=server.Ticker, ClientUpdateCountThisTick=server.ClientUpdateCountThisTick,
                 Timer=nullcontext, logger=server.logger)
    stack = ExitStack()
    try:
        for filename, cls, names in [
            ("_baseticker", server.BaseTicker, ("_expand_entry", "_find_bubble_id_for_event", "_is_valid_event_bubble")),
            ("_ticker", server.Ticker, ("_apply_history",)),
            ("_parkupdatebatcher", server.ParkUpdateBatcher, ("send_batch",)),
        ]:
            path = f"python/destiny/net/server/{filename}.py"
            source = subprocess.check_output(["git", "-C", repo, "show", f"{pin}:{path}"], encoding="utf-8")
            node = next(c for c in ast.parse(source).body if isinstance(c, ast.ClassDef) and c.name == cls.__name__)
            for name in names:
                fn = next(f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == name)
                exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), scope)
                stack.enter_context(patch.object(cls, name, scope[name]))
    except BaseException:
        stack.close()
        raise
    return stack


def routes(rows):
    views = defaultdict(dict)
    for known, live, forward, who, count in rows:
        views[known, live, forward][who] = count
    for conditions, counts in views.items():
        sent = route_scene(*conditions)
        equal(f"route {conditions}", dict(sent), {i: n for i, n in counts.items() if n})


def field_queries(rows):
    park = server.DestinyPark(is_master=True)
    for bid, x in [(3, 16), (4, 16), (6, 40), (9, 64)]:
        park.AddBall(bid, 2e6, 1., 300., 1, 0, int(bid == 6), 1, 0,
                     x, 16., 16., 0., 0., 0., 1., 1.)
    s = client.Session(park, {3: 0, 4: 1})
    s.start()
    p = park.lane
    def candidates(bid):
        return [b.id for b in p._nearby_balls(p.balls[bid], False, False, False, True, False)]
    for harmonic, corp, alliance, *expected in rows:
        s.actions.set_ball_harmonic(3, harmonic, corp, alliance, False)
        s.actions.set_ball_harmonic(4, 6, 7, 8, False)
        s.actions.set_ball_harmonic(9, 7, 8, 9, True)
        s.tick()
        equal("queued field matching ship", candidates(3), expected)
        equal("queued field nonmatching control", candidates(4), [6, 9])
    s.actions.set_ball_harmonic(9, -1, -1, -1, True)
    equal("field before pre-tick", candidates(4), [6, 9])
    s.tick()
    equal("disabled field after pre-tick", candidates(4), [6])


def rotations():
    p = full.FullBallpark(dynamical_orientation=True)
    leader = p.add_ball(1, 2e6, 1, 300, True, x=16, y=32, z=64, agility=1)
    follower = p.add_ball(2, 2e6, 2, 300, True, agility=1)
    p.load_formations([[(3, 4, 0)], [(0, 0, 0)]])
    p.set_ball_formation(1, 0)
    p.formation_follow(2, 1, p.reserve_formation_slot(1))
    for quaternion, offset in [((0, 0, 0, 1), (4.8, 6.4, 0)),
                               ((.5, .5, .5, .5), (0, 4.8, 6.4)),
                               ((0, 0, 1, 0), (-4.8, -6.4, 0))]:
        p.set_ball_rotation(1, *quaternion)
        p._evolve_formation(follower)
        equal("rotated formation target", list(map(word, follower.goto)),
              [word(x + full.f32(d)) for x, d in zip(leader.pos, offset, strict=True)])
    p.set_ball_formation(1, 1)
    p._evolve_formation(follower)
    equal("zero slot target", follower.goto, leader.pos)


def warp_trace(rows, exps, logs, drop):
    seen = {}
    def record(name, fn):
        def call(x):
            value = fn(x)
            seen[name, word(x)] = word(value)
            return value
        return call
    p = ModeBallpark(friction=1, is_master=False)
    with patch.object(math, "exp", record("exp", math.exp)), patch.object(math, "log", record("log", math.log)):
        b = p.add_ball(1, 1, 1, 100, (0, 0, 0), (80, 0, 0), 1, 1)
        p.warp_to(1, 131072, 0, 0, 0, 4000)
        with patch.object(p, "integrate", wraps=p.integrate) as integrate:
            for stamp, mode, x, vx in rows:
                p.evolve()
                equal(f"warp commit {stamp}", (p.current_time, b.mode, word(b.pos[0]), word(b.vel[0])),
                      (stamp, mode, x, vx))
        starts = [[word(call.args[0][0]), word(call.args[1][0])] for call in integrate.call_args_list]
        equal("ordinary steps at dropout and the next tick", starts, [drop, rows[-2][2:]])
    equal("warp elementary-function inputs and outputs", seen,
          {(name, arg): value for name, table in [("exp", exps), ("log", logs)] for arg, value in table})
    equal("warp transitions", [(e[0], e[1]) for e in p.events if e[1].startswith("On")],
          [(0, "OnActivatingWarp"), (9, "OnDeactivatingWarp"), (9, "OnExitWarp")])



def upstream_warp(repo, pin, rows):
    """Use the existing conformance compiler and parser for the same 11 axial commits."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tests"))
    import mode_cases

    ship = dict(id=1, mass=1, radius=1, max_vel=100, agility=1, sf=1,
                pos=(0, 0, 0), vel=(80, 0, 0), free=True, massive=False, interactive=False,
                cloaked=0, bubble=-1, moribund=False, mode=2)
    case = dict(id=1, friction=1., tick=1000, start=0, new_orbit=False, master=False, steps=len(rows),
                balls=[ship], cmds=[(1, mode_cases.WARP_OP, 1, 0, 131072, 0, 0, 0, 4000)])
    with TemporaryDirectory() as build:
        run = subprocess.run([sys.executable, root / "conformance/build_modes.py", repo, build],
                             capture_output=True, encoding="utf-8")
        if run.returncode:
            raise SystemExit(f"Could not build the pinned Destiny flight oracle: {run.stderr.strip()}")
        manifest = json.loads(run.stdout)
        equal("Destiny flight source revision", manifest["destiny_commit"], pin)
        for path, digest in manifest["source_sha256"].items():
            original = subprocess.check_output(["git", "-C", repo, "show", f"{pin}:{path}"])
            equal(f"Destiny flight source {path}", digest, hashlib.sha256(original).hexdigest())
        with patch.object(mode_cases, "HARNESS", Path(build) / "harness"):
            states, events = mode_cases.oracle([case])
    states = [dict(zip(mode_cases.FIELDS, row, strict=True)) for row in states]
    equal("compiled Destiny warp commits", [[s["step"], s["mode"], word(s["px"]), word(s["vx"])]
                                            for s in states], rows)
    equal("compiled Destiny warp events", [(e[1], e[2]) for e in events if e[2].startswith("On")],
          [(0, "OnActivatingWarp"), (9, "OnDeactivatingWarp"), (9, "OnExitWarp")])

def ship_state(park):
    b = park.lane.balls[101]
    return [b.mode, *map(word, [*b.pos, *b.vel])]


def command_rewind(rows):
    # Two otherwise identical sessions. Delay only the packet with this GOTO in the second.
    wanted = {stamp: (sent, state) for stamp, sent, *state in rows}
    for delay in [0, 1]:
        run = session([(101, 0, 40)], {101: 0}, up=1)
        run.tick()
        run.command("go_to_direction", 101, 1., 0., 0.)
        sent = []
        pilot, ticker, desync = run.clients[0]
        for stamp in wanted:
            if stamp > run.tick_no:
                run.down = delay if stamp == 5 else 0
                run.tick()
            sent.extend(st for _, _, _, payload, _, _ in run.sent_log[-1][1]
                        for st, (action, _) in payload if action == "GotoDirection")
            stored_stamp, expected = wanted[stamp]
            equal("GOTO outgoing stamps", sent, [] if stored_stamp < 0 else [stored_stamp])
            equal("server movement", ship_state(run.server), expected)
            client_stamp = pilot.currentTime
            equal("client rewind clock", client_stamp, stamp - int(delay and stamp >= 6))
            if delay and stamp == 5:
                equal("speculative movement", ship_state(pilot), wanted[4][1])
                if not any(s[1] <= 4 for s in ticker._states):
                    raise SystemExit("Missing retained snapshot for the late GOTO")
            else:
                equal("client movement at common stamp", ship_state(pilot), wanted[client_stamp][1])
        equal("rewind desyncs", (desync.fatal, desync.recoverable), (0, 0))


def combat(rows):
    fixtures = [((.03, .04, .12), (3, 4, 12), 1e30), ((4, -3, 0), (3, 4, 12), 1), ((3, 0, 0), (4, 0, 0), 1)]
    for (accepted, sq, nan), (v, r, res) in zip(rows, fixtures, strict=True):
        p = full.FullBallpark()
        p.add_ball(1, 1, 1, 1)
        p.add_ball(2, 1, 1, 1, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
        with patch.object(full, "_sqrt", wraps=full._sqrt) as sqrt:
            chance, _, angular = p.get_accuracy(1, 2, 20, 1, 1, 1, res)
        equal("tracking square-root inputs", [word(c.args[0]) for c in sqrt.call_args_list], [sq])
        equal("tracking result", (math.isnan(angular), math.isinf(chance)), (bool(nan), not bool(accepted)))


def worlds(rows):
    import torch
    from exact_ballpark.vec_modes import VecModesPark
    from exact_ballpark.vec_server import ModesAdapter, VecServer

    def build(members):
        p = VecModesPark(len(members), device="cpu")
        p.load([dict(id=101, world=i, mass=2e6, radius=40, max_vel=300, agility=1, speed_fraction=1,
                     pos=(20, 0, -0.) if w else (0, 0, 0), vel=(0, -8, 3) if w else (0, 0, 0),
                     interactive=True) for i, w in enumerate(members)])
        p.goto_direction(torch.tensor([not w for w in members]),
                         torch.tensor([[1., 0., 0.]] * len(members), dtype=torch.float64))
        return p, VecServer(ModesAdapter(p))

    def read(p, i):
        return [int(p.clock[i]), int(p.mode[i]), *map(word, p.pos[i].tolist() + p.vel[i].tolist())]

    mixed, batched = build([0, 1])
    alone = [build([w]) for w in [0, 1]]
    interval = 10_000_000
    times = [(interval-1, interval), (interval, interval), (2*interval, interval+1),
             (2*interval, 2*interval), (3*interval, 3*interval)]
    if len(rows) != 2 * len(times):
        raise SystemExit("Incomplete mixed-world trace")
    for k, pair in enumerate(times):
        batched.on_tick(pair)
        for w, (one, solo) in enumerate(alone):
            solo.on_tick([pair[w]])
            equal("batched world / solo clock and bits", read(mixed, w), read(one, 0))
            equal("batched world / Lean movement", [w, *read(mixed, w)], rows[2*k+w])


def energy():
    # Independent analytic integrals over three unit intervals, for m=k=1.
    with localcontext() as ctx:
        ctx.prec = 70
        q = Decimal(-1).exp()
        v = [Decimal(1), Decimal(0), Decimal(0)]
        net_work, loss = Decimal(0), Decimal(0)
        dot = lambda a, b: sum(x*y for x, y in zip(a, b, strict=True))
        initial = dot(v, v)/2
        p = Ballpark(friction=1)
        pos, vel = [0.]*3, list(map(float, v))
        for raw in [(3, 0, 0), (0, -2, 0), (0, 0, 0)]:
            a = list(map(Decimal, raw))
            w = [x-y for x, y in zip(v, a, strict=True)]
            net_work += dot(a, a) + dot(a, w)*(1-q)
            loss += dot(a, a) + 2*dot(a, w)*(1-q) + dot(w, w)*(1-q*q)/2
            v = [x+y*q for x, y in zip(a, w, strict=True)]
            pos, vel = p.integrate(pos, vel, raw, 1, 1, float(q), 1)
            if abs((dot(v, v)/2-initial) - (net_work-loss)) > Decimal('1e-65') or loss < 0:
                raise SystemExit("Piecewise thrust / drag work budget failed")
            if any(abs(Decimal.from_float(x)-y) > Decimal('2e-15') for x, y in zip(vel, v, strict=True)):
                raise SystemExit("Scalar velocity left the three-burn reference tolerance")


def compare(run_lake, destiny_repo=None):
    data = defaultdict(list)
    for row in run_lake("env", "lean", "--run", "scripts/LawTrace.lean", capture=True).splitlines():
        name, *values = row.split()
        data[name].append(list(map(int, values)))
    equal("Lean trace rows", {key: len(v) for key, v in data.items()},
          dict(sensor=8, route=24, field=5, exp=11, log=2, warp=11, drop=1, goto=6, combat=3, world=10))
    # The server logs initial SetState and the normal producer/consumer stamp difference.
    with patch.object(server.logger, "error") as messages:
        for forward in [False, True]:
            sensor_scene(data["sensor"], forward)
        field_queries(data["field"])
        command_rewind(data["goto"])
        for msg in messages.call_args_list:
            if not str(msg.args[0]).startswith("Found ballpark action for mismatched timestamp"):
                raise SystemExit(f"Unexpected server diagnostic in movement replay: {msg.args[0]}")
    routes(data["route"])
    if destiny_repo:
        from replay_discrepancies import PINNED
        with upstream_routing(destiny_repo, PINNED):
            routes(data["route"])
        upstream_warp(destiny_repo, PINNED, data["warp"])
    rotations()
    warp_trace(data["warp"], data["exp"], data["log"], data["drop"][0])
    combat(data["combat"])
    worlds(data["world"])
    energy()
