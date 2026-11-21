# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare the client merge to Lean and replay packet admission and blocking on both clients."""
import ast
import collections
import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from exact_ballpark import client, server
from exact_ballpark.ref import STOP


def server_park():
    park = server.DestinyPark(is_master=True)
    for i in (1, 2, 3):
        args = (i, 1e6, 1., 10., True, False, True, True, False, i*100., 100., 100., 0., 0., 0., 1., 1.)
        park.AddBall(*(args + ((1., 1.) if server.is_dynamical_orientation_enabled() else ())))
    return park


def follower_cleanup():
    for duplicate in (False, True):
        park = server_park()
        session = client.Session(park, {})
        for i in (1, 3):
            park.FollowBall(i, 2, 1000.)
        park.SetBallPosition(1, 500000000., 0., 0.)
        if park.GetRemoteFollowers() != [1] or park.lane.balls[2].followers != {1, 3}:
            raise SystemExit("Pre-tick fixture did not establish the two follower memberships")
        if duplicate:
            session.actions.stop(1)
        else:
            session.actions.set_ball_mass(1, 12.)
        ticker = session.ticker
        update = ticker._update_bubbles
        observed = []

        def at_bubbles():
            b = park.lane.balls[1]
            observed.append((session.actions._stamp_for_system, list(session.actions._system_history),
                             [(stamp, event[0]) for _, stamp, event in ticker.current_system_history],
                             b.mode, b.follow_id, b.owner_id, b.effect_stamp, b.follow_ptr, b.follow_range,
                             set(park.lane.balls[2].followers), b.mass))
            update()

        with patch.object(ticker, "_update_bubbles", at_bubbles):
            ticker.tick()
        queue = [(0, "Stop")] if duplicate else [(0, "SetBallMass"), (0, "Stop")]
        if observed != [(1, [], queue, STOP, 0, 0, 0, None, 0., {3}, 1e6 if duplicate else 12.)]:
            raise SystemExit(f"Remote-follower state at the bubble update changed: {observed}")


def producer_packets():
    """Drive both the server and the real client. Only the first post-tick send is postponed."""
    for delay in (False, True):
        park = server_park()
        session = client.Session(park, {1: 0})
        # The batcher logs stale stamps at post-tick, including a normal session startup.
        # Other logger methods and all park actions still run.
        with patch.object(server.logger, "error") as errors, patch.object(client.logger, "error") as client_errors:
            session.start()
            session.command("cloak_ball", 1, 2, 0.)
            session.tick()
            recipient = session.clients[0][1]
            desync = session.clients[0][2]
            fatal_before = desync.fatal
            if park.balls[1].isCloaked != 2 or recipient._history:
                raise SystemExit("Private-history fixture requires a cloaked pilot and an empty client queue")
            session.command("set_ball_mass", 1, 120.)
            session.command("set_ball_mass", 2, 240.)
            if delay:
                with patch.object(session.batcher, "send_batch"):
                    session.tick()
            else:
                session.tick()
            session.command("set_ball_mass", 1, 121.)
            session.tick()
            if any(not str(call.args[0]).startswith("Found ballpark action for mismatched timestamp")
                   for call in errors.call_args_list):
                raise SystemExit("Unexpected server error in producer replay")
            if (desync.fatal - fatal_before != int(delay) or
                    [c.args[0] for c in client_errors.call_args_list] !=
                    (["Found update batch with %s items and %s timestamps", "Action: %s", "Action: %s"] if delay else [])):
                raise SystemExit("Unexpected client diagnostics for the producer's packets")
        packets = []
        for _, frame in session.sent_log[-2:]:
            packets.append([(kind, wait, int(count), state)
                            for kind, _, _, state, wait, count in frame])
        early, visible, late = [(s, ("SetBallMass", (i, m))) for s, i, m in
                                ((3, 1, 120.), (3, 2, 240.), (4, 1, 121.))]
        first = ("single", True, 2, [early])
        reply = ("narrow", False, 2, [visible])
        expected = ([[], [("single", True, 2, [early, late]), reply]] if delay else
                    [[first, reply], [("single", False, 1, [late])]])
        if packets != expected or recipient._history != ([[[late], True]] if delay else []):
            raise SystemExit(f"Server packets or final client wait changed at delay={delay}: {packets}")


def compare(run_lake):
    """Mix all nonempty incoming stamp sets with every old stamp set, in both wait modes."""
    expected = []
    for old in range(8):
        for new in range(1, 8):
            for wait in (False, True):
                history = [[[(i, 10+i)], i == 1] for i in range(3) if old & (1 << i)]
                packet = [(i, base+i) for base in (100, 200) for i in (2, 1, 0) if new & (1 << i)]
                client.merge_state_into_history(packet, history, wait)
                expected.append("|".join(f"{events[0][0]}:{int(w)}:" +
                                         ",".join(str(event) for _, event in events) for events, w in history))
    actual = run_lake("env", "lean", "--run", "scripts/MixedHistoryProbe.lean", capture=True).splitlines()
    if expected != actual:
        raise SystemExit("Mixed-stamp Lean histories differ from the scalar client")
    follower_cleanup()
    producer_packets()
    return len(expected)


def upstream_methods(repo, pin, logger):
    """Load just the three pinned functions. Clock, logging and action dispatch belong to the fixture."""
    scope = {"collections": collections, "logger": logger}
    for file, owner, names in [("_util.py", None, ["merge_state_into_history"]),
                               ("_ticker.py", "Ticker", ["_flush_state", "do_pre_tick"])]:
        path = f"python/destiny/net/client/{file}"
        source = subprocess.check_output(["git", "-C", repo, "show", f"{pin}:{path}"], encoding="utf-8")
        body = ast.parse(source).body
        if owner:
            body = next((node.body for node in body if isinstance(node, ast.ClassDef) and node.name == owner), [])
        funcs = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
        if [node.name for node in funcs] != names:
            raise SystemExit(f"Could not locate the pinned functions in {path}")
        exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), scope)
    return scope["_flush_state"], scope["do_pre_tick"]


def packet_trace(flush, tick):
    def fixture():
        applied = []
        ticker = SimpleNamespace(_history=[], _latest_set_state_time=0, _current_time=12,
                                 _dump_history=lambda: None, _state_is_valid=False,
                                 _real_flush_state=lambda group: applied.append(group[0][0]))
        return ticker, applied

    def state(ticker):
        return tuple((events[0][0], waiting) for events, waiting in ticker._history)

    def packet(*stamps):
        return [(stamp, ("GotoDirection", (1, 1, 0, 0))) for stamp in stamps]

    t, applied = fixture()
    flush(t, packet(11, 10), True)
    waits = state(t)
    flush(t, packet(10), False)
    tick(t)
    t._current_time = 100
    tick(t)
    blocked = (tuple(applied), state(t))
    flush(t, packet(11), False)
    tick(t)
    resumed = (tuple(applied), state(t))

    def after_reset(*stamps):
        t, _ = fixture()
        flush(t, [(10, ("SetState", ()))], False)
        tick(t)
        flush(t, packet(*stamps), False)
        return tuple(stamp for stamp, _ in state(t))

    return waits, blocked, resumed, after_reset(11, 9), after_reset(9, 11)


def check_log(log, repo, pin):
    quiet = logging.Logger("mixed-history-replay")
    quiet.addHandler(logging.NullHandler())
    with patch.object(client, "logger", quiet):
        local = packet_trace(client.Ticker._flush_state, client.Ticker.do_pre_tick)
    expected = (((10, True), (11, True)), ((10,), ((11, True),)), ((10, 11), ()), (9, 11), ())
    if local != expected:
        raise SystemExit(f"Client packet outcomes changed: {local!r}")
    if repo is not None and packet_trace(*upstream_methods(repo, pin, quiet)) != local:
        raise SystemExit("The pinned Destiny client disagrees on the mixed-stamp packets")
    waits, (applied, queued), (done, _), fresh_first, stale_first = local
    observed = [
        f"| Private packet | {[stamp for stamp, wait in waits if wait]} |",
        f"| Reply at 10, then advance the clock to 100 | {list(applied)} applied, {[k for k, w in queued if w]} waiting |",
        f"| Reply at 11 | {list(done)} applied, queue empty |",
        f"| Fresh event first | {list(fresh_first)} |",
        f"| Stale event first | {list(stale_first)} |",
    ]
    if any(line not in log.splitlines() for line in observed):
        raise SystemExit("The discrepancy log differs from the client packet trace")
