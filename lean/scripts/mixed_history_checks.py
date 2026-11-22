# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare the client merge to Lean and replay packet admission and blocking on both clients."""
import ast
import collections
from itertools import zip_longest
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
    """Capture both possible companion stamps from the real server, with and without a postponed send."""
    batches = {}
    for delay, reply_stamp in ((False, 3), (False, 4), (True, 3), (True, 4)):
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
            if reply_stamp == 3:
                session.command("set_ball_mass", 2, 240.)
            if delay:
                with patch.object(session.batcher, "send_batch"):
                    session.tick()
            else:
                session.tick()
            session.command("set_ball_mass", 1, 121.)
            if reply_stamp == 4:
                session.command("set_ball_mass", 2, 240.)
            before_last_send = errors.call_count
            session.tick()
            if any(not str(call.args[0]).startswith("Found ballpark action for mismatched timestamp")
                   for call in errors.call_args_list):
                raise SystemExit("Unexpected server error in producer replay")
            old_pilot_event = str((3, ("SetBallMass", (1, 120.))))
            reported = any(c.args[0].endswith(old_pilot_event) for c in errors.call_args_list[before_last_send:])
            if reported != delay:
                raise SystemExit("The server stamp check did not distinguish the postponed pilot event")
            if (desync.fatal - fatal_before != int(delay) or
                    [c.args[0] for c in client_errors.call_args_list] !=
                    (["Found update batch with %s items and %s timestamps", "Action: %s", "Action: %s"] if delay else [])):
                raise SystemExit("Unexpected client diagnostics for the producer's packets")
        packets = []
        for _, frame in session.sent_log[-2:]:
            packets.append([(kind, wait, int(count), state)
                            for kind, _, _, state, wait, count in frame])
        early, visible, late = [(s, ("SetBallMass", (i, m))) for s, i, m in
                                ((3, 1, 120.), (reply_stamp, 2, 240.), (4, 1, 121.))]
        reply = ("narrow", False, 2, [visible])
        if delay:
            expected = [[], [("single", True, 2, [early, late]), reply]]
            queue = [[[late], True]] if reply_stamp == 3 else [[[early], True], [[late, visible], False]]
        else:
            expected = [[("single", False, 1, [change])] if change[0] != reply_stamp else
                        [("single", True, 2, [change]), reply] for change in (early, late)]
            queue = []
        if packets != expected or recipient._history != queue:
            raise SystemExit(f"Server packet or queue changed for reply {reply_stamp}, delay={delay}: {packets}")
        if delay:
            batches[reply_stamp] = [state for _, _, _, state in packets[-1]]
    return batches


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
    drain_rows = run_lake("env", "lean", "--run", "scripts/MixedHistoryProbe.lean", "drain", capture=True).splitlines()
    quiet = logging.Logger("queue-comparison")
    quiet.addHandler(logging.NullHandler())
    with patch.object(client, "logger", quiet):
        scalar = queue_traces(client.Ticker.update, client.Ticker._report_update_if_bad,
                              client.Ticker._flush_state, client.Ticker.do_pre_tick)
    for cpu_row, lean_row in zip_longest(scalar, drain_rows):
        if cpu_row != lean_row:
            raise SystemExit(f"Queue dispatch differs: scalar {cpu_row!r}, Lean {lean_row!r}")
    follower_cleanup()
    return len(expected)


def upstream_methods(repo, pin, logger):
    """Use the public update path from the pinned client. The fixture supplies the clock and callbacks."""
    scope = {"collections": collections, "logger": logger}
    for file, owner, names in [("_util.py", None, ["merge_state_into_history"]),
                               ("_baseticker.py", None, ["_expand_states"]),
                               ("_baseticker.py", "BaseTicker", ["update", "_report_update_if_bad"]),
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
    return scope["update"], scope["_report_update_if_bad"], scope["_flush_state"], scope["do_pre_tick"]


def reply_controls(batches, fixture, receive, tick):
    """Deliver the captured events to one recipient, checking both queues and dispatched payload order."""
    results = {}
    for reply_stamp, (private, bubble) in batches.items():
        stamps = sorted({stamp for stamp, _ in private})
        private_groups = [[event for event in private if event[0] == stamp] for stamp in stamps]
        complete = [[event for event in private + bubble if event[0] == stamp] for stamp in stamps]
        schedules = {
            "engine": [(private, True), (bubble, False)],
            "earliest": [(group, i == 0) for i, group in enumerate(private_groups)] + [(bubble, False)],
            "complete": [(group, False) for group in complete],
        }
        for policy, packets in schedules.items():
            ticker, dispatched = fixture()
            ticker._real_flush_state = dispatched.append
            for events, wait in packets:
                receive(ticker, events, wait)
            tick(ticker)
            if dispatched + [group for group, _ in ticker._history] != complete:
                raise SystemExit(f"Lost or reordered payloads with {policy} packet delivery at {reply_stamp}")
            waiting = tuple(group[0][0] for group, wait in ticker._history if wait)
            results[reply_stamp, policy] = waiting, tuple(group[0][0] for group in dispatched)
    return results


def packet_harness(update, report, flush):
    def fixture():
        applied = []
        ticker = SimpleNamespace(_ballpark=SimpleNamespace(), _history=[], _latest_set_state_time=0, _current_time=12,
                                 calls=[],
                                 _dump_history=lambda: None, _state_is_valid=False,
                                 _real_flush_state=lambda group: applied.append(group[0][0]))
        ticker._error_handler = SimpleNamespace(on_fatal_desync=lambda: ticker.calls.append("fatal"))
        ticker._report_update_if_bad = lambda events: report(ticker, events)

        def admit(events, wait):
            ticker.calls.append("admission")
            flush(ticker, events, wait)

        def evolve():
            ticker._current_time += 1

        ticker._flush_state = admit
        ticker._ballpark._parent_Evolve = evolve
        return ticker, applied

    def receive(ticker, events, wait):
        ticker.calls.clear()
        update(ticker, events, wait)
        mixed = len({stamp for stamp, _ in events}) > 1
        if ticker.calls != (["fatal", "admission"] if mixed else ["admission"]):
            raise SystemExit(f"Unexpected diagnostic order for {events}: {ticker.calls}")
        return ticker.calls.count("fatal")

    return fixture, receive


def queue_traces(update, report, flush, tick):
    """Compare full payload order, wait bits and the reached clock after one pre-tick."""
    fixture, receive = packet_harness(update, report, flush)
    rows = []

    def event(stamp, token):
        return stamp, ("GotoDirection", (1, token, 0, 0))

    def encode(groups):
        return "|".join(f"{events[0][0]}:{int(wait)}:" +
                        ",".join(str(action[1][1]) for _, action in events) for events, wait in groups)

    for count in (2, 3, 5):
        groups = [[event(3+i, 10+i), event(3+i, 20+i)] for i in range(count)]
        reply = [event(2+count, 99)]
        complete_groups = [*groups[:-1], groups[-1] + reply]
        for complete in (False, True):
            packets = ([(group, False) for group in complete_groups] if complete else
                       [([entry for group in groups for entry in group], True), (reply, False)])
            for now in (2, 12, 100):
                ticker, applied = fixture()
                ticker._current_time = now
                ticker._real_flush_state = applied.append
                for packet, wait in packets:
                    receive(ticker, packet, wait)
                tick(ticker)
                if applied + [group for group, _ in ticker._history] != complete_groups:
                    raise SystemExit(f"Lost or reordered a payload in the {count}-stamp queue")
                done = encode((group, False) for group in applied)
                rows.append(f"{count} {str(complete).lower()} {now} {ticker._current_time} "
                            f"[{done}] [{encode(ticker._history)}]")
    return rows


def packet_trace(update, report, flush, tick, batches=None):
    fixture, receive = packet_harness(update, report, flush)

    def state(ticker):
        return tuple((events[0][0], waiting) for events, waiting in ticker._history)

    def packet(*stamps):
        return [(stamp, ("GotoDirection", (1, 1, 0, 0))) for stamp in stamps]

    t, applied = fixture()
    private_reports = receive(t, packet(11, 10), True)
    waits = state(t)
    receive(t, packet(10), False)
    tick(t)
    t._current_time = 100
    tick(t)
    blocked = (tuple(applied), state(t))
    receive(t, packet(11), False)
    tick(t)
    resumed = (tuple(applied), state(t))

    def after_reset(*stamps):
        t, _ = fixture()
        receive(t, [(10, ("SetState", ()))], False)
        tick(t)
        reports = receive(t, packet(*stamps), False)
        return reports, tuple(stamp for stamp, _ in state(t))

    trace = private_reports, waits, blocked, resumed, after_reset(11, 9), after_reset(9, 11)
    if batches is None:
        return trace
    return (*trace, reply_controls(batches, fixture, receive, tick))


def check_log(log, repo, pin):
    batches = producer_packets()
    quiet = logging.Logger("mixed-history-replay")
    quiet.addHandler(logging.NullHandler())
    methods = (client.Ticker.update, client.Ticker._report_update_if_bad,
               client.Ticker._flush_state, client.Ticker.do_pre_tick)
    with patch.object(client, "logger", quiet):
        local = packet_trace(*methods, batches)
    complete = ((), (3, 4))
    expected_schedules = {(3, "engine"): ((4,), (3,)), (4, "engine"): ((3,), ()),
                          (3, "earliest"): complete, (4, "earliest"): ((3,), ()),
                          (3, "complete"): complete, (4, "complete"): complete}
    expected = (1, ((10, True), (11, True)), ((10,), ((11, True),)), ((10, 11), ()),
                (1, (9, 11)), (1, ()), expected_schedules)
    if local != expected:
        raise SystemExit(f"Client packet outcomes changed: {local!r}")
    if repo is not None:
        pinned = upstream_methods(repo, pin, quiet)
        if packet_trace(*pinned, batches) != local:
            raise SystemExit("The pinned Destiny client disagrees on the mixed-stamp packets")
        with patch.object(client, "logger", quiet):
            if queue_traces(*pinned) != queue_traces(*methods):
                raise SystemExit("The pinned Destiny client disagrees on queue payloads, waits or clocks")
    reports, waits, (applied, queued), (done, _), fresh, stale, schedules = local
    observed = [
        f"| Private packet | {reports} fatal report, {[stamp for stamp, wait in waits if wait]} waiting |",
        f"| Reply at 10, then advance the clock to 100 | {list(applied)} applied, {[k for k, w in queued if w]} waiting |",
        f"| Reply at 11 | {list(done)} applied, queue empty |",
        f"| Fresh event first | {fresh[0]} | {list(fresh[1])} |",
        f"| Stale event first | {stale[0]} | {list(stale[1])} |",
    ]
    for stamp in batches:
        held = list(schedules[stamp, "engine"][0])
        first_only = list(schedules[stamp, "earliest"][0])
        released = list(schedules[stamp, "complete"][1])
        observed.append(f"| {stamp} | {held} | {first_only} | {released} |")
    if any(line not in log.splitlines() for line in observed):
        raise SystemExit("The discrepancy log differs from the client packet trace")
