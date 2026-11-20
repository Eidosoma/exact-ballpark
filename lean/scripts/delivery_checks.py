# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare send_batch packets and client snapshot operations with the Lean models."""
import json
from contextlib import nullcontext
from unittest import TestCase
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def batch_cases():
    # The same client can have interests through several balls and several characters.
    for ids, events, client, private in product(
        (None, [], [3], [3, 4]), ([], [11, 12]), (None, 0, 8), ([], [31, 32])
    ):
        yield dict(interests=[[3, [0, 7]], [4, [7, 8]]],
                   bubbles=[[ids, events], [[4], [21]]],
                   chars=[[client, private], [7, [41]], [7, [51, 52]]])
    yield dict(interests=[], bubbles=[], chars=[])
    yield dict(interests=[], bubbles=[], chars=[[8, [61]]])


def step(op, now, flag=False, running=True, stamp=None):
    return dict(op=op, now=now, stamp=now if stamp is None else stamp, flag=flag, running=running)


def cache_cases():
    # Cross the capacity boundary twice and rewind around a discarded run of stamps.
    yield ([step("store", i, flag=i % 2 == 0) for i in range(17)]
           + [step("sync", 20, stamp=i) for i in (-1, 0, 5, 15, 16, 20, 22)])
    # Equal stamps identify distinct streams. List order can also differ from stamp order.
    yield [step("store", i, flag=True) for i in (0, 8, 8, 12)] + [
        step("sync", 13, stamp=8), step("store", 5), step("sync", 13, stamp=8)]
    for previous, mid, running, new_base in product((3, 4, 5), (False, True), (False, True), (False, True)):
        yield [step("store", 4, flag=True), step("store", previous, flag=mid),
               step("rebase", 5, flag=new_base, running=running), step("sync", 7, stamp=4)]
    yield [step("store", 4), step("store", 9, running=False), step("post", 15, stamp=14),
           step("post", 16, stamp=15), step("post", 17, flag=True), step("post", 28, running=False)]


def compare_batch(case):
    from exact_ballpark.client import RecordingNetwork
    from exact_ballpark.server import ParkUpdateBatcher

    class Network(RecordingNetwork):
        def __init__(self):
            super().__init__()
            self.calls = []

        def singlecast(self, packets):
            self.calls.append("private")
            super().singlecast(packets)

        def narrowcast(self, packets):
            count = len(self.calls)
            self.calls.append("bubble")
            if any(packet[-1] != count for packet in packets):
                raise SystemExit("Bubble packet in the wrong send_batch interface call")
            super().narrowcast(packets)

    interests = dict(case["interests"])
    chars = dict(enumerate(client for client, _ in case["chars"]))
    bubbles = {b: ids for b, (ids, _) in enumerate(case["bubbles"]) if ids is not None}
    net = Network()
    batcher = ParkUpdateBatcher(
        SimpleNamespace(currentTime=1, bubbleInteractives=bubbles), net,
        SimpleNamespace(get_client_id_for_character=chars.get),
        SimpleNamespace(get_interested_client_ids_for_ball=lambda ball: interests.get(ball, [])), None)
    for i, (_, events) in enumerate(case["bubbles"]):
        batcher._bubble_history[i] = [(1, ("probe", token)) for token in events]
    for i, (_, events) in enumerate(case["chars"]):
        batcher._character_history[i] = [(1, ("probe", token)) for token in events]
    batcher.send_batch()
    sent = net.take()
    first_calls = net.calls
    net.calls = []
    batcher.send_batch()
    if first_calls != ["private", "bubble", "bubble"] or net.take():
        raise SystemExit("Batch call order or history draining changed")
    output = []
    for kind, ids, method, payload, wait, count in sent:
        if method != "DoDestinyUpdate" or any(stamp != 1 or name != "probe" for stamp, (name, _) in payload):
            raise SystemExit("Changed destination method, action or event stamp")
        tokens = [token for _, (_, token) in payload]
        output.append([[ids] if kind == "single" else sorted(ids), tokens, wait, int(count)])
    return output


def compare_cache(steps, expected):
    from exact_ballpark.client import ClientPark, ClientTickerInterface, TickErrorHandlerInterface, Ticker

    class Park(ClientPark):
        def WriteFullStateToStream(self, stream, src_id=-1):
            super().WriteFullStateToStream(stream, src_id)
            stream.probe_token = self.token

        def _parent_ReadFullStateFromStream(self, stream, partial=0):
            if partial != 1:
                raise SystemExit("Snapshot restore changed its partial-read flag")
            self.restored = stream.probe_token
            super()._parent_ReadFullStateFromStream(stream, partial)

        def _parent_Evolve(self):
            self.ticks += 1
            super()._parent_Evolve()

    park = Park()
    park.lane.add_ball(1, 1., 1., 1., agility=1.)
    ticker = Ticker(TickErrorHandlerInterface(), ClientTickerInterface())
    ticker.set_ballpark(park)
    result = []
    for token, s in enumerate(steps):
        park.token, park.restored, park.ticks = token, None, 0
        park.currentTime = s["now"]
        if s["running"]:
            park.Start()
        else:
            park.Pause()
        success = True
        if s["op"] == "store":
            ticker.store_state(mid_tick=s["flag"])
        elif s["op"] == "rebase":
            ticker.flush_simulation_history(new_base_snapshot=s["flag"])
        elif s["op"] == "post":
            ticker._should_rebase = s["flag"]
            ticker.do_post_tick(s["stamp"])
            if ticker._should_rebase:
                raise SystemExit("Client failed to clear its post-tick rebase flag")
        elif s["op"] == "sync":
            missing = not expected[token][3]
            capture = TestCase().assertLogs("exact_ballpark.client", level="WARNING") if missing else nullcontext()
            with capture as logs:
                success = bool(ticker.synchronize_to_simulation_time(s["stamp"]))
            if missing and [r.getMessage() for r in logs.records] != [
                    "SynchroniseToSimulationTime: Did not find any state"]:
                raise SystemExit(f"Unexpected snapshot warnings: {logs.output}")
            if park.currentTime != (s["stamp"] if success else s["now"]):
                raise SystemExit("Client synchronize returned with an unexpected clock")
        saved = [[stream.probe_token, stamp, mid] for stream, stamp, mid in ticker._states]
        result.append([saved, ticker._last_stamp, park.restored, success, park.ticks])
    return result


def compare(run_lake):
    batches, traces = list(batch_cases()), list(cache_cases())
    with TemporaryDirectory() as temp:
        path = Path(temp) / "fixtures.json"
        path.write_text(json.dumps(dict(batches=batches, traces=traces)))
        expected_batches, expected_traces = json.loads(run_lake(
            "env", "lean", "--run", "scripts/DeliveryProbe.lean", str(path), capture=True))
    for case, expected in zip(batches, expected_batches, strict=True):
        observed = compare_batch(case)
        if observed != expected:
            raise SystemExit(f"send_batch mismatch for {case}: {observed} != {expected}")
    for steps, expected in zip(traces, expected_traces, strict=True):
        for s, observed, want in zip(steps, compare_cache(steps, expected), expected, strict=True):
            if observed != want:
                raise SystemExit(f"Snapshot mismatch at {s}: {observed} != {want}")
    return len(batches), sum(map(len, traces))
