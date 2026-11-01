# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the bubble partition: what each client
receives when a world holds several bubbles. `SpaceServer` extends the `exact_ballpark.vec_server.VecServer`
(its queue, stamps, duplicate test, slots and adapters, unchanged) from one static bubble a world to the master's
partition kept by `exact_ballpark.vec_space.VecSpace`, so that the gym can observe exactly what a pilot's client knows.

It must reproduce `exact_ballpark.server` (Destiny's net/server transcribed: Actions, Ticker, BubbleUpdater,
ParkUpdateBatcher on a master DestinyPark over FullBallpark) bit for bit, driven as Destiny's test helper drives it
(test_ticker.py:172-182: the Ticker's tick, the park's Evolve, the batcher's send_batch). Checked by
tests/test_movement_vec_server_space.py on generated command streams over worlds of several bubbles. What it adds to
the layer, each in Destiny's order (_baseticker.py:23-32, _ticker.py:56-196, _parkupdatebatcher.py:52-316):

- MOV-731, the follower clean-up: at each pre-tick, before the stamp moves, a Stop is queued for every free ball
  following a ball missing or in another bubble (GetRemoteFollowers, ascending id), behind the queued commands and
  under the duplicate test (Actions.clean_up_followers).
- The partition through the actions: SetBallPosition, and SetBallRadius and SetMaxSpeed when accepted, and SetBallFree when it fixes a free ball, re-file their ball (its bubble may change); SetBallInteractive moves it in or out of its bubble's interactive set; CloakBall's
  normal cloak adds its 2 s proximity sensor and UncloakBall removes it and drops the ball from the sensor lists of its
  bubble (Ballpark.cpp:5223-5290); FollowBall and Orbit refuse a target in another bubble (the park's guard reads the
  bubble the layer keeps).
- BubbleUpdater (AdditionsAndDeletions and CopyBubbles, once per park time): per user ball its bubble's additions and
  removals when it stayed, its own lists across its old and new bubbles when it moved (its client is told to remove
  the old bubble's balls and given the new bubble's: RemoveBalls, then a PackagedAction of AddBallsToPark), each to
  the ship's character (_parkupdatebatcher.py:225-316).
- SEN-712's routes per bubble: an event goes to its ball's bubble (valid: a dictionary and an interactive ball), or to
  its ball's character while the ball is cloaked; CloakBall to the bubble it was queued in (the old bubble while the
  two differ, _actions.py:628-638); an uncloak's AddBallsToPark to the other interactive balls of its bubble.
- The send (send_batch, after Evolve): each bubble with history and an interactive ball narrowcasts to the clients of
  its interactive balls, so a pilot hears its ship's bubble only; wait_for_bubble and the counts as Destiny sets them.
- The tick: Evolve with the dead bubbles' balls left out (TIME-714), the commits' re-filing and HandleProximities
  (the master's sensor events stay on the server: Destiny sends them to the park's Python side, not to clients).

`SpaceClientView` extends the ClientView: besides the delayed fields, which balls each client's park holds (the
SetState-equivalent first additions, the removals and additions as its ship changes bubble, the cloaks and uncloaks;
the global balls' notices and the SetState's global balls), delayed by `down` ticks like the rest.

Global balls (_actions.py:328-362, _ticker.py:173-196): over a VecCollidePark made with
`global_balls` (SpaceCollideAdapter on CollideView), and over VecModesPark and VecOrientPark with the
flags held by the view (`space_for(adapter, global_balls)`: their parks have no collisions, so the layer's sensors
and bubble lists and the server's streams read it), `push` takes make_ball_global and make_ball_local (the kinds
AddBallsToPark and BallNotGlobal): SetBallGlobal is applied at the call (the park's flag, which its collisions and
the layer's sensors and bubble lists read), and the notice is queued under the duplicate test: AddBallsToPark with the
ball's stream as written at the call (the fields and the park's time then, kept with the call: a world's first
OnTick evolves without a pre-tick, so the queue's calls may be older than the park's state), BallNotGlobal with the
ball's bubble at the call (not queued for a ball not in the park); the duplicate test compares AddBallsToPark's
stream by its bytes (as Python compares the queued tuples). Neither
reaches the park at the pre-tick (no park method); in the history AddBallsToPark goes to every bubble with an
interactive ball and BallNotGlobal's RemoveBall to every such bubble but the one it names (route EVERY), so a client
in another bubble hears of a global ball; the streams carry DSTBALL_ISGLOBAL. make_ball_global of a
ball not in the park queues its empty stream (the header and the park's time: WriteBallsToStream skips a ball it does
not find), routed as any; SpaceClientView applies the notices (a client in another bubble adds the ball on
AddBallsToPark and removes it on RemoveBall; `set_state` holds the SetState's global balls). Refused: the kernel path
(exact_ballpark.vec_server_space_cuda: its adapters do not take the kinds), RemoveGlobalBall (no removal here).

OnTick's intervals (Ballpark.cpp:219-330, TIME-702 to TIME-705; exact_ballpark.full's
FullBallpark.on_tick): `on_tick(sim_time)` drives every world by its own simulation clock (100 ns units, as Destiny's
server calls OnTick): a world whose time has not moved by a whole interval does nothing (its time kept), the master's
backlog of more than five intervals is cut to one, a world's first call that holds an interval evolves once without
the tick events, and every other world runs its whole intervals, each a pre-tick, the park's tick and a post-tick,
the remainder carried. The worlds tick together under a per-world mask (`pre_tick`, `evolve` and `post_tick` take
`mask`): a world without a tick keeps its queue, its stamp, its last action and its lists, the layer's operations and
the park's tick leave it as it is (VecCollidePark's `active`), and it sends nothing. Over every adapter (VecModesPark's and VecOrientPark's ticks take the mask as rows, exact_ballpark.vec_space's `row_mask`, inside the
dead bubbles' skip; the kernel path, vec_server_space_cuda, refuses on_tick); BringOutTheDeads has nothing to do here
(no removal); the balls' mNewTime and mOldTime, which OnTick's timestamp sets, are kept by no batched park and sent in
no stream.

Not here (each refused or left out): global balls but the above, InitializeBubbles during the server's play and
UpdateBallBubble (a ball in two interactive sets is flagged), missiles, the game's SetState (send_full_state_update)
and DoBubbleTransitions' game-side handling, the server kernel (vec_server_cuda: one bubble a world).

Over VecCollidePark (`SpaceCollideAdapter`, the layer on its CollideView) the master's
collisions keep within bubbles: the park skips a candidate in another bubble and the dead bubbles' balls, its contacts
ordered by the layer's filing (exact_ballpark.vec_space's "Collisions within bubbles"); GOTO, STOP, the speed fraction,
SetBallPosition, SetBallMassive, SetBallInteractive and every cloak, the normal cloak's sensor included, are applied.
A plain CollideAdapter (the client's-view adapter of the one-bubble layer) runs here too, with its own refusals.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from . import vec_server as VS
from . import vec_space as SP
from .full import MAX_CATCH_UP_TICKS                 # Ballpark.cpp:248 (TIME-703)
from .modes import WARP

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
STOP = VS.CODE["Stop"]
REFILERS = (VS.CODE["SetBallPosition"], VS.CODE["SetBallRadius"], VS.CODE["SetMaxSpeed"])


def _with_ok(s: VS.Slot, ok: torch.Tensor) -> VS.Slot:
    """A copy of a queued call with its queued flag [W] replaced (its other attributes, a stream's fields among them,
    kept)."""
    t = copy.copy(s)
    t.ok = ok
    return t


class SpaceModesAdapter(VS.ModesAdapter):
    """ModesAdapter whose proximity sensor is the partition layer's (the cloak's sensor added there). 
    SetBallFree is applied (SetBallFree(false) re-files its ball: the layer's REFILE, as SetBallPosition's); SetBallTroll
    is not (its PetrifyTroll, in the park's tick, takes the troll out of its bubble's interactive set and re-files it,
    which the partition layer's tick does not do: vec_space's)."""

    def __init__(self, park, master: bool = True):
        super().__init__(park, master)
        self.kinds = frozenset(self.kinds - {VS.TROLL})

    def set_sensor(self, m: torch.Tensor) -> None:
        pass

    def clear_sensor(self, m: torch.Tensor) -> None:
        pass


class SpaceOrientAdapter(VS.OrientAdapter):
    """OrientAdapter whose proximity sensor is the partition layer's."""

    def set_sensor(self, m: torch.Tensor) -> None:
        pass

    def clear_sensor(self, m: torch.Tensor) -> None:
        pass


class SpaceCollideAdapter(VS.CollideAdapter):
    """CollideAdapter (VecCollidePark: GOTO and STOP with Destiny's collisions) on the master's partition: its proximity sensor is the partition layer's (`sensor` [R] here, written by the layer), so every cloak
    mode is applied, the normal cloak's 2 s sensor included (its re-filings are the layer's, and the park's filing
    follows the layer's: exact_ballpark.vec_space.CollideView); each world's rows are listed in the order its balls
    were added (`added`: per world the ball ids in AddBall's order, the bubble dictionary's order; by default
    ascending id, the park's slots); and the setters VecCollidePark has for this layer (SETTERS). The park's
    collisions skip other bubbles' balls through its `bubble` field, which the layer keeps."""

    def __init__(self, park, interactive: list | None = None, added: list | None = None):
        super().__init__(park, interactive=interactive, master=True)
        W, N, d = park.W, park.N, self.dev
        self.sensor = torch.zeros(self.R, dtype=BOOL, device=d)
        self.kinds = self.kinds | {VS.CODE[k] for k in self.SETTERS}
        self.cloak_modes = frozenset(range(256))
        if added is not None:
            if len(added) != W:
                raise ValueError(f"added: {len(added)} worlds for a park of {W}")
            ids = park.id.cpu().tolist()
            ex = park.exists.cpu().tolist()
            rows = []
            for w, order in enumerate(added):
                slot = {ids[w][s]: s for s in range(N) if ex[w][s]}
                if sorted(int(b) for b in order) != sorted(slot):
                    raise ValueError(f"world {w}: added lists {sorted(order)}, the park holds {sorted(slot)}")
                rows.append([w * N + slot[int(b)] for b in order] + [-1] * (N - len(order)))
            self.rows_of_world = torch.tensor(rows, dtype=I64, device=d).reshape(W, N)
            safe = self.rows_of_world.clamp(min=0)
            self.ids_of_world = torch.where(self.rows_of_world >= 0, self.ids[safe], 0)

    def set_sensor(self, m: torch.Tensor) -> None:
        pass

    def clear_sensor(self, m: torch.Tensor) -> None:
        pass

    SETTERS = ("SetBallPosition", "SetBallVelocity", "SetBallRadius", "SetMaxSpeed", "SetBallMass", "SetBallAgility")

    def apply(self, kind: int, m: torch.Tensor, trow: torch.Tensor, f: torch.Tensor, n: torch.Tensor) -> None:
        """CollideAdapter's kinds, and the setters VecCollidePark has for the master's layer (the re-filings that
        follow SetBallPosition, and SetBallRadius and SetMaxSpeed when accepted, are SpaceServer's layer's)."""
        name = VS.KINDS[kind]
        if name not in self.SETTERS:
            super().apply(kind, m, trow, f, n)
            return
        p, W, N = self.park, self.park.W, self.park.N
        mm = m.reshape(W, N)
        one = f[:, 0].reshape(W, N)
        xyz = f[:, :3].reshape(W, N, 3)
        if name == "SetBallPosition":                   # the new and the old position
            p.pos = torch.where(mm[..., None], xyz, p.pos)
            p.old_pos = torch.where(mm[..., None], xyz, p.old_pos)
        elif name == "SetBallVelocity":
            p.set_velocity(mm, xyz)
        elif name == "SetBallRadius":
            p.set_radius(mm, one)
        elif name == "SetMaxSpeed":
            p.set_max_velocity(mm, one)
        elif name == "SetBallMass":
            p.set_mass(mm, one)
        else:
            p.set_agility(mm, one)


def space_for(adapter, global_balls=None) -> SP.VecSpace:
    """The partition layer over an adapter's park: the adapter's worlds and rows (each world's slots in the order its
    balls were added), the views of exact_ballpark.vec_space. `global_balls`: each row's global flag at
    the load [R], held by the view over VecModesPark or VecOrientPark (a VecCollidePark holds its own: refused)."""
    if isinstance(adapter, VS.ModesAdapter):
        view = SP.ModesView(adapter.park, global_balls)
    elif isinstance(adapter, VS.OrientAdapter):
        view = SP.OrientView(adapter.park, adapter.ids, flags=adapter, global_balls=global_balls)
    elif isinstance(adapter, VS.CollideAdapter):
        if global_balls is not None:
            raise ValueError("a VecCollidePark holds its global flags itself (made with global_balls=True)")
        view = SP.CollideView(adapter.park, flags=adapter)
    else:
        raise ValueError(f"{type(adapter).__name__}: the partition layer runs over VecModesPark, VecOrientPark or "
                         "VecCollidePark")
    return SP.VecSpace(view, adapter.rows_of_world, server=True)


@dataclass
class SpaceBatch(VS.Batch):
    """VecServer's Batch with the bubbles: the bubble each BUBBLE route goes to, and the bubble update's lists per row
    (its kind, additions and removals as rows, the streams' fields)."""
    rbubble: torch.Tensor | None = None             # [W, J] the route's bubble
    ad: dict = field(default_factory=dict)          # kind [R], adds [R, N] rows, addn [R], dels [R, N], deln [R]
    ad_fields: dict | None = None
    narrow: torch.Tensor | None = None              # [R] the bubble whose narrowcast the row's pilot gets, or -1
    bubble_live: dict = field(default_factory=dict)  # w -> set of bubbles with interactives at the send
    every: dict = field(default_factory=dict)       # slot j -> [W, I] the bubbles with interactive
                                                    # balls when its EVERY route is distributed (-1 padded)
    ticked: torch.Tensor | None = None              # [W] the worlds this tick ran for (OnTick's mask)


class SpaceServer(VS.VecServer):
    """VecServer over the master's partition (the torch path, every world several bubbles). `space`: the
    exact_ballpark.vec_space.VecSpace over the adapter's park (by default `space_for(adapter)`), loaded here."""

    def __init__(self, adapter, pilots=None, space: SP.VecSpace | None = None):
        super().__init__(adapter, pilots, kernel="torch")
        self.space = space if space is not None else space_for(adapter)
        self.space.load()
        A, sp = self.a, self.space
        self.slot_of_row = torch.full((A.R,), -1, dtype=I64, device=self.dev)
        sel = sp.rows >= 0
        self.slot_of_row[sp.rows[sel]] = torch.nonzero(sel)[:, 1]
        self._ad_time = torch.full((self.W,), -1, dtype=I64, device=self.dev)
        self._ad_keep: dict | None = None
        self.cleanup_stops = 0                              # the follower clean-up's Stops pushed (a count, for tests)
        # global balls where the park holds the flag the layer reads (a VecCollidePark made with them)
        self._globals = self.space.view.global_flags() is not None
        # OnTick's clock per world (Be::Time mTime, 100 ns units) and mFirstTime; the last call's counts
        self.time = torch.zeros(self.W, dtype=I64, device=self.dev)
        self.first_time = torch.ones(self.W, dtype=BOOL, device=self.dev)
        self.tick_counts: dict = {}
        self._gstream: dict = {}                            # w -> (the last queued AddBallsToPark's bytes, its serial)
        self._gserial = 0

    def _stream_fields(self, copy: bool = False) -> dict:
        """The adapter's stream fields, with each row's global flag where the park holds one; `copy`:
        every tensor a copy (on the CPU, .cpu() returns the park's own tensor, which the queued actions then change in
        place: a snapshot taken before them must not follow them)."""
        fl = self.a.stream_fields()
        gl = self.space.view.global_flags()
        if gl is not None:
            fl["global"] = gl.detach().cpu()
        if copy:
            fl = {k: v.clone() if torch.is_tensor(v) else v for k, v in fl.items()}
        return fl

    # ---- helpers ----

    def _slots_of(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.where(rows >= 0, self.slot_of_row[rows.clamp(min=0)], -1)

    def _tape_one(self, name: str, slot: torch.Tensor, n=None, f=None) -> None:
        """One layer operation for the worlds whose slot >= 0."""
        W, d = self.W, self.dev
        code = torch.where(slot >= 0, SP.OP[name], 0).to(I64).reshape(W, 1)
        nn = torch.zeros(W, 1, dtype=I64, device=d) if n is None else n.to(I64).reshape(W, 1)
        ff = torch.zeros(W, 1, 4, dtype=F64, device=d)
        if f is not None:
            ff[:, 0, :f.shape[1]] = f
        self.space.tape(code, slot.reshape(W, 1), nn, ff)

    def bubbles_of(self, rows: torch.Tensor) -> torch.Tensor:
        """The layer's bubble (newBubbleId) of each row [R]."""
        sp = self.space
        out = torch.full((self.a.R,), -1, dtype=I64, device=self.dev)
        sel = sp.rows >= 0
        out[sp.rows[sel]] = sp.st["bubble"][sel]
        return out[rows]

    def _ia(self) -> tuple:
        """The interactive pairs as rows: (bubble [W, I], row [W, I], valid [W, I])."""
        sp = self.space
        n = sp.st["ian"]
        k = torch.arange(sp.I, device=self.dev)
        valid = k[None, :] < n[:, None]
        rows = torch.gather(sp.rows, 1, sp.st["ias"].clamp(min=0, max=sp.N - 1))
        return sp.st["iab"], torch.where(valid, rows, -1), valid

    def in_ia_of(self, bubble_w: torch.Tensor) -> torch.Tensor:
        """[R]: whether each row is in the interactive set of its world's bubble bubble_w[W] (-1: none)."""
        b, rows, valid = self._ia()
        hit = valid & (b == bubble_w[:, None]) & (bubble_w[:, None] != -1)
        out = torch.zeros(self.a.R + 1, dtype=BOOL, device=self.dev)
        idx = torch.where(hit, rows, self.a.R)
        out.index_fill_(0, idx.flatten(), True)
        return out[:self.a.R]

    def ia_bubble_of_rows(self) -> torch.Tensor:
        """[R]: the bubble whose interactive set holds each row (-1: none; a row in two is flagged by the layer)."""
        b, rows, valid = self._ia()
        out = torch.full((self.a.R + 1,), -1, dtype=I64, device=self.dev)
        idx = torch.where(valid, rows, self.a.R)
        out.scatter_(0, idx.flatten(), torch.where(valid, b, -1).flatten())
        return out[:self.a.R]

    # ---- Actions ----

    def push(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
        """VecServer.push; CloakBall captures its ball's bubble from the layer (its old bubble while the two differ,
        _actions.py:628-638)."""
        k = VS.CODE[kind] if isinstance(kind, str) else int(kind)
        if k in VS.GLOBAL_KINDS:
            return self._push_global(k, mask, ball)
        if k == VS.CLOAK:
            row = self.a.lookup(ball.to(I64).to(self.dev))
            sp = self.space
            nb = torch.full((self.a.R,), -1, dtype=I64, device=self.dev)
            ob = nb.clone()
            sel = sp.rows >= 0
            nb[sp.rows[sel]] = sp.st["bubble"][sel]
            ob[sp.rows[sel]] = sp.st["old_bubble"][sel]
            safe = row.clamp(min=0)
            loc = torch.where(nb[safe] != ob[safe], ob[safe], nb[safe])
            return self._push_with_loc(k, mask, ball, target, f, n, loc)
        return super().push(kind, mask, ball, target, f, n, loc)

    def _cloak_loc(self, row: torch.Tensor, pushed: torch.Tensor) -> torch.Tensor:
        """CloakBall's bubble as the queue holds it (push above; the kernel's rule): the ball's old
        bubble while it differs from its bubble, else its bubble; the pushed argument for a ball not in the park (not
        queued). For the pending queries' test on the kernel path (VecServer._resolve)."""
        sp = self.space
        nb = torch.full((self.a.R,), -1, dtype=I64, device=self.dev)
        ob = nb.clone()
        sel = sp.rows >= 0
        nb[sp.rows[sel]] = sp.st["bubble"][sel]
        ob[sp.rows[sel]] = sp.st["old_bubble"][sel]
        safe = row.clamp(min=0)
        return torch.where(row >= 0, torch.where(nb[safe] != ob[safe], ob[safe], nb[safe]), pushed)

    def _push_global(self, k: int, mask: torch.Tensor, ball: torch.Tensor):
        """make_ball_global (GLOBAL_ON) and make_ball_local (GLOBAL_OFF) for every world in mask [W] (_actions.py:
        328-362): SetBallGlobal at the call, then the notice queued: AddBallsToPark, or BallNotGlobal
        with the ball's bubble at the call (`loc`; not queued for a ball not in the park). make_ball_global
        of a ball not in the park queues AddBallsToPark with its empty stream (WriteBallsToStream skips a ball it does
        not find: the header and the park's time alone), routed as any (_ticker.py:173-176 reads no ball)."""
        if not self._globals:
            raise ValueError("global balls need SpaceServer over a VecCollidePark made with global_balls=True, or a "
                             "view made with global_balls (space_for)")
        A, d = self.a, self.dev
        mask, ball = mask.to(BOOL).to(d), ball.to(I64).to(d)
        row = A.lookup(ball)
        on = mask & (row >= 0)
        rows = torch.zeros(A.R + 1, dtype=BOOL, device=d)
        rows.index_fill_(0, torch.where(on, row, A.R), True)
        self.space.view.set_global(rows[:A.R], k == VS.GLOBAL_ON)              # SetBallGlobal, at once (the view)
        loc = torch.zeros(self.W, dtype=I64, device=d)
        if k == VS.GLOBAL_OFF:
            loc = torch.where(on, self.bubbles_of(row.clamp(min=0)), 0)        # ball.newBubbleId at the call
        if k == VS.GLOBAL_OFF:
            return self._push_with_loc(k, on, ball, None, None, None, loc)
        # AddBallsToPark: WriteBallsToStream at the call (its fields and time then, kept with the call); TIME-710's
        # duplicate test compares the queued tuple, the stream's bytes among them, which can differ for one ball
        # within one queue (a world's first OnTick evolves without a pre-tick): `n` is the serial of the stream's
        # bytes, the last queued AddBallsToPark's where they are equal, so the test is exact
        fields, clk = self._stream_fields(copy=True), A.clock().clone()
        serial = torch.zeros(self.W, dtype=I64)
        streams = {}
        for w, (r, o, c) in enumerate(zip(row.tolist(), mask.tolist(), clk.tolist())):
            if not o:
                continue
            s = VS.ball_stream(fields, [r] if r >= 0 else [], c)  # (a ball not in the park, none)
            prev = self._gstream.get(w)
            if prev is not None and prev[0] == s:
                serial[w] = prev[1]
            else:
                self._gserial += 1
                serial[w] = self._gserial
            streams[w] = (s, int(serial[w]))
        slot = self._push_with_loc(k, mask, ball, None, None, serial.to(d), loc)
        ok = slot.ok.tolist()
        for w, v in streams.items():
            if ok[w]:
                self._gstream[w] = v
        slot.fields, slot.fields_clock = fields, clk
        return slot

    def _push_with_loc(self, k, mask, ball, target, f, n, loc):
        """VecServer.push's queueing with a given local argument (CloakBall's bubble)."""
        W, d = self.W, self.dev
        z = lambda dt: torch.zeros(W, dtype=dt, device=d)                    # noqa: E731
        mask = mask.to(BOOL).to(d)
        ball = ball.to(I64).to(d)
        target = z(I64) if target is None else target.to(I64).to(d)
        ff = torch.zeros(W, 4, dtype=F64, device=d)
        if f is not None:
            f = f.to(F64).to(d)
            ff[:, :f.shape[1]] = f.reshape(W, -1)
        n = z(I64) if n is None else n.to(I64).to(d)
        row = self.a.lookup(ball)
        trow = z(I64) - 1
        mask = mask & ((row >= 0) | (k == VS.GLOBAL_ON))     # (AddBallsToPark's empty stream queued)
        last = self.last
        dup = ((last["kind"] == k) & (last["ball"] == ball) & (last["target"] == target)
               & (last["f"] == ff).all(dim=1) & (last["n"] == n) & (last["loc"] == loc))
        ok = mask & ~dup
        for key, val in (("kind", torch.full((W,), k, dtype=I64, device=d)), ("ball", ball), ("target", target),
                         ("n", n), ("loc", loc)):
            last[key] = torch.where(ok, val, last[key])
        last["f"] = torch.where(ok[:, None], ff, last["f"])
        slot = VS.Slot(k, ok, ball, row, target, trow, ff, n, loc)
        self.slots.append(slot)
        return slot

    # ---- the tick ----

    def pre_tick(self, snapshot: bool | None = None, mask: torch.Tensor | None = None) -> None:
        """DoPreTick's Ticker.tick with the partition: the clean-up's Stops, the stamp, the actions in queue order
        (with the layer's re-filings and sets), the bubble update, the history's routes (CloakBall applied there).
        `mask` [W] (OnTick's): the worlds that tick; the others keep their queue, stamp, last action
        and lists, and route nothing."""
        A, sp, W, d = self.a, self.space, self.W, self.dev
        on = torch.ones(W, dtype=BOOL, device=d) if mask is None else mask.to(BOOL).to(d).reshape(W)
        # (1) the follower clean-up: a Stop for each remote follower, queued behind the commands (_actions.py:677)
        out, _ = sp.query("REMOTE_FOLLOWERS")
        count = torch.where(on, out[:, 0], 0)
        top = int(count.max()) if W else 0
        self.cleanup_stops += int(count.sum())
        ids_w = A.ids_of_world
        for k in range(top):
            has = count > k
            slot = torch.where(has, out[:, 1 + k], 0).clamp(min=0)
            ball = torch.gather(ids_w, 1, slot[:, None])[:, 0]
            self.push("Stop", has, torch.where(has, ball, 0))
        clock = A.clock().clone()
        stamp = self.stamp_for_system
        self.stamp_for_system = torch.where(on, clock + 1, stamp)
        slots, self.slots = self.slots, []
        if mask is None:
            self._reset_last()
        else:                                               # a world without a tick keeps its queue
            kept = [_with_ok(s, s.ok & ~on) for s in slots]
            slots = [_with_ok(s, s.ok & on) for s in slots]
            self.slots = [s for s in kept if bool(s.ok.any())]
            last = self.last
            self._reset_last()
            self.last = {k: torch.where(on.reshape(W, *([1] * (v.dim() - 1))), self.last[k], v)
                         for k, v in last.items()}
        kinds = {s.kind for s in slots if bool(s.ok.any())} if mask is not None else {s.kind for s in slots}
        if snapshot is None:
            snapshot = True
        # (2) every action but CloakBall, in queue order, the layer after each
        for s in slots:
            if s.kind == VS.CLOAK or s.kind in VS.GLOBAL_KINDS:    # (the global notices call no park method)
                continue
            m, trow, f, n = self._rows(s)
            if s.kind == VS.INTERACTIVE:
                slot = torch.where(s.ok & (s.row >= 0), self._slots_of(s.row), -1)
                self._tape_one("SET_INTERACTIVE", slot, n=s.n)
                continue
            if s.kind == VS.UNCLOAK:
                slot = torch.where(s.ok & (s.row >= 0), self._slots_of(s.row), -1)
                self._tape_one("UNCLOAK", slot)
                A.apply(s.kind, m, trow, f, n)
                continue
            accepted = None
            if s.kind in REFILERS[1:]:                      # SetBallRadius, SetMaxSpeed: a negative float is refused
                accepted = ~(VS.L.f32(s.f[:, 0]) < 0.0)
            if s.kind == VS.FREE:                           # SetBallFree(false) of a free ball re-files it (
                accepted = (s.n == 0) & A.park.free[s.row.clamp(min=0)]     # full.py's set_free)
            A.apply(s.kind, m, trow, f, n)
            if s.kind in REFILERS or s.kind == VS.FREE:
                ok = s.ok & (s.row >= 0)
                if accepted is not None:
                    ok = ok & accepted
                self._tape_one("REFILE", torch.where(ok, self._slots_of(s.row), -1))
        # (3) BubbleUpdater: AdditionsAndDeletions and CopyBubbles once per park time
        fresh = (clock != self._ad_time) & on
        if bool(fresh.any()):
            code = torch.where(fresh, SP.OP["ADDITIONS_AND_DELETIONS"], 0).to(I64).reshape(W, 1)
            sp.tape(code)
            keep = {k: sp.st[k].clone() for k in ("ad_kind", "ad_add", "ad_addn", "ad_del", "ad_deln")}
            if self._ad_keep is not None:
                for k in keep:
                    keep[k] = torch.where(fresh.reshape(W, *([1] * (keep[k].dim() - 1))), keep[k], self._ad_keep[k])
            self._ad_keep = keep
        self._ad_time = torch.where(on, clock, self._ad_time)
        ad = self._ad_rows(self._ad_keep)
        ad_fields = self._stream_fields() if snapshot else None
        self._seen_first = True
        # (4) the history in queue order
        route = torch.zeros(W, len(slots), dtype=torch.int8, device=d)
        rbubble = torch.full((W, len(slots)), -1, dtype=I64, device=d)
        notices = []
        every = {}
        dicts_ok = self._dict_exists
        for j, s in enumerate(slots):
            valid = s.ok & (s.row >= 0)
            safe = s.row.clamp(min=0)
            if s.kind in VS.GLOBAL_KINDS:                   # to every bubble with an interactive ball
                b_ia, _, v_ia = self._ia()                  # (BallNotGlobal's RemoveBall: but the one it names)
                every[j] = torch.where(v_ia, b_ia, -1)
                r = torch.where(s.ok, VS.EVERY, VS.NONE)
                rbubble[:, j] = torch.where(s.ok & (s.kind == VS.GLOBAL_OFF), s.loc, -1)
            elif s.kind == VS.CLOAK:                        # to the bubble queued in; then applied
                r = torch.where(s.ok, VS.BUBBLE, VS.NONE)
                rbubble[:, j] = s.loc
                m, _, f, n = self._rows(s)
                slot = torch.where(valid, self._slots_of(s.row), -1)
                self._tape_one("CLOAK_SENSOR", slot, n=s.n, f=s.f[:, :1])
                A.cloak(m, n, f[:, 0])
            elif s.kind == VS.UNCLOAK:                      # its own character; AddBallsToPark to its bubble's others
                bub = self.bubbles_of(safe)
                ia = self.in_ia_of(torch.where(valid, bub, -1))
                own = valid & ia[safe] & (self.pilot[safe] >= 0)
                r = torch.where(own, VS.CHARACTER, VS.NONE)
                rec = ia & A.exists & (A.arange != s.row[A.world]) & (self.pilot >= 0)
                notices.append((j, rec, self._stream_fields() if snapshot else None))
            else:
                bub = self.bubbles_of(safe)
                live_b = valid & (s.ball > 0) & dicts_ok(torch.where(valid, bub, -1)) & \
                    self._has_ia(torch.where(valid, bub, -1))
                if s.kind == VS.SETPOS:
                    live_b = live_b & (s.loc == 0)
                cl = A.cloaked[safe] != 0
                r = torch.where(live_b, torch.where(~cl, VS.BUBBLE,
                                                    torch.where(self.pilot[safe] >= 0, VS.CHARACTER, VS.NONE)),
                                VS.NONE)
                rbubble[:, j] = torch.where(live_b & ~cl, bub, -1)
            route[:, j] = r.to(torch.int8)
        self.batch = SpaceBatch(stamp=stamp, clock=clock, slots=slots, route=route, notices=notices, additions=None,
                                rbubble=rbubble, ad=ad, ad_fields=ad_fields, every=every, ticked=on)

    def _dict_exists(self, bubble_w: torch.Tensor) -> torch.Tensor:
        """[W]: whether each world's bubble bubble_w[W] has a dictionary (park.bubbles)."""
        sp = self.space
        k = torch.arange(sp.D, device=self.dev)
        valid = k[None, :] < sp.st["dn"][:, None]
        return ((sp.st["dbub"] == bubble_w[:, None]) & valid).any(dim=1) & (bubble_w != -1)

    def _has_ia(self, bubble_w: torch.Tensor) -> torch.Tensor:
        """[W]: whether each world's bubble has an interactive ball (park.bubbleInteractives)."""
        b, rows, valid = self._ia()
        return ((b == bubble_w[:, None]) & valid).any(dim=1) & (bubble_w != -1)

    def _ad_rows(self, keep: dict | None) -> dict:
        """The bubble update's lists per row (slots made rows)."""
        sp, R, N = self.space, self.a.R, self.space.N
        kind = torch.zeros(R, dtype=I64, device=self.dev)
        adds = torch.full((R, N), -1, dtype=I64, device=self.dev)
        dels = torch.full((R, N), -1, dtype=I64, device=self.dev)
        addn = torch.zeros(R, dtype=I64, device=self.dev)
        deln = torch.zeros(R, dtype=I64, device=self.dev)
        if keep is None:
            return {"kind": kind, "adds": adds, "addn": addn, "dels": dels, "deln": deln}
        sel = sp.rows >= 0
        rows = sp.rows[sel]
        kind[rows] = keep["ad_kind"][sel]
        addn[rows] = keep["ad_addn"][sel]
        deln[rows] = keep["ad_deln"][sel]
        world_rows = sp.rows                                 # slot -> row, per world
        for name, src in (("adds", "ad_add"), ("dels", "ad_del")):
            lst = keep[src]                                  # [W, N, N] slots
            as_rows = torch.gather(world_rows[:, None, :].expand(-1, N, -1), 2, lst.clamp(min=0))
            out = adds if name == "adds" else dels
            out[rows] = as_rows[sel]
        return {"kind": kind, "adds": adds, "addn": addn, "dels": dels, "deln": deln}

    def evolve(self, mask: torch.Tensor | None = None) -> None:
        """The park's tick with the dead bubbles' balls left out (TIME-714), then the commits' re-filings and
        HandleProximities (exact_ballpark.vec_space). `mask` [W] (OnTick's): those worlds alone, over
        VecCollidePark (its `active`); over VecModesPark and VecOrientPark too (their `evolve(active)` on
        the worlds' rows, vec_space's `row_mask`; VecModesPark's clock kept by its masked tick, the others' here)."""
        A = self.a
        if mask is not None:
            m = mask.to(BOOL).to(self.dev).reshape(self.W)
            self.space.tick(A.park, worlds=m)
            if isinstance(A, (VS.OrientAdapter, VS.CollideAdapter)):
                A.time.add_(m.to(I64))
            return
        self.space.tick(A.park)
        if isinstance(A, (VS.OrientAdapter, VS.CollideAdapter)):
            A.time.add_(1)

    def post_tick(self, mask: torch.Tensor | None = None) -> None:
        """DoPostTick's send_batch: per row (its pilot), the bubble whose narrowcast it gets (its ship in that bubble's
        interactive set, the bubble with history) and whether it has a character update. `mask` [W] (OnTick's): the worlds that ticked; the others send nothing."""
        A, b = self.a, self.batch
        W, R = self.W, A.R
        ia_b = self.ia_bubble_of_rows()
        # bubbles with history: the routes' bubbles
        hist = torch.zeros(R, dtype=BOOL, device=self.dev)
        for j in range(b.route.shape[1]):
            rb = b.rbubble[:, j]
            on = (b.route[:, j] == VS.BUBBLE) & (rb != -1)
            hit = on[A.world] & (ia_b == rb[A.world])
            hist = hist | hit
            if j in b.every:                                # EVERY, the bubbles at its distribution
                ev = (b.route[:, j] == VS.EVERY)[A.world]
                listed = ((b.every[j][A.world] == ia_b[:, None]) & (ia_b[:, None] != -1)).any(dim=1)
                hist = hist | (ev & listed & (ia_b != rb[A.world]))
        has = torch.zeros(R, dtype=BOOL, device=self.dev)
        for j, s in enumerate(b.slots):
            m = s.ok[A.world] & (s.row[A.world] == A.arange) & (b.route[:, j] == VS.CHARACTER)[A.world]
            has = has | m
        for j, rec, _ in b.notices:
            has = has | rec
        ad = b.ad
        same = ad["kind"] == SP.AD["SAME_BUBBLE"]
        lists = (ad["kind"] == SP.AD["SAME_OWN"]) | (ad["kind"] == SP.AD["CHANGED"])
        has = has | ((same | lists) & ((ad["addn"] > 0) | (ad["deln"] > 0)))
        pilot = self.pilot >= 0
        if mask is not None:
            pilot = pilot & mask.to(BOOL).to(self.dev).reshape(W)[A.world]
        b.has_character = has & pilot
        b.gets_bubble = hist & pilot
        b.narrow = torch.where(b.gets_bubble, ia_b, -1)

    # ---- OnTick (Ballpark.cpp:219-330, TIME-702 to TIME-705) ----

    def on_tick(self, sim_time) -> list:
        """Ballpark::OnTick for every world at once, each at its own simulation time (sim_time [W], 100 ns units;
        exact_ballpark.full.FullBallpark.on_tick): less than a whole interval since the world's time: nothing (the time
        kept); on the master a backlog of more than five intervals cut to one; a world's first call evolves once
        without the tick events; every other world runs its whole intervals (a pre-tick, the park's tick and a
        post-tick each, the worlds with an interval left ticking together under a mask), the remainder carried in its
        time. Returns the batches of the ticks run, in order (a world's sends are in those whose `ticked` holds it);
        `tick_counts` holds the call's counts. Host synchronisations: the interval counts' maximum and the masks'.
        Over every adapter (SpaceModesAdapter and SpaceOrientAdapter too; the kernel path,
        vec_server_space_cuda.CudaSpaceServer, refuses it)."""
        W, d = self.W, self.dev
        sim = torch.as_tensor(sim_time, dtype=I64).to(d).reshape(W)
        interval = int(round(self.a.park.dt * 1000.0)) * 10000          # mTickInterval x 10,000 (dt = ms x 0.001)
        last = sim - self.time
        go = ~(last < interval)                                         # too early: nothing (Ballpark.cpp:235)
        n = torch.where(go, torch.div(last, interval, rounding_mode="floor"), 0)
        resync = go & (n > MAX_CATCH_UP_TICKS)                          # the master's resynchronisation (246-257)
        last = torch.where(resync, interval, last)
        n = torch.where(resync, 1, n)
        first = go & self.first_time                                    # the first call: Evolve once, no events
        rest = go & ~first
        self.tick_counts = {"worlds too early": int((~go).sum()), "first calls": int(first.sum()),
                            "resynchronisations": int(resync.sum()),
                            "intervals run": int(torch.where(rest, n, 0).sum()),
                            "remainders carried": int((rest & (last - n * interval > 0)).sum())}
        if bool(first.any()):
            self.evolve(mask=first)
        self.first_time = self.first_time & ~first
        batches = []
        top = int(torch.where(rest, n, 0).max()) if W else 0
        for k in range(top):
            m = rest & (n > k)
            self.pre_tick(mask=m)
            self.evolve(mask=m)
            self.post_tick(mask=m)
            batches.append(self.batch)
        self.time = torch.where(go, sim - torch.where(rest, last - n * interval, 0), self.time)
        return batches

    def check(self) -> None:
        """Raise on a world outside the layer's domain (exact_ballpark.vec_space's flags)."""
        self.space.check()


# ======================================================================================== what each client holds

class SpaceClientView(VS.ClientView):
    """the ClientView (the observed fields `down` ticks back) with which balls each pilot's client park holds,
    maintained from the update batches as the client applies them (exact_ballpark.client's Ticker): the uncloaks'
    AddBallsToPark (in front), the bubble update's RemoveBalls and AddBallsToPark, then the bubble's events, of which
    a CloakBall removes its ball from every client but its own (_ticker.py:213-216; even a cloak the server refused,
    see the note's findings); the first additions are what a SetState would hold. A client whose ship is in no
    interactive set at a send hears nothing more of its bubble, so its park runs on alone and leaves the server's
    state (Destiny sends it no correction): from then on it is not `synced`, and the view holds for synced clients
    only. A client also keeps a ball the server cloaked without telling it (the finding 1: a non-interactive ball's
    CloakBall goes to bubble -1), whose later commands it never hears: that ball is held but not `fresh` until an
    AddBallsToPark of it arrives (an uncloak's notice, the bubble's additions). `view()` returns (time, fields,
    present [R, R]: ball row x pilot ship row, synced [R] by pilot ship row, fresh [R, R]), the holdings `down` ticks
    back like the fields; a field is the client's where present, fresh and synced.

    Global balls (net/client _ticker.py:344-374, net/server _ticker.py:173-196): a client applies its
    narrowcast's events in queue order: a CloakBall removes its ball (as above), a global AddBallsToPark (its stream
    as written at the call) adds its ball or reads it again (a ball not in the park: an empty stream, nothing added),
    and BallNotGlobal's RemoveBall removes its ball, the client's own ship included (a ship that changed bubble after
    its make_ball_local hears it). `set_state()` gives each pilot's park its SetState (WriteFullStateToStream with
    its ship as the source, Thunkers.cpp:2145-2200: its bubble's balls and its world's global balls, no cloaked ball
    but its own), where the clients get it (exact_ballpark.client.Session.start: after the first tick); over a park
    holding a global ball, a batch before it is refused (the first additions leave global balls out). A client holds
    another bubble's global ball whose changes it does not hear, so `fresh` follows what each client applies: a held
    ball stops being fresh when the server changes it by an action whose event the client does not apply (any applied
    action but the cloaks', the uncloaks' and SetBallInteractive, which change no field the view keeps), when the
    client applies its character events and its narrowcast's events of it in one tick (the singlecast comes first, so
    their order is not the server's), when its bubble is dead in the tick (the server leaves its balls out of Evolve,
    TIME-714; the client's park does not), and while it is cloaked on the server (as before); a global AddBallsToPark
    makes it fresh again when its stream is the server's state (written at the tick's park time, no action of the
    tick applied to its ball before it in the queue, no character event of its ball to this client in the tick:
    those come first and the stream overwrites them)."""

    NEUTRAL = frozenset((VS.CLOAK, VS.UNCLOAK, VS.INTERACTIVE)) | VS.GLOBAL_KINDS   # change no field
                                                                                   # the view keeps

    def __init__(self, server: SpaceServer, down: int, fields: tuple = ("pos", "vel", "mode")):
        super().__init__(server, down, fields)
        R, d = server.a.R, server.dev
        self.member = torch.zeros(R, R, dtype=BOOL, device=d)
        self.mring = torch.zeros(self.depth, R, R, dtype=BOOL, device=d)
        self.synced = torch.ones(R, dtype=BOOL, device=d)
        self.sring = torch.ones(self.depth, R, dtype=BOOL, device=d)
        self.fresh = torch.ones(R, R, dtype=BOOL, device=d)
        self.fring = torch.ones(self.depth, R, R, dtype=BOOL, device=d)
        self._applied = None
        self._state_set = False                             # set_state() called

    def _global_rows(self) -> torch.Tensor:
        """[R]: each row's global flag where the park holds one, else none global."""
        A = self.server.a
        gl = self.server.space.view.global_flags()
        if gl is None:
            return torch.zeros(A.R, dtype=BOOL, device=self.member.device)
        return gl.reshape(A.R).to(BOOL).to(self.member.device)

    def set_state(self, rows: torch.Tensor | None = None) -> None:
        """Each pilot's SetState now (the batcher's send_full_state_update, _parkupdatebatcher.py:
        195-217: WriteFullStateToStream with its ship as the source): its park holds exactly its ship's bubble's balls
        and its world's global balls, no cloaked ball but its own, each fresh. `rows` [R] bool: the pilots' ship rows
        (by default every pilot's)."""
        S, A = self.server, self.server.a
        d = self.member.device
        pilot = S.pilot >= 0
        sel = pilot if rows is None else (rows.to(BOOL).to(d).reshape(A.R) & pilot)
        bub = S.bubbles_of(A.arange)
        eye = torch.eye(A.R, dtype=BOOL, device=d)
        same = (A.world[:, None] == A.world[None, :]) & A.exists[:, None] & A.exists[None, :]
        near = ((bub[:, None] == bub[None, :]) & (bub[:, None] != -1)) | self._global_rows()[:, None]
        held = same & near & ((A.cloaked == 0)[:, None] | eye)
        self.member[:, sel] = held[:, sel]
        self.fresh[:, sel] = True
        self._state_set = True

    def apply_batch(self, b: SpaceBatch) -> None:
        S, A = self.server, self.server.a
        R, world = A.R, A.world
        pilot = S.pilot >= 0
        u = A.arange
        m, fr = self.member, self.fresh
        if not self._state_set and bool(self._global_rows().any()):
            raise ValueError("over a park holding global balls, call set_state() where the clients get their "
                             "SetState (the first additions leave global balls out)")
        for j, rec, _ in b.notices:                         # the uncloaked ball added (in front: first)
            s = b.slots[j]
            ball = s.row[world]
            ok = rec & s.ok[world] & (ball >= 0) & pilot
            m[ball[ok], u[ok]] = True
            fr[ball[ok], u[ok]] = True
        ad = b.ad
        sent = pilot & (ad["kind"] != SP.AD["NONE"])
        N = ad["dels"].shape[1]
        for k in range(N):                                  # RemoveBalls, then AddBallsToPark
            ok = sent & (ad["deln"] > k)
            m[ad["dels"][ok, k], u[ok]] = False
        for k in range(N):
            ok = sent & (ad["addn"] > k)
            m[ad["adds"][ok, k], u[ok]] = True
            fr[ad["adds"][ok, k], u[ok]] = True
        # which of the tick's applied actions each client applies (its narrowcast's, its character's)
        gets, narrow, has = b.gets_bubble, b.narrow, b.has_character
        heard = {}
        by_char = torch.zeros(R, R, dtype=BOOL, device=m.device)       # [ball, client]: a character event of it
        by_narrow = torch.zeros(R, R, dtype=BOOL, device=m.device)     # [ball, client]: a narrowcast event of it
        for j, s in enumerate(b.slots):
            if s.kind in self.NEUTRAL:
                continue
            ball = s.row[world]
            valid = s.ok[world] & (ball >= 0)
            route = b.route[:, j][world]
            nar = valid & gets & (route == VS.BUBBLE) & (b.rbubble[:, j][world] == narrow)
            chr_ = valid & has & (route == VS.CHARACTER) & (ball == u)
            by_narrow[ball[nar], u[nar]] = True
            by_char[ball[chr_], u[chr_]] = True
            heard[j] = (ball, valid, nar | chr_)
        touched = torch.zeros(R + 1, dtype=BOOL, device=m.device)     # [ball]: an action of the tick applied so far
        for j, s in enumerate(b.slots):                     # the narrowcast in queue order (with the actions' effects)
            ball = s.row[world]
            if s.kind == VS.CLOAK:                          # removes its ball from every client but its own
                ok = (gets & (b.route[:, j] == VS.BUBBLE)[world] & (b.rbubble[:, j][world] == narrow)
                      & s.ok[world] & (ball >= 0) & (ball != u))
                m[ball[ok], u[ok]] = False
            elif s.kind in VS.GLOBAL_KINDS:                 # to every bubble with an interactive ball
                if j not in b.every:                        # (BallNotGlobal's RemoveBall: but the one it names)
                    continue
                listed = ((b.every[j][world] == narrow[:, None]) & (narrow[:, None] != -1)).any(dim=1)
                got = (gets & (b.route[:, j] == VS.EVERY)[world] & listed & (narrow != b.rbubble[:, j][world])
                       & s.ok[world] & (ball >= 0))
                safe = ball.clamp(min=0)
                if s.kind == VS.GLOBAL_ON:                  # the ball read from its stream, as written at the call
                    cur = ((s.fields_clock.to(m.device)[world] == b.clock[world]) & ~touched[safe]
                           & ~by_char[safe, u])
                    m[ball[got], u[got]] = True
                    fr[ball[got], u[got]] = cur[got]
                else:                                       # RemoveBall
                    m[ball[got], u[got]] = False
            elif j in heard:                                # an applied action: stale where the client misses it
                bl, valid, hd = heard[j]
                miss = valid & ~hd
                fr[bl[miss], u[miss]] = False
                touched[torch.where(s.ok & (s.row >= 0), s.row, R)] = True
        fr &= ~(by_char & by_narrow)                        # both routes in one tick: out of order
        self.synced &= (S.ia_bubble_of_rows() >= 0) | ~pilot       # out of every interactive set: on its own
        cloaked = A.cloaked != 0                            # cloaked on the server, still held by another client
        stale = m & cloaked[:, None] & (A.arange[:, None] != A.arange[None, :])
        fr &= ~stale

    def record(self, time: int) -> None:
        b = self.server.batch
        if b is not None and b is not self._applied:
            self.apply_batch(b)
            self._applied = b
        dead = getattr(self.server.space, "last_dead", None)    # the tick's dead bubbles' balls, left
        if dead is not None:                                    # out of the server's Evolve, not the clients'
            self.fresh &= ~dead.to(BOOL).to(self.fresh.device).reshape(-1)[:, None]
        slot = self.k % self.depth
        self.mring[slot].copy_(self.member)
        self.sring[slot].copy_(self.synced)
        self.fring[slot].copy_(self.fresh)
        super().record(time)

    def view(self) -> tuple:
        if self.k <= self.down:
            return -1, {}, None, None, None
        t, out, _ = super().view()
        slot = (self.k - 1 - self.down) % self.depth
        return t, out, self.mring[slot], self.sring[slot], self.fring[slot]


# ================================================================================ decoding (the host, the tests)

def decode(server: SpaceServer, batch: SpaceBatch | None = None, client_base: int = 10000) -> list:
    """The batch as Destiny's network sends it, per world: {client id: [(kind, method, state, wait_for_bubble,
    count)]} in the order each client receives them (its singlecast, then its ship's bubble's narrowcast); a state is
    [(stamp, (name, args))] with PackagedAction payloads as exact_ballpark.server.Marshalled. Host only."""
    from .server import ClientUpdateCountThisTick, Marshalled
    A, b = server.a, batch or server.batch
    pilot = server.pilot.tolist()
    ids = A.ids.tolist()
    rows_of = [[r for r in rw if r >= 0] for rw in A.rows_of_world.tolist()]
    route = b.route.cpu().tolist()
    rbub = b.rbubble.cpu().tolist()
    stamp, clock = b.stamp.tolist(), b.clock.tolist()
    oks = [s.ok.tolist() for s in b.slots]
    has, gets, narrow = b.has_character.tolist(), b.gets_bubble.tolist(), b.narrow.tolist()
    ad = {k: v.cpu().tolist() for k, v in b.ad.items()}
    ticked = b.ticked.tolist() if getattr(b, "ticked", None) is not None else [True] * A.W
    out = []
    for w in range(A.W):
        if not ticked[w]:                                   # no tick for this world, no send
            out.append({})
            continue
        bubbles, chars = {}, {}
        for j, s in enumerate(b.slots):
            if not oks[j][w]:
                continue
            if route[w][j] == VS.EVERY:                     # the global balls' notices
                if s.kind == VS.GLOBAL_ON and int(s.row[w]) < 0:  # a ball not in the park, its
                    ev = ("AddBallsToPark", (VS.ball_stream(s.fields, [], int(s.fields_clock[w])),))   # empty stream
                elif s.kind == VS.GLOBAL_ON:                # the stream as written at the call
                    fl = dict(s.fields)
                    fl["global"] = fl["global"].clone()
                    fl["global"][int(s.row[w])] = True
                    ev = ("AddBallsToPark", (VS.ball_stream(fl, [int(s.row[w])], int(s.fields_clock[w])),))
                else:
                    ev = ("RemoveBall", (int(s.ball[w]),))
                for bub in sorted({x for x in b.every[j][w].tolist() if x != -1} - {rbub[w][j]}):
                    bubbles.setdefault(bub, []).append((stamp[w], ev))
                continue
            ev = VS.event_of(s, w, ids)
            if route[w][j] == VS.BUBBLE:
                bubbles.setdefault(rbub[w][j], []).append((stamp[w], ev))
            elif route[w][j] == VS.CHARACTER:
                r = int(s.row[w])
                chars.setdefault(pilot[r], []).append((stamp[w], ev))
        front = {}
        for j, rec, fl in b.notices:
            s = b.slots[j]
            if not oks[j][w] or int(s.row[w]) < 0 or not any(bool(rec[r]) for r in rows_of[w]):
                continue
            if fl is None:
                raise ValueError("this tick ran without snapshots")
            payload = ("AddBallsToPark", (VS.ball_stream(fl, [int(s.row[w])], clock[w]),))
            for r in rows_of[w]:
                if rec[r]:
                    front.setdefault(pilot[r], []).insert(0, (stamp[w], payload))
        upd = {}
        fl = b.ad_fields
        for r in rows_of[w]:
            p = pilot[r]
            k = ad["kind"][r]
            if p < 0 or k == SP.AD["NONE"]:
                continue
            adds = [x for x in ad["adds"][r][:ad["addn"][r]]]
            dels = [ids[x] for x in ad["dels"][r][:ad["deln"][r]]]
            entries = []
            if k == SP.AD["SAME_BUBBLE"]:
                if dels:
                    entries.append((clock[w], ("PackagedAction", Marshalled(((clock[w], ("RemoveBalls", (dels,))),)))))
                if adds:
                    stream = VS.ball_stream(fl, adds, clock[w])
                    entries.append((clock[w], ("PackagedAction",
                                               Marshalled([(clock[w], ("AddBallsToPark", (stream,)))]))))
            else:
                if dels:
                    entries.append((clock[w], ("RemoveBalls", (dels,))))
                if adds:
                    stream = VS.ball_stream(fl, adds, clock[w])
                    entries.append((clock[w], ("PackagedAction",
                                               Marshalled([(clock[w], ("AddBallsToPark", (stream,)))]))))
            upd[p] = entries
        per = {}
        for r in rows_of[w]:
            p = pilot[r]
            if p < 0:
                continue
            c = client_base + p
            state = front.get(p, []) + upd.get(p, []) + chars.get(p, [])
            got = []
            if has[r]:
                got.append(("single", "DoDestinyUpdate", state, bool(gets[r]),
                            ClientUpdateCountThisTick.TWO if gets[r] else ClientUpdateCountThisTick.ONE))
            if gets[r]:
                got.append(("narrow", "DoDestinyUpdate", bubbles.get(narrow[r], []), False,
                            ClientUpdateCountThisTick.TWO if has[r] else ClientUpdateCountThisTick.ONE))
            if got:
                per[c] = got
        out.append(per)
    return out
