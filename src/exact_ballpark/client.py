# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's client layer (destiny.net.client), transcribed: the client Ticker that merges the server's update batches into
a time-sorted history and applies them to the client's ballpark at their stamps, rewinding to a stored snapshot or
running ahead as needed; and a session harness (this port's, not Destiny's) that ticks one server park, the network and
one client park per pilot in lockstep, with whole-tick delays, so that what a pilot's client knows at each tick can be
read and tested.

A transcription of Carbon's `destiny` Python package (github.com/carbonengine/destiny at 114e89fa, MIT, (c) CCP Games). Destiny's Python is downloaded material: it was read, never imported or run. Every
class and function cites its source by file and line under python/destiny/net/:
- client/_baseticker.py:12-140 (`BaseTicker`, `_expand_states`), client/_ticker.py:14-415 (`LOCAL_ACTIONS`, `Ticker`),
  client/_util.py:6-68 (`merge_state_into_history`), client/_interface.py:6-118 (`ClientTickerInterface`,
  `TickErrorHandlerInterface`);
- __init__.py:3-23, the package's own account of the timing: the server's Ticker applies the actions at each pre-tick,
  "the client side ticker can receive 0-2 updates for the same tick", one for the bubble and one for the user, "User
  events get sent out first", and "the client ticker should always receive the user updates before the bubble ones".

Kept exactly: the order of every loop and call, the history's structure ([state, wait_for_bubble] entries sorted by
stamp), the rewind and run-ahead rules, the snapshots' bookkeeping (and its slicing), the local actions, the CloakBall
removal, and the surprises below. The client's ballpark is `ClientPark`: exact_ballpark.server's `DestinyPark` (Destiny's
Python ballpark API over this port's FullBallpark, a client's view unless told otherwise) with the three members of
Ballpark_Blue.cpp the ticker reads that DestinyPark leaves out: `ego` (mEgo, 0 at construction, Ballpark.cpp:88,
READWRITE), `isRunning` (mHaveTicks, READ) and `Start` / `Pause` (Ballpark.cpp:3013-3031: register for, or drop,
the engine's ticks). Abstractions, each labelled where it is made: blue.MemStream and blue.marshal are
exact_ballpark.server's stand-ins; blue.statistics' zones are no-ops; logging is Python's.

Surprises in Destiny's code, kept as coded:
- An entry stamped one or two ticks ahead of the client's park waits; three or more ahead is applied at once, the park
  run forward to its stamp (_ticker.py:100-102, 265-305).
- An entry older than the park's time rewinds to the last snapshot at or before its stamp and evolves up to it; the park
  is left at the entry's stamp, so the client's clock goes back by the difference and is not advanced again here
  (_ticker.py:265-305). With no such snapshot the update is dropped as a recoverable desync (_ticker.py:195-203).
- After an applied entry, when more than one entry remains and the next is not waiting, the park evolves once more
  (_ticker.py:118-123): a backlog is worked off by extra ticks.
- A head entry waiting for its bubble part blocks every later entry; the flag clears only when an update with the same
  stamp is merged (_util.py:41-68). A user update whose bubble part comes with a different stamp waits for ever.
- Every update is checked to carry one stamp; more than one is a fatal desync, reported and then merged all the same
  (_baseticker.py:46-58, 107-117).
- store_state keeps the first snapshot and the last five once there are more than ten, `states[:1] + states[3:3] +
  states[-5:]` (the middle slice is empty, _ticker.py:260-261); do_post_tick stores one every eleventh tick when nothing
  was applied (`stamp > last_stamp + 10`, _ticker.py:224-236).
- A CloakBall event removes the cloaked ball from the client's park unless it is the ego (_ticker.py:213-216).
- A SetState clears the history entries older than its stamp, and later updates older than the last SetState are
  dropped (_ticker.py:57-82).
- Python 2 remnants: `except StandardError` (_baseticker.py:133; here Exception), `logger.warn` (here warning), and
  `logger.info("Removing ball %s")` without its argument (_ticker.py:366-368; kept: logging reports the formatting error
  at emission if INFO is enabled).

The session (`Session`) is the conformance kit's harness around Destiny's two layers, with its protocol stated in its docstring.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from . import server as srv
from .server import (BallInfoInterface, CharacterInterestsInterface, ClientInterestsInterface, DestinyPark,
                     MemStream, NetworkInterface, marshal_load)

logger = logging.getLogger(__name__)


# ====================================================================================== the client's ballpark

class ClientPark(DestinyPark):
    """destiny.Ballpark() on the client: DestinyPark (a client's view, isMaster false) with `ego`, `isRunning`, `Start`
    and `Pause` (Ballpark_Blue.cpp:15-21, 114-120, 556-567; Ballpark.cpp:88, 3013-3031)."""

    def __init__(self, lane_park=None, *, is_master: bool = False):
        super().__init__(lane_park, is_master=is_master)
        self.ego = 0                                        # mEgo(0), READWRITE
        self._have_ticks = False                            # mHaveTicks

    @property
    def isRunning(self):
        return self._have_ticks

    def Start(self):
        """Ballpark::Start: registered for the engine's ticks once."""
        if not self._have_ticks:
            self._have_ticks = True

    def Pause(self):
        """Ballpark::Pause."""
        if self._have_ticks:
            self._have_ticks = False

    _parent_Start = Start
    _parent_Pause = Pause


class _Statistics:
    """blue.statistics' zones (profiling): no-ops."""

    @staticmethod
    def EnterZone(name):
        return None

    @staticmethod
    def LeaveZone():
        return None


_statistics = _Statistics()


# ====================================================================================== _interface.py:6-118

class ClientTickerInterface:
    """_interface.py:6-95 (abstract in Destiny; `__metaclass__` is Python 2 syntax, inert in Python 3)."""

    def set_ballpark(self, ballpark):
        self._ballpark = ballpark

    def on_set_state(self):
        pass

    def on_ballpark_local_action(self, func_name, args):
        pass

    def should_log_actions(self):
        pass

    def get_ball_destruction_delay(self, ball, is_terminal=False):
        pass

    def get_ball_destruction_delays(self, ball_ids, is_release=False):
        pass

    def clean_up_after_ball_removal(self, ball_id, ball, is_terminal=False):
        pass

    def clean_up_after_multiple_ball_removal(self, ball_ids, is_release=False):
        pass


class TickErrorHandlerInterface:
    """_interface.py:98-118."""

    def set_ballpark(self, ballpark):
        self._ballpark = ballpark

    def on_fatal_desync(self):
        pass

    def on_recoverable_desync(self):
        pass


# ===================================================================================== _baseticker.py:12-140

class BaseTicker(object):
    """A base class for client-side PreTick handlers (_baseticker.py:12-18)."""

    def __init__(self, error_handler):                                                  # _baseticker.py:21-29
        self._ballpark = None
        self._history = []  # list of state updates
        self._latest_set_state_time = 0
        self._error_handler = error_handler

    def set_ballpark(self, ballpark):                                                    # _baseticker.py:31-37
        self._ballpark = ballpark
        self._error_handler.set_ballpark(ballpark)

    def do_pre_tick(self):                                                               # _baseticker.py:39-44
        raise NotImplementedError

    def update(self, state, wait_for_bubble):                                            # _baseticker.py:46-58
        """Call this when you get state from the server. wait_for_bubble means we just got state for this user only,
        but have an upcoming state for the bubble we are in."""
        if self._ballpark is None:
            return

        state = _expand_states(state)
        self._report_update_if_bad(state)
        self._flush_state(state, wait_for_bubble)

    def _flush_state(self, state, wait_for_bubble):                                      # _baseticker.py:60-65
        raise NotImplementedError

    @property
    def _current_time(self):                                                             # _baseticker.py:67-73
        return self._ballpark.currentTime

    def _dump_history(self):                                                             # _baseticker.py:75-105
        if logger.level > logging.INFO:
            return
        logger.info("------ %s History Dump %s -------", self.__class__.__name__, self._current_time)
        rev = self._history[:]
        rev.reverse()
        for state, wait_for_bubble in rev:
            logger.info('state waiting: %s', "yes" if wait_for_bubble else "no")
            last_state = None
            last_state_count = 0
            for entry in state:
                event_stamp, event = entry
                func_name, args = event
                next_state = ["[", event_stamp, "]", func_name]
                if next_state != last_state:
                    if last_state is not None:
                        if last_state_count != 1:
                            last_state.append("(x %d)" % last_state_count)
                        logger.info(" ".join([str(s) for s in last_state]))
                    last_state = next_state
                    last_state_count = 1
                else:
                    last_state_count += 1
            if last_state is not None:
                if last_state_count != 1:
                    last_state.append("(x %d)" % last_state_count)
                logger.info(" ".join([str(s) for s in last_state]))
            logger.info(" ")

    def _report_update_if_bad(self, state):                                              # _baseticker.py:107-117
        timestamps = {action[0] for action in state}
        if len(timestamps) > 1:
            logger.error("Found update batch with %s items and %s timestamps", len(state), len(timestamps))
            for action in state:
                logger.error("Action: %s", action)

            self._error_handler.on_fatal_desync()


def _expand_states(state):                                                               # _baseticker.py:120-140
    """Expand any packaged updates back into their full state (a PackagedAction is a pre-serialised stream of regular
    actions; assumed not to nest)."""
    expanded_states = []
    for action in state:
        if action[1][0] == 'PackagedAction':
            try:
                unpackaged_actions = marshal_load(action[1][1])
                expanded_states.extend(unpackaged_actions)
            except Exception:                               # (Python 2's StandardError)
                logger.exception("Exception whilst expanding a PackagedAction")
        else:
            expanded_states.append(action)
    state = expanded_states
    return state


# ============================================================================================= _util.py:6-68

def merge_state_into_history(state, history_list, wait_for_bubble):                    # _util.py:6-68
    """Merge the incoming state with the states we already have: history is a list sorted by stamp, the incoming
    states may not be; an incoming stamp equal to an entry's is merged into it and clears its wait_for_bubble; at most
    one new entry waits for its bubble."""
    entries_by_time = defaultdict(list)
    for entry in state:
        entries_by_time[entry[0]].append(entry)
    time_list = sorted(entries_by_time.keys())
    # Insert events happening earlier than (some of) our current history
    time_list_idx = 0
    for history_idx, history_time_entries in enumerate(history_list):
        history_time = history_time_entries[0][0][0]
        entry_time = time_list[time_list_idx]
        if history_time < entry_time:  # new action happens later
            wait_for_bubble = False  # Because if we need to wait for bubble, then surely we must wait for it earlier
            continue
        if history_time == entry_time:  # merge the two and don't wait for bubble
            history_time_entries[0].extend(entries_by_time[entry_time])
            history_time_entries[1] = False
        else:  # history_time > entry_time
            history_list.insert(history_idx, [entries_by_time[entry_time], wait_for_bubble])
            wait_for_bubble = False  # Max 1 wait for bubble
        time_list_idx += 1
        if time_list_idx >= len(time_list):
            break  # time_list (i.e. the new actions) has now been exhausted
    # Insert entries after current history (0 iterations when all of time_list got processed above)
    for tl_idx in range(time_list_idx, len(time_list)):
        history_list.append([entries_by_time[time_list[tl_idx]], wait_for_bubble])


# ============================================================================================ _ticker.py:14-415

LOCAL_ACTIONS = {                                                                       # _ticker.py:14-18
    'AddBallsToPark',
    'RemoveBall',
    'RemoveBalls',
}


class Ticker(BaseTicker):
    """The client's ticker (_ticker.py:21-415)."""

    def __init__(self, error_handler, client_ticker_interface):                         # _ticker.py:22-35
        super(Ticker, self).__init__(error_handler)
        self._should_rebase = False
        self._last_stamp = -1
        self._states = []
        self._state_is_valid = False
        self._client_ticker_interface = client_ticker_interface

    def set_ballpark(self, ballpark):                                                    # _ticker.py:37-45
        super(Ticker, self).set_ballpark(ballpark)
        self._client_ticker_interface.set_ballpark(ballpark)

    def _flush_state(self, state, wait_for_bubble):                                      # _ticker.py:47-86
        """Merge state into self._history."""
        logger.info("Server Update with %s event(s) added to history", len(state))

        if len(state) == 0:
            logger.warning('Empty state received from remote ballpark')
            return

        if state[0][1][0] == 'SetState':
            # A SetState is a reset: clear the history from stamps BEFORE its stamp (machonet does not guarantee the
            # order of delivery)
            self._latest_set_state_time = state[0][0]
            logger.info("Michelle received a SetState at time %s. Clearing out-of-date entries...",
                        self._latest_set_state_time)
            self._history = [
                history_entry
                for history_entry in self._history
                if history_entry[0][0][0] >= self._latest_set_state_time
            ]
            self._dump_history()
        else:
            # An update from before the previous SetState is useless (and may break things): discarded
            entry_time = state[0][0]
            if entry_time < self._latest_set_state_time:
                logger.warning(
                    "Michelle discarded a state that was too old %s < %s",
                    entry_time,
                    self._latest_set_state_time
                )
                self._dump_history()
                return

        merge_state_into_history(state, self._history, wait_for_bubble)

        self._dump_history()

    def do_pre_tick(self):                                                               # _ticker.py:88-123
        """If we are not waiting for bubble events, apply history to ballpark."""
        while len(self._history) > 0:
            state, wait_for_bubble = self._history[0]
            if wait_for_bubble:
                return

            # The stamp of the first event of the entry: every event of an entry has the same stamp
            event_stamp = state[0][0]
            if event_stamp > self._current_time and event_stamp - self._current_time < 3:
                break

            self._real_flush_state(state)
            del self._history[0]

            # A snapshot of the current tick, mid-tick, in case straggling events for this tick come with the next's
            if self._state_is_valid and self._should_rebase:
                self.store_state(mid_tick=True)

            # If there are more state updates in the queue and the next one is still waiting, then return
            if len(self._history) > 1:
                if self._history[0][1]:
                    return

                self._ballpark._parent_Evolve()

    def _set_state(self, state, ego_ball_id):                                            # _ticker.py:125-144
        """Read the entire state of the ballpark from "state" and set the ego ball to "ego_ball_id"."""
        self._client_ticker_interface.on_set_state()
        self._ballpark.ClearAll()

        ms = MemStream()
        ms.Write(state)
        # Make a full SetState
        self._ballpark.ReadFullStateFromStream(ms)
        self._ballpark.ego = int(ego_ball_id)  # This is a ballID int.
        self._ballpark.Start()
        self._state_is_valid = True
        self.flush_simulation_history()

    def _real_flush_state(self, state):                                                  # _ticker.py:146-222
        """Apply the history: a list of actions in chronological order."""
        logger.info("Handling Server Update with %s event(s)", len(state))

        if len(state) == 0:
            logger.warning('Empty state received from remote ballpark')
            return

        # First of all, check if the first event is a 'SetState' (they always come first)
        event_stamp, event = state[0]
        func_name, args = event

        if func_name == 'SetState':
            # Replace the existing state with the one received and forget everything else
            if self._client_ticker_interface.should_log_actions():
                logger.info("Action: %12.12s %s - current: %s %s", func_name, event_stamp, self._current_time, args)

            self._set_state(*args)  # Rebase is implicit in this

        if self._state_is_valid:
            # A 'normal' update: a series of actions relative to the same stamp, applied in order
            self._should_rebase = False
            synchronised = False

            for action in state:
                event_stamp, event = action
                func_name, args = event

                if func_name == 'SetState':
                    continue

                if self._client_ticker_interface.should_log_actions():
                    logger.info("Action: %12.12s %s - current: %s %s", func_name, event_stamp, self._current_time,
                                args)
                try:
                    # The simulation must be at the stamp of these updates; once it is, the group is applied
                    if not synchronised:
                        synchronised = self.synchronize_to_simulation_time(event_stamp)

                    if not synchronised:
                        # The last recourse: only the simulation can be reset, not the game state on top of it
                        logger.warning("Failed to synchronize to %s", event_stamp)
                        self._error_handler.on_recoverable_desync()
                        return

                    self._should_rebase = True
                    if func_name in LOCAL_ACTIONS:  # These are functions that are applied to ourself
                        getattr(self, func_name)(*args)
                    else:
                        # These are functions that are applied directly to the ballpark
                        getattr(self._ballpark, '_parent_' + func_name)(*args)
                        self._client_ticker_interface.on_ballpark_local_action(func_name, args)
                        if func_name == 'CloakBall':
                            event_ball_id = args[0]
                            if self._ballpark.ego and self._ballpark.ego != event_ball_id:
                                self.RemoveBall(event_ball_id)
                except Exception:
                    message = "{} failed.".format(func_name)
                    logger.error(message, exc_info=1)
                    continue
        else:
            logger.info("Events ignored")

    def do_post_tick(self, stamp):                                                       # _ticker.py:224-236
        """Call this in the ballparks PostTick method: the state history for rewinding."""
        if self._should_rebase:
            self.flush_simulation_history()
            self._should_rebase = False
        elif stamp > self._last_stamp + 10:
            self.store_state()

    def clear_states(self):                                                              # _ticker.py:238-242
        self._states = []

    def store_state(self, mid_tick=False):                                               # _ticker.py:244-263
        """Updates our history of full ballpark states."""
        if not self._ballpark.isRunning:
            return

        ms = MemStream()
        self._ballpark.WriteFullStateToStream(ms)

        self._states.append([ms, self._current_time, mid_tick])

        logger.info("store_state: %s mid_tick: %s", self._current_time, mid_tick)

        if len(self._states) > 10:
            self._states = self._states[:1] + self._states[3:3] + self._states[-5:]

        self._last_stamp = self._current_time

    def synchronize_to_simulation_time(self, stamp):                                     # _ticker.py:265-305
        """Rewind or run forward so that an update applies at its stamp; rewinding reads a stored snapshot, losing what
        was applied at later stamps."""
        logger.info("SynchroniseToSimulationTime looking for: %s - current: %s", stamp, self._current_time)

        if stamp < self._current_time:
            # We are ahead of this state: go back to the last state with a stamp at or before it
            last_stamp = 0
            last_state = None
            for item in self._states:
                if item[1] <= stamp:
                    last_stamp = item[1]
                    last_state = item[0]
                else:
                    continue

            # No such state: a SetState is needed to be in sync again
            if not last_state:
                logger.warning('SynchroniseToSimulationTime: Did not find any state')
                return 0

            self._ballpark._parent_ReadFullStateFromStream(last_state, 1)
        else:
            # We are behind this state
            last_stamp = self._current_time

        # 'last_stamp' is at or before 'stamp' and the park is at 'last_stamp': evolve to 'stamp'
        for i in range(stamp - last_stamp):
            self._ballpark._parent_Evolve()

        logger.info("SynchroniseToSimulationTime found: %s", self._current_time)
        return 1

    def flush_simulation_history(self, new_base_snapshot=True):                          # _ticker.py:307-342
        """Clear the simulation history and start a new one with the current state (keeping the previous tick's
        mid-state if it is the last stored), unless new_base_snapshot is False (the ballpark is being released)."""
        logger.info(
            "State history rebased at %s newBaseSnapshot %s",
            self._current_time,
            new_base_snapshot,
        )

        last_mid_state = None
        if new_base_snapshot and len(self._states):
            last_mid_state = self._states[-1]
            if not last_mid_state[2] or last_mid_state[1] != self._current_time-1:
                last_mid_state = None

        self.clear_states()

        if new_base_snapshot:
            if last_mid_state:
                self._states.append(last_mid_state)

            self.store_state()
            for state in self._states:
                logger.info("State entry %s", state)

    def AddBallsToPark(self, state):                                                    # _ticker.py:344-353
        """A 'partial' SetState: balls added to the park, nothing cleared."""
        ms = MemStream()
        ms.Write(state)
        self._ballpark.ReadFullStateFromStream(ms, 2)

    def RemoveBall(self, ball_id, terminal=False):                                       # _ticker.py:355-374
        """The RemoveBall action."""
        _statistics.EnterZone("RemoveBall")
        try:
            if terminal:
                logger.info("Removing ball %s (terminal)")          # (sic) no argument for %s
            else:
                logger.info("Removing ball %s")                     # (sic)
            ball = self._ballpark.balls.get(ball_id, None)
            delay = self._client_ticker_interface.get_ball_destruction_delay(ball, is_terminal=terminal)
            self._ballpark._parent_RemoveBall(ball_id, delay)
            self._client_ticker_interface.clean_up_after_ball_removal(ball_id, ball, is_terminal=terminal)
        finally:
            _statistics.LeaveZone()

    def RemoveBalls(self, ball_ids, is_release=False):                                   # _ticker.py:376-392
        """The RemoveBalls action; also called when the ballpark is released."""
        _statistics.EnterZone("RemoveBalls")
        try:
            self._client_ticker_interface.clean_up_after_multiple_ball_removal(ball_ids, is_release=is_release)
            removal_delays = self._client_ticker_interface.get_ball_destruction_delays(ball_ids, is_release=is_release)

            for ball_id, delay in removal_delays.items():
                self._ballpark._parent_RemoveBall(ball_id, delay)
        finally:
            _statistics.LeaveZone()

    def invalidate_state(self):                                                          # _ticker.py:394-400
        self._state_is_valid = False

    def release(self):                                                                   # _ticker.py:402-415
        """Clean up the balls and the snapshots; the state is no longer valid."""
        self.RemoveBalls(self._ballpark.balls.keys(), is_release=True)
        self._ballpark.ClearAll()
        self.flush_simulation_history(new_base_snapshot=False)
        self._history = []
        self._latest_set_state_time = 0
        self._state_is_valid = False


# =========================================================================== the session (the conformance kit's harness)

class PilotInterests(BallInfoInterface, CharacterInterestsInterface, ClientInterestsInterface):
    """The server layer's four lookups for ships flown by pilots: pilot p is character `char_base + p` on client
    `client_base + p`, interested in its own ship only; a ball without a pilot has no character (None) and no client.
    (The game's own bookkeeping; Destiny leaves it to the interfaces, _interface.py:50-195.)"""

    def __init__(self, pilots: dict, char_base: int = 90000000, client_base: int = 10000):
        self.pilots = dict(pilots)                          # ship id -> pilot index
        self.char_base, self.client_base = int(char_base), int(client_base)
        self.char_of = {s: self.char_base + p for s, p in self.pilots.items()}
        self.client_of_char = {self.char_base + p: self.client_base + p for p in self.pilots.values()}

    # BallInfoInterface
    def get_character_for_ball(self, ball_id):
        return self.char_of.get(ball_id)

    def get_characters_for_ball(self, ball_id):
        c = self.char_of.get(ball_id)
        return [] if c is None else [c]

    # CharacterInterestsInterface
    def add_interest(self, ball_id, char_id):
        raise NotImplementedError("fixed at construction")

    def get_interested_character_ids_for_ball(self, ball_id):
        return self.get_characters_for_ball(ball_id)

    def remove_character_interest(self, ball_id, character_id):
        raise NotImplementedError("fixed at construction")

    def get_client_id_for_character(self, character_id):
        return self.client_of_char.get(character_id)

    # ClientInterestsInterface
    def add_client_interest(self, ball_id, client_id):
        raise NotImplementedError("fixed at construction")

    def get_all_interested_ball_ids(self):
        return list(self.char_of)

    def get_interested_client_ids_for_ball(self, ball_id):
        c = self.char_of.get(ball_id)
        return [] if c is None else [self.client_of_char[c]]

    def remove_client_interest(self, ball_id, client_id):
        raise NotImplementedError("fixed at construction")

    def has_client_interest(self, ball_id):
        return ball_id in self.char_of


class RecordingNetwork(NetworkInterface):
    """The server's network: each send_batch's singlecasts and narrowcasts, in the order sent."""

    def __init__(self):
        self.sent: list = []

    def singlecast(self, updates):
        for client_id, method, state, wait, count in updates:
            self.sent.append(("single", client_id, method, state, wait, count))

    def narrowcast(self, updates):
        for client_ids, method, state, wait, count in updates:
            self.sent.append(("narrow", list(client_ids), method, state, wait, count))

    def take(self) -> list:
        out, self.sent = self.sent, []
        return out


class _SilentClient(ClientTickerInterface):
    """The game's client hooks (Destiny's test helpers' choices, test/net/client/helpers.py:67-90): no ball destruction
    delay, nothing to clean up, actions not logged."""

    def on_set_state(self):
        pass

    def on_ballpark_local_action(self, func_name, args):
        pass

    def should_log_actions(self):
        return False

    def get_ball_destruction_delay(self, ball, is_terminal=False):
        return 0

    def get_ball_destruction_delays(self, ball_ids, is_release=False):
        return {ball_id: 0 for ball_id in ball_ids}

    def clean_up_after_ball_removal(self, ball_id, ball, is_terminal=False):
        pass

    def clean_up_after_multiple_ball_removal(self, ball_ids, is_release=False):
        pass


class DesyncCounter(TickErrorHandlerInterface):
    """Counts the client ticker's desync reports."""

    def __init__(self):
        self.fatal = 0
        self.recoverable = 0

    def on_fatal_desync(self):
        self.fatal += 1

    def on_recoverable_desync(self):
        self.recoverable += 1


class Session:
    """One server park with Destiny's server layer and one client park per pilot, ticked in lockstep with whole-tick
    delays: what a pilot's client knows at each tick. The conformance kit's harness around Destiny's code (exact_ballpark.server,
    this module), with this protocol:

    - The parks tick by Ballpark::OnTick (exact_ballpark.full.FullBallpark.on_tick, TIME-702 to TIME-705), one whole
      interval a call: the server's DoPreTick runs its Ticker, DoPostTick its batcher's send_batch; a client's DoPreTick
      runs its ticker's do_pre_tick, DoPostTick its do_post_tick with the C++'s stamp (mCurrentTime after the Evolve).
      Every park's first OnTick evolves once without the two events (TIME-704).
    - Session tick n (n = 1, 2, ...): the uplink commands due at n are issued on the server's Actions in their order;
      the server's OnTick; the server's sends of tick n queued for delivery at tick n + `down`; everything due at n
      delivered to each client's ticker (`update`), in the order sent (singlecasts first: Destiny requires the user
      updates before the bubble ones, net/__init__.py:14-22); then each client's OnTick. `command` issued after tick n
      reaches the server's Actions at the start of tick n + 1 + `up`.
    - Start-up (`start`): the balls are in the server park at time 0; tick 1 (every park's first OnTick, no pre-tick);
      then a SetState for each pilot (the batcher's send_full_state_update, stamped 1, the state at 1); tick 2 (the
      first pre-tick: the first bubble additions, appended after each SetState). Commands start after tick 2: one
      issued before tick 2 would carry Actions' first stamp, 0, while applied at time 1, and its client could neither
      synchronise to it nor clear its SetState's wait for the bubble (see the module's surprises).

    With these delays, each client's park settles `down` ticks behind the server's and applies each update at its
    stamp, so after tick n it holds the server's state at time n - down (tests/test_destiny_client_layer.py checks
    this field by field), and a command issued after tick n first moves its ship in Evolve n + up -> n + up + 1."""

    def __init__(self, server_park: DestinyPark, pilots: dict, down: int = 0, up: int = 0, char_base: int = 90000000,
                 client_base: int = 10000):
        if down < 0 or up < 0:
            raise ValueError("the delays are whole ticks, at least 0")
        if not server_park.isMaster:
            raise ValueError("the server's park is a master park")
        self.server = server_park
        self.down, self.up = int(down), int(up)
        self.interests = PilotInterests(pilots, char_base, client_base)
        self.network = RecordingNetwork()
        self.actions = srv.Actions(server_park)
        self.batcher = srv.ParkUpdateBatcher(server_park, self.network, self.interests, self.interests,
                                             srv.BubbleUpdater(server_park))
        self.ticker = srv.Ticker(server_park, self.interests, self.actions, self.batcher)
        lane = server_park.lane
        lane.pre_tick_handlers.append(lambda current_time: self.ticker.tick())
        lane.post_tick_handlers.append(lambda current_time: self.batcher.send_batch())
        self.interval = lane.tick_interval * 10000
        self.tick_no = 0
        self.clients: dict = {}                             # pilot index -> (ClientPark, Ticker, DesyncCounter)
        for ship, p in sorted(self.interests.pilots.items(), key=lambda kv: kv[1]):
            park = ClientPark(is_master=False)
            errors = DesyncCounter()
            ticker = Ticker(errors, _SilentClient())
            ticker.set_ballpark(park)
            park.lane.pre_tick_handlers.append(lambda current_time, t=ticker: t.do_pre_tick())
            park.lane.post_tick_handlers.append(lambda current_time, t=ticker: t.do_post_tick(current_time))
            self.clients[p] = (park, ticker, errors)
        self.downlink: dict = defaultdict(list)             # tick -> [(client id, state, wait_for_bubble)]
        self.uplink: dict = defaultdict(list)               # tick -> [(action name, args)]
        self.sent_log: list = []                            # (tick, the network's sends of that tick)

    def start(self) -> None:
        """Tick 1, a SetState for every pilot, tick 2 (the docstring's start-up)."""
        if self.tick_no != 0:
            raise RuntimeError("start() once, first")
        self.tick()
        for ship, p in sorted(self.interests.pilots.items(), key=lambda kv: kv[1]):
            self.batcher.send_full_state_update(self.interests.char_of[ship], ship)
        self.tick()

    def command(self, name: str, *args) -> None:
        """An Actions method (e.g. "go_to_direction", ship, x, y, z), issued after this tick: it reaches the server's
        Actions at the start of tick (this tick + 1 + up)."""
        self.uplink[self.tick_no + 1 + self.up].append((name, args))

    def tick(self) -> None:
        """One session tick (the docstring's order)."""
        self.tick_no += 1
        n = self.tick_no
        for name, args in self.uplink.pop(n, []):
            getattr(self.actions, name)(*args)
        self.server.lane.on_tick(n * self.interval)
        sent = self.network.take()
        self.sent_log.append((n, sent))
        for kind, to, method, state, wait, count in sent:
            for client_id in ([to] if kind == "single" else to):
                self.downlink[n + self.down].append((client_id, state, wait))
        for client_id, state, wait in self.downlink.pop(n, []):
            p = client_id - self.interests.client_base
            self.clients[p][1].update(state, wait)
        for p in sorted(self.clients):
            self.clients[p][0].lane.on_tick(n * self.interval)

    def client_park(self, pilot: int) -> ClientPark:
        return self.clients[pilot][0]
