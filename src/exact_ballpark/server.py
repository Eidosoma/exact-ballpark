# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer (destiny.net.server), transcribed: the queued Actions, the Ticker that applies them at the
pre-tick, the bubble updater and the update batcher that sends each tick's events to the clients.

A transcription of Carbon's `destiny` Python package (github.com/carbonengine/destiny at 114e89fa, MIT, (c) CCP Games). Destiny's Python is downloaded material: it was read, never imported or run. Every
class and method cites its source by file and line under python/destiny/:

- net/server/_actions.py:14-766 (`Actions`), net/server/_baseticker.py:10-115 (`BaseTicker`),
  net/server/_ticker.py:15-196 (`Ticker`), net/server/_bubbleupdater.py:6-54 (`BubbleUpdater`),
  net/server/_parkupdatebatcher.py:15-359 (`ParkUpdateBatcher`), net/server/_interface.py:16-195 (the four interfaces),
  net/const.py:6-8 (`ClientUpdateCountThisTick`), and the helpers they use: _util/signal.py:8-101 (`Signal`),
  _util/timing.py:7-31 (`TimedFunction`, `Timer`), _util/settings.py:6-31 (the dynamical-orientation switch).

Kept exactly: the order of every loop and every call, the stamps (an action is stamped with stamp_for_system, which the
Ticker sets to currentTime + 1 at each pre-tick after the follower clean-up has queued its Stops), the event tuples
(ball id, stamp, (name, args[, local args])), the duplicate check against the last queued action only, the signals and
when they fire, the dictionaries' and sets' iteration orders (CPython's, as Destiny's), and the bugs (below).

The layer drives a ballpark through Destiny's Python API. `DestinyPark` provides that API over this port's ballpark
(`exact_ballpark.full.FullBallpark` when that module defines it, else `exact_ballpark.space.SpaceBallpark`): each exposed
method parses its arguments as its C++ thunker does (Thunkers.cpp, PyArg_ParseTuple formats) and calls this port's
method; `_parent_<Name>` exists for every method Ballpark_Blue.cpp:292-754 exposes, as on a Blue object (the Ticker
applies an action only through `_parent_<action>`; a Blue wrapper provides these for the C++ methods, which Destiny's
own test, test_ticker.py:57-61, relies on). `BallView` gives a ball Ball_Blue.cpp's attribute names; a write stores
the member and then runs Ball::OnModified's setter (Ball.cpp:1507-1631). Three C++ thunkers the layer depends on are
transcribed here, citing
Thunkers.cpp: AdditionsAndDeletions with PopulateAddDelForBubble (2509-2895), AddTransitionToList and
NotifyOfBubbleTransitions (Ballpark.cpp:6009-6057); WriteBallsToStream, WriteFullStateToStream and WriteBallToStream
(2145-2258, 3180-3377), producing the same bytes as the C++ on a little-endian machine; ReadFullStateFromStream with
ReadBallFromStream (2083-2143, 2463-2507, 2897-3178). Abstractions, each labelled where it is made:
- blue.MemStream is a byte buffer with Write, Read and Seek (`MemStream`);
- blue.marshal.Save / Load keep the marshalled structure itself (`Marshalled`), not Blue's byte format;
- blue.pyos.taskletTimer (profiling) is a no-op;
- the process-global settings (destiny.settings, Settings.h) are a module-level `settings` object; only
  useDynamicalOrientation is read by the layer;
- the events the C++ sends to the park's Python side (DoBubbleTransitions) are recorded in `DestinyPark.events`.

Surprises in Destiny's code, kept as coded:
- Actions reads `park._currentTime` when the park's time is not zero (_actions.py:34), an attribute the C++ ballpark
  does not expose (Ballpark_Blue.cpp); Destiny's tests construct Actions only at time 0.
- Actions.remove_ball removes at once through `_parent_RemoveBall` (_actions.py:378-382); only a global ball's
  notification is queued.
- An action is dropped only when it equals the LAST queued action (_actions.py:62): the same command twice in a row,
  same ball and stamp; on_add_to_system_history fires for the dropped one all the same (_actions.py:64).
- make_ball_global and make_ball_local call SetBallGlobal at once (_actions.py:338, 360) and queue only the
  notifications; remove_ball removes at once. Every other action waits for the pre-tick.
- ParkUpdateBatcher checks that every event it sends carries the park's current time (_parkupdatebatcher.py:121-135),
  but in the documented order (tick at DoPreTick, send_batch at DoPostTick: _parkupdatebatcher.py:16-18, 53-56) Evolve
  has advanced the time by one in between (Ballpark.cpp:289-319), so every event after the first tick is logged as
  mismatched; Destiny's own test helper `_perform_tick` (test_ticker.py:172-182) runs in that order.
- Actions.set_max_angular_velocity queues 'SetMaxAngularVelocity' (_actions.py:205-209), a method the ballpark does not
  expose (Ballpark_Blue.cpp:292-754): the Ticker never applies it to the server's park (_ticker.py:115-116) and sends it
  to the bubble as an event all the same.
- has_pending_set_free and has_pending_set_not_free return 0, not False, when nothing is pending (_actions.py:746, 750).
- The Ticker caches the list of `_parent_` methods on the class, once per process (_ticker.py:47-54).
- The follower clean-up queues its Stops with the stamp of the tick before (it runs before the stamp moves,
  _baseticker.py:27-28) and behind any command already queued, so a remote follower given a new command in the same
  tick is stopped all the same (the Stop is applied last, _ticker.py:106-122).
- CloakBall is applied after the bubbles are updated (_ticker.py:110-112, 124-131): a cloak and an uncloak queued in
  the same tick leave the ball cloaked (Destiny's own test says so, test_ticker.py:134-146).
- The CloakBall event goes to the bubble captured when the action was queued (the old bubble if the ball changed
  bubbles in that tick, _actions.py:628-638), other events to the ball's bubble when they are distributed.
- The interfaces declare `__metaclass__ = abc.ABCMeta`, Python 2 syntax: in Python 3 their abstract methods are not
  enforced (_interface.py:17, 56, 119, 172; _baseticker.py:15).
- BubbleUpdater updates the park only once per tick, keyed on currentTime (_bubbleupdater.py:26-29).
- AdditionsAndDeletions sets each user ball's old bubble to its new one (Thunkers.cpp:2887) and posts
  DoBubbleTransitions with the list of transitions even when it is empty (2891).
- WriteBallToStream writes a ball's header and then, for a mode it does not know (BOID, MINIBALL), stops with an error
  (Thunkers.cpp:3333-3338); the reader then adds that ball and stops reading at it (3104-3109), dropping every later
  ball of the stream. WriteFullStateToStream leaves out balls with ids below DSTLOCALBALLS (Thunkers.cpp:2177), which
  includes a ball the server added with id -1.

Rules (rule ids): TIME-709 (actions queued and applied at the pre-tick), TIME-710 (a duplicate of the last action
dropped), TIME-711 (immediate removal), TIME-712 (the pre-tick's order), SEN-712 (events by bubble, cloaked balls to
their own character, dead bubbles dropped) and MOV-731's pre-tick part (Actions.clean_up_followers in the Ticker).
"""

from __future__ import annotations

import abc
import copy
import functools
import inspect
import itertools
import logging
import struct
import weakref
from collections import defaultdict
from collections.abc import Mapping
from enum import IntEnum

from .modes import _float32

logger = logging.getLogger(__name__)

# ==================================================================================================== constants

# DSTBALLMODE (IDstConstants.h:11-26)
(DSTBALL_GOTO, DSTBALL_FOLLOW, DSTBALL_STOP, DSTBALL_WARP, DSTBALL_ORBIT, DSTBALL_MISSILE, DSTBALL_MUSHROOM,
 DSTBALL_BOID, DSTBALL_TROLL, DSTBALL_MINIBALL, DSTBALL_FIELD, DSTBALL_RIGID, DSTBALL_FORMATION) = range(13)
DESTINY_FULLSTATE, DESTINY_BALLS = 0, 1                     # DESTINYPACKETS (IDstConstants.h:28-32)
DSTBALL_ISFREE, DSTBALL_ISGLOBAL, DSTBALL_ISMASSIVE = 0x01, 0x02, 0x04             # BALLFLAGS (IDstConstants.h:35-45)
DSTBALL_ISINTERACTIVE, DSTBALL_ISSPACEJUNK, DSTBALL_HASMINIBOXES = 0x08, 0x10, 0x20
DSTBALL_HASMINIBALLS, DSTBALL_HASMINICAPSULES = 0x40, 0x80
DSTLOCALBALLS = -1073741824                                 # DSTCONSTANTS (IDstConstants.h:74-80)
DSTNORMALCLOAK, DSTRESTORECLOAK, DSTGMCLOAK = 1, 2, 3


class ClientUpdateCountThisTick(IntEnum):
    """net/const.py:6-8."""
    ONE = 1
    TWO = 2


# ===================================================================================== destiny._util.settings

class _Settings:
    """The process-global switches (destiny.settings; Settings.h's defaults). The layer reads only
    useDynamicalOrientation (_util/settings.py:6-8)."""
    __slots__ = ("useDynamicalOrientation", "useNewOrbit")

    def __init__(self):
        self.useDynamicalOrientation = False
        self.useNewOrbit = False


settings = _Settings()


def is_dynamical_orientation_enabled() -> bool:
    """_util/settings.py:6-8."""
    return settings.useDynamicalOrientation


def enable_dynamical_orientation() -> None:
    """_util/settings.py:11-14."""
    settings.useDynamicalOrientation = True


def disable_dynamical_orientation() -> None:
    """_util/settings.py:17-20."""
    settings.useDynamicalOrientation = False


def reset_settings_to_default() -> None:
    """_util/settings.py:23-25."""
    settings.useDynamicalOrientation = False
    settings.useNewOrbit = False


def enable_new_orbit() -> None:
    """_util/settings.py:28-31."""
    settings.useNewOrbit = True


# ======================================================================================= destiny._util.signal

class Signal(object):
    """_util/signal.py:8-101: a weakref signal / slot (functions in a WeakSet, bound methods by their object)."""

    _monitor_enabled = False
    _monitor_callback = None

    def __init__(self, signalName=None):
        self._signalName = signalName
        self._functions = weakref.WeakSet()
        self._methods = weakref.WeakKeyDictionary()

    def __repr__(self):
        if self._signalName:
            nameText = self._signalName + " "
        else:
            nameText = ""
        if len(self) > 0:
            return "%s%s connected to %s>" % (nameText, object.__repr__(self)[:-1], ",".join([str(f) for f in self]))
        return nameText + object.__repr__(self)

    def __call__(self, *args, **kwargs):
        if Signal._monitor_enabled and Signal._monitor_callback:
            Signal._monitor_callback(self._signalName, args, kwargs)
        for call in self:
            try:
                call(*args, **kwargs)
            except Exception:
                logging.exception("Exception in signal handler: {}".format(call))

    def __len__(self):
        return len(self._functions) + sum([len(methods) for methods in self._methods.values()])

    def __iter__(self):
        callables = []
        callables.extend(self._functions)
        for obj, funcs in self._methods.items():
            callables.extend([func.__get__(obj) for func in funcs])
        return iter(callables)

    def connect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = weakref.WeakSet()
            self._methods[slot.__self__].add(slot.__func__)
        elif inspect.isfunction(slot) and slot.__name__ == "<lambda>":
            raise TypeError("Signal cannot connect lambda methods")
        elif callable(slot):
            self._functions.add(slot)
        else:
            raise TypeError("Signal connect requires a callable slot")

    def disconnect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].discard(slot.__func__)
        else:
            self._functions.discard(slot)

    def clear(self):
        self._functions.clear()
        self._methods.clear()

    @property
    def debugDisplayName(self):
        return self._signalName

    @classmethod
    def StartSignalMonitoring(cls, callback):
        cls._monitor_enabled = True
        cls._monitor_callback = callback

    @classmethod
    def StopSignalMonitoring(cls):
        cls._monitor_enabled = False
        cls._monitor_callback = None


# ======================================================================================= destiny._util.timing

def TimedFunction(context):
    """_util/timing.py:7-17: blue.pyos.taskletTimer around the call (profiling only: here a plain call)."""
    def decorator(function):
        @functools.wraps(function)
        def Wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return Wrapper
    return decorator


class Timer(object):
    """_util/timing.py:20-31: a profiling context (no-op here)."""
    __slots__ = ["ctxt"]

    def __init__(self, context):
        self.ctxt = context

    def __enter__(self):
        return None

    def __exit__(self, *_):
        return None


# ============================================================================================= blue stand-ins

class MemStream:
    """blue.MemStream as the layer and Destiny's C++ use it: Write at the position, Read from it, Seek."""

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0

    def Write(self, data) -> int:
        data = bytes(data)
        end = self._pos + len(data)
        if end > len(self._buf):
            self._buf.extend(b"\0" * (end - len(self._buf)))
        self._buf[self._pos:end] = data
        self._pos = end
        return len(data)

    def Read(self, n: int = -1) -> bytes:
        end = len(self._buf) if n is None or n < 0 else min(len(self._buf), self._pos + n)
        out = bytes(self._buf[self._pos:end])
        self._pos = end
        return out

    def Seek(self, pos: int) -> None:
        self._pos = int(pos)

    @property
    def size(self) -> int:
        return len(self._buf)


class Marshalled:
    """blue.marshal.Save's result, kept as the structure it encodes (a deep copy) rather than Blue's bytes; Load
    returns a deep copy of it. Lists stay lists and tuples tuples, as in a marshal round trip."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def __eq__(self, other):
        return isinstance(other, Marshalled) and self.value == other.value

    def __repr__(self):
        return f"Marshalled({self.value!r})"

    __hash__ = None


def marshal_save(value) -> Marshalled:
    """blue.marshal.Save."""
    return Marshalled(value)


def marshal_load(blob: Marshalled):
    """blue.marshal.Load."""
    return copy.deepcopy(blob.value)


# ============================================================================== net/server/_interface.py:16-195

class NetworkInterface(object):
    """_interface.py:16-47."""
    __metaclass__ = abc.ABCMeta                             # Python 2 syntax: inert in Python 3, as in Destiny

    @abc.abstractmethod
    def narrowcast(self, updates):
        pass

    @abc.abstractmethod
    def singlecast(self, updates):
        pass


class ClientInterestsInterface(object):
    """_interface.py:50-110."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_client_interest(self, ball_id, client_id):
        pass

    @abc.abstractmethod
    def get_all_interested_ball_ids(self):
        pass

    @abc.abstractmethod
    def get_interested_client_ids_for_ball(self, ball_id):
        pass

    @abc.abstractmethod
    def remove_client_interest(self, ball_id, client_id):
        pass

    @abc.abstractmethod
    def has_client_interest(self, ball_id):
        pass


class CharacterInterestsInterface(object):
    """_interface.py:113-164."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_interest(self, ball_id, char_id):
        pass

    @abc.abstractmethod
    def get_interested_character_ids_for_ball(self, ball_id):
        pass

    @abc.abstractmethod
    def remove_character_interest(self, ball_id, character_id):
        pass

    @abc.abstractmethod
    def get_client_id_for_character(self, character_id):
        pass


class BallInfoInterface(object):
    """_interface.py:167-195."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_character_for_ball(self, ball_id):
        pass

    @abc.abstractmethod
    def get_characters_for_ball(self, ball_id):
        pass


# ================================================================================ net/server/_actions.py:14-766

class Actions(object):
    """Exposes a simple API for destiny actions. Actions get added to the system history to be applied on the next
    tick (_actions.py:14-19)."""

    def __init__(self, park):                                                             # _actions.py:20-29
        self._park = park
        self._system_history = []
        self.on_ball_made_global = Signal()
        self.on_add_to_system_history = Signal()
        self._stamp_for_system = self._get_initial_stamp_for_system()

    def _get_initial_stamp_for_system(self):                                             # _actions.py:31-34
        if self._park.currentTime == 0:
            return 0
        return self._park._currentTime + 1                  # (sic) an attribute the C++ park does not expose

    def _add_to_system_history(self, ball_id, eventName, *args, **kwargs):              # _actions.py:36-64
        insert_ball_id_into_args = kwargs.get("insert_ball_id_into_args", True)
        if insert_ball_id_into_args:
            args = (ball_id,) + args
        local_args = kwargs.get("local_args", None)
        if local_args is None:
            event = (eventName, args)
        else:
            event = (eventName, args, local_args)
        action = ball_id, self._stamp_for_system, event
        if not self._system_history or self._system_history[-1] != action:  # No use adding the same action twice.
            self._system_history.append(action)
        self.on_add_to_system_history(eventName)

    def set_stamp_for_system(self, stamp):                                               # _actions.py:66-71
        self._stamp_for_system = stamp

    def flush_history(self):                                                             # _actions.py:73-79
        history, self._system_history = self._system_history, []
        return history

    def go_to_point(self, src_id, x, y, z):                                              # _actions.py:81-97
        self._add_to_system_history(src_id, 'GotoPoint', x, y, z)

    def go_to_direction(self, src_id, x, y, z):                                          # _actions.py:99-115
        self._add_to_system_history(src_id, "GotoDirection", x, y, z)

    def orbit(self, src_id, dst_id, orbit_range_meters):                                 # _actions.py:117-131
        self._add_to_system_history(src_id, "Orbit", dst_id, orbit_range_meters)

    def set_ball_troll(self, src_id, delay_ticks):                                       # _actions.py:133-148
        self._add_to_system_history(src_id, 'SetBallTroll', delay_ticks)

    def set_ball_velocity(self, src_id, vx, vy, vz):                                     # _actions.py:150-167
        self._add_to_system_history(src_id, 'SetBallVelocity', vx, vy, vz)

    def set_ball_angular_velocity(self, src_id, wx, wy, wz):                             # _actions.py:169-188
        if not is_dynamical_orientation_enabled():
            raise RuntimeError("Attempted to set ball angular velocity, but Dynamical Orientation is disabled.")
        self._add_to_system_history(src_id, 'SetBallAngularVelocity', wx, wy, wz)

    def set_max_angular_velocity(self, src_id, wx, wy, wz):                              # _actions.py:190-209
        if not is_dynamical_orientation_enabled():
            raise RuntimeError("Attempted to set max angular velocity, but Dynamical Orientation is disabled.")
        self._add_to_system_history(src_id, 'SetMaxAngularVelocity', wx, wy, wz)

    def set_ball_rotation(self, src_id, rx, ry, rz, rw):                                 # _actions.py:211-232
        if not is_dynamical_orientation_enabled():
            raise RuntimeError("Attempted to set ball rotation, but Dynamical Orientation is disabled.")
        self._add_to_system_history(src_id, 'SetBallRotation', rx, ry, rz, rw)

    def set_ball_massive(self, src_id, is_massive):                                      # _actions.py:234-248
        self._add_to_system_history(src_id, 'SetBallMassive', is_massive)

    def stop(self, src_id):                                                              # _actions.py:250-259
        self._add_to_system_history(src_id, 'Stop')

    def set_ball_mass(self, src_id, mass_kg):                                            # _actions.py:261-274
        self._add_to_system_history(src_id, 'SetBallMass', mass_kg)

    def set_ball_agility(self, src_id, agility):                                         # _actions.py:276-291
        self._add_to_system_history(src_id, 'SetBallAgility', agility)

    def set_ball_angular_agility(self, src_id, angular_agility):                         # _actions.py:293-309
        if not is_dynamical_orientation_enabled():
            raise RuntimeError("Attempting to set angular agility, but Dynamical Orientation is disabled.")
        self._add_to_system_history(src_id, 'SetBallAngularAgility', angular_agility)

    def add_ball_to_client_parks(self, src_id):                                          # _actions.py:311-326
        dump = MemStream()
        self._park.WriteBallsToStream([src_id], dump)
        ball_stream = dump.Read()
        self._add_to_system_history(src_id, 'AddBallsToPark', ball_stream, insert_ball_id_into_args=False)

    def make_ball_local(self, src_id):                                                   # _actions.py:328-348
        self._park.SetBallGlobal(src_id, False)
        if src_id not in self._park.balls:
            logger.error("Attempted to make ball local, but it's not in the park %s", src_id)
            return
        ball = self._park.balls[src_id]
        self._add_to_system_history(src_id, 'BallNotGlobal', ball.newBubbleId, insert_ball_id_into_args=False)

    def make_ball_global(self, src_id):                                                  # _actions.py:350-362
        self._park.SetBallGlobal(src_id, True)
        self.add_ball_to_client_parks(src_id)
        self.on_ball_made_global(src_id)

    def remove_ball(self, src_id):                                                       # _actions.py:364-382
        """IMPORTANT: Ball removal is immediate (_actions.py:369-373)."""
        ball = self._park.balls[src_id]
        if ball.isGlobal:
            self._notify_remove_global_ball(src_id)
        self._park._parent_RemoveBall(src_id)

    def _notify_remove_global_ball(self, src_id):                                        # _actions.py:384-388
        self._add_to_system_history(src_id, 'RemoveGlobalBall')

    def set_speed_fraction(self, src_id, speed_fraction):                                # _actions.py:390-403
        self._add_to_system_history(src_id, "SetSpeedFraction", speed_fraction)

    def warp_to(self, src_id, x, y, z, minimum_range, warp_speed):                       # _actions.py:405-427
        self._add_to_system_history(src_id, "WarpTo", x, y, z, minimum_range, warp_speed)

    def entity_warp_in(self, src_id, x, y, z, warp_speed):                               # _actions.py:429-450
        self._add_to_system_history(src_id, 'EntityWarpIn', x, y, z, warp_speed)

    def set_ball_position(self, src_id, x, y, z, is_local_ball=False):                   # _actions.py:452-473
        self._add_to_system_history(src_id, 'SetBallPosition', x, y, z, local_args=is_local_ball)

    def set_ball_harmonic(self, src_id, harmonic_value, corporation_id, alliance_id, is_forcefield):  # :475-508
        if is_forcefield:
            if harmonic_value != -1 or corporation_id != -1 or alliance_id != -1:
                self.set_ball_massive(src_id, True)
            else:
                self.set_ball_massive(src_id, False)
        self._add_to_system_history(src_id, 'SetBallHarmonic', harmonic_value, corporation_id, alliance_id,
                                    is_forcefield)

    def set_ball_radius(self, src_id, radius_meters):                                    # _actions.py:510-523
        self._add_to_system_history(src_id, 'SetBallRadius', radius_meters)

    def set_ball_free(self, src_id, is_free=True):                                       # _actions.py:525-538
        self._add_to_system_history(src_id, 'SetBallFree', is_free)

    def set_ball_interactive(self, src_id, is_interactive):                              # _actions.py:540-552
        self._add_to_system_history(src_id, 'SetBallInteractive', is_interactive)

    def follow_ball(self, src_id, dst_id, range_meters):                                 # _actions.py:554-568
        self._add_to_system_history(src_id, 'FollowBall', dst_id, range_meters)

    def set_max_speed(self, src_id, max_meters_per_second):                              # _actions.py:570-582
        self._add_to_system_history(src_id, 'SetMaxSpeed', max_meters_per_second)

    def set_max_angular_speed(self, src_id, max_radians_per_second):                     # _actions.py:584-598
        if not is_dynamical_orientation_enabled():
            raise RuntimeError("Attempting to set angular speed, but Dynamical Orientation is disabled.")
        self._add_to_system_history(src_id, 'SetMaxAngularSpeed', max_radians_per_second)

    def cloak_ball(self, src_id, cloak_mode, uncloak_range_meters):                      # _actions.py:600-639
        ball = self._park.balls.get(src_id, None)
        if ball is None:
            return False
        if ball.newBubbleId != ball.oldBubbleId:
            logger.info("cloak_ball( %s ) found ball changing bubbles this tick. Using old bubbleID for cloak", src_id)
            bubble_id = ball.oldBubbleId
        else:
            bubble_id = ball.newBubbleId
        self._add_to_system_history(src_id, 'CloakBall', cloak_mode, uncloak_range_meters, local_args=bubble_id)
        return True

    def uncloak_ball(self, src_id):                                                      # _actions.py:641-651
        self._add_to_system_history(src_id, 'UncloakBall')

    def launch_missile(self, src_id, dst_id, owner_id, is_aimed_launch, is_missile_massive):  # _actions.py:653-675
        self._add_to_system_history(src_id, 'LaunchMissile', dst_id, owner_id, is_aimed_launch, is_missile_massive)

    @TimedFunction("destiny.net::actions::clean_up_followers")
    def clean_up_followers(self):                                                        # _actions.py:677-684
        """Stop all balls that are following a ball which is located in another bubble."""
        for followerID in self._park.GetRemoteFollowers():
            self.stop(followerID)

    @TimedFunction("destiny.net::actions::undo_pending_cloak")
    def undo_pending_cloak(self, src_id):                                                # _actions.py:686-693
        removed_count = self._prune_pending_actions(src_id, "CloakBall")
        logger.info("undo_pending_cloak trying to prune pending CloakBall calls from the system history for ship %s, "
                    "count=%s", src_id, removed_count)

    def _prune_pending_actions(self, src_id, event_name):                                # _actions.py:695-707
        history_length_before_removal = len(self._system_history)
        self._system_history = [
            event
            for event in self._system_history if
            event[0] != src_id or event[2][0] != event_name
        ]
        removed_count = history_length_before_removal - len(self._system_history)
        return removed_count

    def _get_pending_actions(self, src_id, event_name):                                  # _actions.py:709-718
        return [
            event
            for ball_id, timestamp, event in self._system_history
            if ball_id == src_id and event[0] == event_name
        ]

    def _has_pending_action(self, src_id, event_name):                                   # _actions.py:720-728
        for ball_id, timestamp, event in self._system_history:
            if ball_id == src_id and event[0] == event_name:
                return True
        return False

    def has_pending_cloak(self, src_id):                                                 # _actions.py:730-731
        return self._has_pending_action(src_id, "CloakBall")

    def has_pending_uncloak(self, src_id):                                               # _actions.py:733-734
        return self._has_pending_action(src_id, "UncloakBall")

    def get_pending_cloak_mode(self, src_id):                                            # _actions.py:736-742
        events = self._get_pending_actions(src_id, "CloakBall")
        if events:
            return events[-1][1][1]
        if src_id not in self._park.balls:
            return None
        return self._park.balls[src_id].isCloaked

    def has_pending_set_free(self, src_id):                                              # _actions.py:744-746
        events = self._get_pending_actions(src_id, "SetBallFree")
        return len(events) and events[-1][1][1] == 1

    def has_pending_set_not_free(self, src_id):                                          # _actions.py:748-750
        events = self._get_pending_actions(src_id, "SetBallFree")
        return len(events) and events[-1][1][1] == 0

    def get_pending_interactive_state(self, src_id):                                     # _actions.py:752-758
        events = self._get_pending_actions(src_id, "SetBallInteractive")
        if events:
            return bool(events[-1][1][1])
        if src_id not in self._park.balls:
            return None
        return self._park.balls[src_id].isInteractive

    def get_pending_speed_fraction(self, src_id):                                        # _actions.py:760-766
        events = self._get_pending_actions(src_id, "SetSpeedFraction")
        if events:
            return events[-1][1][1]
        if src_id not in self._park.balls:
            return None
        return self._park.balls[src_id].speedFraction


# ============================================================================ net/server/_baseticker.py:10-115

class BaseTicker(object):
    """A base class for tickers that send out updates in synchronization with the ballpark ticks
    (_baseticker.py:10-14)."""
    __metaclass__ = abc.ABCMeta                             # Python 2 syntax: inert in Python 3, as in Destiny

    def __init__(self, park, ball_info):                                                 # _baseticker.py:17-21
        self._park = park
        self._ball_info = ball_info
        self.stamp_for_system = self._park.currentTime + 1
        self.current_system_history = []

    def tick(self):                                                                      # _baseticker.py:23-32
        """Call this from the ballparks DoPreTick method."""
        self._finalize_tick()
        self._increment_timestamp()
        self._flush_history()
        self._pre_update_bubbles()
        self._update_bubbles()
        self._post_update_bubbles()

    @abc.abstractmethod
    def _finalize_tick(self):                                                            # _baseticker.py:34-40
        pass

    def _increment_timestamp(self):                                                      # _baseticker.py:42-46
        self.stamp_for_system = self._park.currentTime + 1

    @abc.abstractmethod
    def _flush_history(self):                                                            # _baseticker.py:48-53
        pass

    @abc.abstractmethod
    def _pre_update_bubbles(self):                                                       # _baseticker.py:55-61
        pass

    @abc.abstractmethod
    def _update_bubbles(self):                                                           # _baseticker.py:63-73
        pass

    @abc.abstractmethod
    def _post_update_bubbles(self):                                                      # _baseticker.py:75-81
        pass

    def _expand_entry(self, entry):                                                      # _baseticker.py:83-97
        event_ball_id, stamp, event = entry
        if stamp != self._park.currentTime and stamp > 0:
            # The first tick is a special case.  Ignore it.
            logger.error("Found systemHistory entry with incorrect stamp %s %s", self._park.currentTime, entry)
        action, args = event[0], event[1]
        local_args = None
        if len(event) > 2:
            local_args = event[2]
            event = event[:-1]
        return action, args, event, event_ball_id, local_args, stamp

    def _find_bubble_id_for_event(self, event_ball_id, action, args):                    # _baseticker.py:99-106
        if event_ball_id in self._park.balls:
            event_bubble_id = self._park.balls[event_ball_id].newBubbleId
        elif action == 'RemoveBall':
            event_bubble_id = args[1]
        else:
            event_bubble_id = -1
        return event_bubble_id

    def _is_valid_event_bubble(self, event_bubble_id, event_ball_id):                    # _baseticker.py:108-115
        if event_bubble_id not in self._park.bubbles:
            return False
        if event_bubble_id not in self._park.bubbleInteractives:
            return False  # A dead bubble probably
        if event_ball_id <= 0 or event_ball_id not in self._park.balls:
            return False
        return True


# ================================================================================= net/server/_ticker.py:15-196

PARK_METHODS_THAT_GET_CALLED_DIRECTLY = {                                                # _ticker.py:15-19
    "RemoveBall",
}


class Ticker(BaseTicker):
    """Applies history from destiny actions and updates bubble and user histories on the update batcher, which should
    be sent out in the ballparks DoPostTick method (_ticker.py:22-27)."""
    _parent_methods = None  # Lazy init.                                                 # _ticker.py:28

    ACTION_HANDLERS = {                                                                  # _ticker.py:30-37
        "CloakBall": "_handle_cloak_ball_action",
        "UncloakBall": "_handle_uncloak_action",
        "SetBallPosition": "_handle_set_ball_position_action",
        "AddBallsToPark": "_handle_add_balls_to_park_action",
        "BallNotGlobal": "_handle_ball_not_global_action",
        "RemoveGlobalBall": "_handle_remove_global_ball_action",
    }

    def __init__(self, park, ball_info, actions, update_batcher):                        # _ticker.py:39-45
        super(Ticker, self).__init__(park, ball_info)
        self._actions = actions
        self._update_batcher = update_batcher
        self.on_ball_cloaking = Signal()
        self.on_ball_uncloaking = Signal()
        self.on_set_ball_position = Signal()

    def _get_parent_methods(self):                                                       # _ticker.py:47-54
        if Ticker._parent_methods is None:
            Ticker._parent_methods = [
                "_parent_" + x
                for x in dir(self._park)
                if not x.startswith("__") and callable(getattr(self._park, "_parent_" + x, None))
            ]
        return Ticker._parent_methods

    def _clean_up_followers(self):                                                       # _ticker.py:56-59
        with Timer("Ticker::_clean_up_followers"):
            self._actions.clean_up_followers()

    def _finalize_tick(self):                                                            # _ticker.py:61-62
        self._clean_up_followers()

    def _increment_timestamp(self):                                                      # _ticker.py:64-66
        super(Ticker, self)._increment_timestamp()
        self._actions.set_stamp_for_system(self.stamp_for_system)

    def _flush_history(self):                                                            # _ticker.py:68-69
        self.current_system_history = self._actions.flush_history()

    def _pre_update_bubbles(self):                                                       # _ticker.py:71-72
        self._apply_park_parent_methods_except_cloak_ball()

    def _update_bubbles(self):                                                           # _ticker.py:74-76
        with Timer("Ticker::_update_bubbles"):
            self._update_batcher.update_bubbles()

    def _post_update_bubbles(self):                                                      # _ticker.py:78-79
        self._apply_history()

    def _apply_history(self):                                                            # _ticker.py:81-104
        with Timer("Ticker::_apply_history"):
            for entry in self.current_system_history:
                action, args, event, event_ball_id, local_args, stamp = self._expand_entry(entry)

                handler_method_name = Ticker.ACTION_HANDLERS.get(action, None)
                if handler_method_name is not None:
                    handler = getattr(self, handler_method_name)
                    handled = handler(args, event, event_ball_id, local_args, stamp)
                    if handled:
                        continue

                event_bubble_id = self._find_bubble_id_for_event(event_ball_id, action, args)
                with Timer("Ticker::_apply_history::distribute_to_bubbles"):
                    if not self._is_valid_event_bubble(event_bubble_id, event_ball_id):
                        continue
                    if not self._park.balls[event_ball_id].isCloaked:
                        self._update_batcher.add_to_bubble_history(event_bubble_id, (stamp, event))
                    else:  # Don't send any events about cloaked balls to anyone else
                        charID = self._ball_info.get_character_for_ball(event_ball_id)
                        if charID is not None:
                            self._update_batcher.add_to_character_history(charID, (stamp, event))

    def _apply_park_parent_methods_except_cloak_ball(self):                              # _ticker.py:106-122
        with Timer("Ticker::_ApplyParkParentMethodsExceptCloakBall"):
            for entry in self.current_system_history:
                event_ball_id, stamp, event = entry
                action, args = event[0], event[1]
                if action == 'CloakBall':
                    continue  # Skip the cloaking bit in destiny until after we update bubbles
                elif action in PARK_METHODS_THAT_GET_CALLED_DIRECTLY:
                    continue  # RemoveBall gets called directly by gameplay server park.
                parent_action_name = '_parent_' + action
                if parent_action_name in self._get_parent_methods():
                    fn = getattr(self._park, parent_action_name)
                    try:
                        fn(*args)
                    except Exception:
                        logger.exception("Bummer in park, fn %s args %s", action, args)
                        continue

    def _handle_cloak_ball_action(self, args, event, event_ball_id, local_args, stamp):  # _ticker.py:124-131
        ball_id = args[0]
        if ball_id in self._park.balls:
            self.on_ball_cloaking(ball_id)
        event_bubble_id = local_args
        self._update_batcher.add_to_bubble_history(event_bubble_id, (stamp, event))
        self._park.CloakBall(*args)
        return True

    def generate_add_balls_update(self, ball_id):                                        # _ticker.py:133-137
        dump = MemStream()
        self._park.WriteBallsToStream([ball_id], dump)
        ball_stream = dump.Read()
        return ('AddBallsToPark', (ball_stream,))

    def _handle_uncloak_action(self, args, event, event_ball_id, local_args, stamp):     # _ticker.py:139-163
        if args[0] != event_ball_id:
            logger.error("Malformed 'UncloakBall' entry in system history: %s", str(event))
            event_ball_id = args[0]
        if event_ball_id not in self._park.balls:
            logger.info("'UncloakBall' event for a ball not in park: %s", str(event))
            return True
        event_ball = self._park.balls[event_ball_id]
        self.on_ball_uncloaking(event_ball_id)
        event_bubble_id = event_ball.newBubbleId
        bubble_interactives = self._park.bubbleInteractives.get(event_bubble_id, [])
        add_balls_update = None
        for ball_id in bubble_interactives:
            char_id = self._ball_info.get_character_for_ball(event_ball_id)
            if ball_id == event_ball_id:
                self._update_batcher.add_to_character_history(char_id, (stamp, event))
            else:
                if add_balls_update is None:
                    add_balls_update = self.generate_add_balls_update(event_ball_id)
                char_ids = self._ball_info.get_characters_for_ball(ball_id)
                for charID in filter(None, char_ids):
                    self._update_batcher.add_to_character_history(charID, (stamp, add_balls_update), in_front=True)
        return True

    def _handle_set_ball_position_action(self, args, event, event_ball_id, local_args, stamp):  # _ticker.py:165-171
        is_local_ball = local_args is True
        self.on_set_ball_position(event_ball_id, is_local_ball)
        if is_local_ball:
            logger.info("Local ball moved, notifying noone. %s", event)
            return True
        return False

    def _handle_add_balls_to_park_action(self, args, event, event_ball_id, local_args, stamp):  # _ticker.py:173-176
        for bubble_id in self._park.bubbleInteractives:
            self._update_batcher.add_to_bubble_history(bubble_id, (stamp, event))
        return True

    def _handle_ball_not_global_action(self, args, event, event_ball_id, local_args, stamp):  # _ticker.py:178-187
        for bubble_id in self._park.bubbleInteractives:
            if bubble_id != args[0]:
                self._update_batcher.add_to_bubble_history(bubble_id, (stamp, ('RemoveBall', (event_ball_id,))))
        return True

    def _handle_remove_global_ball_action(self, args, event, event_ball_id, local_args, stamp):  # _ticker.py:189-196
        for bubble_id in self._park.bubbleInteractives:
            self._update_batcher.add_to_bubble_history(bubble_id, (stamp, ('RemoveBall', (event_ball_id,))))
        return True


# ============================================================================ net/server/_bubbleupdater.py:6-54

class BubbleUpdater(object):
    """Updates ball bubble presence in the "bubbles" dict on the park (_bubbleupdater.py:6-10)."""

    def __init__(self, park):                                                            # _bubbleupdater.py:11-17
        self._park = park
        self._tick = None
        self.additions_per_player = {}
        self.deletions_per_player = {}
        self.additions_per_bubble = {}
        self.deletions_per_bubble = {}

    def additions_and_deletions(self):                                                   # _bubbleupdater.py:19-29
        if self._tick != self._park.currentTime:
            self._process_additions_and_deletions_for_tick()
            self._park.CopyBubbles()  # Update the parks "bubbles" dict.
            self._tick = self._park.currentTime

    def _process_additions_and_deletions_for_tick(self):                                 # _bubbleupdater.py:31-54
        self.user_balls = list(itertools.chain(*self._park.bubbleInteractives.values()))
        self.additions_per_player = {}
        self.deletions_per_player = {}
        self.additions_per_bubble = {}
        self.deletions_per_bubble = {}
        self._park.AdditionsAndDeletions(
            self.additions_per_player,
            self.deletions_per_player,
            self.additions_per_bubble,
            self.deletions_per_bubble,
            self.user_balls
        )


# ======================================================================== net/server/_parkupdatebatcher.py:15-359

class ParkUpdateBatcher(object):
    """Accumulates updates for users and bubbles that should get sent out in the ballparks DoPostTick method
    (_parkupdatebatcher.py:15-19)."""

    def __init__(self, ballpark, network_interface, character_interests, client_interests, bubble_updater):  # :20-33
        self._park = ballpark
        self._character_history = defaultdict(list)  # Map of character_id to list of update actions (stamp, event)
        self._bubble_history = {}  # Map of bubble_id to list of updates
        self._network_interface = network_interface
        self._character_interests = character_interests
        self._client_interests = client_interests
        self._bubble_updater = bubble_updater
        self.on_add_to_character_history = Signal(signalName="on_add_to_character_history")
        self.on_add_to_bubble_history = Signal(signalName="on_add_to_bubble_history")

    def get_clients_with_character_history(self):                                        # :35-50
        client_ids = set()
        for char_id, state in self._character_history.items():
            client_id = self._character_interests.get_client_id_for_character(char_id)
            if client_id is None:
                continue
            if state:
                client_ids.add(client_id)
        return client_ids

    def send_batch(self):                                                                # :52-119
        clients_waiting_for_bubble = set()
        clients_with_character_history = self.get_clients_with_character_history()
        single_batch_narrowcasts = []  # For characters who only get DoDestinyUpdate for the bubble
        dual_batch_narrowcasts = []    # For characters who also get DoDestinyUpdate for their character specifically.
        singlecasts = []

        for bubble_id, state in self._bubble_history.items():
            if bubble_id in self._park.bubbleInteractives:
                client_ids = set()
                for ball_id in self._park.bubbleInteractives[bubble_id]:
                    client_ids.update(self._client_interests.get_interested_client_ids_for_ball(ball_id))
                self._check_state_timestamp(state)
                dual_update_narrowcast_clients = client_ids & clients_with_character_history
                single_update_narrowcast_clients = client_ids - clients_with_character_history
                if single_update_narrowcast_clients:
                    single_batch_narrowcasts.append(
                        (
                            list(single_update_narrowcast_clients),
                            'DoDestinyUpdate',
                            state,
                            False,
                            ClientUpdateCountThisTick.ONE
                        )
                    )
                if dual_update_narrowcast_clients:
                    dual_batch_narrowcasts.append(
                        (
                            list(dual_update_narrowcast_clients),
                            'DoDestinyUpdate',
                            state,
                            False,
                            ClientUpdateCountThisTick.TWO
                        )
                    )
                clients_waiting_for_bubble.update(client_ids)
        self._bubble_history = {}

        for char_id, state in self._character_history.items():
            client_id = self._character_interests.get_client_id_for_character(char_id)
            if client_id is None:
                continue
            if state:
                wait_for_bubble = client_id in clients_waiting_for_bubble
                if wait_for_bubble:
                    update_count = ClientUpdateCountThisTick.TWO
                else:
                    update_count = ClientUpdateCountThisTick.ONE
                self._check_state_timestamp(state)
                singlecasts.append((client_id, 'DoDestinyUpdate', state, wait_for_bubble, update_count))

        self._character_history.clear()
        self._network_interface.singlecast(singlecasts)
        self._network_interface.narrowcast(single_batch_narrowcasts)
        self._network_interface.narrowcast(dual_batch_narrowcasts)

    def _check_state_timestamp(self, state):                                             # :121-135
        for entry in state:
            if entry[0] != self._park.currentTime and entry[0] > 0:
                logger.error(
                    "Found ballpark action for mismatched timestamp {time} {state_id} {entry}".format(
                        time=self._park.currentTime,
                        state_id=id(state),
                        entry=entry
                    )
                )

    def add_to_character_history(self, character_id, action, in_front=False):           # :137-160
        if in_front:
            self._character_history[character_id].insert(0, action)
        else:
            self._character_history[character_id].append(action)
        try:
            action_name = action[1][0]
        except IndexError:
            logger.error("Failed to retrieve action name for \"add to user history\" signal")
        else:
            self.on_add_to_character_history(action_name)

    def clear_character_history(self, char_id):                                          # :162-171
        if char_id in self._character_history:
            del self._character_history[char_id]

    def add_to_bubble_history(self, bubble_id, action):                                  # :173-193
        if bubble_id in self._bubble_history:
            self._bubble_history[bubble_id].append(action)
        else:
            self._bubble_history[bubble_id] = [action]
        try:
            action_name = action[1][0]
        except IndexError:
            logger.error("Failed to retrieve action name for \"add to bubble history\" signal")
        else:
            self.on_add_to_bubble_history(action_name)

    def send_full_state_update(self, char_id, src_id):                                   # :195-217
        bubble_dump = MemStream()
        self._park.WriteFullStateToStream(bubble_dump, src_id)
        state = bubble_dump.Read()
        set_state_event = self._park.currentTime, ('SetState', (state, src_id))
        logger.info("SetState issued for %s at time %s", src_id, self._park.currentTime)
        self.add_to_character_history(char_id, set_state_event, in_front=True)

    def _dump_ball_list(self, ball_id_list):                                             # :219-223
        dump = MemStream()
        self._park.WriteBallsToStream(ball_id_list, dump)
        ball_stream = dump.Read()
        return 'AddBallsToPark', ((ball_stream),)

    def update_bubbles(self):                                                            # :225-316
        self._bubble_updater.additions_and_deletions()
        additions_per_player = self._bubble_updater.additions_per_player
        deletions_per_player = self._bubble_updater.deletions_per_player
        additions_per_bubble = self._bubble_updater.additions_per_bubble
        deletions_per_bubble = self._bubble_updater.deletions_per_bubble

        add_actions_per_bubble = {}

        with Timer("destiny.net::parkupdatebatcher::update_bubbles::PreBuildDeletions"):
            for bubble_id, dels in deletions_per_bubble.items():
                if len(dels) > 0:
                    user_actions = (
                        (self._park.currentTime, ('RemoveBalls', (dels,))),
                    )
                    packaged_action = marshal_save(user_actions)
                    deletions_per_bubble[bubble_id] = (
                        self._park.currentTime, ('PackagedAction', packaged_action)
                    )

        with Timer("destiny.net::parkupdatebatcher::update_bubbles::PreBuildAdditions"):
            for bubble_id, adds in additions_per_bubble.items():
                if len(adds) > 0:
                    user_actions = [
                        (self._park.currentTime, self._dump_ball_list(adds)),
                    ]
                    with Timer("destiny.net::parkupdatebatcher::update_bubbles::PreBuildAdditions::Marshal"):
                        packaged_action = marshal_save(user_actions)
                        add_actions_per_bubble[bubble_id] = (
                            self._park.currentTime, ('PackagedAction', packaged_action)
                        )

        getter = self._character_interests.get_interested_character_ids_for_ball

        with Timer("destiny.net::parkupdatebatcher::update_bubbles::Deletions"):
            for ball_id, dels in deletions_per_player.items():
                if isinstance(dels, int):
                    deletions = deletions_per_bubble[dels]
                    if deletions:
                        for character_id in getter(ball_id):
                            self.add_to_character_history(character_id, deletions)
                elif len(dels) > 0:
                    for character_id in getter(ball_id):
                        self.add_to_character_history(
                            character_id, (self._park.currentTime, ('RemoveBalls', (dels,)))
                        )

        with Timer("destiny.net::parkupdatebatcher::update_bubbles::Additions"):
            for ball_id, adds in additions_per_player.items():
                if isinstance(adds, int):
                    additions = add_actions_per_bubble.get(adds, [])
                    if additions:
                        for character_id in getter(ball_id):
                            self.add_to_character_history(character_id, additions)
                elif len(adds) > 0:
                    characters = getter(ball_id)
                    if characters:
                        user_actions = [
                            (self._park.currentTime, self._dump_ball_list(adds)),
                        ]
                        with Timer("destiny.net::parkupdatebatcher::update_bubbles::AddToCharacterHistory"):
                            packaged_action = marshal_save(user_actions)
                            for character_id in characters:
                                self.add_to_character_history(
                                    character_id, (self._park.currentTime, ('PackagedAction', packaged_action))
                                )

    def get_bubble_history(self, bubble_id):                                             # :318-325
        return copy.copy(self._bubble_history.get(bubble_id, []))

    def bubble_has_history(self, bubble_id):                                             # :327-337
        return bool(
            bubble_id in self._bubble_history and
            self._bubble_history[bubble_id]
        )

    def get_character_history(self, char_id):                                            # :339-346
        return copy.copy(self._character_history.get(char_id, []))

    def character_has_history(self, char_id):                                            # :348-359
        return bool(
            char_id in self._character_history and
            self._character_history[char_id]
        )


# ===================================================================== the ballpark's Python API over a lane park

def _parse_tuple(args: tuple, fmt: str) -> list:
    """PyArg_ParseTuple for the units the thunkers use: L, i, b (integers; a float there is a TypeError; i and b range
    checked), f (a C float), d (a double), '|' (the rest optional, missing ones None)."""
    units = [c for c in fmt if c != "|"]
    required = len(fmt.split("|")[0])
    if not required <= len(args) <= len(units):
        raise TypeError(f"function takes {'at least ' if required < len(units) else ''}{required} arguments "
                        f"({len(args)} given)")
    out = []
    for c, a in zip(units, args):
        if c in "Lib":
            if isinstance(a, float) or not hasattr(a, "__index__"):
                raise TypeError(f"integer argument expected, got {type(a).__name__}")
            v = a.__index__()
            if c == "L" and not -(1 << 63) <= v < (1 << 63):
                raise OverflowError("signed long long is out of range")
            if c == "i" and not -(1 << 31) <= v < (1 << 31):
                raise OverflowError("signed integer is out of range")
            if c == "b" and not 0 <= v <= 255:
                raise OverflowError("unsigned byte integer is out of range")
            out.append(int(v))
        elif c in "fd":
            v = float(a)                                    # a float or an int (__float__ or __index__)
            out.append(_float32(v) if c == "f" else v)
        else:
            raise ValueError(f"format unit {c!r} not used by these thunkers")
    return out + [None] * (len(units) - len(out))


def _new_lane_park(is_master: bool):
    """This port's ballpark: FullBallpark when exact_ballpark.full defines it, else SpaceBallpark."""
    try:
        from . import full
    except ModuleNotFoundError as exc:
        if exc.name != __package__ + ".full":
            raise
        full = None
    FullBallpark = getattr(full, "FullBallpark", None)
    if FullBallpark is not None:
        kw = {"is_master": is_master}
        if is_dynamical_orientation_enabled():
            kw["dynamical_orientation"] = True
        return FullBallpark(**kw)
    if is_dynamical_orientation_enabled():
        raise NotImplementedError("dynamical orientation needs exact_ballpark.full.FullBallpark")
    from .space import SpaceBallpark
    return SpaceBallpark(is_master=is_master)


_NOT_IN_LANE = object()


def _lane_field(ball, name: str, default=_NOT_IN_LANE):
    value = getattr(ball, name, default)
    if value is _NOT_IN_LANE:
        raise NotImplementedError(f"this port's ball has no {name!r}")
    return value


class BallView:
    """A ball as Destiny's Python sees it: Ball_Blue.cpp's attribute names (19-480) over a lane ball, read live.
    formationID and the mini lists default to Ball::Ball's values (-1, empty: Ball.cpp:61-118) when this port's ball does
    not carry them (only formations and minis change them). A write to a READWRITE attribute stores the member (float
    members as floats), then, for a NOTIFY attribute, runs Ball::OnModified (Ball.cpp:1507-1631): the park's setter for
    it, while the ball is in a park. isInteractive, isCloaked and ownerId have no setter there: the member only."""
    __slots__ = ("_b", "_p")

    def __init__(self, lane_ball, park=None):
        object.__setattr__(self, "_b", lane_ball)
        object.__setattr__(self, "_p", park)

    def __repr__(self):
        return f"BallView({self._b.id})"

    def _in_park(self) -> bool:
        """Ball::mPark is set: the ball is the park's (Ball.cpp:1511-1512)."""
        return self._p is not None and self._p.lane.balls.get(self._b.id) is self._b

    def __setattr__(self, name, value):
        b, park = self._b, self._p
        if name in ("x", "y", "z"):                                       # Ball.cpp:1526-1537
            pos = list(b.pos)
            pos["xyz".index(name)] = float(value)
            b.pos = pos
            if self._in_park():
                park.SetBallPosition(b.id, *b.pos)
        elif name in ("vx", "vy", "vz"):                                  # Ball.cpp:1538-1549
            vel = list(b.vel)
            vel["xyz".index(name[1])] = float(value)
            b.vel = vel
            if self._in_park():
                park.SetBallVelocity(b.id, *b.vel)
        elif name in ("gotoX", "gotoY", "gotoZ"):                         # Ball.cpp:1514-1525
            goto = list(b.goto)
            goto["XYZ".index(name[-1])] = float(value)
            b.goto = goto
            if self._in_park():
                park.GotoPoint(b.id, *b.goto)
        elif name == "radius":                                            # Ball.cpp:1586-1589
            b.radius = _float32(float(value))
            if self._in_park():
                park.SetBallRadius(b.id, b.radius)
        elif name == "maxVelocity":                                       # Ball.cpp:1590-1593
            b.max_vel = _float32(float(value))
            if self._in_park():
                park.SetMaxSpeed(b.id, b.max_vel)
        elif name == "mass":                                              # Ball.cpp:1594-1597
            b.mass = float(value)
            if self._in_park():
                park.SetBallMass(b.id, b.mass)
        elif name == "isFree":                                            # Ball.cpp:1598-1601: SetBallFree then
            b.is_free = bool(value)                                       # finds the flag already set (no effect)
            if self._in_park():
                park.SetBallFree(b.id, b.is_free)
        elif name == "isMassive":                                         # Ball.cpp:1601-1604
            b.is_massive = bool(value)
            if self._in_park():
                park.SetBallMassive(b.id, b.is_massive)
        elif name == "isGlobal":                                          # Ball.cpp:1605-1608
            b.is_global = bool(value)
            if self._in_park():
                park.SetBallGlobal(b.id, b.is_global)
        elif name == "Agility":                                           # Ball.cpp:1609-1612
            b.agility = _float32(float(value))
            if self._in_park():
                park.SetBallAgility(b.id, b.agility)
        elif name == "speedFraction":                                     # Ball.cpp:1613-1616
            b.speed_fraction = _float32(float(value))
            if self._in_park():
                park.SetSpeedFraction(b.id, b.speed_fraction)
        elif name in ("harmonic", "corporationID", "allianceID"):         # Ball.cpp:1617-1628
            field = {"harmonic": "harmonic", "corporationID": "corporation_id", "allianceID": "alliance_id"}[name]
            setattr(b, field, value.__index__() if name == "harmonic" else _int32(value.__index__()))
            if self._in_park():
                park.SetBallHarmonic(b.id, b.harmonic, b.corporation_id, b.alliance_id, 0)
        elif name == "isInteractive":                                     # no OnModified branch: the member only
            b.is_interactive = bool(value)
        elif name == "isCloaked":                                         # no OnModified branch: the member only
            b.is_cloaked = value.__index__() & 0xFF
        elif name == "ownerId":                                           # READWRITE without NOTIFY
            b.owner_id = value.__index__()
        elif name in ("rx", "ry", "rz", "rw"):                            # Ball.cpp:1550-1565
            rot = list(_lane_field(b, "rot"))
            rot["xyzw".index(name[1])] = _float32(float(value))
            b.rot = tuple(rot)
            if self._in_park():
                park.SetBallRotation(b.id, *b.rot)
        elif name in ("wx", "wy", "wz"):                                  # Ball.cpp:1566-1577
            ang_vel = list(_lane_field(b, "ang_vel"))
            ang_vel["xyz".index(name[1])] = _float32(float(value))
            b.ang_vel = tuple(ang_vel)
            if self._in_park():
                park.SetBallAngularVelocity(b.id, *b.ang_vel)
        elif name == "maxAngularVelocity":                                # Ball.cpp:1578-1581
            _lane_field(b, "max_ang_vel")
            b.max_ang_vel = _float32(float(value))
            if self._in_park():
                park.SetMaxAngularSpeed(b.id, b.max_ang_vel)
        elif name == "RotationalAgility":                                 # Ball.cpp:1582-1585
            _lane_field(b, "rot_agility")
            b.rot_agility = _float32(float(value))
            if self._in_park():
                park.SetBallAngularAgility(b.id, b.rot_agility)
        else:
            raise AttributeError(f"attribute {name!r} is read-only (Ball_Blue.cpp)")

    id = property(lambda s: s._b.id)                                        # mId
    mass = property(lambda s: s._b.mass)                                    # mMass
    radius = property(lambda s: s._b.radius)                                # mRadius (float)
    maxVelocity = property(lambda s: s._b.max_vel)                          # mMaxVel (float)
    isFree = property(lambda s: bool(s._b.is_free))
    isGlobal = property(lambda s: bool(s._b.is_global))
    isMassive = property(lambda s: bool(s._b.is_massive))
    isInteractive = property(lambda s: bool(s._b.is_interactive))
    isCloaked = property(lambda s: int(s._b.is_cloaked))                    # uint8
    isMoribund = property(lambda s: bool(s._b.is_moribund))
    harmonic = property(lambda s: s._b.harmonic)
    corporationID = property(lambda s: s._b.corporation_id)
    allianceID = property(lambda s: s._b.alliance_id)
    x = property(lambda s: s._b.pos[0])                                     # mNewPos
    y = property(lambda s: s._b.pos[1])
    z = property(lambda s: s._b.pos[2])
    vx = property(lambda s: s._b.vel[0])                                    # mNewVel
    vy = property(lambda s: s._b.vel[1])
    vz = property(lambda s: s._b.vel[2])
    maxAngularVelocity = property(lambda s: _lane_field(s._b, "max_ang_vel"))      # mMaxAngVel (float)
    wx = property(lambda s: _lane_field(s._b, "ang_vel")[0])                # mNewAngVel (floats)
    wy = property(lambda s: _lane_field(s._b, "ang_vel")[1])
    wz = property(lambda s: _lane_field(s._b, "ang_vel")[2])
    rx = property(lambda s: _lane_field(s._b, "rot")[0])                    # mNewRot (floats)
    ry = property(lambda s: _lane_field(s._b, "rot")[1])
    rz = property(lambda s: _lane_field(s._b, "rot")[2])
    rw = property(lambda s: _lane_field(s._b, "rot")[3])
    RotationalAgility = property(lambda s: _lane_field(s._b, "rot_agility"))  # mRotAgility (float)
    yaw = property(lambda s: _lane_field(s._b, "yaw"))                      # mNewYaw
    pitch = property(lambda s: _lane_field(s._b, "pitch"))                  # mNewPitch
    roll = property(lambda s: _lane_field(s._b, "roll_angle"))              # mNewRoll
    Agility = property(lambda s: s._b.agility)                              # mAgility (float)
    speedFraction = property(lambda s: s._b.speed_fraction)                 # mSpeedFraction (float)
    mode = property(lambda s: s._b.mode)
    gotoX = property(lambda s: s._b.goto[0])                                # mGoto
    gotoY = property(lambda s: s._b.goto[1])
    gotoZ = property(lambda s: s._b.goto[2])
    followId = property(lambda s: s._b.follow_id)
    followRange = property(lambda s: s._b.follow_range)                     # mFollowRange (float)
    ownerId = property(lambda s: s._b.owner_id)
    effectStamp = property(lambda s: s._b.effect_stamp)
    newBubbleId = property(lambda s: s._b.bubble)                           # mNewBubble
    oldBubbleId = property(lambda s: s._b.old_bubble)                       # mOldBubble
    formationID = property(lambda s: getattr(s._b, "formation_id", -1))     # mFormationID
    miniBalls = property(lambda s: [MiniBallView(m) for m in getattr(s._b, "mini_balls", ())])
    miniCapsules = property(lambda s: [MiniCapsuleView(m) for m in getattr(s._b, "mini_capsules", ())])
    miniBoxes = property(lambda s: [MiniBoxView(m) for m in getattr(s._b, "mini_boxes", ())])

    def AddMiniBall(self, *args):
        """Ball::PyAddMiniBall ("dddf", Ball.cpp:2129-2141) and AddMiniBall: this port's add_mini_ball while the ball is
        the park's (a ball outside a park only calls the not-in-park callback, which is not modelled)."""
        if self._in_park():
            self._p.lane.add_mini_ball(self._b.id, *args)

    def AddMiniCapsule(self, *args):
        """Ball::PyAddMiniCapsule ("ddddddf", Ball.cpp:2143-2166): a radius <= 0 is a ValueError."""
        if self._in_park():
            self._p.lane.add_mini_capsule(self._b.id, *args)

    def AddMiniBox(self, *args):
        """Ball::PyAddMiniBox (twelve "d", Ball.cpp:2168-2190)."""
        if self._in_park():
            self._p.lane.add_mini_box(self._b.id, *args)


class MiniBallView:
    """A MiniBall as Python sees it (Miniball_Blue.cpp: id, radius, x, y, z) over this port's record."""
    __slots__ = ("_m",)

    def __init__(self, rec):
        self._m = rec

    id = property(lambda s: s._m.id)
    radius = property(lambda s: s._m.radius)
    x = property(lambda s: s._m.pos[0])
    y = property(lambda s: s._m.pos[1])
    z = property(lambda s: s._m.pos[2])


class MiniCapsuleView:
    """A MiniCapsule (MiniCapsule_Blue.cpp: id, radius, ax, ay, az, bx, by, bz) over this port's record."""
    __slots__ = ("_m",)

    def __init__(self, rec):
        self._m = rec

    id = property(lambda s: s._m.id)
    radius = property(lambda s: s._m.radius)
    ax = property(lambda s: s._m.a[0])
    ay = property(lambda s: s._m.a[1])
    az = property(lambda s: s._m.a[2])
    bx = property(lambda s: s._m.b[0])
    by = property(lambda s: s._m.b[1])
    bz = property(lambda s: s._m.b[2])


class MiniBoxView:
    """A MiniBox (MiniBox_Blue.cpp: id, the corner c0-c2 and the edges x0-x2, y0-y2, z0-z2) over this port's record."""
    __slots__ = ("_m",)

    def __init__(self, rec):
        self._m = rec

    id = property(lambda s: s._m.id)
    c0 = property(lambda s: s._m.corner[0])
    c1 = property(lambda s: s._m.corner[1])
    c2 = property(lambda s: s._m.corner[2])
    x0 = property(lambda s: s._m.lx[0])
    x1 = property(lambda s: s._m.lx[1])
    x2 = property(lambda s: s._m.lx[2])
    y0 = property(lambda s: s._m.ly[0])
    y1 = property(lambda s: s._m.ly[1])
    y2 = property(lambda s: s._m.ly[2])
    z0 = property(lambda s: s._m.lz[0])
    z1 = property(lambda s: s._m.lz[1])
    z2 = property(lambda s: s._m.lz[2])


class BallsView(Mapping):
    """park.balls: mBalls, a Python dict in insertion order (MapOfBalls.h), values as BallView."""

    def __init__(self, lane_balls: dict, park=None):
        self._d = lane_balls
        self._p = park

    def __getitem__(self, k):
        return BallView(self._d[k], self._p)

    def __contains__(self, k):
        return k in self._d

    def __iter__(self):
        return iter(list(self._d))

    def __len__(self):
        return len(self._d)


# the methods Ballpark_Blue.cpp:292-754 exposes to Python, in its order; `_parent_<name>` is made for each
BLUE_METHODS = (
    "AddBall", "FollowBall", "MissileFollow", "WarpTo", "GotoPoint", "GotoDirection", "Orbit", "Stop", "RemoveBall",
    "GetBallIdsInRangeOfTriangle", "GetBallIdsInCapsule", "GetBallIdsInCone", "ClearAll", "SetBallMass",
    "SetBallAgility", "SetBallRadius", "SetMaxSpeed", "SetBallGlobal", "SetBallMassive", "SetBallInteractive",
    "SetBallFree", "SetBallVelocity", "SetMaxAngularSpeed", "SetBallAngularVelocity", "SetBallRotation",
    "SetBallAngularAgility", "SetSpeedFraction", "SetNotificationRange", "SetTargetTracking", "SetBallPosition",
    "SetBallHarmonic", "InitializeBubbles", "WriteFullStateToStream", "WriteBallsToStream", "ReadFullStateFromStream",
    "AddProximitySensor", "AddMushroom", "RemoveProximitySensor", "CheckVisibility", "CloakBall", "UncloakBall",
    "GetAccuracy", "Start", "Pause", "Evolve", "GetBallIdsInRange", "GetBallIdsAndDistInRange", "CopyBubbles",
    "AdditionsAndDeletions", "SetBoid", "GetSurfaceDist", "GetCenterDist", "ScanCone", "GetBallBox",
    "GetStaticCollidableBox", "GetRemoteFollowers", "GetFollowers", "GetActiveBoxes", "SetBoxObject", "GetBoxObject",
    "GetBubbleAtCoordinates", "GetBoxCenter", "SetBallTroll", "SetBallRigid", "GetCurrentEgoPos", "LaunchMissile",
    "SetBallFormation", "FormationFollow", "LoadFormations", "SetBallNotInParkCallback", "EntityWarpIn", "AdjustTimes",
    "GetBallBoxKeys", "GetBoxChildren", "GetBoxKey",
)


class DestinyPark:
    """destiny.Ballpark's Python API (Ballpark_Blue.cpp) over a lane ballpark; a client's view unless told otherwise,
    as destiny.Ballpark() (Ballpark.cpp:97, isMaster false)."""

    def __init__(self, lane_park=None, *, is_master: bool = False):
        self._lane = lane_park if lane_park is not None else _new_lane_park(is_master)
        self.events: list = []                              # events the C++ sends to the park's Python side

    # ---- attributes (Ballpark_Blue.cpp:15-290) ----

    @property
    def lane(self):
        return self._lane

    @property
    def isMaster(self):
        return self._lane.is_master

    @isMaster.setter
    def isMaster(self, value):
        self._lane.is_master = bool(value)

    @property
    def currentTime(self):
        return self._lane.current_time

    @currentTime.setter
    def currentTime(self, value):
        self._lane.current_time = int(value)

    @property
    def tickInterval(self):
        return self._lane.tick_interval

    @property
    def friction(self):
        return self._lane.friction

    @property
    def balls(self):
        return BallsView(self._lane.balls, self)

    @property
    def globals(self):
        return BallsView(self._lane.globals, self)

    @property
    def bubbles(self):
        return self._lane.bubbles

    @property
    def bubbleInteractives(self):
        return self._lane.bubble_interactives

    @property
    def bubbleKeepAlives(self):
        return self._lane.bubble_keep_alives

    @property
    def bubbleKeepAliveBalls(self):
        return self._lane.bubble_keep_alive_balls

    # ---- helpers ----

    def _call(self, name: str, *args):
        fn = getattr(self._lane, name, None)
        if fn is None:
            raise NotImplementedError(f"this port's ballpark has no {name}()")
        return fn(*args)

    # ---- methods (Thunkers.cpp: each parses as its thunker, then calls this port) ----

    def AddBall(self, *args):
        """PyAddBall (Thunkers.cpp:150-160): PyAddDynamicallyOrientedBall with dynamical orientation on (51-110),
        otherwise PyAddOldStyleOrientedBall (113-147); this port's add_ball is that path."""
        if is_dynamical_orientation_enabled():
            (src, mass, radius, max_vel, free, glob, massive, interactive, junk, x, y, z, vx, vy, vz, agility, sf,
             max_ang_vel, ang_agility) = _parse_tuple(args, "Ldffiiiiiddddddffff")
            if self._lane.in_evolve:
                raise RuntimeError("Can not add ball during Evolve().")
            ball = self._lane.add_ball(src, mass, radius, max_vel, free != 0, glob != 0, massive != 0,
                                       interactive != 0, junk != 0, x, y, z, vx, vy, vz, agility, sf,
                                       max_ang_vel=max_ang_vel, rot_agility=ang_agility)
        else:
            (src, mass, radius, max_vel, free, glob, massive, interactive, junk, x, y, z, vx, vy, vz, agility,
             sf) = _parse_tuple(args, "Ldffiiiiiddddddff")
            if self._lane.in_evolve:
                raise RuntimeError("Can not add ball during Evolve().")
            ball = self._lane.add_ball(src, mass, radius, max_vel, free != 0, glob != 0, massive != 0,
                                       interactive != 0, junk != 0, x, y, z, vx, vy, vz, agility, sf)
        return BallView(ball, self)

    def FollowBall(self, *args):
        """PyFollowBall (Thunkers.cpp:244-270)."""
        src, dst, rng = _parse_tuple(args, "LL|f")
        if dst < 0:
            raise RuntimeError(f"You can not Follow a negative ballID, src({src}), dst({dst})")
        self._lane.follow_ball(src, dst, 1.0 if rng is None else rng)

    def MissileFollow(self, *args):
        """PyMissileFollow (Thunkers.cpp:275-300)."""
        src, dst, owner = _parse_tuple(args, "LLL")
        if dst < 0:
            raise RuntimeError(f"You can not MissileFollow a negative ballID, src({src}), dst({dst})")
        self._call("missile_follow", src, dst, owner)

    def Orbit(self, *args):
        """PyOrbit (Thunkers.cpp:330-355)."""
        src, dst, rng = _parse_tuple(args, "LL|f")
        if dst < 0:
            raise RuntimeError(f"You can not Orbit a negative ballID, src({src}), dst({dst})")
        self._lane.orbit(src, dst, 1.0 if rng is None else rng)

    def WarpTo(self, *args):
        """PyWarpTo (Thunkers.cpp:360-383): minimum range 20,000 m and warp factor 20 by default."""
        src, x, y, z, min_range, factor = _parse_tuple(args, "Lddd|di")
        self._lane.warp_to(src, x, y, z, 20000.0 if min_range is None else min_range, 20 if factor is None else factor)

    def EntityWarpIn(self, *args):
        """PyEntityWarpIn (Thunkers.cpp:389-410)."""
        self._lane.entity_warp_in(*_parse_tuple(args, "Ldddi"))

    def GotoPoint(self, *args):
        """PyGotoPoint (Thunkers.cpp:435-456)."""
        self._lane.goto_point(*_parse_tuple(args, "Lddd"))

    def GotoDirection(self, *args):
        """PyGotoDirection (Thunkers.cpp:461-482)."""
        self._lane.goto_direction(*_parse_tuple(args, "Lddd"))

    def Stop(self, *args):
        """PyStop (Thunkers.cpp:596-611)."""
        self._lane.stop(*_parse_tuple(args, "L"))

    def RemoveBall(self, src_id, delay=0):
        """RemoveBall(ID, delay = 0), exposed with its optional argument (Ballpark_Blue.cpp:346-352)."""
        src, d = _parse_tuple((src_id, delay), "Li")
        self._lane.remove_ball(src, d)

    def SetBallMass(self, *args):
        """PySetBallMass (Thunkers.cpp:487-504)."""
        self._lane.set_mass(*_parse_tuple(args, "Ld"))

    def SetBallAgility(self, *args):
        """PySetBallAgility (Thunkers.cpp:687-704)."""
        self._lane.set_agility(*_parse_tuple(args, "Lf"))

    def SetBallRadius(self, *args):
        """PySetBallRadius (Thunkers.cpp:552-569)."""
        self._lane.set_radius(*_parse_tuple(args, "Lf"))

    def SetMaxSpeed(self, *args):
        """PySetMaxSpeed (Thunkers.cpp:573-590)."""
        self._lane.set_max_velocity(*_parse_tuple(args, "Lf"))

    def SetBallGlobal(self, *args):
        """PySetBallGlobal (Thunkers.cpp:1033-1050)."""
        src, flag = _parse_tuple(args, "Li")
        self._lane.set_global(src, flag != 0)

    def SetBallMassive(self, *args):
        """PySetBallMassive (Thunkers.cpp:1054-1071)."""
        src, flag = _parse_tuple(args, "Li")
        self._lane.set_massive(src, flag != 0)

    def SetBallInteractive(self, *args):
        """PySetBallInteractive (Thunkers.cpp:1076-1093)."""
        src, flag = _parse_tuple(args, "Li")
        self._lane.set_interactive(src, flag != 0)

    def SetBallFree(self, *args):
        """PySetBallFree (Thunkers.cpp:980-997)."""
        src, flag = _parse_tuple(args, "Li")
        self._lane.set_free(src, flag != 0)

    def SetBallVelocity(self, *args):
        """PySetBallVelocity (Thunkers.cpp:777-798)."""
        self._lane.set_velocity(*_parse_tuple(args, "Lddd"))

    def SetMaxAngularSpeed(self, *args):
        """PySetMaxAngularSpeed (Thunkers.cpp:800-811)."""
        self._call("set_max_angular_speed", *_parse_tuple(args, "Lf"))

    def SetBallAngularVelocity(self, *args):
        """PySetBallAngularVelocity (Thunkers.cpp:813-824)."""
        self._call("set_ball_angular_velocity", *_parse_tuple(args, "Lddd"))

    def SetBallRotation(self, *args):
        """PySetBallRotation (Thunkers.cpp:826-837)."""
        self._call("set_ball_rotation", *_parse_tuple(args, "Ldddd"))

    def SetBallAngularAgility(self, *args):
        """PySetBallAngularAgility (Thunkers.cpp:838-849)."""
        self._call("set_ball_angular_agility", *_parse_tuple(args, "Lf"))

    def SetSpeedFraction(self, *args):
        """PySetSpeedFraction (Thunkers.cpp:709-726)."""
        self._lane.set_speed_fraction(*_parse_tuple(args, "Lf"))

    def SetNotificationRange(self, *args):
        """PySetNotificationRange (Thunkers.cpp:732-749)."""
        self._lane.set_notification_range(*_parse_tuple(args, "Ld"))

    def SetTargetTracking(self, *args):
        """PySetTargetTracking (Thunkers.cpp:754-771)."""
        self._lane.set_target_tracking(*_parse_tuple(args, "LLd"))

    def SetBallPosition(self, *args):
        """PySetBallPosition (Thunkers.cpp:661-682)."""
        self._lane.set_position(*_parse_tuple(args, "Lddd"))

    def SetBallHarmonic(self, *args):
        """PySetBallHarmonic (Thunkers.cpp:952-975)."""
        src, harmonic, corp, alliance, field = _parse_tuple(args, "LLiii")
        self._lane.set_harmonic(src, harmonic, corp, alliance, field != 0)

    def InitializeBubbles(self, *args):
        """PyInitializeBubbles (Thunkers.cpp:1123-1141)."""
        (src,) = _parse_tuple(args, "|L")
        self._lane.initialize_bubbles(0 if src is None else src)

    def AddProximitySensor(self, *args):
        """PyAddProximitySensor (Thunkers.cpp:1284-1307)."""
        src, rng, period, shuffle, only = _parse_tuple(args, "Lf|dib")
        self._lane.add_proximity_sensor(src, rng, 2.0 if period is None else period, 0 if shuffle is None else shuffle,
                                        bool(only))

    def RemoveProximitySensor(self, *args):
        """PyRemoveProximitySensor (Thunkers.cpp:1312-1328)."""
        self._lane.remove_proximity_sensor(*_parse_tuple(args, "L"))

    def CheckVisibility(self, *args):
        """PyCheckVisibility (Thunkers.cpp:1824-1840)."""
        return self._lane.check_visibility(*_parse_tuple(args, "LL"))

    def CloakBall(self, *args):
        """PyCloakBall (Thunkers.cpp:1846-1865): the range is optional (2000 m)."""
        src, mode, rng = _parse_tuple(args, "Lb|f")
        self._call("cloak_ball", src, mode, 2000.0 if rng is None else rng)

    def UncloakBall(self, *args):
        """PyUncloakBall (Thunkers.cpp:1870-1885)."""
        self._call("uncloak_ball", *_parse_tuple(args, "L"))

    def ClearAll(self):
        """PyClearAll and Ballpark::ClearAll (Ballpark.cpp:5896-5952): this port's clear_all."""
        self._call("clear_all")

    def Evolve(self):
        """PyEvolve (Thunkers.cpp:1244-1252): Evolve(0)."""
        self._lane.evolve()

    def GetBallIdsInRange(self, *args):
        """PyGetBallIdsInRange (Thunkers.cpp:1330-1335): this port returns None where Destiny raises."""
        out = self._lane.ball_ids_in_range(*args)
        if out is None:
            raise TypeError("GetBallIdsInRange: bad arguments or no such ball")
        return out

    def GetBallIdsAndDistInRange(self, *args):
        """PyGetBallIdsAndDistInRange (Thunkers.cpp:1337-1342)."""
        out = self._lane.ball_ids_in_range(*args, include_dist=True)
        if out is None:
            raise TypeError("GetBallIdsAndDistInRange: bad arguments or no such ball")
        return out

    def CopyBubbles(self):
        """PyCopyBubbles (Thunkers.cpp:2035-2042)."""
        self._lane.copy_bubbles()

    def SetBoid(self, *args):
        """PySetBoid (Thunkers.cpp:1003-1028)."""
        self._call("set_boid", *_parse_tuple(args, "Li"))

    def GetRemoteFollowers(self):
        """PyGetRemoteFollowers (Thunkers.cpp:1193-1237)."""
        return list(self._lane.remote_followers())

    def GetBubbleAtCoordinates(self, *args):
        """PyGetBubbleAtCoordinates (Thunkers.cpp:2011-2030)."""
        x, y, z = _parse_tuple(args, "ddd")
        return self._lane.bubble_at(x, y, z)

    def SetBallTroll(self, *args):
        """PySetBallTroll (Thunkers.cpp:530-547)."""
        self._call("set_ball_troll", *_parse_tuple(args, "Li"))

    def SetBallRigid(self, *args):
        """PySetBallRigid (Thunkers.cpp:509-524)."""
        self._call("set_ball_rigid", *_parse_tuple(args, "L"))

    def LaunchMissile(self, *args):
        """PyLaunchMissile (Thunkers.cpp:857-946): the parse and the negative-target check here; the rest (from
        Thunkers.cpp:878) is this port's launch_missile."""
        src, dst, owner, aimed, massive = _parse_tuple(args, "LLLbb")
        if dst < 0:
            raise RuntimeError("Can not launch missile on a negative ballID")
        self._call("launch_missile", src, dst, owner, aimed, massive)

    def SetBallFormation(self, *args):
        """PySetBallFormation (Thunkers.cpp:1899-1916)."""
        self._call("set_ball_formation", *_parse_tuple(args, "Li"))

    def FormationFollow(self, *args):
        """PyFormationFollow (Thunkers.cpp:1098-1118)."""
        self._call("formation_follow", *_parse_tuple(args, "LLi"))

    # ---- AdditionsAndDeletions (Thunkers.cpp:2509-2895) with Ballpark.cpp:6009-6057 ----

    def _populate_add_del_for_bubble(self, bubble: dict, add_list: list, del_list: list) -> None:
        """Ballpark::PopulateAddDelForBubble (Thunkers.cpp:2509-2568)."""
        balls = self._lane.balls
        for ball_id, action in list(bubble.items()):
            if ball_id < DSTLOCALBALLS:
                continue
            ball = balls.get(ball_id)
            if action != -1:                                # the ball might already be deleted
                if ball is None or ball.is_global or ball.is_cloaked:
                    continue
            else:
                if ball is not None and ball.is_cloaked:    # no removals of cloaked balls
                    continue
            if action == 1:
                add_list.append(ball_id)
            elif action == -1:
                del_list.append(ball_id)

    def _add_transition_to_list(self, ball, transitions: list) -> None:
        """Ballpark::AddTransitionToList (Ballpark.cpp:6009-6042): (id, old bubble, new bubble); none for a troll."""
        if ball.mode == DSTBALL_TROLL:
            return
        transitions.append((ball.id, ball.old_bubble, ball.bubble))

    def _notify_of_bubble_transitions(self, transitions: list) -> None:
        """Ballpark::NotifyOfBubbleTransitions (Ballpark.cpp:6048-6057): SendEvent DoBubbleTransitions to the park's
        Python side, recorded here."""
        self.events.append(("DoBubbleTransitions", list(transitions)))
        handler = getattr(self, "DoBubbleTransitions", None)
        if handler is not None:
            handler(transitions)

    def AdditionsAndDeletions(self, additions_per_player, deletions_per_player, additions_per_bubble,
                              deletions_per_bubble, user_ships):
        """Ballpark::PyAdditionsAndDeletions (Thunkers.cpp:2570-2895): per bubble with interactives the balls added and
        deleted in it; per user ship its bubble's id when it stayed, else its own lists across its old and new
        bubbles; then each user ship's old bubble becomes its new one, and the transitions are posted."""
        for d in (additions_per_player, deletions_per_player, additions_per_bubble, deletions_per_bubble):
            if not isinstance(d, dict):
                raise TypeError("AdditionsAndDeletions expects four dicts and a list")
        if not isinstance(user_ships, list):
            raise TypeError("AdditionsAndDeletions expects four dicts and a list")
        bubbles = self._lane.bubbles
        interactives = self._lane.bubble_interactives
        balls = self._lane.balls
        additions_per_player.clear()
        deletions_per_player.clear()
        additions_per_bubble.clear()
        deletions_per_bubble.clear()
        for key1, bubble in list(bubbles.items()):                          # Thunkers.cpp:2627-2645
            if key1 not in interactives:
                continue
            bubble_add_list = []
            additions_per_bubble[key1] = bubble_add_list
            bubble_del_list = []
            deletions_per_bubble[key1] = bubble_del_list
            self._populate_add_del_for_bubble(bubble, bubble_add_list, bubble_del_list)
        transitions: list = []
        for val in user_ships:                                              # Thunkers.cpp:2657-2889
            if isinstance(val, float) or not hasattr(val, "__index__"):
                raise TypeError(f"Value is not __int64: {val!r}")
            user_id = val.__index__()
            u_ball = balls.get(user_id)
            if u_ball is None:
                continue
            bubble = bubbles.get(u_ball.bubble) if u_ball.bubble != -1 else None
            old_bubble = bubbles.get(u_ball.old_bubble) if u_ball.old_bubble != -1 else None
            same = u_ball.bubble == u_ball.old_bubble and u_ball.bubble != -1 and bubble is not None
            if same:
                bubble_id = u_ball.bubble
                if bubble_id in additions_per_bubble and bubble_id in deletions_per_bubble:
                    additions_per_player[val] = bubble_id
                    deletions_per_player[val] = bubble_id
                else:                                       # no lists pre-built: the pilot boarded in space
                    add_list, del_list = [], []
                    self._populate_add_del_for_bubble(bubble, add_list, del_list)
                    additions_per_player[val] = add_list
                    deletions_per_player[val] = del_list
            else:
                add_list = []
                additions_per_player[val] = add_list
                del_list = []
                deletions_per_player[val] = del_list
                for key, action in (list(bubble.items()) if bubble is not None else []):   # :2757-2817
                    ball_id = key
                    if ball_id < DSTLOCALBALLS:             # no miniballs
                        continue
                    ball = balls.get(ball_id)
                    if ball is not None and (ball.is_global or ball.is_cloaked):
                        continue
                    was_in = old_bubble is not None and key in old_bubble
                    if action >= 0 and not was_in:
                        add_list.append(ball_id)
                    elif action == -1 and (was_in or old_bubble is None):
                        del_list.append(ball_id)
                for key, action in (list(old_bubble.items()) if old_bubble is not None else []):   # :2820-2883
                    ball_id = key
                    if ball_id < DSTLOCALBALLS:
                        continue
                    ball = balls.get(ball_id)
                    if action != -1:
                        if ball is None:
                            continue
                        if ball.is_global or ball.is_cloaked:
                            continue
                    else:
                        if ball is not None and ball.is_cloaked:
                            continue
                    is_in = bubble is not None and key in bubble
                    if not is_in and action <= 0:
                        del_list.append(ball_id)
                self._add_transition_to_list(u_ball, transitions)
            u_ball.old_bubble = u_ball.bubble                               # Thunkers.cpp:2887
        self._notify_of_bubble_transitions(transitions)

    # ---- the ball streams (Thunkers.cpp:2083-2258, 2463-2507, 2897-3377) ----

    def _write_ball_to_stream(self, b, s: MemStream) -> None:
        """Ballpark::WriteBallToStream (Thunkers.cpp:3180-3377): the C++'s fields in its order and sizes,
        little-endian, unpadded."""
        if b.is_moribund:
            return
        do = is_dynamical_orientation_enabled()
        mini_balls = list(getattr(b, "mini_balls", ()))
        mini_capsules = list(getattr(b, "mini_capsules", ()))
        mini_boxes = list(getattr(b, "mini_boxes", ()))
        flags = ((DSTBALL_ISFREE if b.is_free else 0) | (DSTBALL_ISGLOBAL if b.is_global else 0)
                 | (DSTBALL_ISMASSIVE if b.is_massive else 0) | (DSTBALL_ISINTERACTIVE if b.is_interactive else 0)
                 | (DSTBALL_ISSPACEJUNK if b.is_space_junk else 0))
        if len(mini_balls):
            flags |= DSTBALL_HASMINIBALLS
        if len(mini_capsules):
            flags |= DSTBALL_HASMINICAPSULES
        if len(mini_boxes):
            flags |= DSTBALL_HASMINIBOXES
        mode = b.mode & 0xFF
        s.Write(struct.pack("<qBf3dB", b.id, mode, b.radius, *b.pos, flags))
        if mode != DSTBALL_RIGID:
            s.Write(struct.pack("<dbqii", b.mass, _int8(b.is_cloaked), b.harmonic, _int32(b.corporation_id),
                                _int32(b.alliance_id)))
        if flags & DSTBALL_ISFREE:
            s.Write(struct.pack("<f3dff", b.max_vel, *b.vel, b.agility, b.speed_fraction))
            if do:
                rot = _lane_field(b, "rot")
                ang_vel = _lane_field(b, "ang_vel")
                s.Write(struct.pack("<4ff3ff", *rot, _lane_field(b, "max_ang_vel"), *ang_vel,
                                    _lane_field(b, "rot_agility")))
        s.Write(struct.pack("<b", _int8(getattr(b, "formation_id", -1))))
        effect_stamp = _int32(b.effect_stamp)
        if mode == DSTBALL_FOLLOW:
            s.Write(struct.pack("<qf", b.follow_id, b.follow_range))
        elif mode == DSTBALL_FORMATION:
            s.Write(struct.pack("<qfi", b.follow_id, b.follow_range, effect_stamp))
        elif mode == DSTBALL_MISSILE:
            s.Write(struct.pack("<qfqi3d", b.follow_id, b.follow_range, b.owner_id, effect_stamp, *b.goto))
        elif mode == DSTBALL_ORBIT:
            s.Write(struct.pack("<qf", b.follow_id, b.follow_range))
        elif mode == DSTBALL_GOTO:
            s.Write(struct.pack("<3d", *b.goto))
        elif mode == DSTBALL_WARP:
            s.Write(struct.pack("<3didqq", *b.goto, effect_stamp, b.last_collision, b.follow_id, b.owner_id))
        elif mode == DSTBALL_MUSHROOM:
            s.Write(struct.pack("<fdiq", b.follow_range, b.goto[0], effect_stamp, b.owner_id))
        elif mode == DSTBALL_TROLL:
            s.Write(struct.pack("<i", effect_stamp))
        elif mode in (DSTBALL_STOP, DSTBALL_FIELD, DSTBALL_RIGID):
            pass
        else:
            logger.error("WriteBallToStream: Unknown ball mode %d in dump", mode)
            return                                          # (sic) the header is already written
        if mini_balls:
            s.Write(struct.pack("<H", len(mini_balls) & 0xFFFF))
        for mb in mini_balls:
            s.Write(struct.pack("<3df", *mb.pos, mb.radius))
        if mini_capsules:
            s.Write(struct.pack("<H", len(mini_capsules) & 0xFFFF))
        for mc in mini_capsules:
            s.Write(struct.pack("<3d3df", *mc.a, *mc.b, mc.radius))
        if mini_boxes:
            s.Write(struct.pack("<H", len(mini_boxes) & 0xFFFF))
        for mx in mini_boxes:
            s.Write(struct.pack("<3d3d3d3d", *mx.corner, *mx.lx, *mx.ly, *mx.lz))

    def WriteBallsToStream(self, ball_list, s: MemStream):
        """PyWriteBallsToStream (Thunkers.cpp:2202-2257): DESTINY_BALLS, the time as int32, then the listed balls."""
        if not isinstance(ball_list, list):
            raise TypeError("WriteBallsToStream expects a list")
        s.Seek(0)
        s.Write(struct.pack("<Bi", DESTINY_BALLS, _int32(self._lane.current_time)))
        for val in ball_list:
            if not hasattr(val, "__index__") or isinstance(val, float):
                continue                                    # a bogus id: skipped
            ball = self._lane.balls.get(val.__index__())
            if ball is None:
                continue
            self._write_ball_to_stream(ball, s)
        s.Seek(0)

    def WriteFullStateToStream(self, s: MemStream, src_id=-1):
        """PyWriteFullStateToStream (Thunkers.cpp:2145-2200): DESTINY_FULLSTATE, the time, then every ball not below
        DSTLOCALBALLS; on the server with a source ball, only its bubble's and global balls, and no cloaked ball but
        itself."""
        (src_id,) = _parse_tuple((src_id,), "L")
        s.Seek(0)
        s.Write(struct.pack("<Bi", DESTINY_FULLSTATE, _int32(self._lane.current_time)))
        src = self._lane.balls.get(src_id) if src_id != -1 else None
        for b in list(self._lane.balls.values()):
            if b.id < DSTLOCALBALLS:
                continue
            if src is not None and self._lane.is_master:
                if src.bubble != b.bubble and not b.is_global:
                    continue
                if b.is_cloaked and b is not src:
                    continue
            self._write_ball_to_stream(b, s)
        s.Seek(0)

    def ReadFullStateFromStream(self, s: MemStream, partial=0):
        """PyReadFullStateFromStream (Thunkers.cpp:2083-2143) with ReadFullStateFromStream (2463-2507)."""
        (partial,) = _parse_tuple((partial,), "i")
        if not partial:
            self._lane.free_balls.clear()
            self._lane.proximity_balls.clear()
            self._lane.bubble_interactives.clear()
        s.Seek(0)
        while True:
            packet = s.Read(1)
            if not packet:
                break
            if packet[0] not in (DESTINY_FULLSTATE, DESTINY_BALLS):
                raise ValueError("Unknown packet type")
            (timestamp,) = struct.unpack("<i", _read(s, 4))
            self._lane.current_time = timestamp
            loaded = []
            while True:
                ball = self._read_ball_from_stream(s, partial)
                if ball is None:
                    break
                loaded.append(ball)
            for b in loaded:
                if b.mode in (DSTBALL_FOLLOW, DSTBALL_MISSILE, DSTBALL_ORBIT, DSTBALL_FORMATION):
                    dest = self._lane.balls.get(b.follow_id)
                    if dest is None:
                        logger.error("ball %s to follow not found for ball %s", b.follow_id, b.id)
                        b.mode = DSTBALL_GOTO
                    else:
                        b.follow_ptr = dest.id
                        dest.followers.add(b.id)

    def _read_ball_from_stream(self, s: MemStream, partial: int):
        """Ballpark::ReadBallFromStream (Thunkers.cpp:2897-3178)."""
        raw = s.Read(8)
        if len(raw) == 0:
            return None
        (bid,) = struct.unpack("<q", raw.ljust(8, b"\0"))
        mass, is_cloaked, max_vel, vel, agility, sf = 1.0e34, 0, 0.0, (0.0, 0.0, 0.0), 1.0, 0.0
        harmonic, corp, alliance = -1, -1, -1
        mode, radius, x, y, z, flags = struct.unpack("<Bf3dB", _read(s, 30))
        if mode != DSTBALL_RIGID:
            mass, is_cloaked, harmonic, corp, alliance = struct.unpack("<dbqii", _read(s, 25))
        if flags & DSTBALL_ISFREE:
            max_vel, vx, vy, vz, agility, sf = struct.unpack("<f3dff", _read(s, 36))
            vel = (vx, vy, vz)
            if is_dynamical_orientation_enabled():
                raise NotImplementedError("reading a dynamically oriented ball needs this port's "
                                          "AddDynamicallyOrientedBall")
        ball = self._lane.add_ball(bid, mass, radius, max_vel, bool(flags & DSTBALL_ISFREE),
                                   bool(flags & DSTBALL_ISGLOBAL), bool(flags & DSTBALL_ISMASSIVE),
                                   bool(flags & DSTBALL_ISINTERACTIVE), bool(flags & DSTBALL_ISSPACEJUNK), x, y, z,
                                   *vel, agility, sf)
        (formation_id,) = struct.unpack("<b", _read(s, 1))
        if hasattr(ball, "formation_id"):
            ball.formation_id = formation_id
        elif formation_id != -1:
            raise NotImplementedError("this port's ball has no formation_id")
        if partial != 1 and not (flags & DSTBALL_ISFREE):
            for name, remove in (("mini_balls", "remove_ball"), ("mini_capsules", "remove_capsule"),
                                 ("mini_boxes", "remove_oriented_box")):
                minis = getattr(ball, name, None)
                if minis:
                    for m in reversed(list(minis)):
                        self._call(remove, m.id)
                    minis.clear()
        ball.harmonic, ball.corporation_id, ball.alliance_id = harmonic, corp, alliance
        ball.mode = mode                                    # set directly: no mode-change event
        ball.is_cloaked = is_cloaked & 0xFF
        if mode in (DSTBALL_FOLLOW, DSTBALL_ORBIT):
            ball.follow_id, ball.follow_range = struct.unpack("<qf", _read(s, 12))
        elif mode == DSTBALL_FORMATION:
            ball.follow_id, ball.follow_range, ball.effect_stamp = struct.unpack("<qfi", _read(s, 16))
        elif mode == DSTBALL_MISSILE:
            fid, frange, owner, stamp, gx, gy, gz = struct.unpack("<qfqi3d", _read(s, 48))
            ball.follow_id, ball.follow_range, ball.owner_id, ball.effect_stamp = fid, frange, owner, stamp
            ball.goto = [gx, gy, gz]
        elif mode == DSTBALL_GOTO:
            ball.goto = list(struct.unpack("<3d", _read(s, 24)))
        elif mode == DSTBALL_WARP:
            gx, gy, gz, stamp, length, fid, owner = struct.unpack("<3didqq", _read(s, 52))
            ball.goto, ball.effect_stamp, ball.last_collision = [gx, gy, gz], stamp, length
            ball.follow_id, ball.owner_id = fid, owner
        elif mode == DSTBALL_MUSHROOM:
            frange, span, stamp, owner = struct.unpack("<fdiq", _read(s, 24))
            ball.follow_range, ball.effect_stamp, ball.owner_id = frange, stamp, owner
            ball.goto = [span, ball.goto[1], ball.goto[2]]
        elif mode == DSTBALL_TROLL:
            (ball.effect_stamp,) = struct.unpack("<i", _read(s, 4))
        elif mode in (DSTBALL_STOP, DSTBALL_FIELD, DSTBALL_RIGID):
            pass
        else:
            logger.error("ReadBallFromStream: Unknown ball mode %d in dump", mode)
            return None                                     # (sic) the ball stays added; reading stops here
        if flags & DSTBALL_HASMINIBALLS:
            (cnt,) = struct.unpack("<H", _read(s, 2))
            for _ in range(cnt):
                cx, cy, cz, r = struct.unpack("<3df", _read(s, 28))
                if partial != 1:
                    self._call("add_mini_ball", ball.id, cx, cy, cz, r)
        if flags & DSTBALL_HASMINICAPSULES:
            (cnt,) = struct.unpack("<H", _read(s, 2))
            for _ in range(cnt):
                ax, ay, az, bx, by, bz, r = struct.unpack("<3d3df", _read(s, 52))
                if partial != 1:
                    self._call("add_mini_capsule", ball.id, ax, ay, az, bx, by, bz, r)
        if flags & DSTBALL_HASMINIBOXES:
            (cnt,) = struct.unpack("<H", _read(s, 2))
            for _ in range(cnt):
                v = struct.unpack("<3d3d3d3d", _read(s, 96))
                if partial != 1:
                    self._call("add_mini_box", ball.id, *v)
        return ball


def _read(s: MemStream, n: int) -> bytes:
    """IBlueStream::Read of n bytes; a short read leaves the rest as it was (zero here)."""
    return s.Read(n).ljust(n, b"\0")


def _int8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v >= 128 else v


def _int32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _not_in_lane(name: str):
    def method(self, *args, **kwargs):
        raise NotImplementedError(f"Ballpark.{name} is not in this port's ballpark")
    method.__name__ = name
    return method


for _name in BLUE_METHODS:                                  # a Blue object's `_parent_<method>` for each method
    if not hasattr(DestinyPark, _name):
        setattr(DestinyPark, _name, _not_in_lane(_name))
    setattr(DestinyPark, "_parent_" + _name, getattr(DestinyPark, _name))
del _name
