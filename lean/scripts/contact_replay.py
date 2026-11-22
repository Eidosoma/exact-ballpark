# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Replay fixed-ball selection and free-ball repeat handling through the scalar engine and the pinned space harness."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from fractions import Fraction
from io import StringIO
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from discrepancy_sources import checkout_destiny
from exact_ballpark.full import Capsule, FullBallpark
from exact_ballpark.space import CollisionBallProperties, CollisionImpact, CollisionItem, FUZZY_DELTA, SpaceBallpark


def word(x):
    return struct.pack(">d", x).hex()


class EarliestFixed(FullBallpark):
    """Use the engine's single-obstacle sweep, then compare its absolute contact times."""

    def _next_collision_static(self, path, item, balls, collidables=()):
        assert not collidables and all(not ball.is_free for ball in balls)
        for other in balls:
            candidate = CollisionItem()
            super()._next_collision_static(path, candidate, [other])
            if candidate.collider is not None and (item.time < 0 or candidate.time < item.time):
                for field in CollisionItem.__slots__:
                    setattr(item, field, getattr(candidate, field))
                if item.time == path.t_start:
                    break


def fixed_pair(*, compare_seconds=False, initial_sweep=False):
    return fixed_window(0. if initial_sweep else .5, 500 if initial_sweep else 1000,
                        compare_seconds=compare_seconds)


def fixed_window(start, tick_ms, *, compare_seconds):
    """Keep the chord eight metres long while varying the collision window in seconds."""
    park_type = EarliestFixed if compare_seconds else FullBallpark
    park = park_type(friction=1, iterative=True, tick_interval_ms=tick_ms)
    speed = 8. / (park.dt - start)
    ship = park.add_ball(1, 1, .5, speed, True, is_massive=True, agility=1, vx=speed)
    ship.last_g = [speed, 0., 0.]
    path = park._properties(ship)
    path.t_start = path.locations[0].time = start
    near, far = [park.add_ball(i, 1, .5, 0, x=x, is_massive=True) for i, x in [(3, 3), (4, 5)]]
    if path.position(start)[0][0] != 0 or path.position(park.dt)[0][0] != 8:
        raise SystemExit("The supplied collision window no longer sweeps eight metres")
    results = []
    for candidates in ([near, far], [far, near]):
        item = CollisionItem()
        park._next_collision_static(path, item, candidates)
        results.append((item.collider.id, word(item.time)))
    return results


def rebound():
    """Same one-second evolution twice. Only the control's fixed-ball comparison changes."""
    results = []
    for cls in (FullBallpark, EarliestFixed):
        park = cls(friction=1, iterative=True)
        for i, (x, vx) in enumerate([(5, -16), (0, 0), (7, 0), (8, 0)], 1):
            park.add_ball(i, 16, .5, 16, i == 1, is_massive=True, agility=1,
                          x=100+x, y=100, z=100, vx=vx)
        park.evolve()
        ship = park.balls[1]
        results.append((ship.pos[0], ship.vel[0]))
    return results


REBOUND_SHIP_X = float.fromhex("-0x1.9821b123583d1p+9")
REBOUND_THIRD_X = float.fromhex("-0x1.738a574389bfdp-2")


@dataclass
class ReboundRun:
    contacts: list[tuple[int, str, int, str]]
    endpoint: tuple[str, str]
    history: list[tuple[int, str, str]]
    stored_end: tuple[str, str]
    endpoint_index: int


def rebound_history(control=False, *, ship_x=REBOUND_SHIP_X, third_x=REBOUND_THIRD_X):
    """Start three balls and observe their actual contacts during one 100 ms tick."""
    trace = []

    class ObservedPark(FullBallpark):
        def _impact_static(self, path, item):
            index = path.find_index(item.time)
            super()._impact_static(path, item)
            trace.append((item.collider.id, word(item.time), index, word(item.impact.dv[0])))

    def exact_index(path, query):
        for i, location in enumerate(path.locations):
            if Fraction(location.time) + Fraction(FUZZY_DELTA) >= Fraction(query):
                return max(0, i - 1)
        return len(path.locations) - 1

    park = ObservedPark(friction=1, iterative=True, tick_interval_ms=100)
    xs = (ship_x, float.fromhex("-0x1.6cffb2ca84a00p+0"), third_x)
    for i, x in enumerate(xs, 1):
        park.add_ball(i, 0.0625, 0.015625, 32768, i == 1, is_massive=True, agility=1,
                      x=x, y=100, z=100, vx=16384 if i == 1 else 0)
    lookup = patch.object(CollisionBallProperties, "find_index", exact_index) if control else nullcontext()
    with lookup:
        park.evolve()
        path = park.collision_locations[1]
        endpoint_index = path.find_index(park.dt)
        end_p, end_v = path.position(park.dt)
    ship = park.balls[1]
    endpoint = (word(ship.pos[0]), word(ship.vel[0]))
    if (word(end_p[0]), word(end_v[0])) != endpoint:
        raise SystemExit("The final history lookup differs from the committed three-ball state")
    last = path.locations[-1]
    history = [(loc.collider, word(loc.time), word(loc.v[0]))
               for loc in path.locations[1:-1]]
    return ReboundRun(trace, endpoint, history, (word(last.p[0]), word(last.v[0])), endpoint_index)


def detected_rebound(control=False):
    """The contact trace, committed endpoint and impact words used by the Lean witness."""
    run = rebound_history(control)
    return run.contacts, run.endpoint, run.history


def free_repeat():
    """A supplied prior contact and a reversal split one free-ball search into two chords."""
    park = FullBallpark(friction=1, iterative=True)
    a, b = [park.add_ball(i, 1, .5, 16, True, is_massive=True, agility=1, x=x, vx=v)
            for i, x, v in [(1, 0, 0), (2, 1, 4)]]
    query = park._properties(a)
    other = park._properties(b)
    for path, time, impulse, collider in [(query, 0., 0., 2), (other, .5, -12., 3)]:
        impact = CollisionImpact(time)
        impact.dv[0], impact.collider = impulse, collider
        path.add_impact(impact)
        path.calculate_end()
    whole, final_leg = CollisionItem(), CollisionItem()
    park._next_collision_free(query, whole, [b])
    query.t_start = .5
    park._next_collision_free(query, final_leg, [b])
    if whole.collider is not None or final_leg.collider is not b:
        raise SystemExit("Free-ball repeat fixture changed")
    return whole.time, final_leg.time


def static_order():
    park = FullBallpark(friction=1, iterative=True)
    ship = park.add_ball(1, 1, .5, 16, True, is_massive=True, agility=1, vx=16)
    ship.last_g = [16., 0., 0.]
    path = park._properties(ship)
    path.t_start = path.locations[0].time = .5
    shapes = []
    for i, x in [(3, 3.), (4, 3.), (5, 5.), (6, 0.)]:
        shape = Capsule()
        shape.initialize(i, 99, (x,-1,0), (x,1,0), .5)
        shapes.append(shape)
    fixed = park.add_ball(7, 1, .5, 0, x=1, is_massive=True)
    for indices, balls, expected in [([2,0,1], [], (3,.625)), ([1,0], [], (4,.625)),
                                     ([3,0], [fixed], (6,.5))]:
        item = CollisionItem()
        park._next_collision_static(path, item, balls, [shapes[i] for i in indices])
        if (item.collider.id, item.time) != expected:
            raise SystemExit(f"Unexpected static shape selection for {indices}")


# Uses conformance/build_space.py's pinned function extraction and its Ballpark.
# Both the half-window selector and the full tick run the original collision loops.
DRIVER = r"""
// Evolve commits the real tick. The same inputs in a second park expose the
// raw and limited impulses, which Destiny does not keep in the finished history.
static bool one_impact_tick(const char* name, double rightMass, float leftMax) {
    auto setup = [&](Ballpark& park) {
        park.mFriction = park.dt = 1;
        Ball* a = park.AddBall(1,1,0.5f,leftMax,true,false,true,false,false,0,0,0,2,0,0,1,2/leftMax);
        Ball* b = park.AddBall(2,rightMass,0.5f,2,true,false,true,false,false,3,0,0,-2,0,0,1/rightMass,1);
        park.GotoPoint(a,Vector3d(64,0,0));
        park.GotoPoint(b,Vector3d(-61,0,0));
        return std::make_pair(a,b);
    };
    Ballpark live, parts;
    const auto balls = setup(live);
    live.Evolve(0);
    const auto prepared = setup(parts);
    Ball* a = prepared.first;
    Ball* b = prepared.second;
    a->mLastG = parts.GotoThrust(a,a->mGoto,false);
    b->mLastG = parts.GotoThrust(b,b->mGoto,false);
    auto& ap = parts.mCollisionLocations[a->mId];
    auto& bp = parts.mCollisionLocations[b->mId];
    ap.Initialize(a,parts.dt);
    bp.Initialize(b,parts.dt);
    VectorOfBalls candidates{b};
    CollisionItem hit;
    parts.GetNextCollisionFree(ap,hit,candidates,0);
    if (hit.collider != b || !(0 < hit.timeOfImpact && hit.timeOfImpact < parts.dt)) return false;
    Vector3d pa,va,pb,vb;
    ap.CalculatePosition(pa,va,hit.timeOfImpact);
    bp.CalculatePosition(pb,vb,hit.timeOfImpact);
    auto other = parts.CalculateCollisionImpactFree(ap,bp,hit);
    if (!hit.impact || !other.impact || hit.impact->collider != 2 || other.impact->collider != 1) return false;
    const auto rawA = hit.impact->velocityChange, rawB = other.impact->velocityChange;
    parts.AddImpact(*other.impact,0,b);
    parts.AddImpact(*hit.impact,0,a);
    auto emit = [&](Ball* ball, const Vector3d& p, const Vector3d& v, const Vector3d& raw, const Vector3d& applied) {
        const auto& locs = live.mCollisionLocations.at(ball->mId).m_locations;
        if (locs.size() != 3 || locs[0].time != 0 || locs[0].collider != -1 ||
            locs[1].time != hit.timeOfImpact || locs[1].collider != 3-ball->mId ||
            locs[2].time != live.dt || locs[2].collider != -1) return false;
        std::printf("REBOUND %s %lld %a %a",name,(long long)ball->mId,ball->mTimeFactor,locs[1].time);
        for (const auto& value : {ball->mLastG,p,v,raw,applied,locs[1].p,locs[1].v,ball->mNewPos,ball->mNewVel})
            std::printf(" %a %a %a",value.x,value.y,value.z);
        std::printf("\n");
        return true;
    };
    return emit(balls.first,pa,va,rawA,hit.impact->velocityChange) &&
           emit(balls.second,pb,vb,rawB,other.impact->velocityChange);
}

int main() {
    g_useIterativeCollision = true;
    if (!one_impact_tick("equal",1,2) || !one_impact_tick("unequal",2,2) || !one_impact_tick("limited",1,0.5f)) {
        std::fprintf(stderr,"Pinned one-impact tick produced an unexpected contact or history\n");
        return 1;
    }
    Ballpark limited;
    limited.mFriction = limited.dt = 1;
    Ball* right = limited.AddBall(1,1,0.5f,1,true,false,true,false,false,1,0,0,-100,0,0,1,0);
    Ball* left = limited.AddBall(2,1,0.5f,2,true,false,true,false,false,0,0,0,-90,0,0,1,0);
    auto& rp = limited.mCollisionLocations[right->mId];
    auto& lp = limited.mCollisionLocations[left->mId];
    rp.Initialize(right,1);
    lp.Initialize(left,1);
    CollisionItem touch;
    VectorOfBalls touching{left};
    limited.GetNextCollisionFree(rp,touch,touching,0);
    if (touch.collider != left || touch.timeOfImpact != 0) {
        std::fprintf(stderr,"Expected contact at the start of the unequal-cap replay\n");
        return 1;
    }
    auto secondImpact = limited.CalculateCollisionImpactFree(rp,lp,touch);
    if (!secondImpact.impact || !touch.impact) {
        std::fprintf(stderr,"Missing one of the two approaching impulses\n");
        return 1;
    }
    limited.AddImpact(*secondImpact.impact,0,left);
    limited.AddImpact(*touch.impact,0,right);
    std::printf("CAPPED_PAIR %a %a\n",rp.m_locations.back().v.x,lp.m_locations.back().v.x);
    Ballpark turn;
    turn.mFriction = turn.dt = 1;
    Ball* pilot = turn.AddBall(1,1,0.25f,10,true,false,true,false,false,
                               0,0,0,10,0,0,1,1);
    const Vector3d start = pilot->mNewPos;
    const auto thrust = turn.GotoThrust(pilot,Vector3d(0,100,0),false);
    auto position = [&](double dt) {
        auto p = start, v = pilot->mNewVel;
        turn.Integrate(p,v,thrust,pilot->mMass*pilot->mAgility,turn.mFriction,pilot->mTimeFactor,dt);
        return p;
    };
    const auto mid = position(0.5), end = position(1);
    const auto low = position(0.435661), high = position(0.435662);
    const Vector3d crossing(4,1,0), clear(4,-4,0);
    Ball* turningObstacle = turn.AddBall(2,2,0.25f,10,true,false,true,false,false,4,1,0,-0.5,-0.25,0,1,1);
    auto movingEnd = turningObstacle->mNewPos, movingVelocity = turningObstacle->mNewVel;
    turn.Integrate(movingEnd,movingVelocity,Vector3d(0,-1,0),turningObstacle->mMass*turningObstacle->mAgility,
                   turn.mFriction,turningObstacle->mTimeFactor,1);
    std::printf("CURVE %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a\n", pilot->mTimeFactor,
                thrust.x,thrust.y,thrust.z,mid.x,mid.y,mid.z,end.x,end.y,end.z,
                CollideTwoSpheres(start,end,crossing,crossing,0.5), CollideTwoSpheres(start,end,clear,clear,0.5),
                low.x,low.y,low.z,high.x,high.y,high.z);
    std::printf("TWO_TURNS %a %a %a %a\n", movingEnd.x,movingEnd.y,movingEnd.z,
                CollideTwoSpheres(start,end,crossing,movingEnd,0.5));
    auto body = [](Ballpark& p, int id, double x, double vx, double m, bool free, double yz) {
        return p.AddBall(id, m, 0.5f, 16, free, false, true, false, false,
                         x,yz,yz,vx,0,0,1,0);
    };
    for (bool initialSweep : {false, true}) {
        Ballpark park;
        park.mFriction = 1;
        park.mTickInterval = initialSweep ? 500 : 1000;
        park.dt = 0.001 * park.mTickInterval;
        Ball* ship = body(park,1,0,16,1,true,0);
        Ball* near = body(park,3,3,0,1,false,0);
        Ball* far = body(park,4,5,0,1,false,0);
        ship->mLastG = Vector3d(16,0,0);
        CollisionBallProperties path;
        path.Initialize(ship,park.dt);
        path.m_tStart = path.m_locations[0].time = initialSweep ? 0.0 : 0.5;
        VectorOfStaticCollidables statics;
        VectorOfBalls candidates{near,far};
        for (int i=0; i<2; ++i) {
            CollisionItem item;
            park.GetNextCollisionStatic(path,item,candidates,statics);
            std::printf("%s %lld %a\n",initialSweep ? "FIRST_SWEEP" : "W pair",
                        (long long)item.collider->mId,item.timeOfImpact);
            std::reverse(candidates.begin(),candidates.end());
        }
    }
    Ballpark freePair;
    freePair.mFriction = freePair.dt = 1;
    Ball* a = body(freePair,1,0,0,1,true,0);
    Ball* b = body(freePair,2,1,4,1,true,0);
    CollisionBallProperties query;
    query.Initialize(a,1);
    auto& other = freePair.mCollisionLocations[b->mId];
    other.Initialize(b,1);
    CollisionImpact contact{};
    contact.timeOfImpact = 0; contact.collider = 2;
    contact.velocityChange = Vector3d(0,0,0);
    query.AddImpact(contact); query.CalculateEnd();
    contact.timeOfImpact = 0.5; contact.collider = 3;
    contact.velocityChange = Vector3d(-12,0,0);
    other.AddImpact(contact); other.CalculateEnd();
    VectorOfBalls partner{b};
    CollisionItem whole, finalLeg;
    freePair.GetNextCollisionFree(query,whole,partner,0);
    query.m_tStart = 0.5;
    freePair.GetNextCollisionFree(query,finalLeg,partner,0);
    std::printf("W free %a %a\n",whole.timeOfImpact,finalLeg.timeOfImpact);
    Ballpark tick;
    tick.mFriction = tick.dt = 1;
    Ball* mover = body(tick,1,105,-16,16,true,100);
    body(tick,2,100,0,16,false,100);
    body(tick,3,107,0,16,false,100);
    body(tick,4,108,0,16,false,100);
    tick.Evolve(0);
    std::printf("W tick %a %a\n",mover->mNewPos.x,mover->mNewVel.x);
    Ballpark cutoff;
    cutoff.mFriction = 1;
    cutoff.mTickInterval = 100;
    cutoff.dt = cutoff.mTickInterval * 0.001;
    const double xs[] = {-0x1.9821b123583d1p+9, -0x1.6cffb2ca84a00p+0, -0x1.738a574389bfdp-2};
    Ball* moving = nullptr;
    for (int i=0; i<3; ++i) {
        Ball* b = cutoff.AddBall(i+1, 0.0625, 0.015625f, 32768, i==0, false, true, false, false,
                                 xs[i],100,100,i==0 ? 16384 : 0,0,0,1,0);
        if (i==0) moving = b;
    }
    cutoff.Evolve(0);
    std::printf("W cutoff %a %a\n", moving->mNewPos.x, moving->mNewVel.x);
    const auto& locs = cutoff.mCollisionLocations[1].m_locations;
    std::printf("CUTOFF_END %a %a %zu\n", locs.back().p.x, locs.back().v.x,
                cutoff.mCollisionLocations[1].FindLocationsIndex(cutoff.dt));
    for (size_t j=1; j+1<locs.size(); ++j)
        std::printf("W record %lld %a %a\n", (long long)locs[j].collider, locs[j].time, locs[j].v.x);
}
"""


def upstream(repo):
    import build_space
    import rebound_checks
    from boundary_checks import limited_contact
    from curve_checks import second_turn, turning_edges, turning_sweep

    # GCC 11's libstdc++ exposes asinf in the global namespace on this host.
    prelude = "#include <cmath>\nnamespace std { using ::asinf; }\n" + build_space.PRELUDE_A
    driver = build_space.DRIVER.replace("int main() {", "int space_driver() {") + DRIVER
    with TemporaryDirectory() as temp, patch.object(build_space, "PRELUDE_A", prelude), patch.object(
            build_space, "DRIVER", driver), redirect_stdout(StringIO()):
        source, build = Path(temp) / "source", Path(temp) / "build"
        checkout_destiny(repo, build_space.PINNED, source)
        build_space.main(source, build)
        lines = subprocess.check_output([build / "harness"], encoding="utf-8").splitlines()
    rebound_checks.compare_cpp(lines)
    capped = [line.split()[1:] for line in lines if line.startswith("CAPPED_PAIR ")]
    velocities = limited_contact()[:2]
    if len(capped) != 1 or [float.fromhex(x) for x in capped[0]] != velocities:
        raise SystemExit(f"Independently limited pair differs from the pinned C++ contact: {capped}")
    cache, thrust, mid, end, times = turning_sweep()
    low, high = turning_edges()
    expected = [word(v) for v in [cache, *thrust, *mid, *end, *times, *low, *high]]
    curve_rows = [[word(float.fromhex(v)) for v in row.split()[1:]] for row in lines if row.startswith("CURVE ")]
    if curve_rows != [expected]:
        raise SystemExit(f"Turning GOTO sweep differs between scalar and pinned C++: {curve_rows}, expected {expected}")
    other_start, other_end, _, _ = second_turn()
    pair_time = SpaceBallpark.collide_two_spheres([0., 0., 0.], end, other_start, other_end, .5)
    pair_words = [[word(float.fromhex(v)) for v in row.split()[1:]] for row in lines if row.startswith("TWO_TURNS ")]
    if pair_words != [[word(v) for v in [*other_end, pair_time]]]:
        raise SystemExit("Two turning bodies differ between scalar and pinned C++")
    actual = [row.split()[2:] for row in lines if row.startswith("W ")]
    near_first, far_first, free, final, cutoff, *records = actual
    run = rebound_history()
    stored = [row.split()[1:] for row in lines if row.startswith("CUTOFF_END ")]
    if len(stored) != 1 or (tuple(word(float.fromhex(v)) for v in stored[0][:2]), int(stored[0][2])) != (
            run.stored_end, run.endpoint_index):
        raise SystemExit("Three-ball stored endpoint or final lookup index differs from pinned C++")
    if (tuple(word(float.fromhex(v)) for v in cutoff) != run.endpoint or
            [(int(i), word(float.fromhex(t)), word(float.fromhex(v))) for i, t, v in records] != run.history):
        raise SystemExit("Detected rebound differs between the scalar and pinned C++ histories or endpoints")
    pairs = [(int(i), word(float.fromhex(t))) for i, t in [near_first, far_first]]
    initial = [(int(i), word(float.fromhex(t))) for _, i, t in
               (row.split() for row in lines if row.startswith("FIRST_SWEEP "))]
    if initial != fixed_pair(initial_sweep=True):
        raise SystemExit(f"Non-unit first-sweep results differ from pinned C++: {initial}")
    tick = [word(float.fromhex(v)) for v in final]
    if pairs != fixed_pair() or tick != [word(v) for v in rebound()[0]]:
        raise SystemExit(f"Fixed-ball CPU mismatch: Destiny {pairs}, {tick}")
    intervals = [word(float.fromhex(t)) for t in free]
    if intervals != [word(v) for v in free_repeat()]:
        raise SystemExit(f"Free-ball CPU mismatch: Destiny {intervals}")
    return pairs, tick, intervals
