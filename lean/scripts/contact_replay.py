# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Replay fixed-ball selection and free-ball repeat handling through the scalar engine and the pinned space harness."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from exact_ballpark.full import Capsule, FullBallpark
from exact_ballpark.space import CollisionImpact, CollisionItem


def word(x):
    return struct.pack(">d", x).hex()


def fixed_pair():
    park = FullBallpark(friction=1, iterative=True)
    ship = park.add_ball(1, 1, .5, 16, True, is_massive=True, agility=1, vx=16)
    ship.last_g = [16., 0., 0.]
    path = park._properties(ship)
    path.t_start = path.locations[0].time = .5
    near, far = [park.add_ball(i, 1, .5, 0, x=x, is_massive=True) for i, x in [(3, 3), (4, 5)]]
    if path.position(.5)[0][0] != 0 or path.position(1.)[0][0] != 8:
        raise SystemExit("Supplied half-tick chord changed")
    results = []
    for candidates in ([near, far], [far, near]):
        item = CollisionItem()
        park._next_collision_static(path, item, candidates)
        results.append((item.collider.id, word(item.time)))
    return results


def rebound():
    """Same one-second evolution twice. Only the control's fixed-ball comparison changes."""
    class EarliestFixed(FullBallpark):
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
int main() {
    g_useIterativeCollision = true;
    Ballpark half;
    half.mFriction = half.dt = 1;
    auto body = [](Ballpark& p, int id, double x, double vx, double m, bool free, double yz) {
        return p.AddBall(id, m, 0.5f, 16, free, false, true, false, false,
                         x,yz,yz,vx,0,0,1,0);
    };
    Ball* ship = body(half,1,0,16,1,true,0);
    Ball* near = body(half,3,3,0,1,false,0);
    Ball* far = body(half,4,5,0,1,false,0);
    ship->mLastG = Vector3d(16,0,0);
    CollisionBallProperties path;
    path.Initialize(ship,1);
    path.m_tStart = path.m_locations[0].time = 0.5;
    VectorOfStaticCollidables statics;
    VectorOfBalls candidates{near,far};
    for (int i=0; i<2; ++i) {
        CollisionItem item;
        half.GetNextCollisionStatic(path,item,candidates,statics);
        std::printf("W pair %lld %a\n",(long long)item.collider->mId,item.timeOfImpact);
        std::reverse(candidates.begin(),candidates.end());
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
}
"""


def upstream(repo):
    import build_space

    # GCC 11's libstdc++ exposes asinf in the global namespace on this host.
    prelude = "#include <cmath>\nnamespace std { using ::asinf; }\n" + build_space.PRELUDE_A
    driver = build_space.DRIVER.replace("int main() {", "int space_driver() {") + DRIVER
    with TemporaryDirectory() as temp, patch.object(build_space, "PRELUDE_A", prelude), patch.object(
            build_space, "DRIVER", driver), redirect_stdout(StringIO()):
        # Isolate the pinned checkout from the supplied clone, including any edits or a different HEAD.
        source, build = Path(temp) / "source", Path(temp) / "build"
        subprocess.run(["git", "clone", "--quiet", "--local", "--shared", "--no-checkout", "--",
                        repo.resolve(), source], check=True)
        subprocess.run(["git", "-C", source, "checkout", "--quiet", "--detach", build_space.PINNED], check=True)
        build_space.main(source, build)
        lines = subprocess.check_output([build / "harness"], encoding="utf-8").splitlines()
    actual = [row.split()[2:] for row in lines if row.startswith("W ")]
    near_first, far_first, free, final = actual
    pairs = [(int(i), word(float.fromhex(t))) for i, t in [near_first, far_first]]
    tick = [word(float.fromhex(v)) for v in final]
    if pairs != fixed_pair() or tick != [word(v) for v in rebound()[0]]:
        raise SystemExit(f"Fixed-ball CPU mismatch: Destiny {pairs}, {tick}")
    intervals = [word(float.fromhex(t)) for t in free]
    if intervals != [word(v) for v in free_repeat()]:
        raise SystemExit(f"Free-ball CPU mismatch: Destiny {intervals}")
    return pairs, tick, intervals
