# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Build the compiled oracle of the flight modes: Destiny's FOLLOW, ORBIT and WARP modes.

    python3 -I conformance/build_modes.py DESTINY_REPO BUILD_DIR

As build.py (whose source, binary and manifest it leaves alone), Destiny's own code is taken from the pinned source
(carbonengine/destiny at 114e89fa, MIT, (c) CCP Games) and never retyped: functions and the mode enum by brace matching
(comments and literals skipped), and a few whole lines (macros, constants, typedefs) by exact match. This time the tick
itself is extracted (`Ballpark::Evolve`, `EvolveBehaviorForBall`), so the phase order, the warp drop-out step and the
follower release run as Destiny wrote them, with the commands (`FollowBall`, `Orbit`, `WarpTo`, `EntityWarpIn`, `Stop`,
`GotoPoint`, `GotoDirection`) and `Ball`'s warp predicates and `SetMode` from Ball.cpp.

A prelude supplies what the extracted code touches: the members, with Destiny's types (float members stay float) and the
constructors' defaults; the global switches at Destiny's defaults (dynamical orientation and iterative collision off;
useNewOrbit set per case); and stubs for what lies outside the slice: collisions (the gradient finds no neighbours),
bubbles (all live), proximity sensors (none), the partition (each `InsertBallInBoxes` call is printed as an event), and
missiles, formations, trolls, mushrooms and dynamical orientation (those stubs abort if ever reached). Python's event
calls (`PostEvent`, `SendEvent`) are printed. A driver reads scenarios on stdin (balls, then commands applied before
given ticks) and prints every ball's state after every tick (%.17g) and every event.

The build directory receives the source, the binary and a manifest: the SHA-256 of every source file read and of every
extracted piece, the compiler, its command and warnings, and the references to the ballpark's mWarpSpeed under src/
(TRV-716). Compiled with g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math, as build.py. The repository is read, never
executed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PINNED = "114e89fa584f0e1a81eb0a679aabf41af8870072"

# (file, name, regular expression for the start of the definition); extracted to the end of the block that follows
BLOCKS = [
    ("src/IDstConstants.h", "enum DSTBALLMODE", r"enum DSTBALLMODE\b"),
    ("src/Ballpark.cpp", "_finite", r"inline float _finite\( double x \)"),
    ("src/Ballpark.cpp", "DynamicalOrientationDisabledForBall", r"bool DynamicalOrientationDisabledForBall\("),
    ("src/Ballpark.cpp", "Ballpark::Evolve", r"void Ballpark::Evolve\("),
    ("src/Ballpark.cpp", "Ballpark::GetOrbitalNormal", r"Vector3d Ballpark::GetOrbitalNormal\("),
    ("src/Ballpark.cpp", "Ballpark::DefaultOrbitalNormal", r"Vector3d Ballpark::DefaultOrbitalNormal\("),
    ("src/Ballpark.cpp", "Ballpark::Integrate", r"void Ballpark::Integrate\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveBehaviorForBall", r"void Ballpark::EvolveBehaviorForBall\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveWarp", r"Vector3d Ballpark::EvolveWarp\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveGoto", r"Vector3d Ballpark::EvolveGoto\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveFollow", r"Vector3d Ballpark::EvolveFollow\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveOrbit", r"Vector3d Ballpark::EvolveOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveNewStyleOrbit", r"Vector3d Ballpark::EvolveNewStyleOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveOldStyleOrbit", r"Vector3d Ballpark::EvolveOldStyleOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveStop", r"void Ballpark::EvolveStop\("),
    ("src/Ballpark.cpp", "Ballpark::CapAcceleration", r"void Ballpark::CapAcceleration\("),
    ("src/Ballpark.cpp", "Ballpark::GotoThrust", r"Vector3d Ballpark::GotoThrust\("),
    ("src/Ballpark.cpp", "Ballpark::OrbitThrust", r"Vector3d Ballpark::OrbitThrust\("),
    ("src/Ballpark.cpp", "Ballpark::GotoThrustFollow", r"Vector3d Ballpark::GotoThrustFollow\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateBallPositionVelocity", r"void Ballpark::CalculateBallPositionVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::FollowBall", r"void Ballpark::FollowBall\("),
    ("src/Ballpark.cpp", "Ballpark::Orbit", r"void Ballpark::Orbit\("),
    ("src/Ballpark.cpp", "Ballpark::WarpTo", r"void Ballpark::WarpTo\("),
    ("src/Ballpark.cpp", "Ballpark::RealWarp", r"void Ballpark::RealWarp\("),
    ("src/Ballpark.cpp", "Ballpark::SetupWarpConstants", r"void Ballpark::SetupWarpConstants\("),
    ("src/Ballpark.cpp", "Ballpark::WarpDistance", r"double Ballpark::WarpDistance\("),
    ("src/Ballpark.cpp", "Ballpark::EntityWarpIn", r"void Ballpark::EntityWarpIn\("),
    ("src/Ballpark.cpp", "Ballpark::GotoDirection(ID)", r"void Ballpark::GotoDirection\(\s*const ID& srcId,"),
    ("src/Ballpark.cpp", "Ballpark::GotoDirection(Ball*)", r"void Ballpark::GotoDirection\(\s*Ball \*ball,"),
    ("src/Ballpark.cpp", "Ballpark::GotoPoint(ID)", r"void Ballpark::GotoPoint\(\s*const ID& srcId,"),
    ("src/Ballpark.cpp", "Ballpark::GotoPoint(Ball*)", r"void Ballpark::GotoPoint\(\s*Ball \*ball,"),
    ("src/Ballpark.cpp", "Ballpark::Stop(ID)", r"void Ballpark::Stop\(\s*const ID& srcId\s*\)"),
    ("src/Ballpark.cpp", "Ballpark::Stop(Ball*)", r"void Ballpark::Stop\(\s*Ball \*ball\s*\)"),
    ("src/Ballpark.cpp", "Ballpark::SetBallTimeFactor", r"void Ballpark::SetBallTimeFactor\("),
    ("src/Ballpark.cpp", "Ballpark::StopAllFollowers", r"void Ballpark::StopAllFollowers\("),
    ("src/Ball.cpp", "Ball::SetMode", r"void Ball::SetMode\("),
    ("src/Ball.cpp", "Ball::IsWarpish", r"bool Ball::IsWarpish\(\)"),
    ("src/Ball.cpp", "Ball::IsAligningForWarp", r"bool Ball::IsAligningForWarp\(\)"),
    ("src/Ball.cpp", "Ball::IsWarping", r"bool Ball::IsWarping\(\)"),
    ("src/Ball.cpp", "Ball::IsAlignedForWarp", r"bool Ball::IsAlignedForWarp\(\)"),
]

# (file, name, the line's start after leading whitespace); the whole line is copied
LINES = [
    ("src/Ball.h", "typedef VectorOfBalls", "typedef std::vector<Ball *> VectorOfBalls;"),
    ("src/Ball.h", "typedef ID", "typedef int64_t ID;"),
    ("src/Ball.h", "typedef VectorOfIDs", "typedef std::vector<ID> VectorOfIDs;"),
    ("src/Ball.h", "typedef SetOfIDs", "typedef std::unordered_set<ID> SetOfIDs;"),
    ("src/Ball.h", "MAX_ALIGN_TICKS", "#define MAX_ALIGN_TICKS"),
    ("src/Ballpark.h", "AU", "const double AU ="),
    ("src/Ballpark.h", "typedef DictOfFreeBalls", "typedef std::map<ID, Ball *> DictOfFreeBalls;"),
    ("src/Ballpark.cpp", "SIGNUM (Ballpark.cpp)", "#define SIGNUM(X)"),
    ("src/Ballpark.cpp", "ABS (Ballpark.cpp)", "#define ABS(X)"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_AU_PER_SECOND", "double WARP_FACTOR_TO_AU_PER_SECOND"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_ACCELERATION", "double WARP_FACTOR_TO_ACCELERATION"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_DECELERATION", "double WARP_FACTOR_TO_DECELERATION"),
    ("src/Ballpark.cpp", "followModes", "extern const std::array<int,4> followModes"),
    ("src/Ballpark.cpp", "ORBITAL_PRECESSION", "double ORBITAL_PRECESSION"),
    ("src/Ball.cpp", "ABS (Ball.cpp)", "#define ABS(X)"),
]

PRELUDE_A = r"""// Harness prelude: what the extracted Destiny code touches, with Destiny's types.
#include <math.h>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace Be { typedef int64_t Time; }                   // a 64-bit time stamp
struct Vector3 {                                        // Blue's float vector: used only in branches never reached here
    float x, y, z;
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float a, float b, float c) : x(a), y(b), z(c) {}
};
struct Quaternion {                                     // only copied in the dynamical-orientation branches
    float x, y, z, w;
    Quaternion() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    Quaternion(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
};
#define CCP_LOG_CH(...) ((void)0)                       // logging and profiling compiled out
#define CCP_LOGWARN_CH(...) ((void)0)
#define CCP_LOGERR_CH(...) ((void)0)
#define CCP_ASSERT(x) ((void)0)
#define CCP_STATS_ZONE(x)
#define TASKLET(A)
struct IEveBallpark {};                                 // event targets (PyOS below)
struct IBall {};
struct Ball;
struct Ballpark;
"""

PRELUDE_B = r"""
// ---- prelude, continued ----
static bool g_useNewOrbit = false;                      // Settings.h's defaults; useNewOrbit is set per case
static bool g_useDynamicalOrientation = false;
static bool g_useIterativeCollision = false;
static bool g_disableDynamicalOrientationForMissiles = false;
static long long g_case = 0;                            // the driver's case id, printed with every event

struct QuaternionD {                                    // CapAcceleration's code after its `return;`: never run
    QuaternionD(const Vector3d&, double) {}
    Vector3d RotateVector(const Vector3d&) const { std::abort(); }
};
struct Matrix {};                                       // the dynamical-orientation helpers: never reached
inline Matrix RotationMatrix(const Quaternion&) { std::abort(); }
inline Vector3 TransformCoord(const Vector3&, const Matrix&) { std::abort(); }
inline float Dot(const Vector3&, const Vector3&) { std::abort(); }
inline float LengthSq(const Quaternion&) { std::abort(); }
inline float LengthSq(const Vector3&) { std::abort(); }
inline Vector3 operator*(const Vector3&, float) { std::abort(); }

struct ProximitySensor { bool active = false; };        // Ball.h: only `active` is touched here

struct Ball : public IBall {
    ID mId = 0;                                         // Partitionable.h, with its constructor's defaults
    Vector3d mNewPos, mOldPos;
    long mNewBubble = -1;
    bool isInteractive = false;
    long mEffectStamp = 0;
    bool isMoribund = false;
    ID mFollowId = 0;                                   // Ball.h, with Ball's constructor's defaults
    ID mOwnerId = 0;
    Ballpark *mPark = nullptr;
    Ball *mFollowPtr = nullptr;
    uint8_t isCloaked = 0;
    bool isFree = false;
    bool isMassive = false;
    bool mInDeadBubble = false;
    float mRadius = 0.0f;
    float mMaxVel = 0.0f;
    float mAgility = 1.0f;
    float mRoll = 0.0f;
    float mRotAgility = 0.9f;
    float mSpeedFraction = 1.0f;
    float mFollowRange = 10.0f;
    double mMass = 0.0;
    double mTimeFactor = 1.0;
    double mLastCollision = -1.0;
    Vector3d mNewVel, mOldVel, mGoto;
    Vector3 mNewAngVel, mOldAngVel, mTorque;
    Vector3d mLastG, mLastC;
    Quaternion mNewRot, mOldRot;
    DSTBALLMODE mMode = DSTBALL_STOP;
    Be::Time mNewTime = 0, mOldTime = 0;
    SetOfIDs mFollowers;
    VectorOfIDs mCollisions;
    ProximitySensor mSensor;
    void SetMode(DSTBALLMODE mode);                     // extracted from Ball.cpp below
    bool IsWarpish();
    bool IsAligningForWarp();
    bool IsWarping();
    bool IsAlignedForWarp();
    void CalculateYawPitchRoll(bool snap = false) {}    // orientation only, which no mode here reads back
    void SnapOrientation() { std::abort(); }            // dynamical orientation, missiles, formations: never reached
    void DispatchCollisions() { std::abort(); }
    void FreeFormationSlot(size_t) { std::abort(); }
    void GetRotatedVector(Vector3&) { std::abort(); }
};

struct MapOfBalls {                                     // MapOfBalls.h: operator[] finds, and never inserts
    std::map<ID, Ball*> balls;
    Ball* operator[](const ID id) { auto it = balls.find(id); return it == balls.end() ? nullptr : it->second; }
};
struct CollisionLocation { ID collider; };              // the iterative-collision store, always empty here
struct CollisionBallProperties {
    std::vector<CollisionLocation> m_locations;
    void CalculatePosition(Vector3d&, Vector3d&, double) { std::abort(); }
};

struct Ballpark : public IEveBallpark {
    int mTickInterval = 1000;                           // Ballpark.h, with the constructor's defaults
    ID mProbe = 0;
    long mCurrentTime = 0;
    double mFriction = 1000000.0;
    double dt = 1.0;
    MapOfBalls mBalls;
    bool isMaster = false;
    bool inEvolve = false;
    DictOfFreeBalls mFreeBalls;
    std::unordered_map<ID, CollisionBallProperties> mCollisionLocations;
    // extracted below, declared as in Ballpark.h
    void Evolve(Be::Time timestamp);
    void EvolveBehaviorForBall(Ball* ball);
    Vector3d EvolveWarp(Ball* ball);
    Vector3d EvolveGoto(Ball* ball);
    Vector3d EvolveFollow(Ball* ball);
    Vector3d EvolveOrbit(Ball* ball, long currentTime);
    Vector3d EvolveOldStyleOrbit( Ball* ball, long currentTime );
    Vector3d EvolveNewStyleOrbit( Ball* ball, long currentTime );
    void EvolveStop(Ball* ball);
    void CapAcceleration(const Ball *ball, Vector3d& a);
    void SetBallTimeFactor(Ball *ball);
    void StopAllFollowers(Ball *ball);
    void Integrate(Vector3d& p, Vector3d& v, const Vector3d& a, double mass, double k, double timefactor, double step);
    Vector3d GotoThrust(const Ball *b,const Vector3d& target,bool missile = false);
    Vector3d OrbitThrust( const Ball* b, const Vector3d& target );
    Vector3d GotoThrustFollow( const Ball* b, const Vector3d& target, const Vector3d& targetVelocity, double weight );
    Vector3d GetOrbitalNormal( Ball* ball, Vector3d& toVector, double& dist );
    Vector3d DefaultOrbitalNormal( Ball* ball );
    void CalculateBallPositionVelocity(Ball* ball, double t, Vector3d& position, Vector3d& velocity);
    void FollowBall(const ID& srcId, const ID& dstId, float range);
    void Orbit(const ID& srcId, const ID& dstId, float range);
    void WarpTo(const ID& srcId, double x, double y, double z, double minRange, int warpFactor);
    void RealWarp(Ball *ball);
    void EntityWarpIn(const ID& srcId, double x, double y, double z, int warpFactor);
    double WarpDistance(Ball *b,Vector3d& p, Vector3d& v,double t,bool interpolating=false);
    void SetupWarpConstants(double warpFactor, double warpDistance, double &accelDuration, double &cruiseDuration,
                            double &decelDuration, double &accelDistance, double &cruiseDistance,
                            double &decelDistance, double &accelRate, double &decelRate, double &warpSpeed);
    void GotoPoint(const ID& srcId, double x, double y, double z);
    void GotoPoint(Ball *ball, const Vector3d& p);
    void GotoDirection(const ID& srcId, double x, double y, double z);
    void GotoDirection(Ball *ball, Vector3d dir);
    void Stop(const ID& srcId);
    void Stop(Ball *ball);
    // outside this slice
    bool InDeadBubble(Ball*) { return false; }          // bubbles: every bubble is live
    void Gradient(Ball*) {}                             // collisions: no neighbours, so mLastC stays zero
    void HandleProximities() {}                         // proximity sensors: none
    void InsertBallInBoxes(Ball* ball);                 // the partition: each call is printed as an event
    void CalculateIterativeCollisionResponses() { std::abort(); }   // switched off or modes absent: never reached
    bool TrollReady(Ball*) { std::abort(); }
    void PetrifyTroll(Ball*) { std::abort(); }
    void RemoveBall(const ID&, int delay = 0) { std::abort(); }
    Vector3d EvolveMissile(Ball*) { std::abort(); }
    Vector3d EvolveFormation(Ball*) { std::abort(); }
    Vector3 GotoTorqueBreaks(const Ball*, const Vector3&) { std::abort(); }
    float GotoRollBreaks(const Ball*, const Vector3&) { std::abort(); }
    void ApplyTorque(Quaternion&, Vector3&, const Vector3&, const float, const float, const float) { std::abort(); }
    void CalculateBallRotationVelocity(Ball*, double, Quaternion&, Vector3&) { std::abort(); }
};

struct PyOSStub {                                       // Python's event calls: recorded and printed
    bool PostEvent(IEveBallpark* park, const char* name, const char* method, const char* fmt, ...);
    bool SendEvent(IBall* ball, const char* name, const char* method, void* kw, const char* fmt, ...);
    void PyError() {}
};
static PyOSStub g_pyos;
static PyOSStub* const PyOS = &g_pyos;
"""

PRELUDE_C = r"""
// ---- prelude: the event printers ----
static int read_args(const char* fmt, va_list ap, long long out[2]) {
    int n = 0;
    for (const char* f = fmt; *f && n < 2; ++f)
        out[n++] = (*f == 'L') ? va_arg(ap, long long) : (long long)va_arg(ap, int);
    return n;
}
bool PyOSStub::PostEvent(IEveBallpark* park, const char*, const char* method, const char* fmt, ...) {
    long long v[2] = {0, 0};
    va_list ap;
    va_start(ap, fmt);
    read_args(fmt, ap, v);
    va_end(ap);
    std::printf("E %lld %ld %s %lld %lld 0\n", g_case, static_cast<Ballpark*>(park)->mCurrentTime, method, v[0], v[1]);
    return true;
}
bool PyOSStub::SendEvent(IBall* ib, const char*, const char* method, void*, const char* fmt, ...) {
    long long v[2] = {0, 0};
    va_list ap;
    va_start(ap, fmt);
    read_args(fmt, ap, v);
    va_end(ap);
    Ball* b = static_cast<Ball*>(ib);
    std::printf("E %lld %ld %s %lld %lld %lld\n", g_case, b->mPark->mCurrentTime, method, (long long)b->mId,
                v[0], v[1]);
    return true;
}
void Ballpark::InsertBallInBoxes(Ball* ball) {
    std::printf("E %lld %ld partition_insert %lld 0 0\n", g_case, mCurrentTime, (long long)ball->mId);
}
"""

DRIVER = r"""
// ---- driver: scenarios on stdin; every ball's state after every tick, and every event, on stdout ----
struct Command { int step, op; long long a, b; double x, y, z, r; int w; };

static void print_ball(long long cid, int step, Ball* b) {
    std::vector<long long> fs(b->mFollowers.begin(), b->mFollowers.end());
    std::sort(fs.begin(), fs.end());
    std::printf("S %lld %d %lld %d %ld %d %d %lld %lld %lld "
                "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %zu",
                cid, step, (long long)b->mId, (int)b->mMode, b->mEffectStamp, (int)b->isMassive, (int)b->mSensor.active,
                (long long)b->mFollowId, (long long)b->mOwnerId, b->mFollowPtr ? (long long)b->mFollowPtr->mId : -1LL,
                b->mNewPos.x, b->mNewPos.y, b->mNewPos.z, b->mNewVel.x, b->mNewVel.y, b->mNewVel.z,
                b->mGoto.x, b->mGoto.y, b->mGoto.z, b->mLastG.x, b->mLastG.y, b->mLastG.z,
                (double)b->mFollowRange, b->mLastCollision, fs.size());
    for (long long f : fs) std::printf(" %lld", f);
    std::printf("\n");
}

int main() {
    char tag[8];
    while (std::scanf("%7s", tag) == 1) {
        long long cid; double friction; int tick, steps, newOrbit, master, nb, nk; long start;
        if (tag[0] != 'C' || std::scanf("%lld %lf %d %d %d %d %ld %d %d", &cid, &friction, &tick, &steps, &newOrbit,
                                        &master, &start, &nb, &nk) != 9)
            return 3;
        g_case = cid;
        g_useNewOrbit = newOrbit != 0;
        Ballpark bp;
        bp.mFriction = friction;
        bp.mTickInterval = tick;
        bp.dt = bp.mTickInterval * 0.001;
        bp.mCurrentTime = start;
        bp.isMaster = master != 0;
        std::vector<Ball*> owned;
        for (int i = 0; i < nb; ++i) {
            long long id; double mass, radius, maxvel, agility, sf, px, py, pz, vx, vy, vz; long bubble;
            int fr, ma, in, cl, mo, mode;
            if (std::scanf("%7s %lld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d %d %ld %d %d", tag, &id,
                           &mass, &radius, &maxvel, &agility, &sf, &px, &py, &pz, &vx, &vy, &vz, &fr, &ma, &in, &cl,
                           &bubble, &mo, &mode) != 20 || tag[0] != 'B')
                return 4;
            Ball* b = new Ball();
            b->mId = id; b->mPark = &bp; b->mMass = mass;
            b->mRadius = (float)radius; b->mMaxVel = (float)maxvel; b->mAgility = (float)agility;
            b->mSpeedFraction = (float)sf;
            b->mNewPos = Vector3d(px, py, pz); b->mNewVel = Vector3d(vx, vy, vz);
            b->isFree = fr != 0; b->isMassive = ma != 0; b->isInteractive = in != 0; b->isCloaked = (uint8_t)cl;
            b->mNewBubble = bubble; b->isMoribund = mo != 0; b->mMode = (DSTBALLMODE)mode;
            bp.SetBallTimeFactor(b);
            bp.mBalls.balls[id] = b;
            if (b->isFree) bp.mFreeBalls[id] = b;
            owned.push_back(b);
        }
        std::vector<Command> cmds(nk);
        for (auto& c : cmds)
            if (std::scanf("%7s %d %d %lld %lld %lf %lf %lf %lf %d", tag, &c.step, &c.op, &c.a, &c.b, &c.x, &c.y, &c.z,
                           &c.r, &c.w) != 10 || tag[0] != 'K')
                return 5;
        for (int s = 1; s <= steps; ++s) {
            for (const auto& c : cmds) {
                if (c.step != s) continue;
                const ID a = c.a, b = c.b;
                switch (c.op) {
                    case 1: bp.GotoPoint(a, c.x, c.y, c.z); break;
                    case 2: bp.GotoDirection(a, c.x, c.y, c.z); break;
                    case 3: bp.Stop(a); break;
                    case 4: bp.FollowBall(a, b, (float)c.r); break;
                    case 5: bp.Orbit(a, b, (float)c.r); break;
                    case 6: bp.WarpTo(a, c.x, c.y, c.z, c.r, c.w); break;
                    case 7: bp.EntityWarpIn(a, c.x, c.y, c.z, c.w); break;
                    default: return 6;
                }
            }
            bp.Evolve(0);
            for (auto& kv : bp.mBalls.balls) print_ball(cid, s, kv.second);
        }
        for (Ball* b : owned) delete b;
    }
    return 0;
}
"""


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def block_end(src: str, start: int) -> int:
    """The index just past the brace closing the first block opened at or after `start`; comments, string and
    character literals are skipped."""
    i, n, depth = start, len(src), 0
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i) + 2
            continue
        c = src[i]
        if c in "\"'":
            j = i + 1
            while src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                break
        i += 1
    raise SystemExit(f"unbalanced braces after offset {start}")


def extract_block(src: str, pattern: str, name: str) -> str:
    hits = list(re.finditer(pattern, src))
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} matches for {pattern!r}, not exactly one")
    end = block_end(src, hits[0].end())
    if name.startswith("enum "):                         # keep the declaration's semicolon
        semi = re.match(r"\s*;", src[end:])
        if not semi:
            raise SystemExit(f"{name}: no semicolon after the block")
        end += semi.end()
    return src[hits[0].start():end]


def extract_line(src: str, prefix: str, name: str) -> str:
    hits = [line for line in src.splitlines() if line.lstrip().startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} lines start with {prefix!r}, not exactly one")
    return hits[0]


def head_commit(repo: Path) -> str:
    head = (repo / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):                         # a symbolic HEAD: its loose or packed ref
        ref = head[5:]
        loose = repo / ".git" / ref
        head = loose.read_text().strip() if loose.is_file() else next(
            line.split()[0] for line in (repo / ".git" / "packed-refs").read_text().splitlines()
            if line.endswith(" " + ref))
    return head


def main(repo: Path, build: Path) -> None:
    head = head_commit(repo)
    if head != PINNED:
        raise SystemExit(f"destiny is at {head}, not the pinned {PINNED}")
    files = sorted({f for f, _, _ in BLOCKS} | {f for f, _, _ in LINES} | {"src/Vector3d.h"})
    text = {f: (repo / f).read_text() for f in files}
    blocks = {name: extract_block(text[f], pat, name) for f, name, pat in BLOCKS}
    lines = {name: extract_line(text[f], prefix, name) for f, name, prefix in LINES}
    where = {name: (f, re.search(pat, text[f]).start()) for f, name, pat in BLOCKS}
    park = sorted((n for n in blocks if where[n][0] == "src/Ballpark.cpp" and n != "_finite"),
                  key=lambda n: where[n][1])     # Ballpark.cpp's pieces in source order
    ball = [n for f, n, _ in BLOCKS if f == "src/Ball.cpp"]

    def piece(name: str, body: str, origin: str) -> str:
        return f"\n// ---- {origin}, {name}, verbatim ----\n{body}\n"

    source = (PRELUDE_A
              + piece("enum DSTBALLMODE", blocks["enum DSTBALLMODE"], "src/IDstConstants.h")
              + "".join(piece(n, lines[n], "src/Ball.h") for n in ("typedef VectorOfBalls", "typedef ID",
                                                                   "typedef VectorOfIDs", "typedef SetOfIDs",
                                                                   "MAX_ALIGN_TICKS"))
              + piece("AU", lines["AU"], "src/Ballpark.h")
              + "\n// ---- src/Vector3d.h, verbatim ----\n" + text["src/Vector3d.h"]
              + piece("typedef DictOfFreeBalls", lines["typedef DictOfFreeBalls"], "src/Ballpark.h")
              + PRELUDE_B
              + "".join(piece(n, lines[n], "src/Ballpark.cpp") for n in (
                  "SIGNUM (Ballpark.cpp)", "ABS (Ballpark.cpp)", "WARP_FACTOR_TO_AU_PER_SECOND",
                  "WARP_FACTOR_TO_ACCELERATION", "WARP_FACTOR_TO_DECELERATION", "followModes", "ORBITAL_PRECESSION"))
              + "\nnamespace {   // as in Ballpark.cpp, for compilers other than MSVC\n"
              + piece("_finite", blocks["_finite"], "src/Ballpark.cpp") + "}\n"
              + "".join(piece(n, blocks[n], "src/Ballpark.cpp") for n in park)
              + "\n#undef ABS   // Ball.cpp has its own\n"
              + piece("ABS (Ball.cpp)", lines["ABS (Ball.cpp)"], "src/Ball.cpp")
              + "".join(piece(n, blocks[n], "src/Ball.cpp") for n in ball)
              + PRELUDE_C + DRIVER)
    build.mkdir(parents=True, exist_ok=True)
    (build / "harness.cpp").write_text(source)
    cmd = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-o", str(build / "harness"),
           str(build / "harness.cpp")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"g++ failed with exit status {res.returncode}")
    gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    warp_speed_refs = [f"src/{p.name}:{i}" for p in sorted((repo / "src").iterdir()) if p.is_file()
                       for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1) if "mWarpSpeed" in line]
    manifest = {"destiny_commit": head, "compiler": gxx, "command": cmd,
                "compiler_warnings": [w for w in res.stderr.splitlines() if "warning:" in w],
                "source_sha256": {f: sha(t) for f, t in text.items()},
                "extracted_sha256": {**{n: sha(b) for n, b in blocks.items()}, **{n: sha(s) for n, s in lines.items()}},
                "mWarpSpeed_references_under_src": warp_speed_refs,
                "mWarpSpeed_in_extracted_code": any("mWarpSpeed" in b for b in [*blocks.values(), *lines.values()]),
                "harness_cpp_sha256": sha(source),
                "binary_sha256": sha((build / "harness").read_bytes())}
    (build / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
