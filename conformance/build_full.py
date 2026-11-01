# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Build the compiled oracle of the full ballpark: Destiny's missiles, static collidables
(capsules, oriented boxes, miniballs, minicapsules, miniboxes), cloaking, dynamical orientation and the orientation
itself, mushrooms, trolls, formations, the accuracy helpers and OnTick, around the tick, partition, bubbles, sensors and
collisions of slices 1 and 2.

    python3 -I conformance/build_full.py DESTINY_REPO MATH_REPO BUILD_DIR [DIRECTXMATH_REPO]

As build_space.py (whose source, binary and manifest it leaves alone), Destiny's own code is taken from the pinned
source (carbonengine/destiny at 114e89fa, MIT, (c) CCP Games) and never retyped: functions, classes and enums by brace
matching (comments and literals skipped), and whole lines (macros, constants, typedefs, globals) by exact match. New
here, beside slice 2's pieces: the missile, formation and mushroom modes (`EvolveMissile`, `EvolveFormation`,
`MissileFollow`, `FormationFollow`, `AddMushroom`, `PyLaunchMissile`, `SetBallFormation`, `AddFormation`,
`PyLoadFormations`, `Ball::ReserveFormationSlot`, `Ball::FreeFormationSlot`); trolls and boids (`SetBallTroll`,
`TrollReady`, `PetrifyTroll`, `SetBallRigid`, `PySetBoid`); cloaking (`CloakBall`, `UncloakBall`); the static
collidables (Capsule.cpp, OrientedBox.cpp, BoxShape.cpp, Triangle.cpp, Plane.cpp, AABB.cpp, Collision.cpp in full,
`AddCapsule`, `AddOrientedBox`, `AddMiniball`, `InsertStaticCollidableInBoxes`, `RemoveCapsule`, `RemoveOrientedBox`,
the minis of Ball.cpp and their classes); dynamical orientation (`GotoTorqueBreaks`, `GotoRollBreaks`, `ApplyTorque`,
`CalculateBallRotationVelocity`, `SnapOrientation`, the rotation setters, `AddDynamicallyOrientedBall`); the
orientation without it (`CalculateYawPitchRoll` was already compiled in slice 2; `AddOldStyleOrientedBall`,
`Ball::GetRotatedVector`); the Python AddBall (`PyAddBall`); missile collision events (`Ball::DispatchCollisions`); the
accuracy helpers (`GetAccuracy`, `Ball::CalculateRelativeVelocities`, `PyGetCenterDist`, `PyGetSurfaceDist`); `OnTick`
and `ClientBall::GetShiftedTime`.

Destiny's float vector, quaternion and matrix types come from carbonengine/math (CcpMath.h; MIT, (c) CCP Games), read
from its snapshot (MATH_REPO, pinned at 8273322 by .git/HEAD as for Destiny; Destiny's vcpkg.json asks for carbon-math
>= 1.2.1 and pins no revision, so which revision Destiny was built with is not known). The struct declarations are
extracted with their DirectXMath conversion lines removed (each removed line is in the manifest), and every function
body is extracted verbatim. One function needs DirectXMath: RotationQuaternion(yaw, pitch, roll) (src/Quaternion.cpp:60)
calls XMQuaternionRotationRollPitchYaw. It is reached by Ball::GetRotatedVector without dynamical orientation (a
formation leader's offsets, a stationary missile launcher's direction).

- **With DIRECTXMATH_REPO** (Microsoft's DirectXMath, MIT, (c) Microsoft Corporation), pinned at f5d82574 by .git/HEAD, and every header the harness includes
  checked against its pinned SHA-256 before the build: the function is extracted verbatim, with the one declaration it
  needs kept in `struct Quaternion` (its constructor from XMVECTOR, and that constructor's definition verbatim), and
  DirectXMath.h is included by path from the snapshot (nothing of it is copied into this port). It is compiled with
  `-D_XM_NO_INTRINSICS_`, DirectXMath's scalar path: on aarch64 the header would otherwise take its NEON path, and
  Destiny's Windows x64 build likely takes the SSE path (with Intel's SVML under MSVC 2019 and later), so the bits are
  this commit's scalar path's (whose XMQuaternionRotationRollPitchYaw calls the C library's sinf and cosf). The
  header includes "sal.h", Microsoft's source annotations, which Linux does not have: the build writes its own stub
  (the ten annotations DirectXMath uses, as empty macros; they carry no semantics for code generation). The compiler's
  dependency list is recorded, and every DirectXMath file it read must be one of the checked headers. A second binary,
  `dxm_probe`, is the same RotationQuaternion(yaw, pitch, roll) alone (read three float bit patterns, print the
  quaternion's four), so tests can hold the reference's transcription to it over any angles.
- **Without it** (the build of 16 October, byte-identical): the harness supplies only the function's value at
  yaw = pitch = roll = +0, the identity (exact in any implementation), and stops the scenario otherwise (an "X" row);
  the reference before 23 October refused the same calls.

The prelude supplies what is not Destiny's simulation code: the CPython stand-in of build_space.py, extended with
tuples, `O!` and `b` arguments; the `Ball`, `Ballpark`, `MapOfBalls` and `BallIterator` member lists (Ball.h,
Ballpark.h); Blue's lists of minis (`BlueList`: Insert appends, Remove(-1) clears), Blue's class macros, object wrappers
and error setter; the switches at Settings.h's defaults, with useNewOrbit, iterative collision, dynamical orientation
and its missile exclusion set per case. A client's ball is a `ClientBall` carrying only `GetShiftedTime`; the display
state ClientBall adds (mLastYaw, mLastRot, and the lines its CalculateYawPitchRoll override adds, which call the
DirectXMath function above) is not part of the simulation and is not compiled. Events (`PostEvent`, `SendEvent`) are
printed. A driver reads scenarios on stdin and prints after every step every ball's state (with its orientation,
rotation, angular velocity, torque, roll, formation and minis), every static collidable, the partition (with the static
collidables in its boxes), the bubble dictionaries, the park's clock, and the answers to queries.

The build directory receives the source, the binary and a manifest (the SHA-256 of every source file read and every
extracted piece, the extracted lines, the lines removed from the math declarations, the compiler, its command and
warnings). Compiled with g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math, as the earlier harnesses, and
-fsigned-char: plain char is unsigned on aarch64 Linux and signed on x86 and under MSVC, Destiny's platforms; this
slice reaches Destiny's char values (mFormationID, whose "none" is -1; a formation slot; the cloak mode; Python's "b"
arguments), which the earlier slices never did. The repositories are read, never executed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PINNED = "114e89fa584f0e1a81eb0a679aabf41af8870072"
MATH_PINNED = "8273322093eb1f80168ae60a35891f2d454369f9"

# (file, name, regular expression for the start of the definition); extracted to the end of the block that follows
# (and its semicolon for classes, structs, enums and arrays)
BLOCKS = [
    # ---- declarations ----
    ("src/IDstConstants.h", "enum DSTBALLMODE", r"enum DSTBALLMODE\b"),
    ("src/IDstConstants.h", "enum DSTCONSTANTS", r"enum DSTCONSTANTS\b"),
    ("src/Quaternion.h", "typedef struct QuaternionD", r"typedef struct QuaternionD\s*\{"),
    ("src/SortedSets.h", "class BoxSortComparer", r"class BoxSortComparer\s*\{"),
    ("src/SortedSets.h", "class BallSortComparer", r"class BallSortComparer\s*\{"),
    ("src/SortedSets.h", "class StaticCollidableSortComparer", r"class StaticCollidableSortComparer\s*\{"),
    ("src/Partition.h", "class BoxAllocator", r"class BoxAllocator\s*\{"),
    ("src/Partition.h", "class Key", r"class Key\s*\{"),
    ("src/Partition.h", "struct KeyHash", r"struct KeyHash\s*\{"),
    ("src/Partition.h", "class Partition", r"class Partition\s*\{"),
    ("src/Partitionable.h", "class Partitionable", r"class Partitionable\s*\{"),
    ("src/Box.h", "class Box", r"class Box\s*\{"),
    ("src/StaticCollidable.h", "class StaticCollidable", r"class StaticCollidable\s*:\s*public Partitionable"),
    ("src/MapOfLongLongs.h", "class MapOfLongLongs", r"class MapOfLongLongs\s*:"),
    ("src/Ball.h", "class ProximitySensor", r"class ProximitySensor\s*\{"),
    ("src/CollisionStructures.h", "struct CollisionImpact", r"struct CollisionImpact\s*\{"),
    ("src/CollisionStructures.h", "struct CollisionItem", r"struct CollisionItem\s*\{"),
    ("src/CollisionStructures.h", "struct BallLocation", r"struct BallLocation\s*\{"),
    ("src/CollisionBallProperties.h", "class CollisionBallProperties", r"class CollisionBallProperties\s*\{"),
    ("src/Triangle.h", "class Triangle", r"class Triangle\s*\{"),
    ("src/Plane.h", "class Planed", r"class Planed\s*\{"),
    ("src/AABB.h", "class AABB", r"class AABB\s*\{"),
    ("src/BoxShape.h", "enum BoxCorner", r"enum BoxCorner\b"),
    ("src/BoxShape.h", "enum BoxSide", r"enum BoxSide\b"),
    ("src/BoxShape.h", "class RectangleShape", r"class RectangleShape\s*\{"),
    ("src/BoxShape.h", "class BoxShape", r"class BoxShape\s*\{"),
    ("src/MiniBall.h", "class MiniBall", r"class MiniBall\s*:"),
    ("src/MiniBox.h", "class MiniBox", r"class MiniBox\s*:"),
    ("src/MiniCapsule.h", "class MiniCapsule", r"class MiniCapsule\s*:"),
    ("src/Capsule.h", "class Capsule", r"BLUE_CLASS\( Capsule \)"),
    ("src/OrientedBox.h", "class OrientedBox", r"BLUE_CLASS\( OrientedBox \)"),
    # ---- Collision.cpp, Triangle.cpp, Plane.cpp, AABB.cpp, BoxShape.cpp: the static collidables' geometry ----
    ("src/Collision.cpp", "GetPointOnBallThatFirstIntersectsPlane",
     r"Vector3d GetPointOnBallThatFirstIntersectsPlane\(Vector3d ballPos"),
    ("src/Collision.cpp", "IntersectSegmentPlane", r"bool IntersectSegmentPlane\(Vector3d p0"),
    ("src/Collision.cpp", "IntersectSegmentSphere", r"bool IntersectSegmentSphere\(Vector3d p0"),
    ("src/Collision.cpp", "IntersectSphereTriangle", r"bool IntersectSphereTriangle\(Vector3d spherePosition"),
    ("src/Collision.cpp", "CollideTwoSpheres", r"double CollideTwoSpheres\("),
    ("src/Collision.cpp", "Quadratic", r"bool Quadratic\(double& v1"),
    ("src/Triangle.cpp", "Triangle::Triangle", r"Triangle::Triangle\(Vector3d a"),
    ("src/Triangle.cpp", "Triangle::GetNormal", r"Vector3d Triangle::GetNormal\(\)"),
    ("src/Triangle.cpp", "Triangle::GetClosestPoint", r"Vector3d Triangle::GetClosestPoint\("),
    ("src/Triangle.cpp", "Triangle::ContainsPoint", r"bool Triangle::ContainsPoint\("),
    ("src/Plane.cpp", "Planed::Planed", r"Planed::Planed\(Vector3d point"),
    ("src/AABB.cpp", "AABB::AABB()", r"AABB::AABB\(\)"),
    ("src/AABB.cpp", "AABB::AABB(low, high)", r"AABB::AABB\(Vector3d low"),
    ("src/AABB.cpp", "AABB::Update", r"void AABB::Update\("),
    ("src/AABB.cpp", "AABB::CanExcludeCollision", r"bool AABB::CanExcludeCollision\("),
    ("src/AABB.cpp", "AABB::GetLongestWidth", r"double AABB::GetLongestWidth\(\)"),
    ("src/BoxShape.cpp", "RectangleShape::RectangleShape", r"RectangleShape::RectangleShape\(Vector3d a"),
    ("src/BoxShape.cpp", "RectangleShape::CollideWithSphere", r"bool RectangleShape::CollideWithSphere\("),
    ("src/BoxShape.cpp", "BoxShape::BoxShape", r"BoxShape::BoxShape\(Vector3d corner"),
    ("src/BoxShape.cpp", "BoxShape::GetClosestCornerToPoint", r"BoxCorner BoxShape::GetClosestCornerToPoint\("),
    ("src/BoxShape.cpp", "BoxShape::GetClosestPointOnBox", r"Vector3d BoxShape::GetClosestPointOnBox\("),
    ("src/BoxShape.cpp", "BoxShape::RayIntersect", r"bool BoxShape::RayIntersect\("),
    ("src/BoxShape.cpp", "array CONNECTED_SIDES", r"static BoxSide CONNECTED_SIDES\[8\]\[3\] ="),
    ("src/BoxShape.cpp", "BoxShape::CollideWithSphere", r"bool BoxShape::CollideWithSphere\("),
    ("src/BoxShape.cpp", "BoxShape::GetCenter", r"Vector3d BoxShape::GetCenter\(\)"),
    ("src/BoxShape.cpp", "BoxShape::GetBoundingRadius", r"double BoxShape::GetBoundingRadius\(\)"),
    # ---- CollisionStructures.cpp, CollisionBallProperties.cpp: the iterative method's store ----
    ("src/CollisionStructures.cpp", "CollisionItem::CollisionItem", r"CollisionItem::CollisionItem\(\) :"),
    ("src/CollisionStructures.cpp", "BallLocation::BallLocation", r"BallLocation::BallLocation\(\) :"),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::CollisionBallProperties",
     r"CollisionBallProperties::CollisionBallProperties\(\) :"),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::Initialize",
     r"void CollisionBallProperties::Initialize\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::AddImpact",
     r"std::vector<std::pair<double, ID>> CollisionBallProperties::AddImpact\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::RemoveImpact",
     r"std::vector<std::pair<double, ID>> CollisionBallProperties::RemoveImpact\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::CalculateEnd",
     r"void CollisionBallProperties::CalculateEnd\(\)"),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::IgnoreCollision",
     r"bool CollisionBallProperties::IgnoreCollision\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::FindLocationsIndex",
     r"size_t CollisionBallProperties::FindLocationsIndex\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::CalculatePosition",
     r"void CollisionBallProperties::CalculatePosition\("),
    ("src/CollisionBallProperties.cpp", "CollisionBallProperties::CalculateRotation",
     r"void CollisionBallProperties::CalculateRotation\("),
    # ---- Partition.cpp ----
    ("src/Partition.cpp", "Partition::Partition", r"Partition::Partition\(\):"),
    ("src/Partition.cpp", "Partition::GetBoxKey", r"void Partition::GetBoxKey\("),
    ("src/Partition.cpp", "Partition::GetBox(int64)", r"Box\* Partition::GetBox\(int64_t ix"),
    ("src/Partition.cpp", "Partition::CheckBox", r"Box\* Partition::CheckBox\("),
    ("src/Partition.cpp", "Partition::GetBox(double)", r"Box\* Partition::GetBox\(double radius"),
    ("src/Partition.cpp", "Partition::GetExistingTopBox", r"Box\* Partition::GetExistingTopBox\("),
    ("src/Partition.cpp", "Partition::GetBoxCenter", r"void Partition::GetBoxCenter\("),
    ("src/Partition.cpp", "Partition::GetSize", r"int Partition::GetSize\(\)"),
    ("src/Partition.cpp", "Partition::GetTopmost", r"Box\* Partition::GetTopmost\("),
    ("src/Partition.cpp", "Partition::GetOffsetBox", r"Box\* Partition::GetOffsetBox\("),
    ("src/Partition.cpp", "Partition::CheckOffsetBox", r"Box\* Partition::CheckOffsetBox\("),
    ("src/Partition.cpp", "Partition::ClearTraversalFlag", r"void Partition::ClearTraversalFlag\(\)"),
    ("src/Partition.cpp", "Partition::ExplodeChildren", r"void Partition::ExplodeChildren\("),
    ("src/Partition.cpp", "Partition::GetProximityCandidates", r"void Partition::GetProximityCandidates\("),
    ("src/Partition.cpp", "Partition::GetCollisionCandidates", r"void Partition::GetCollisionCandidates\("),
    ("src/Partition.cpp", "Partition::GetNearbyBalls", r"void Partition::GetNearbyBalls\("),
    ("src/Partition.cpp", "BoxAllocator::BoxAllocator", r"BoxAllocator::BoxAllocator\(\)"),
    ("src/Partition.cpp", "BoxAllocator::~BoxAllocator", r"BoxAllocator::~BoxAllocator\(\)"),
    ("src/Partition.cpp", "BoxAllocator::GetNewBox", r"Box \* BoxAllocator::GetNewBox\("),
    ("src/Partition.cpp", "BoxAllocator::ReleaseBox", r"void BoxAllocator::ReleaseBox\("),
    # ---- Box.cpp ----
    ("src/Box.cpp", "Box::Box", r"Box::Box\(int64_t _ix"),
    ("src/Box.cpp", "Box::~Box", r"Box::~Box\(\)"),
    ("src/Box.cpp", "Box::ReleaseBox", r"void Box::ReleaseBox\(\)"),
    ("src/Box.cpp", "Box::SetParent", r"void Box::SetParent\(\)"),
    ("src/Box.cpp", "Box::AddChildren", r"void Box::AddChildren\("),
    ("src/Box.cpp", "Box::AddBall", r"void Box::AddBall\("),
    ("src/Box.cpp", "Box::AddStaticCollidable", r"void Box::AddStaticCollidable\("),
    ("src/Box.cpp", "Box::RemoveChildren", r"void Box::RemoveChildren\("),
    ("src/Box.cpp", "Box::RemoveBall", r"void Box::RemoveBall\("),
    ("src/Box.cpp", "Box::RemoveStaticCollidable", r"void Box::RemoveStaticCollidable\("),
    ("src/Box.cpp", "Box::Traverse", r"void Box::Traverse\("),
    ("src/Box.cpp", "Box::NearbyBubbles", r"long Box::NearbyBubbles\(\)"),
    ("src/Box.cpp", "BoxSortComparer::operator()", r"bool BoxSortComparer::operator \(\)"),
    # ---- Partitionable.cpp, StaticCollidable.cpp, the minis' classes ----
    ("src/Partitionable.cpp", "Partitionable::Partitionable", r"Partitionable::Partitionable\(\):"),
    ("src/StaticCollidable.cpp", "StaticCollidable::StaticCollidable", r"StaticCollidable::StaticCollidable\(\):"),
    ("src/StaticCollidable.cpp", "StaticCollidable::ClearBoxes", r"void StaticCollidable::ClearBoxes\(\)"),
    ("src/StaticCollidable.cpp", "StaticCollidable::DeleteFromBoxes", r"void StaticCollidable::DeleteFromBoxes\(\)"),
    ("src/StaticCollidable.cpp", "StaticCollidable::InsertInBox", r"void StaticCollidable::InsertInBox\("),
    ("src/StaticCollidable.cpp", "StaticCollidable::GetInflatedRadius",
     r"double StaticCollidable::GetInflatedRadius\("),
    ("src/StaticCollidable.cpp", "StaticCollidableSortComparer::operator()",
     r"bool StaticCollidableSortComparer::operator \(\)"),
    ("src/MiniBall.cpp", "MiniBall::MiniBall", r"MiniBall::MiniBall\(IRoot\* lockobj\) :"),
    ("src/MiniBall.cpp", "MiniBall::~MiniBall", r"MiniBall::~MiniBall\(\)"),
    ("src/MiniBox.cpp", "MiniBox::MiniBox", r"MiniBox::MiniBox\(IRoot\* lockobj\) :"),
    ("src/MiniBox.cpp", "MiniBox::~MiniBox", r"MiniBox::~MiniBox\(\)"),
    ("src/MiniCapsule.cpp", "MiniCapsule::MiniCapsule", r"MiniCapsule::MiniCapsule\(IRoot\* lockobj\) :"),
    ("src/MiniCapsule.cpp", "MiniCapsule::~MiniCapsule", r"MiniCapsule::~MiniCapsule\(\)"),
    # ---- Capsule.cpp, OrientedBox.cpp: the static collidables ----
    ("src/Capsule.cpp", "Capsule::Initialize", r"void Capsule::Initialize\(ID theID"),
    ("src/Capsule.cpp", "Capsule::GetBoundingRadius", r"float Capsule::GetBoundingRadius\(\)"),
    ("src/Capsule.cpp", "Capsule::GetCenter", r"Vector3d Capsule::GetCenter\(\)"),
    ("src/Capsule.cpp", "Capsule::CheckCollision", r"bool Capsule::CheckCollision\("),
    ("src/Capsule.cpp", "Capsule::HandleCollisionNonIteratively", r"void Capsule::HandleCollisionNonIteratively\("),
    ("src/Capsule.cpp", "Capsule::CollideWithBall", r"void Capsule::CollideWithBall\("),
    ("src/Capsule.cpp", "Capsule::ReactToCollision", r"void Capsule::ReactToCollision\("),
    ("src/Capsule.cpp", "Capsule::InsertInBoxes", r"void Capsule::InsertInBoxes\("),
    ("src/OrientedBox.cpp", "OrientedBox::Initialize", r"void OrientedBox::Initialize\("),
    ("src/OrientedBox.cpp", "OrientedBox::GetBoundingRadius", r"float OrientedBox::GetBoundingRadius\(\)"),
    ("src/OrientedBox.cpp", "OrientedBox::GetCenter", r"Vector3d OrientedBox::GetCenter\(\)"),
    ("src/OrientedBox.cpp", "OrientedBox::CollideWithBall", r"void OrientedBox::CollideWithBall\("),
    ("src/OrientedBox.cpp", "OrientedBox::ReactToCollision", r"void OrientedBox::ReactToCollision\("),
    ("src/OrientedBox.cpp", "OrientedBox::InsertInBoxes", r"void OrientedBox::InsertInBoxes\("),
    ("src/OrientedBox.cpp", "OrientedBox::CheckCollision", r"bool OrientedBox::CheckCollision\("),
    # ---- Ballpark.cpp ----
    ("src/Ballpark.cpp", "_finite", r"inline float _finite\( double x \)"),
    ("src/Ballpark.cpp", "DynamicalOrientationDisabledForBall", r"bool DynamicalOrientationDisabledForBall\("),
    ("src/Ballpark.cpp", "Ballpark::Ballpark", r"Ballpark::Ballpark\(IRoot\* lockobj\) :"),
    ("src/Ballpark.cpp", "Ballpark::~Ballpark", r"Ballpark::~Ballpark\(\)"),
    ("src/Ballpark.cpp", "Ballpark::OnTick", r"void Ballpark::OnTick\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateIterativeCollisionResponses",
     r"void Ballpark::CalculateIterativeCollisionResponses\(\)"),
    ("src/Ballpark.cpp", "Ballpark::Evolve", r"void Ballpark::Evolve\("),
    ("src/Ballpark.cpp", "Ballpark::GetOrbitalNormal", r"Vector3d Ballpark::GetOrbitalNormal\("),
    ("src/Ballpark.cpp", "Ballpark::DefaultOrbitalNormal", r"Vector3d Ballpark::DefaultOrbitalNormal\("),
    ("src/Ballpark.cpp", "Ballpark::Integrate", r"void Ballpark::Integrate\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveBehaviorForBall", r"void Ballpark::EvolveBehaviorForBall\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveWarp", r"Vector3d Ballpark::EvolveWarp\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveGoto", r"Vector3d Ballpark::EvolveGoto\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveMissile", r"Vector3d Ballpark::EvolveMissile\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveFormation", r"Vector3d Ballpark::EvolveFormation\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveFollow", r"Vector3d Ballpark::EvolveFollow\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveOrbit", r"Vector3d Ballpark::EvolveOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveNewStyleOrbit", r"Vector3d Ballpark::EvolveNewStyleOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveOldStyleOrbit", r"Vector3d Ballpark::EvolveOldStyleOrbit\("),
    ("src/Ballpark.cpp", "Ballpark::EvolveStop", r"void Ballpark::EvolveStop\("),
    ("src/Ballpark.cpp", "Ballpark::CapAcceleration", r"void Ballpark::CapAcceleration\("),
    ("src/Ballpark.cpp", "Ballpark::GotoThrust", r"Vector3d Ballpark::GotoThrust\("),
    ("src/Ballpark.cpp", "Ballpark::OrbitThrust", r"Vector3d Ballpark::OrbitThrust\("),
    ("src/Ballpark.cpp", "Ballpark::GotoThrustFollow", r"Vector3d Ballpark::GotoThrustFollow\("),
    ("src/Ballpark.cpp", "CalculateN2Alpha2(Vector3)",
     r"static inline void CalculateN2Alpha2\( int& n2, float& kappa2, Vector3& alpha2"),
    ("src/Ballpark.cpp", "CalculateN2Alpha2(float)",
     r"static inline void CalculateN2Alpha2\( int& n2, float& kappa2, float& alpha2"),
    ("src/Ballpark.cpp", "Ballpark::GotoTorqueBreaks", r"Vector3 Ballpark::GotoTorqueBreaks\("),
    ("src/Ballpark.cpp", "Ballpark::GotoRollBreaks", r"float Ballpark::GotoRollBreaks\("),
    ("src/Ballpark.cpp", "Ballpark::ApplyTorque", r"void Ballpark::ApplyTorque\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateCollisionResponse", r"bool Ballpark::CalculateCollisionResponse\("),
    ("src/Ballpark.cpp", "Ballpark::GetNextCollisionStatic", r"void Ballpark::GetNextCollisionStatic\("),
    ("src/Ballpark.cpp", "Ballpark::GetNextCollisionFree", r"void Ballpark::GetNextCollisionFree\("),
    ("src/Ballpark.cpp", "Ballpark::ApproximateEllipsePointOfImpact",
     r"Vector3d Ballpark::ApproximateEllipsePointOfImpact\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateCollisionImpactStatic",
     r"void Ballpark::CalculateCollisionImpactStatic\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateCollisionImpactFree",
     r"CollisionItem Ballpark::CalculateCollisionImpactFree\("),
    ("src/Ballpark.cpp", "Ballpark::AddImpact", r"void Ballpark::AddImpact\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveImpactFromFree", r"void Ballpark::RemoveImpactFromFree\("),
    ("src/Ballpark.cpp", "Ballpark::CollisionBallNeighborIgnore",
     r"inline bool Ballpark::CollisionBallNeighborIgnore\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateBallRotationVelocity",
     r"void Ballpark::CalculateBallRotationVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::CalculateBallPositionVelocity", r"void Ballpark::CalculateBallPositionVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::Gradient", r"void Ballpark::Gradient\(Ball \*ball\)"),
    ("src/Ballpark.cpp", "Ballpark::Potential", r"void Ballpark::Potential\(Ball \*me"),
    ("src/Ballpark.cpp", "Ballpark::Start", r"void Ballpark::Start\(\)"),
    ("src/Ballpark.cpp", "Ballpark::Pause", r"void Ballpark::Pause\(\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertInBoxes", r"void Ballpark::InsertInBoxes\(Partitionable \*p\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertBallInBoxes", r"void Ballpark::InsertBallInBoxes\(Ball \*ball\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertInBox", r"void Ballpark::InsertInBox\(Box \*box,Partitionable \*p\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertStaticCollidableInBoxes",
     r"void Ballpark::InsertStaticCollidableInBoxes\("),
    ("src/Ballpark.cpp", "Ballpark::BallExists", r"Ball \*Ballpark::BallExists\("),
    ("src/Ballpark.cpp", "Ballpark::GetIDForCollisionObject", r"ID Ballpark::GetIDForCollisionObject\("),
    ("src/Ballpark.cpp", "Ballpark::AddBall", r"Ball \* Ballpark::AddBall\("),
    ("src/Ballpark.cpp", "Ballpark::AddOldStyleOrientedBall", r"Ball\* Ballpark::AddOldStyleOrientedBall\("),
    ("src/Ballpark.cpp", "Ballpark::AddDynamicallyOrientedBall", r"Ball\* Ballpark::AddDynamicallyOrientedBall\("),
    ("src/Ballpark.cpp", "Ballpark::AddMiniball", r"Ball \* Ballpark::AddMiniball\("),
    ("src/Ballpark.cpp", "Ballpark::AddCapsule", r"Capsule \* Ballpark::AddCapsule\("),
    ("src/Ballpark.cpp", "Ballpark::AddOrientedBox", r"OrientedBox \* Ballpark::AddOrientedBox\("),
    ("src/Ballpark.cpp", "Ballpark::AddMushroom", r"void Ballpark::AddMushroom\("),
    ("src/Ballpark.cpp", "Ballpark::MissileFollow", r"void Ballpark::MissileFollow\("),
    ("src/Ballpark.cpp", "Ballpark::FollowBall", r"void Ballpark::FollowBall\("),
    ("src/Ballpark.cpp", "Ballpark::FormationFollow", r"void Ballpark::FormationFollow\("),
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
    ("src/Ballpark.cpp", "Ballpark::SetBallMass", r"void Ballpark::SetBallMass\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallRadius(ID)", r"void Ballpark::SetBallRadius\(\s*const ID& srcId,"),
    ("src/Ballpark.cpp", "Ballpark::SetBallRadius(Ball*)", r"void Ballpark::SetBallRadius\(\s*Ball \*ball,"),
    ("src/Ballpark.cpp", "Ballpark::SetMaxSpeed", r"void Ballpark::SetMaxSpeed\("),
    ("src/Ballpark.cpp", "Ballpark::SetMaxAngularSpeed", r"void Ballpark::SetMaxAngularSpeed\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallPosition", r"void Ballpark::SetBallPosition\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallVelocity", r"void Ballpark::SetBallVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallAngularVelocity", r"void Ballpark::SetBallAngularVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallRotation", r"void Ballpark::SetBallRotation\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallHarmonic", r"void Ballpark::SetBallHarmonic\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallFree", r"void Ballpark::SetBallFree\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallAgility", r"void Ballpark::SetBallAgility\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallAngularAgility", r"void Ballpark::SetBallAngularAgility\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallTimeFactor", r"void Ballpark::SetBallTimeFactor\("),
    ("src/Ballpark.cpp", "Ballpark::SetSpeedFraction", r"void Ballpark::SetSpeedFraction\("),
    ("src/Ballpark.cpp", "Ballpark::SetNotificationRange", r"void Ballpark::SetNotificationRange\("),
    ("src/Ballpark.cpp", "Ballpark::SetTargetTracking", r"void Ballpark::SetTargetTracking\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallMassive", r"void Ballpark::SetBallMassive\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallGlobal", r"void Ballpark::SetBallGlobal\("),
    ("src/Ballpark.cpp", "Ballpark::AddProximitySensor", r"void Ballpark::AddProximitySensor\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveProximitySensor", r"void Ballpark::RemoveProximitySensor\("),
    ("src/Ballpark.cpp", "Ballpark::GetAccuracy", r"double Ballpark::GetAccuracy\("),
    ("src/Ballpark.cpp", "Ballpark::CloakBall", r"void Ballpark::CloakBall\("),
    ("src/Ballpark.cpp", "Ballpark::UncloakBall", r"void Ballpark::UncloakBall\("),
    ("src/Ballpark.cpp", "Ballpark::CheckVisibility", r"ID Ballpark::CheckVisibility\("),
    ("src/Ballpark.cpp", "Ballpark::StopAllFollowers", r"void Ballpark::StopAllFollowers\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveBall", r"void Ballpark::RemoveBall\(\s*const ID& srcId,"),
    ("src/Ballpark.cpp", "Ballpark::RemoveCapsule", r"void Ballpark::RemoveCapsule\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveOrientedBox", r"void Ballpark::RemoveOrientedBox\("),
    ("src/Ballpark.cpp", "Ballpark::BringOutTheDeads", r"void Ballpark::BringOutTheDeads\(\)"),
    ("src/Ballpark.cpp", "Ballpark::BringOutDeadBalls", r"void Ballpark::BringOutDeadBalls\(\)"),
    ("src/Ballpark.cpp", "Ballpark::BringOutDeadCapsules", r"void Ballpark::BringOutDeadCapsules\(\)"),
    ("src/Ballpark.cpp", "Ballpark::HandleProximities", r"void Ballpark::HandleProximities\(\)"),
    ("src/Ballpark.cpp", "Ballpark::ClearBubbles", r"void Ballpark::ClearBubbles\(\)"),
    ("src/Ballpark.cpp", "Ballpark::ClearAll", r"void Ballpark::ClearAll\("),
    ("src/Ballpark.cpp", "Ballpark::UpdateBallBubble", r"void Ballpark::UpdateBallBubble\("),
    ("src/Ballpark.cpp", "Ballpark::CopyBubbles", r"void Ballpark::CopyBubbles\(\)"),
    ("src/Ballpark.cpp", "Ballpark::InDeadBubble", r"bool Ballpark::InDeadBubble\("),
    ("src/Ballpark.cpp", "Ballpark::GetInteractiveCnt", r"size_t Ballpark::GetInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallRigid", r"void Ballpark::SetBallRigid\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallTroll", r"void Ballpark::SetBallTroll\("),
    ("src/Ballpark.cpp", "Ballpark::TrollReady", r"bool Ballpark::TrollReady\("),
    ("src/Ballpark.cpp", "Ballpark::PetrifyTroll", r"void Ballpark::PetrifyTroll\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallInteractive", r"void Ballpark::SetBallInteractive\("),
    ("src/Ballpark.cpp", "Ballpark::DecreaseInteractiveCnt", r"void Ballpark::DecreaseInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::UpdateInteractiveCnt", r"void Ballpark::UpdateInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::IncreaseInteractiveCnt", r"void Ballpark::IncreaseInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::AddToBubble", r"void Ballpark::AddToBubble\("),
    ("src/Ballpark.cpp", "Ballpark::InitializeBubbles", r"void Ballpark::InitializeBubbles\(\)"),
    ("src/Ballpark.cpp", "Ballpark::AddFormation", r"void Ballpark::AddFormation\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallFormation", r"void Ballpark::SetBallFormation\("),
    ("src/Ballpark.cpp", "Ballpark::UpdateKeepAliveBalls", r"void Ballpark::UpdateKeepAliveBalls\("),
    ("src/Ballpark.cpp", "Ballpark::AddKeepAliveBall", r"void Ballpark::AddKeepAliveBall\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveKeepAliveBall", r"void Ballpark::RemoveKeepAliveBall\("),
    # ---- Thunkers.cpp: the Python entry points the driver uses ----
    ("src/Thunkers.cpp", "Ballpark::PyAddDynamicallyOrientedBall",
     r"PyObject\* Ballpark::PyAddDynamicallyOrientedBall\("),
    ("src/Thunkers.cpp", "Ballpark::PyAddOldStyleOrientedBall", r"PyObject\* Ballpark::PyAddOldStyleOrientedBall\("),
    ("src/Thunkers.cpp", "Ballpark::PyAddBall", r"PyObject\* Ballpark::PyAddBall\("),
    ("src/Thunkers.cpp", "Ballpark::PyAddCapsule", r"PyObject\* Ballpark::PyAddCapsule\("),
    ("src/Thunkers.cpp", "Ballpark::PyAddOrientedBox", r"PyObject\* Ballpark::PyAddOrientedBox\("),
    ("src/Thunkers.cpp", "Ballpark::PyMissileFollow", r"PyObject\* Ballpark::PyMissileFollow\("),
    ("src/Thunkers.cpp", "Ballpark::PyAddMushroom", r"PyObject\* Ballpark::PyAddMushroom\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallRigid", r"PyObject\* Ballpark::PySetBallRigid\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallTroll", r"PyObject\* Ballpark::PySetBallTroll\("),
    ("src/Thunkers.cpp", "Ballpark::PySetMaxAngularSpeed", r"PyObject\* Ballpark::PySetMaxAngularSpeed\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallAngularVelocity", r"PyObject\* Ballpark::PySetBallAngularVelocity\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallRotation", r"PyObject\* Ballpark::PySetBallRotation\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallAngularAgility", r"PyObject\* Ballpark::PySetBallAngularAgility\("),
    ("src/Thunkers.cpp", "Ballpark::PyLaunchMissile", r"PyObject\* Ballpark::PyLaunchMissile\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBoid", r"PyObject\* Ballpark::PySetBoid\("),
    ("src/Thunkers.cpp", "Ballpark::PyFormationFollow", r"PyObject\* Ballpark::PyFormationFollow\("),
    ("src/Thunkers.cpp", "Ballpark::PyInitializeBubbles", r"PyObject\* Ballpark::PyInitializeBubbles\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetRemoteFollowers", r"PyObject\* Ballpark::PyGetRemoteFollowers\("),
    ("src/Thunkers.cpp", "Ballpark::PyEvolve", r"PyObject\* Ballpark::PyEvolve\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBallIdsInRange", r"PyObject\* Ballpark::PyGetBallIdsInRange\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBallIdsAndDistInRange", r"PyObject\* Ballpark::PyGetBallIdsAndDistInRange\("),
    ("src/Thunkers.cpp", "Ballpark::GetBallIdsInRange", r"PyObject\* Ballpark::GetBallIdsInRange\(\s*PyObject\* args,"),
    ("src/Thunkers.cpp", "Ballpark::BallIsValidForIteration", r"bool Ballpark::BallIsValidForIteration\("),
    ("src/Thunkers.cpp", "Ballpark::GetNextValidBallInBubble", r"int Ballpark::GetNextValidBallInBubble\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetCenterDist", r"PyObject\* Ballpark::PyGetCenterDist\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetSurfaceDist", r"PyObject\* Ballpark::PyGetSurfaceDist\("),
    ("src/Thunkers.cpp", "Ballpark::PyCheckVisibility", r"PyObject\* Ballpark::PyCheckVisibility\("),
    ("src/Thunkers.cpp", "Ballpark::PyCloakBall", r"PyObject\* Ballpark::PyCloakBall\("),
    ("src/Thunkers.cpp", "Ballpark::PyUncloakBall", r"PyObject\* Ballpark::PyUncloakBall\("),
    ("src/Thunkers.cpp", "Ballpark::PySetBallFormation", r"PyObject\* Ballpark::PySetBallFormation\("),
    ("src/Thunkers.cpp", "Ballpark::PyLoadFormations", r"PyObject\* Ballpark::PyLoadFormations\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBubbleAtCoordinates", r"PyObject\* Ballpark::PyGetBubbleAtCoordinates\("),
    ("src/Thunkers.cpp", "Ballpark::PyCopyBubbles", r"PyObject\* Ballpark::PyCopyBubbles\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetAccuracy", r"PyObject\* Ballpark::PyGetAccuracy\("),
    # ---- Ball.cpp ----
    ("src/Ball.cpp", "floatrand", r"double floatrand\(double a,double b\)"),
    ("src/Ball.cpp", "Modulo2pi", r"void Modulo2pi\(double& a,double& b\)\s*\{"),
    ("src/Ball.cpp", "Ball::Ball", r"Ball::Ball\(IRoot\* lockobj\) :"),
    ("src/Ball.cpp", "Ball::~Ball", r"Ball::~Ball\(\)"),
    ("src/Ball.cpp", "Ball::CalculateRelativeVelocities", r"void Ball::CalculateRelativeVelocities\("),
    ("src/Ball.cpp", "Ball::ClearBoxes", r"void Ball::ClearBoxes\(\)"),
    ("src/Ball.cpp", "Ball::AddProximitySensor", r"long Ball::AddProximitySensor\("),
    ("src/Ball.cpp", "Ball::UpdateProximityStatus", r"void Ball::UpdateProximityStatus\(\)"),
    ("src/Ball.cpp", "Ball::RemoveProximitySensor", r"void Ball::RemoveProximitySensor\(\)"),
    ("src/Ball.cpp", "Ball::SnapOrientation", r"void Ball::SnapOrientation\(\)"),
    ("src/Ball.cpp", "Ball::CalculateYawPitchRoll", r"void Ball::CalculateYawPitchRoll\(bool snap\)"),
    ("src/Ball.cpp", "Ball::DispatchCollisions", r"void Ball::DispatchCollisions\(\)"),
    ("src/Ball.cpp", "Ball::CheckForProximity", r"void Ball::CheckForProximity\(\)"),
    ("src/Ball.cpp", "Ball::CheckForProximity_NotificationRange",
     r"void Ball::CheckForProximity_NotificationRange\(\)"),
    ("src/Ball.cpp", "Ball::CheckForProximity_TargetTracking", r"void Ball::CheckForProximity_TargetTracking\(\)"),
    ("src/Ball.cpp", "Ball::CheckForProximity_Sensor", r"void Ball::CheckForProximity_Sensor\(\)"),
    ("src/Ball.cpp", "Ball::DeleteFromBoxes", r"void Ball::DeleteFromBoxes\(\)"),
    ("src/Ball.cpp", "Ball::InsertInBox", r"void Ball::InsertInBox\(Box\* box\)"),
    ("src/Ball.cpp", "Ball::GetInflatedRadius", r"double Ball::GetInflatedRadius\("),
    ("src/Ball.cpp", "Ball::GetBoundingRadius", r"float Ball::GetBoundingRadius\(\)"),
    ("src/Ball.cpp", "Ball::SetBoxActive", r"void Ball::SetBoxActive\("),
    ("src/Ball.cpp", "Ball::SetMode", r"void Ball::SetMode\("),
    ("src/Ball.cpp", "ClientBall::GetShiftedTime", r"Be::Time ClientBall::GetShiftedTime\("),
    ("src/Ball.cpp", "Ball::AddMiniBalls", r"void Ball::AddMiniBalls\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniBalls", r"void Ball::RemoveMiniBalls\(\)"),
    ("src/Ball.cpp", "Ball::AddActualMiniball", r"void Ball::AddActualMiniball\("),
    ("src/Ball.cpp", "Ball::AddMiniBall", r"void Ball::AddMiniBall\(\s*double x,"),
    ("src/Ball.cpp", "Ball::AddMiniCapsules", r"void Ball::AddMiniCapsules\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniCapsules", r"void Ball::RemoveMiniCapsules\(\)"),
    ("src/Ball.cpp", "Ball::AddActualMinicapsule", r"void Ball::AddActualMinicapsule\("),
    ("src/Ball.cpp", "Ball::AddMiniCapsule", r"void Ball::AddMiniCapsule\(\s*double ax,"),
    ("src/Ball.cpp", "Ball::AddMiniBox", r"void Ball::AddMiniBox\(\s*Vector3d corner,"),
    ("src/Ball.cpp", "Ball::AddMiniBoxes", r"void Ball::AddMiniBoxes\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniBoxes", r"void Ball::RemoveMiniBoxes\(\)"),
    ("src/Ball.cpp", "Ball::AddActualMiniBox", r"void Ball::AddActualMiniBox\("),
    ("src/Ball.cpp", "Ball::IsWarpish", r"bool Ball::IsWarpish\(\)"),
    ("src/Ball.cpp", "Ball::IsAligningForWarp", r"bool Ball::IsAligningForWarp\(\)"),
    ("src/Ball.cpp", "Ball::IsWarping", r"bool Ball::IsWarping\(\)"),
    ("src/Ball.cpp", "Ball::IsAlignedForWarp", r"bool Ball::IsAlignedForWarp\(\)"),
    ("src/Ball.cpp", "Ball::PyAddMiniBall", r"PyObject\* Ball::PyAddMiniBall\("),
    ("src/Ball.cpp", "Ball::PyAddMiniCapsule", r"PyObject\* Ball::PyAddMiniCapsule\("),
    ("src/Ball.cpp", "Ball::PyAddMiniBox", r"PyObject\* Ball::PyAddMiniBox\("),
    ("src/Ball.cpp", "Ball::GetRotatedVector", r"void Ball::GetRotatedVector\(Vector3& vec\)"),
    ("src/Ball.cpp", "Ball::ReserveFormationSlot", r"char Ball::ReserveFormationSlot\(\)"),
    ("src/Ball.cpp", "Ball::FreeFormationSlot", r"void Ball::FreeFormationSlot\(size_t slot\)"),
    ("src/Ball.cpp", "Ball::InsertInBoxes", r"void Ball::InsertInBoxes\(Box\* box1"),
    ("src/Ball.cpp", "BallSortComparer::operator()", r"bool BallSortComparer::operator \(\)"),
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
    ("src/Ballpark.h", "typedef SetOfBalls", "typedef std::unordered_set<Ball *> SetOfBalls;"),
    ("src/Ballpark.h", "typedef SetOfCapsules", "typedef std::unordered_set<Capsule *> SetOfCapsules;"),
    ("src/Ballpark.h", "typedef SetOfOrientedBoxes", "typedef std::unordered_set<OrientedBox *> SetOfOrientedBoxes;"),
    ("src/Ballpark.h", "typedef VectorOfVectors", "typedef std::vector<Vector3d> VectorOfVectors;"),
    ("src/Ballpark.h", "typedef VectorOfCapsules", "typedef std::vector<Capsule *> VectorOfCapsules;"),
    ("src/Settings.h", "g_collisionMaxIterationsDefault", "constexpr size_t g_collisionMaxIterationsDefault"),
    ("src/Settings.h", "g_useIterativeCollisionDefault", "constexpr bool g_useIterativeCollisionDefault"),
    ("src/Settings.h", "g_useDynamicalOrientationDefault", "constexpr bool g_useDynamicalOrientationDefault"),
    ("src/Settings.h", "g_disableDynamicalOrientationForMissilesDefault",
     "constexpr bool g_disableDynamicalOrientationForMissilesDefault"),
    ("src/Settings.h", "g_useNewOrbitDefault", "constexpr bool g_useNewOrbitDefault"),
    ("src/SortedSets.h", "typedef SortedSetOfBoxes", "typedef std::set<Box*, BoxSortComparer> SortedSetOfBoxes;"),
    ("src/SortedSets.h", "typedef SortedSetOfBalls", "typedef std::set<Ball*, BallSortComparer> SortedSetOfBalls;"),
    ("src/SortedSets.h", "typedef SortedSetOfStaticCollidables",
     "typedef std::set<StaticCollidable*, StaticCollidableSortComparer> SortedSetOfStaticCollidables;"),
    ("src/Partition.h", "typedef DictOfBalls", "typedef std::unordered_map<ID, Ball *> DictOfBalls;"),
    ("src/Partition.h", "typedef VectorOfStaticCollidables",
     "typedef std::vector<StaticCollidable *> VectorOfStaticCollidables;"),
    ("src/Partition.h", "typedef DictEntry", "typedef std::pair<ID, Ball*> DictEntry;"),
    ("src/Partition.h", "typedef VectorOfBoxes", "typedef std::vector<Box *> VectorOfBoxes;"),
    ("src/Partition.h", "typedef BoxVectorInsertor",
     "typedef std::back_insert_iterator< VectorOfBoxes > BoxVectorInsertor;"),
    ("src/Partition.h", "typedef MapOfBoxes", "typedef std::unordered_map<Key, Box*, KeyHash> MapOfBoxes;"),
    ("src/Collision.h", "declaration GetPointOnBallThatFirstIntersectsPlane",
     "Vector3d GetPointOnBallThatFirstIntersectsPlane(Vector3d ballPos, double ballRadius, Vector3d planeNormal);"),
    ("src/Collision.h", "declaration IntersectSegmentPlane",
     "bool IntersectSegmentPlane(Vector3d p0, Vector3d p1, Planed plane, Vector3d & collisionPoint, "
     "double & collisionTime);"),
    ("src/Collision.h", "declaration IntersectSegmentSphere",
     "bool IntersectSegmentSphere(Vector3d p0, Vector3d p1, Vector3d spherePos, double sphereRad, "
     "Vector3d & collisionPoint, double & collisionTime);"),
    ("src/Collision.h", "declaration IntersectSphereTriangle",
     "bool IntersectSphereTriangle(Vector3d spherePosition, double sphereRadius, Vector3d sphereVelocity, "
     "Triangle triangle, Vector3d & collisionPoint, double & collisionTime, Vector3d& normal);"),
    ("src/Collision.h", "declaration Quadratic",
     "bool Quadratic(double& v1, double& v2, double a, double b, double c);"),
    ("src/Collision.h", "declaration CollideTwoSpheres",
     "double CollideTwoSpheres(const Vector3d& p0, const Vector3d& p1, const Vector3d& q0, const Vector3d& q1, "
     "const double collRadius);"),
    ("src/Partition.cpp", "collisionCase1, collisionCase3",
     "Partition::NearbyCriteria collisionCase1(1), collisionCase3(3);"),
    ("src/Ballpark.cpp", "Timer_ResolveBubbleConflicts", "PyObject* Timer_ResolveBubbleConflicts;"),
    ("src/Ballpark.cpp", "Timer_HandleProximities", "PyObject* Timer_HandleProximities;"),
    ("src/Ballpark.cpp", "Timer_InsertBallInBoxes", "PyObject* Timer_InsertBallInBoxes;"),
    ("src/Ballpark.cpp", "Timer_Gradient", "PyObject* Timer_Gradient;"),
    ("src/Ballpark.cpp", "Timer_CheckVisibility", "PyObject* Timer_CheckVisibility;"),
    ("src/Ballpark.cpp", "Timer_CopyBubbles", "PyObject* Timer_CopyBubbles;"),
    ("src/Ballpark.cpp", "Timer_AdditionsAndDeletions", "PyObject* Timer_AdditionsAndDeletions;"),
    ("src/Ballpark.cpp", "Timer_GetBallIdsInRange", "PyObject* Timer_GetBallIdsInRange;"),
    ("src/Ballpark.cpp", "Timer_WarpDistance", "PyObject* Timer_WarpDistance;"),
    ("src/Ballpark.cpp", "Timer_CalculateYawPitchRoll", "PyObject* Timer_CalculateYawPitchRoll;"),
    ("src/Ballpark.cpp", "Timer_WriteBallsToStream", "PyObject* Timer_WriteBallsToStream;"),
    ("src/Ballpark.cpp", "Plus", "PyObject *Plus;"),
    ("src/Ballpark.cpp", "Zero", "PyObject *Zero;"),
    ("src/Ballpark.cpp", "Minus", "PyObject *Minus;"),
    ("src/Ballpark.cpp", "TICK_EVOLVE", "const char* TICK_EVOLVE"),
    ("src/Ballpark.cpp", "SIGNUM (Ballpark.cpp)", "#define SIGNUM(X)"),
    ("src/Ballpark.cpp", "SIGNUMF (Ballpark.cpp)", "#define SIGNUMF( X )"),
    ("src/Ballpark.cpp", "ABS (Ballpark.cpp)", "#define ABS(X)"),
    ("src/Ballpark.cpp", "_PI_ (Ballpark.cpp)", "static constexpr double _PI_ ="),
    ("src/Ballpark.cpp", "s_ballNotInParkCallback", "PyObject * Ballpark::s_ballNotInParkCallback"),
    ("src/Ballpark.cpp", "ballparkCounter", "static int ballparkCounter"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_AU_PER_SECOND", "double WARP_FACTOR_TO_AU_PER_SECOND"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_ACCELERATION", "double WARP_FACTOR_TO_ACCELERATION"),
    ("src/Ballpark.cpp", "WARP_FACTOR_TO_DECELERATION", "double WARP_FACTOR_TO_DECELERATION"),
    ("src/Ballpark.cpp", "followModes", "extern const std::array<int,4> followModes"),
    ("src/Ballpark.cpp", "ORBITAL_PRECESSION", "double ORBITAL_PRECESSION"),
    ("src/Ball.cpp", "ABS (Ball.cpp)", "#define ABS(X)"),
    ("src/Ball.cpp", "MIN (Ball.cpp)", "#define MIN(X,Y)"),
    ("src/Ball.cpp", "SIGNUM (Ball.cpp)", "#define SIGNUM(X)"),
    ("src/Ball.cpp", "_HALFPI_", "#define _HALFPI_"),
    ("src/Ball.cpp", "_2PI_", "#define _2PI_"),
    ("src/Ball.cpp", "ballCount", "static int ballCount"),
    ("src/Ball.cpp", "declaration Modulo2pi", "void Modulo2pi(double& a,double& b);"),
    ("src/MiniBall.h", "TYPEDEF_BLUECLASS(MiniBall)", "TYPEDEF_BLUECLASS(MiniBall);"),
    ("src/MiniBox.h", "TYPEDEF_BLUECLASS(MiniBox)", "TYPEDEF_BLUECLASS(MiniBox);"),
    ("src/MiniCapsule.h", "TYPEDEF_BLUECLASS(MiniCapsule)", "TYPEDEF_BLUECLASS(MiniCapsule);"),
    ("src/Capsule.h", "TYPEDEF_BLUECLASS(Capsule)", "TYPEDEF_BLUECLASS( Capsule );"),
    ("src/OrientedBox.h", "TYPEDEF_BLUECLASS(OrientedBox)", "TYPEDEF_BLUECLASS( OrientedBox );"),
]

# carbonengine/math (CcpMath): the struct declarations (lines naming DirectXMath's types, Vector2 or Vector4 removed)
MATH_STRUCTS = [
    ("include/Vector3.h", "struct Vector3", r"struct Vector3\s*\{"),
    ("include/Quaternion.h", "struct Quaternion", r"struct Quaternion\s*\{"),
    ("include/Matrix.h", "struct Matrix", r"struct Matrix\s*\{"),
]
MATH_DROP = re.compile(r"XMVECTOR|XMMATRIX|Vector2|Vector4")

# carbonengine/math (CcpMath): the inline function definitions the extracted Destiny code calls, verbatim
MATH_BLOCKS = [
    ("include/Vector3_inline.h", "Vector3::Vector3()", r"inline constexpr Vector3::Vector3\(\) :"),
    ("include/Vector3_inline.h", "Vector3::Vector3(x, y, z)",
     r"inline constexpr Vector3::Vector3\( float x_, float y_, float z_ \) :"),
    ("include/Vector3_inline.h", "Vector3::operator[] const", r"inline float Vector3::operator\[\]\( int32_t index \) const"),
    ("include/Vector3_inline.h", "Vector3::operator[]", r"inline float& Vector3::operator\[\]\( int32_t index \)"),
    ("include/Vector3_inline.h", "Vector3::operator+=", r"inline Vector3& Vector3::operator\+=\( const Vector3& other \)"),
    ("include/Vector3_inline.h", "Vector3::operator-=", r"inline Vector3& Vector3::operator-=\( const Vector3& other \)"),
    ("include/Vector3_inline.h", "Vector3::operator*=(Vector3)",
     r"inline Vector3& Vector3::operator\*=\( const Vector3& other \)"),
    ("include/Vector3_inline.h", "Vector3::operator*=(float)", r"inline Vector3& Vector3::operator\*=\( float f \)"),
    ("include/Vector3_inline.h", "Vector3::operator/=(float)", r"inline Vector3& Vector3::operator/=\( float f \)"),
    ("include/Vector3_inline.h", "Vector3::operator+()", r"inline Vector3 Vector3::operator\+\(\) const"),
    ("include/Vector3_inline.h", "Vector3::operator-()", r"inline Vector3 Vector3::operator-\(\) const"),
    ("include/Vector3_inline.h", "Vector3::operator+",
     r"inline const Vector3 Vector3::operator\+\( const Vector3& other \) const"),
    ("include/Vector3_inline.h", "Vector3::operator-",
     r"inline const Vector3 Vector3::operator-\( const Vector3& other \) const"),
    ("include/Vector3_inline.h", "Vector3::operator*(Vector3)",
     r"inline const Vector3 Vector3::operator\*\( const Vector3& other \) const"),
    ("include/Vector3_inline.h", "Vector3::operator*(float)", r"inline const Vector3 Vector3::operator\*\( float f \) const"),
    ("include/Vector3_inline.h", "Vector3::operator/(float)", r"inline const Vector3 Vector3::operator/\( float f \) const"),
    ("include/Vector3_inline.h", "Vector3::operator==", r"inline bool Vector3::operator==\( const Vector3& other \) const"),
    ("include/Vector3_inline.h", "Vector3::operator!=", r"inline bool Vector3::operator!=\( const Vector3& other \) const"),
    ("include/Vector3_inline.h", "operator*(float, Vector3)", r"inline const Vector3 operator\*\( float f, const Vector3& other \)"),
    ("include/Vector3_inline.h", "Dot(Vector3)", r"inline float Dot\( const Vector3& v1, const Vector3& v2 \)"),
    ("include/Vector3_inline.h", "LengthSq(Vector3)", r"inline float LengthSq\( const Vector3& v \)"),
    ("include/Vector3_inline.h", "Length(Vector3)", r"inline float Length\( const Vector3& v \)"),
    ("include/Vector3_inline.h", "Normalize(Vector3)", r"inline Vector3 Normalize\( const Vector3& v \)"),
    ("include/Vector3_inline.h", "Cross(Vector3)", r"inline Vector3 Cross\( const Vector3& v1, const Vector3& v2 \)"),
    ("include/Quaternion_inline.h", "Quaternion::Quaternion()", r"inline constexpr Quaternion::Quaternion\(\) :"),
    ("include/Quaternion_inline.h", "Quaternion::Quaternion(x, y, z, w)",
     r"inline constexpr Quaternion::Quaternion\( float x_, float y_, float z_, float w_ \) :"),
    ("include/Quaternion_inline.h", "Quaternion::operator+=",
     r"inline Quaternion& Quaternion::operator\+=\( const Quaternion& other \)"),
    ("include/Quaternion_inline.h", "Quaternion::operator-=",
     r"inline Quaternion& Quaternion::operator-=\( const Quaternion& other \)"),
    ("include/Quaternion_inline.h", "Quaternion::operator*=(Quaternion)",
     r"inline Quaternion& Quaternion::operator\*=\( const Quaternion& other \)"),
    ("include/Quaternion_inline.h", "Quaternion::operator*=(float)",
     r"inline Quaternion& Quaternion::operator\*=\( float f \)"),
    ("include/Quaternion_inline.h", "Quaternion::operator/=(float)",
     r"inline Quaternion& Quaternion::operator/=\( float f \)"),
    ("include/Quaternion_inline.h", "Quaternion::operator+()", r"inline Quaternion Quaternion::operator\+\(\) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator-()", r"inline Quaternion Quaternion::operator-\(\) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator+",
     r"inline const Quaternion Quaternion::operator\+\( const Quaternion& other \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator-",
     r"inline const Quaternion Quaternion::operator-\( const Quaternion& other \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator*(Quaternion)",
     r"inline const Quaternion Quaternion::operator\*\( const Quaternion& other \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator*(float)",
     r"inline const Quaternion Quaternion::operator\*\( float f \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator/(float)",
     r"inline const Quaternion Quaternion::operator/\( float f \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator==",
     r"inline bool Quaternion::operator==\( const Quaternion& other \) const"),
    ("include/Quaternion_inline.h", "Quaternion::operator!=",
     r"inline bool Quaternion::operator!=\( const Quaternion& other \) const"),
    ("include/Quaternion_inline.h", "operator*(float, Quaternion)",
     r"inline const Quaternion operator\*\( float f, const Quaternion& other \)"),
    ("include/Quaternion_inline.h", "IdentityQuaternion", r"inline Quaternion IdentityQuaternion\(\)"),
    ("include/Quaternion_inline.h", "LengthSq(Quaternion)", r"inline float LengthSq\( const Quaternion& q \)"),
    ("include/Quaternion_inline.h", "Length(Quaternion)", r"inline float Length\( const Quaternion& q \)"),
    ("include/Quaternion_inline.h", "Normalize(Quaternion)", r"inline Quaternion Normalize\( const Quaternion& q \)"),
    ("include/Quaternion_inline.h", "Inverse(Quaternion)", r"inline Quaternion Inverse\( const Quaternion& q \)"),
    ("include/Quaternion_inline.h", "Conjugate(Quaternion)", r"inline Quaternion Conjugate\( const Quaternion& q \)"),
    ("include/Quaternion_inline.h", "Dot(Quaternion)",
     r"inline float Dot\( const Quaternion& q1, const Quaternion& q2 \)"),
    ("include/Quaternion_inline.h", "RotationQuaternion(axis, angle)",
     r"inline Quaternion RotationQuaternion\( const Vector3& axis, float angle \)"),
    ("include/Matrix_inline.h", "Matrix::Matrix()", r"inline constexpr Matrix::Matrix\(\) :"),
    ("include/Matrix_inline.h", "Matrix::Matrix(16 floats)", r"inline constexpr Matrix::Matrix\(\s*float f11"),
    ("include/Matrix_inline.h", "Matrix::operator() ", r"inline float& Matrix::operator\(\)\( uint32_t row"),
    ("include/Matrix_inline.h", "Matrix::operator() const", r"inline float Matrix::operator\(\)\( uint32_t row"),
    ("include/Matrix_inline.h", "RotationMatrix(Quaternion)", r"inline Matrix RotationMatrix\( const Quaternion& q \)"),
    ("include/Matrix_inline.h", "RotationMatrix(axis, angle)",
     r"inline Matrix RotationMatrix\( const Vector3& axis, float angle \)"),
    ("include/Matrix_inline.h", "TransformCoord",
     r"inline Vector3 TransformCoord\( const Vector3& coord, const Matrix& transform \)"),
]
# the one function that needs DirectXMath (src/Quaternion.cpp:60-82): extracted verbatim with DIRECTXMATH_REPO, stubbed
# at the identity without it
MATH_UNAVAILABLE = ("src/Quaternion.cpp", "RotationQuaternion(yaw, pitch, roll)",
                    r"Quaternion RotationQuaternion\( float yaw, float pitch, float roll \)")

# DirectXMath (microsoft/DirectXMath, MIT, (c) Microsoft Corporation): the pinned commit, and the headers the harness
# includes (DirectXMath.h and the four files it includes), each checked against its SHA-256 before the build
DXM_PINNED = "f5d82574b03417d905fdd2f85bfa9bf7a2f1deac"
DXM_HEADERS = {
    "Inc/DirectXMath.h": "489726b39be07e945cbbdbc19752c80367286ee2652af1da153d62cfce36ce87",
    "Inc/DirectXMathConvert.inl": "cb764b2e91b79ae3bcae650b1e15fbc99b507cf316f49c3d32644186bb94f6a5",
    "Inc/DirectXMathVector.inl": "8353782a3ef72070c526c5293c2e353b5c9e09cfb770112ed1df1185a0cb2e36",
    "Inc/DirectXMathMatrix.inl": "31fe84d5c869858f5b243b777f7e2f76fdfe57b3dc434844f274d6b7a4c15b6f",
    "Inc/DirectXMathMisc.inl": "3f1d7cb555e7a796c75344897f28f21635a3f83bca201a7650dd2b75f111850f",
}
DXM_DEFINE = "-D_XM_NO_INTRINSICS_"         # the scalar path (aarch64 would take NEON, Windows x64 SSE)
# with DirectXMath: the one conversion RotationQuaternion(yaw, pitch, roll) needs, kept in struct Quaternion, and its
# definition, verbatim; `using namespace DirectX;` as carbonengine/math's include/Requirements.h has it
MATH_KEEP_DXM = {"struct Quaternion": ("Quaternion( const XMVECTOR& other );",)}
MATH_BLOCKS_DXM = [
    ("include/Quaternion_inline.h", "Quaternion::Quaternion(XMVECTOR)",
     r"inline Quaternion::Quaternion\( const XMVECTOR& other \)"),
]
MATH_LINES_DXM = [("include/Requirements.h", "using namespace DirectX", "using namespace DirectX;")]

SAL_STUB = r"""// sal.h of the conformance kit's harness, not Microsoft's: DirectXMath.h includes "sal.h" (Microsoft's source-code annotation
// language), which Linux does not have. The annotations DirectXMath uses describe arguments to static analysers and
// carry no semantics for code generation: empty here.
#pragma once
#define _Analysis_assume_(expr)
#define _In_
#define _In_reads_(size)
#define _In_reads_bytes_(size)
#define _Out_
#define _Out_opt_
#define _Out_writes_(size)
#define _Out_writes_bytes_(size)
#define _Success_(expr)
#define _Use_decl_annotations_
"""

DXM_PRELUDE = r"""
// ---- prelude: DirectXMath (microsoft/DirectXMath at f5d82574, MIT, (c) Microsoft Corporation), included by
// path from its snapshot, scalar path (-D_XM_NO_INTRINSICS_), as carbonengine/math's include/Requirements.h includes it
// (beside DirectXPackedVector.h, not needed here); sal.h is the harness's own stub ----
#include "DirectXMath.h"
"""

PROBE_HEAD = r"""// dxm_probe: carbonengine/math's RotationQuaternion(yaw, pitch, roll) alone, compiled as the harness compiles
// it. Each input line holds three float bit patterns (yaw, pitch, roll; hexadecimal); each output line the quaternion's
// x, y, z, w bit patterns.
#include <math.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
"""

PROBE_MAIN = r"""
int main() {
    unsigned a, b, c;
    while (std::scanf("%x %x %x", &a, &b, &c) == 3) {
        uint32_t in[3] = {a, b, c}, out[4];
        float yaw, pitch, roll;
        std::memcpy(&yaw, &in[0], 4); std::memcpy(&pitch, &in[1], 4); std::memcpy(&roll, &in[2], 4);
        const Quaternion q = RotationQuaternion(yaw, pitch, roll);
        std::memcpy(&out[0], &q.x, 4); std::memcpy(&out[1], &q.y, 4);
        std::memcpy(&out[2], &q.z, 4); std::memcpy(&out[3], &q.w, 4);
        std::printf("%08x %08x %08x %08x\n", (unsigned)out[0], (unsigned)out[1], (unsigned)out[2], (unsigned)out[3]);
    }
    return 0;
}
"""

PRELUDE_A = r"""// Harness prelude: what the extracted Destiny and CcpMath code touches, with their types.
// <math.h> first: libstdc++'s C++ <math.h> brings std's float overloads into the global namespace (so CcpMath's
// unqualified sin(float) is sinf, as on MSVC, whose <cmath> declares them globally).
#include <math.h>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace Be { typedef int64_t Time; }                   // a 64-bit time stamp (100 ns units)
#define CCP_LOG_CH(...) ((void)0)                       // logging and profiling compiled out
#define CCP_LOGWARN_CH(...) ((void)0)
#define CCP_LOGERR_CH(...) ((void)0)
#define CCP_LOGWARN(...) ((void)0)
#define CCP_ASSERT(x) ((void)0)
#define CCP_STATS_ZONE(x)
#define TASKLET(A)
#define CCP_INT64_FORMAT "lld"
#define PARENTLOCK(x) x()                               // Blue's list locks
#define EXPOSE_TO_BLUE() IRoot* GetRawRoot() { return this; }   // Blue's class macros: plain classes here
#define BLUE_CLASS(X) class X
#define TYPEDEF_BLUECLASS(X) typedef X O##X
template <class F> struct OnBlockExit { F f; ~OnBlockExit() { f(); } };   // ON_BLOCK_EXIT, a scope guard
template <class F> OnBlockExit<F> MakeOnBlockExit(F f) { return OnBlockExit<F>{f}; }
#define ON_BLOCK_EXIT_CAT2(a, b) a##b
#define ON_BLOCK_EXIT_CAT(a, b) ON_BLOCK_EXIT_CAT2(a, b)
#define ON_BLOCK_EXIT(...) auto ON_BLOCK_EXIT_CAT(onBlockExit_, __LINE__) = MakeOnBlockExit(__VA_ARGS__)

// ---- prelude: a minimal stand-in for the CPython objects the extracted code uses. Ints are interned; a dict
// keeps its keys in insertion order, a re-set key keeps its place and a deleted key leaves a hole that PyDict_Next
// skips, as CPython's dict; sets are only counted and searched. Reference counts are not kept (nothing is freed). ----
typedef ssize_t Py_ssize_t;
struct PyObject {
    int kind;                                           // 1 int, 2 float, 3 dict, 4 set, 5 list or tuple, 6 None, 7 str
    long long ival = 0;
    double fval = 0.0;
    std::vector<PyObject*> keys, vals;                  // dict entries; a deleted entry's key is null
    std::unordered_map<long long, size_t> index;        // dict keys (ints) to their entry
    std::set<long long> members;                        // set
    std::vector<PyObject*> items;                       // list or tuple
    explicit PyObject(int k) : kind(k) {}
};
static PyObject g_pyNone(6);
#define Py_None (&g_pyNone)
#define Py_RETURN_NONE return Py_None
static PyObject PyTuple_Type(8), PyList_Type(9);        // type objects for "O!"
static inline void Py_INCREF(PyObject*) {}
static inline void Py_DECREF(PyObject*) {}
static inline void Py_XDECREF(PyObject*) {}
static PyObject* PyLong_FromLongLong(long long v) {
    static std::unordered_map<long long, PyObject*> cache;
    auto it = cache.find(v);
    if (it != cache.end()) return it->second;
    PyObject* o = new PyObject(1);
    o->ival = v;
    cache[v] = o;
    return o;
}
static PyObject* PyLong_FromLong(long v) { return PyLong_FromLongLong(v); }
static long long PyLong_AsLongLong(PyObject* o) { return o->ival; }
#define PyLong_AS_LONG(o) ((long)(o)->ival)
static PyObject* PyFloat_FromDouble(double d) { PyObject* o = new PyObject(2); o->fval = d; return o; }
static double PyFloat_AsDouble(PyObject* o) { return o->kind == 1 ? (double)o->ival : o->fval; }
static PyObject* PyUnicode_FromString(const char*) { return new PyObject(7); }
static PyObject* PyDict_New() { return new PyObject(3); }
static PyObject* PyDict_GetItem(PyObject* d, PyObject* k) {
    auto it = d->index.find(k->ival);
    return it == d->index.end() ? nullptr : d->vals[it->second];
}
static int PyDict_SetItem(PyObject* d, PyObject* k, PyObject* v) {
    auto it = d->index.find(k->ival);
    if (it != d->index.end()) { d->vals[it->second] = v; return 0; }
    d->index[k->ival] = d->keys.size();
    d->keys.push_back(k);
    d->vals.push_back(v);
    return 0;
}
static int PyDict_DelItem(PyObject* d, PyObject* k) {
    auto it = d->index.find(k->ival);
    if (it == d->index.end()) return -1;
    d->keys[it->second] = nullptr;
    d->vals[it->second] = nullptr;
    d->index.erase(it);
    return 0;
}
static Py_ssize_t PyDict_Size(PyObject* d) { return (Py_ssize_t)d->index.size(); }
static int PyDict_Next(PyObject* d, Py_ssize_t* pos, PyObject** k, PyObject** v) {
    for (size_t i = (size_t)*pos; i < d->keys.size(); ++i)
        if (d->keys[i]) {
            *pos = (Py_ssize_t)(i + 1);
            if (k) *k = d->keys[i];
            if (v) *v = d->vals[i];
            return 1;
        }
    *pos = (Py_ssize_t)d->keys.size();
    return 0;
}
static int PyDict_Clear(PyObject* d) { d->keys.clear(); d->vals.clear(); d->index.clear(); return 0; }
static PyObject* PySet_New(PyObject*) { return new PyObject(4); }
static int PySet_Add(PyObject* s, PyObject* k) { s->members.insert(k->ival); return 0; }
static int PySet_Discard(PyObject* s, PyObject* k) { return s->members.erase(k->ival) ? 1 : 0; }
static Py_ssize_t PySet_Size(PyObject* s) { return (Py_ssize_t)s->members.size(); }
static int PySet_Contains(PyObject* s, PyObject* k) { return s->members.count(k->ival) ? 1 : 0; }
static int PySet_Clear(PyObject* s) { s->members.clear(); return 0; }
static PyObject* PyList_New(Py_ssize_t n) {
    PyObject* o = new PyObject(5);
    o->items.assign((size_t)n, nullptr);
    return o;
}
static int PyList_Append(PyObject* l, PyObject* v) { l->items.push_back(v); return 0; }
static Py_ssize_t PyList_Size(PyObject* l) { return (Py_ssize_t)l->items.size(); }
#define PyList_GET_ITEM(l, i) ((l)->items[(size_t)(i)])
#define PyList_SET_ITEM(l, i, v) ((l)->items[(size_t)(i)] = (v))
static int PySequence_DelSlice(PyObject* l, Py_ssize_t a, Py_ssize_t b) {
    l->items.erase(l->items.begin() + a, l->items.begin() + b);
    return 0;
}
static PyObject* PyTuple_New(Py_ssize_t n) { return PyList_New(n); }
#define PyTuple_SET_ITEM(t, i, v) ((t)->items[(size_t)(i)] = (v))
static Py_ssize_t PyTuple_Size(PyObject* t) { return (Py_ssize_t)t->items.size(); }
static PyObject* PyTuple_GetItem(PyObject* t, Py_ssize_t i) { return t->items[(size_t)i]; }
static PyObject g_pyExcRuntimeError(7), g_pyExcTypeError(7), g_pyExcValueError(7);
static PyObject* PyExc_RuntimeError = &g_pyExcRuntimeError;
static PyObject* PyExc_TypeError = &g_pyExcTypeError;
static PyObject* PyExc_ValueError = &g_pyExcValueError;
static int g_pyErr = 0;                                 // the error indicator
static void PyErr_Clear() { g_pyErr = 0; }
static PyObject* PyErr_Occurred() { return g_pyErr ? PyExc_RuntimeError : nullptr; }
static PyObject* PyErr_Format(PyObject*, const char*, ...) { g_pyErr = 1; return nullptr; }
static void PyErr_SetString(PyObject*, const char*) { g_pyErr = 1; }
static void PyErr_Print() { g_pyErr = 0; }
// PyArg_ParseTuple for the formats the extracted thunkers use (L, i, b, d, f, O! and '|'), with CPython's checks: the
// count of arguments, an integer where L, i or b stands (a float there is a TypeError), and the type for O!
static int PyArg_ParseTuple(PyObject* args, const char* fmt, ...) {
    size_t required = 0, total = 0;
    bool optional = false;
    for (const char* f = fmt; *f; ++f) {
        if (*f == '|') { optional = true; continue; }
        if (*f == '!') continue;
        ++total;
        if (!optional) ++required;
    }
    const size_t n = args->items.size();
    if (n < required || n > total) { g_pyErr = 1; return 0; }
    va_list ap;
    va_start(ap, fmt);
    size_t i = 0;
    int ok = 1;
    for (const char* f = fmt; *f && i < n && ok; ++f) {
        if (*f == '|' || *f == '!') continue;
        PyObject* o = args->items[i++];
        switch (*f) {
            case 'L': if (o->kind != 1) ok = 0; else *va_arg(ap, long long*) = o->ival; break;
            case 'i': if (o->kind != 1) ok = 0; else *va_arg(ap, int*) = (int)o->ival; break;
            case 'b': if (o->kind != 1) ok = 0; else *va_arg(ap, char*) = (char)o->ival; break;
            case 'd': *va_arg(ap, double*) = o->kind == 1 ? (double)o->ival : o->fval; break;
            case 'f': *va_arg(ap, float*) = (float)(o->kind == 1 ? (double)o->ival : o->fval); break;
            case 'O':
                if (f[1] == '!') {
                    PyObject* type = va_arg(ap, PyObject*);
                    const int want = type == &PyTuple_Type || type == &PyList_Type ? 5 : -1;
                    if (o->kind != want) { ok = 0; break; }
                }
                *va_arg(ap, PyObject**) = o;
                break;
            default: ok = 0;
        }
    }
    va_end(ap);
    if (!ok) g_pyErr = 1;
    return ok;
}

struct IRoot {                                          // Blue's interfaces (IRoot made polymorphic for the printers)
    virtual ~IRoot() {}
    void Lock() {}                                      // Blue's reference counting: nothing is freed here
    void Unlock() {}
};
struct INotify : public IRoot {};
struct IEveBallpark {};
struct IBall { void Lock() {} void Unlock() {} };
template <class T> PyObject* BlueWrapObjectForPython(T*) { return Py_None; }
class Box;
class Ball;
class ClientBall;
class StaticCollidable;
class Key;
class Partition;
class Ballpark;
class Capsule;
class OrientedBox;
class MiniBall;
class MiniBox;
class MiniCapsule;
template <class T> struct BlueListStub {                // Blue's BlueList of minis: Insert(-1, x) appends, Remove(-1) clears
    std::vector<T*> items;
    ssize_t GetSize() { return (ssize_t)items.size(); }
    void* GetAt(ssize_t i) { return items[(size_t)i]; }
    void Insert(int, T* x) { items.push_back(x); }
    void Remove(int i) { if (i == -1) items.clear(); else std::abort(); }
};
struct PyOSStub {                                       // Python's event calls: recorded and printed
    bool PostEvent(IEveBallpark* park, const char* name, const char* method, const char* fmt, ...);
    bool SendEvent(IBall* ball, const char* name, const char* method, void* kw, const char* fmt, ...);
    bool SendEvent(IEveBallpark* park, const char* name, const char* method, void* kw, const char* fmt, ...);
    bool SendEvent(IRoot* root, const char* name, const char* method, void* kw, const char* fmt, ...);
    void PyError() {}
};
static PyOSStub g_pyos;
static PyOSStub* const PyOS = &g_pyos;
#define BEDEF 0
static int Clsid() { return 0; }
struct BeOSStub {                                       // Blue's tick registration (never reached) and error setter
    void RegisterForTicks(void*, void*) { std::abort(); }
    void UnregisterForTicks(void*, void*) { std::abort(); }
    void SetError(int, int, const char*, ...) { g_pyErr = 1; }
};
static BeOSStub g_beos;
static BeOSStub* const BeOS = &g_beos;
struct DirectXMathUnavailable {};                       // thrown where only DirectXMath would compute a value
struct BeTimer { void Reset() {} };                     // Blue's profiling timer (OnTick): unused
"""

MATH_STUB = r"""
// ---- prelude: carbonengine/math src/Quaternion.cpp:60, RotationQuaternion(yaw, pitch, roll), calls DirectXMath's
// XMQuaternionRotationRollPitchYaw, which is not in the snapshot. Only its value at +0, +0, +0 is supplied: the identity,
// exact in any implementation (sines 0, cosines 1). Any other angle stops the scenario. ----
Quaternion RotationQuaternion( float yaw, float pitch, float roll )
{
    const float z = 0.0f;
    if (std::memcmp(&yaw, &z, 4) == 0 && std::memcmp(&pitch, &z, 4) == 0 && std::memcmp(&roll, &z, 4) == 0)
        return Quaternion( 0.0f, 0.0f, 0.0f, 1.0f );
    throw DirectXMathUnavailable();
}
"""

PRELUDE_B = r"""
// ---- prelude, continued: the switches at Settings.h's defaults (useNewOrbit, the iterative collision method,
// dynamical orientation and its missile exclusion are set per case) ----
static size_t g_collisionMaxIterations = g_collisionMaxIterationsDefault;
static bool g_useIterativeCollision = g_useIterativeCollisionDefault;
static bool g_useDynamicalOrientation = g_useDynamicalOrientationDefault;
static bool g_disableDynamicalOrientationForMissiles = g_disableDynamicalOrientationForMissilesDefault;
static bool g_useNewOrbit = g_useNewOrbitDefault;
static long long g_case = 0;                            // the driver's case id, printed with every event
typedef Ball* BallPtr;                                  // Blue's smart pointer, a plain pointer here

class Ball : public INotify, public IBall, public Partitionable {   // Ball.h's members, in its order, with its types
friend class Ballpark;
friend class Partition;
public:
    Ball(IRoot* lockobj = NULL);                        // extracted from Ball.cpp below
    ~Ball();
    void RegisterBallNotInPark() { std::abort(); }      // a Python callback; never reached (mPark is always set)
    IRoot* GetRawRoot() { return this; }                // the ball's Blue object, for RemoveBall's last event
    ID mFollowId;
    ID mOwnerId;
    int64_t mHarmonic;
    Ballpark *mPark;                                    // as in Ball.h, this hides Partitionable::mPark
    Ball *mTrackingPtr;
    Ball *mFollowPtr;
    int32_t mAllianceID;
    int32_t mCorporationID;
    short mMod[3];
    char mFormationID;
    uint8_t isCloaked;
    bool isFree;
    bool isGlobal;
    bool isMassive;
    bool isSpaceJunk;
    bool mWithinTrackingRange;
    bool mHasProximity;
    bool mWithinRange;
    bool mHandled;                                      // as in Ball.h, this hides Partitionable::mHandled
    bool mInDeadBubble;
    float mRadius;
    float mMaxVel;
    float mAgility;
    float mMaxAngVel;
    float mRoll;
    float mRotAgility;
    float mSpeedFraction;
    float mFollowRange;
    double mMass;
    double mTimeFactor;
    double mLastCollision;
    double mNotificationRange;
    double mNewYaw;
    double mOldYaw;
    double mNewPitch;
    double mOldPitch;
    double mNewRoll;
    double mOldRoll;
    double mNewRollSpeed;
    double mOldRollSpeed;
    double mYawDelta;
    double mTrackingRange;
    std::bitset<16> mFormationSlots;
    Vector3d mNewVel;
    Vector3d mOldVel;
    Vector3d mGoto;
    Vector3 mNewAngVel;
    Vector3 mOldAngVel;
    Vector3 mTorque;
    Vector3d mLastG;
    Vector3d mLastC;
    Quaternion mNewRot;
    Quaternion mOldRot;
    DSTBALLMODE mMode;
    typedef BlueListStub<MiniBall> MiniBallList;
    typedef BlueListStub<MiniBox> MiniBoxList;
    typedef BlueListStub<MiniCapsule> MiniCapsuleList;
    MiniBallList mMiniBalls;
    MiniBoxList mMiniBoxes;
    MiniCapsuleList mMiniCapsules;
    Be::Time mNewTime;
    Be::Time mOldTime;
    SetOfIDs mFollowers;
    VectorOfIDs mCollisions;
    bool mActiveBoxes[27];
    ProximitySensor mSensor;
    long AddProximitySensor(float range, double period, int shuffle, bool onlyInteractives=false);
    void RemoveProximitySensor();
    void CheckForProximity();
    void CheckForProximity_NotificationRange();
    void CheckForProximity_TargetTracking();
    void CheckForProximity_Sensor();
    void DeleteFromBoxes() override;
    void ClearBoxes() override;
    void InsertInBox(Box* box) override;
    void InsertInBoxes(Box* box1, Box* top, long newBubbleId) override;
    double GetInflatedRadius(double dt) override;
    float GetBoundingRadius() override;
    void SetBoxActive(int boxId, bool isActive) override;
    void SnapOrientation();
    virtual void CalculateYawPitchRoll(bool snap=false);
    void DispatchCollisions();
    void UpdateProximityStatus();
    char ReserveFormationSlot();
    void FreeFormationSlot(size_t slot);
    void SetMode(DSTBALLMODE mode);
    void AddMiniBalls();
    void RemoveMiniBalls();
    void AddActualMiniball(MiniBall* b);
    void GetRotatedVector(Vector3&);
    void AddMiniBall(double x, double y, double z, float r);
    void AddMiniBoxes();
    void RemoveMiniBoxes();
    void AddActualMiniBox(MiniBox* b);
    void AddMiniBox(Vector3d corner, Vector3d localX, Vector3d localY, Vector3d localZ);
    void AddMiniCapsules();
    void RemoveMiniCapsules();
    void AddActualMinicapsule(MiniCapsule* c);
    void AddMiniCapsule(double ax, double ay, double az, double bx, double by, double bz, float r);
    bool IsWarpish();
    bool IsAligningForWarp();
    bool IsWarping();
    bool IsAlignedForWarp();
    void CalculateRelativeVelocities( Ball* target, bool calcRelative, bool calcRadial, bool calcTransversal,
                                      bool calcAngular, Vector3d& relVel, double& radVel, double& shortDist,
                                      double& transVel, double& angVel );
    PyObject* PyAddMiniBall(PyObject* args);
    PyObject* PyAddMiniCapsule(PyObject* args);
    PyObject* PyAddMiniBox(PyObject* args);
};
class ClientBall : public Ball {                        // a client's ball: only GetShiftedTime of ClientBall is compiled
public:
    Be::Time GetShiftedTime(Be::Time time);
};
typedef Ball OBall;                                     // Blue's object classes
typedef ClientBall OClientBall;

class MapOfBalls {                                      // MapOfBalls.h over a Python dict: CPython's insertion order
public:
    std::vector<std::pair<ID, Ball*>> entries;          // a removed entry keeps its place with a null ball
    std::unordered_map<ID, size_t> index;
    Ball* operator[](const ID id) {
        auto it = index.find(id);
        return it == index.end() ? nullptr : entries[it->second].second;
    }
    void Add(const ID id, Ball* ball) {
        auto it = index.find(id);
        if (it != index.end()) { entries[it->second].second = ball; return; }
        index[id] = entries.size();
        entries.push_back(std::make_pair(id, ball));
    }
    bool RemoveBall(const ID id) {
        auto it = index.find(id);
        if (it == index.end()) return false;
        entries[it->second].second = nullptr;
        index.erase(it);
        return true;
    }
    size_t GetSize() { return index.size(); }
    void Clear() { entries.clear(); index.clear(); }
};
class BallIterator {                                    // PyDict_Next over MapOfBalls
public:
    BallIterator(MapOfBalls* m) : mMapOfBalls(m), pos(0) {}
    MapOfBalls* mMapOfBalls;
    size_t pos;
    Ball* operator++(int) { return Next(); }
    Ball* Next() {
        while (pos < mMapOfBalls->entries.size()) {
            Ball* b = mMapOfBalls->entries[pos++].second;
            if (b) return b;
        }
        return NULL;
    }
    void Begin() { pos = 0; }
};

__COLLISION_BALL_PROPERTIES__

class Ballpark : public IEveBallpark {                  // Ballpark.h's members, with its types
public:
    static PyObject* s_ballNotInParkCallback;
    Be::Time mTime;
    int mTickInterval;
    long mCurrentTime;
    double mLagDamping;
    ID mEgo;
    ID mProbe;
    ID mLocalCnt;
    ID mLocalHiCnt;
    double mFriction;
    double mWarpSpeed;
    ID mSolarsystemID;
    float mSomeWeirdHackToFixSomething;
    double mPara1;
    double mPara2;
    int mInt1;
    int mInt2;
    int mMoribundBallRemovalCount;
    int mMoribundBallRemovalBuffer;
    double mRollSpeedAcceleration;
    double mRollSpeedDecay;
    double mRollAcceleration;
    double mRollDecay;
    double dt;
    ID mLastEgo;
    ClientBall *mLastEgoBall;
    MapOfBalls mBalls;
    MapOfBalls mGlobals;
    std::unordered_map<ID, StaticCollidable*> mStaticCollidables;
    std::unordered_map<ID, Capsule*> mCapsules;
    std::unordered_map<ID, OrientedBox*> mOrientedBoxes;
    SetOfBalls mBubbleConflicts;
    PyObject *bubbleInteractives;
    PyObject *bubbleKeepAlives;
    PyObject *bubbleKeepAliveBalls;
    PyObject *bubbles;
    PyObject *mBubbleSubscriptions;
    SetOfBalls moribundBalls;
    SetOfCapsules moribundCapsules;
    SetOfOrientedBoxes moribundOrientedBoxes;
    bool mHaveTicks;
    bool mFirstTime;
    bool isMaster;
    bool inEvolve;
    Partition *mPartition;
    Vector3d mLastReference;
    Vector3d mOldReference;
    Vector3d mLastDelta;
    Be::Time mLastReferenceTime;
    DictOfFreeBalls mFreeBalls;
    DictOfBalls mProximityBalls;
    std::vector<VectorOfVectors> mFormations;
    long mBubbleId;
    std::unordered_map<ID, CollisionBallProperties> mCollisionLocations;
    std::unordered_map<ID, VectorOfBalls> mPotentialCollisionBalls;
    std::unordered_map<ID, VectorOfStaticCollidables> mPotentialCollidables;
    // extracted below, declared as in Ballpark.h
    Ballpark(IRoot* lockobj = NULL);
    ~Ballpark();
    void Start();
    void Pause();
    void OnTick(Be::Time realTime, Be::Time simTime, void* cookie);
    void Evolve(Be::Time timestamp);
    void EvolveBehaviorForBall(Ball* ball);
    Vector3d EvolveWarp(Ball* ball);
    Vector3d EvolveGoto(Ball* ball);
    Vector3d EvolveMissile(Ball* ball);
    Vector3d EvolveFormation(Ball* ball);
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
    Vector3 GotoTorqueBreaks( const Ball* ball, const Vector3& direction );
    float GotoRollBreaks( const Ball* ball, const Vector3& headingTorque );
    void ApplyTorque( Quaternion& rotation, Vector3& angvel, const Vector3& torque, const float roll, const float tau,
                      float timeStepFraction );
    Vector3d GetOrbitalNormal( Ball* ball, Vector3d& toVector, double& dist );
    Vector3d DefaultOrbitalNormal( Ball* ball );
    void CalculateBallRotationVelocity( Ball* ball, double timeStepFraction, Quaternion& rotation, Vector3& omega );
    void CalculateBallPositionVelocity(Ball* ball, double t, Vector3d& position, Vector3d& velocity);
    void Gradient(Ball* b);
    void Potential(Ball *me, Ball *other, int recurionsDepth = 0);
    void InsertInBoxes(Partitionable *Partitionable);
    void InsertBallInBoxes(Ball *ball);
    void InsertInBox(Box *box,Partitionable *p);
    void InsertStaticCollidableInBoxes(StaticCollidable* collidable);
    Ball *BallExists(const ID& id);
    ID GetIDForCollisionObject(ID objectId);
    Ball* AddBall(const ID& objectId, double mass, float radius, float maxVel, bool isFree, bool isGlobal,
                  bool isMassive, bool isInteractive, bool isSpaceJunk, double x, double y, double z, double vx,
                  double vy, double vz, float agility, float speedFraction);
    Ball* AddOldStyleOrientedBall(const ID& objectId, double mass, float radius, float maxVel, bool isFree,
                                  bool isGlobal, bool isMassive, bool isInteractive, bool isSpaceJunk, double x,
                                  double y, double z, double vx, double vy, double vz, float agility,
                                  float speedFraction);
    Ball* AddDynamicallyOrientedBall(const ID& objectId, double mass, float radius, float maxVel, float maxAngVel,
                                     bool isFree, bool isGlobal, bool isMassive, bool isInteractive, bool isSpaceJunk,
                                     double x, double y, double z, double vx, double vy, double vz, float rx, float ry,
                                     float rz, float rw, float wx, float wy, float wz, float agility, float rotAgility,
                                     float speedFraction);
    Ball* AddMiniball(const ID& parentObjectId, double mass, float radius, double x, double y, double z);
    Capsule* AddCapsule(const ID& objectId, const ID& parentObjectId, double ax, double ay, double az, double bx,
                        double by, double bz, float radius);
    OrientedBox* AddOrientedBox(const ID& objectId, const ID& parentObjectId, double corner_0, double corner_1,
                                double corner_2, double x0, double x1, double x2, double y0, double y1, double y2,
                                double z0, double z1, double z2);
    void AddMushroom(const ID& ownerId, float range, double time);
    void MissileFollow(const ID& srcId, const ID& dstId, const ID& ownerId);
    void FollowBall(const ID& srcId, const ID& dstId, float range);
    void FormationFollow(const ID& srcId, const ID& dstId, char slot);
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
    void SetBallMass(const ID& srcId, double mass);
    void SetBallRadius(const ID& srcId, float radius);
    void SetBallRadius(Ball *ball, float radius);
    void SetMaxSpeed(const ID& srcId, float speed);
    void SetMaxAngularSpeed( const ID& srcId, float maxAngVel );
    void SetBallPosition(const ID& srcId, double x, double y, double z);
    void SetBallVelocity(const ID& srcId, double vx, double vy, double vz);
    void SetBallAngularVelocity( const ID& srcId, double wx, double wy, double wz );
    void SetBallRotation( const ID& srcId, double rx, double ry, double rz, double rw );
    void SetBallHarmonic(const ID& srcId, int64_t harmonic, int corporationID, int allianceID, bool field);
    void SetBallFree(const ID& srcId, bool flag);
    void SetBallAgility(const ID& srcId, float agility);
    void SetBallAngularAgility( const ID& srcId, float rotAgility );
    void SetSpeedFraction(const ID& srcId, float fraction);
    void SetNotificationRange(const ID& srcId, double notificationRange);
    void SetTargetTracking(const ID& srcID, const ID& targetID, double trackingRange);
    void SetBallMassive(const ID& srcId, bool flag);
    void SetBallGlobal(const ID& srcId, bool flag);
    void AddProximitySensor(const ID& srcId, float range, double period, int shuffle, bool onlyInteractives);
    void RemoveProximitySensor(const ID& srcId);
    double GetAccuracy(const ID& srcId, const ID& dstId, double& optimalRange, double falloff, double& trackingSpeed,
                       double signatureRadius, double optimalSignatureRadius);
    void CloakBall(const ID& srcId, char cloakMode, float range);
    void UncloakBall(const ID& srcId);
    ID CheckVisibility(const ID& srcId, const ID& dstId);
    void HandleProximities();
    void ClearBubbles();
    void ClearAll();
    void UpdateBallBubble(const ID& srcID);
    void CopyBubbles();
    bool InDeadBubble(Ball *ball);
    size_t GetInteractiveCnt(Ball *ball);
    void SetBallRigid(const ID& srcId);
    void SetBallTroll(const ID& srcId, int delay);
    bool TrollReady(Ball *ball);
    void PetrifyTroll(Ball *ball);
    void SetBallInteractive(const ID& srcId, bool flag);
    void IncreaseInteractiveCnt(Partitionable *p, long inBubble);
    void DecreaseInteractiveCnt(Partitionable *p, long inBubble);
    void UpdateInteractiveCnt(Partitionable *p, long oldBubble, long newBubble);
    void AddKeepAliveBall(PyObject *ballId, long inBubble);
    void RemoveKeepAliveBall(PyObject *ballId, long inBubble);
    void UpdateKeepAliveBalls(Partitionable *p, long oldBubble, long newBubble);
    void AddToBubble(Partitionable *partitionable);
    void InitializeBubbles();
    void AddFormation(PyObject *vectorList);
    void SetBallFormation(const ID& srcId, char formationID);
    PyObject* PyAddDynamicallyOrientedBall(PyObject* args);
    PyObject* PyAddOldStyleOrientedBall(PyObject* args);
    PyObject* PyAddBall(PyObject* args);
    PyObject* PyAddCapsule(PyObject* args);
    PyObject* PyAddOrientedBox(PyObject* args);
    PyObject* PyMissileFollow(PyObject* args);
    PyObject* PyAddMushroom(PyObject* args);
    PyObject* PySetBallRigid(PyObject* args);
    PyObject* PySetBallTroll(PyObject* args);
    PyObject* PySetMaxAngularSpeed(PyObject* args);
    PyObject* PySetBallAngularVelocity(PyObject* args);
    PyObject* PySetBallRotation(PyObject* args);
    PyObject* PySetBallAngularAgility(PyObject* args);
    PyObject* PyLaunchMissile(PyObject* args);
    PyObject* PySetBoid(PyObject* args);
    PyObject* PyFormationFollow(PyObject* args);
    PyObject* PyInitializeBubbles(PyObject* args);
    PyObject* PyGetRemoteFollowers(PyObject* args);
    PyObject* PyEvolve(PyObject* args);
    PyObject* PyGetBallIdsInRange(PyObject* args);
    PyObject* PyGetBallIdsAndDistInRange(PyObject* args);
    PyObject* GetBallIdsInRange(PyObject* args, bool includeDist);
    bool BallIsValidForIteration(ID ballID, Ball*& ball, const Ball* refBall);
    int GetNextValidBallInBubble(PyObject* theBubble, Py_ssize_t& pos, PyObject*& key, PyObject*& value,
                                 Ball*& otherBall, const Ball* refBall);
    PyObject* PyGetCenterDist(PyObject* args);
    PyObject* PyGetSurfaceDist(PyObject* args);
    PyObject* PyCheckVisibility(PyObject* args);
    PyObject* PyCloakBall(PyObject* args);
    PyObject* PyUncloakBall(PyObject* args);
    PyObject* PySetBallFormation(PyObject* args);
    PyObject* PyLoadFormations(PyObject* args);
    PyObject* PyGetBubbleAtCoordinates(PyObject* args);
    PyObject* PyCopyBubbles(PyObject* args);
    PyObject* PyGetAccuracy(PyObject* args);
    void RemoveBall(const ID& srcId, int delay = 0);
    void RemoveCapsule(const ID& srcId, int delay = 0);
    void RemoveOrientedBox(const ID& srcId, int delay = 0);
    void BringOutTheDeads();
    void BringOutDeadBalls();
    void BringOutDeadCapsules();
    void CalculateIterativeCollisionResponses();
    bool CalculateCollisionResponse(Ball* ball, size_t collisionIteration);
    void GetNextCollisionStatic(CollisionBallProperties& ballProp, CollisionItem& collisionItem, VectorOfBalls& balls,
                                VectorOfStaticCollidables& collidables);
    void GetNextCollisionFree(CollisionBallProperties& ballProp, CollisionItem& collisionItem, VectorOfBalls& balls,
                              size_t collisionIteration);
    Vector3d ApproximateEllipsePointOfImpact(Ball* ball, BallLocation& ballLoc, const CollisionItem& collision);
    void CalculateCollisionImpactStatic(CollisionBallProperties& ballProp, CollisionItem& collision);
    CollisionItem CalculateCollisionImpactFree(CollisionBallProperties& ballProp, CollisionBallProperties& otherProp,
                                               CollisionItem& collision);
    void AddImpact(CollisionImpact& impact, size_t collIter, Ball* ball);
    void RemoveImpactFromFree(double time, ID collider, Ball* ball);
    bool CollisionBallNeighborIgnore(Ball* b, Ball* neighbor);
};
"""

PRELUDE_C = r"""
// ---- prelude: the event printers. Each argument is printed typed: i<integer> or d<double, %.17g>. ----
static void print_args(const char* fmt, va_list ap) {
    int n = 0;
    char buf[1024];
    size_t used = 0;
    for (const char* f = fmt; *f; ++f) {
        if (*f == '(' || *f == ')') continue;
        if (*f == 'L') used += std::snprintf(buf + used, sizeof(buf) - used, " i%lld", va_arg(ap, long long));
        else if (*f == 'i' || *f == 'b') used += std::snprintf(buf + used, sizeof(buf) - used, " i%d", va_arg(ap, int));
        else if (*f == 'd') used += std::snprintf(buf + used, sizeof(buf) - used, " d%.17g", va_arg(ap, double));
        else if (*f == 'O') {
            (void)va_arg(ap, PyObject*);
            used += std::snprintf(buf + used, sizeof(buf) - used, " o");
        }
        else std::abort();
        ++n;
    }
    std::printf(" %d%s\n", n, used ? buf : "");
}
bool PyOSStub::PostEvent(IEveBallpark* park, const char*, const char* method, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("E %lld %ld %s -1", g_case, static_cast<Ballpark*>(park)->mCurrentTime, method);
    print_args(fmt, ap);
    va_end(ap);
    return true;
}
bool PyOSStub::SendEvent(IBall* ib, const char*, const char* method, void*, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    Ball* b = static_cast<Ball*>(ib);
    std::printf("E %lld %ld %s %lld", g_case, b->mPark ? b->mPark->mCurrentTime : -1L, method, (long long)b->mId);
    print_args(fmt, ap);
    va_end(ap);
    return true;
}
bool PyOSStub::SendEvent(IRoot* root, const char*, const char* method, void*, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    Partitionable* p = nullptr;
    Ballpark* park = nullptr;
    if (Ball* b = dynamic_cast<Ball*>(root)) { p = b; park = b->mPark; }
    else if (Capsule* c = dynamic_cast<Capsule*>(root)) { p = c; park = c->mPark; }
    else if (OrientedBox* o = dynamic_cast<OrientedBox*>(root)) { p = o; park = o->mPark; }
    else std::abort();
    std::printf("E %lld %ld %s %lld", g_case, park ? park->mCurrentTime : -1L, method, (long long)p->mId);
    print_args(fmt, ap);
    va_end(ap);
    return true;
}
bool PyOSStub::SendEvent(IEveBallpark* park, const char*, const char* method, void*, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("E %lld %ld %s -2", g_case, static_cast<Ballpark*>(park)->mCurrentTime, method);
    print_args(fmt, ap);
    va_end(ap);
    return true;
}
"""

DRIVER = r"""
// ---- driver: scenarios on stdin; after every step every ball's and static collidable's state, the partition,
// the bubble dictionaries and the park's clock; every event and every query's answer, on stdout ----
struct Record {                                         // B: the Python AddBall; K: a command, a setter or a query
    char kind; int step, op, w; long long a, b, c;
    std::vector<double> d;                              // B: mass radius maxVel x y z vx vy vz agility sf maxAngVel
    bool f[5];                                          //    rotAgility; B flags: free global massive interactive junk
};

static uint64_t mix(uint64_t h, uint64_t v) { h ^= v; h *= 1099511628211ULL; return h; }

static uint64_t box_hash(Box* box) {                    // a box's canonical content, folded
    uint64_t h = 14695981039346656037ULL;
    h = mix(h, (uint64_t)box->mLevel);
    h = mix(h, (uint64_t)box->mKey.ix);
    h = mix(h, (uint64_t)box->mKey.iy);
    h = mix(h, (uint64_t)box->mKey.iz);
    h = mix(h, (uint64_t)(int64_t)box->mBubble);
    h = mix(h, (uint64_t)box->mHandled);
    h = mix(h, (uint64_t)box->balls.size());
    for (Ball* b : box->balls) h = mix(h, (uint64_t)b->mId);
    h = mix(h, (uint64_t)box->mStaticCollidables.size());
    for (StaticCollidable* s : box->mStaticCollidables) h = mix(h, (uint64_t)s->mId);
    h = mix(h, (uint64_t)box->mChildren.size());
    for (Box* c : box->mChildren) {
        h = mix(h, (uint64_t)c->mKey.ix);
        h = mix(h, (uint64_t)c->mKey.iy);
        h = mix(h, (uint64_t)c->mKey.iz);
    }
    h = mix(h, box->mParent ? (uint64_t)box->mParent->mLevel : 99ULL);
    return h;
}

static void print_boxes(const SortedSetOfBoxes& boxes) {
    std::printf(" %zu", boxes.size());
    for (Box* box : boxes)
        std::printf(" %ld %lld %lld %lld", box->mLevel, (long long)box->mKey.ix, (long long)box->mKey.iy,
                    (long long)box->mKey.iz);
}

static void print_d(std::initializer_list<double> xs) { for (double x : xs) std::printf(" %.17g", x); }

static void print_ball(Ballpark* bp, long long cid, int step, Ball* b) {
    Box* mb = b->mMyBox;
    long long active = 0;
    for (int i = 0; i < 27; ++i) if (b->mActiveBoxes[i]) active |= (1LL << i);
    std::printf("S %lld %d %lld %d %ld %d %d %lld %lld %lld %d %d %d %d %d %d %d %ld %ld %d %d %d %lld %d %d %d %d "
                "%lld %ld %lld %lld %lld %lld %d %d %d %d %lu %lld %lld",
                cid, step, (long long)b->mId, (int)b->mMode, b->mEffectStamp, (int)b->isMassive,
                (int)b->mSensor.active, (long long)b->mFollowId, (long long)b->mOwnerId,
                b->mFollowPtr ? (long long)b->mFollowPtr->mId : -1LL, (int)b->isFree,
                (int)(bp->mFreeBalls.count(b->mId) > 0), (int)b->isGlobal, (int)b->isInteractive,
                (int)b->isCloaked, (int)b->isMoribund, (int)b->isSpaceJunk, b->mNewBubble, b->mOldBubble,
                (int)b->mHasProximity, (int)b->mWithinRange, (int)b->mWithinTrackingRange,
                b->mTrackingPtr ? (long long)b->mTrackingPtr->mId : -1LL, (int)b->mInDeadBubble,
                (int)b->mMod[0], (int)b->mMod[1], (int)b->mMod[2], active,
                mb ? mb->mLevel : -1L, mb ? (long long)mb->mKey.ix : 0LL, mb ? (long long)mb->mKey.iy : 0LL,
                mb ? (long long)mb->mKey.iz : 0LL, (long long)b->mHarmonic, (int)b->mCorporationID,
                (int)b->mAllianceID, (int)b->mSensor.onlyInteractives, (int)b->mFormationID,
                b->mFormationSlots.to_ulong(), (long long)b->mNewTime, (long long)b->mOldTime);
    print_d({b->mNewPos.x, b->mNewPos.y, b->mNewPos.z, b->mNewVel.x, b->mNewVel.y, b->mNewVel.z,
             b->mOldPos.x, b->mOldPos.y, b->mOldPos.z, b->mOldVel.x, b->mOldVel.y, b->mOldVel.z,
             b->mGoto.x, b->mGoto.y, b->mGoto.z, b->mLastG.x, b->mLastG.y, b->mLastG.z,
             b->mLastC.x, b->mLastC.y, b->mLastC.z, b->mMass, (double)b->mRadius,
             (double)b->mMaxVel, (double)b->mAgility, (double)b->mSpeedFraction,
             (double)b->mFollowRange, b->mLastCollision, b->mTimeFactor, b->mNotificationRange,
             b->mTrackingRange, (double)b->mSensor.range, b->mSensor.period, b->mSensor.elapsed,
             b->mNewYaw, b->mOldYaw, b->mNewPitch, b->mOldPitch, b->mNewRoll, b->mOldRoll, b->mNewRollSpeed,
             b->mOldRollSpeed, b->mYawDelta, (double)b->mMaxAngVel, (double)b->mRoll, (double)b->mRotAgility,
             (double)b->mNewRot.x, (double)b->mNewRot.y, (double)b->mNewRot.z, (double)b->mNewRot.w,
             (double)b->mOldRot.x, (double)b->mOldRot.y, (double)b->mOldRot.z, (double)b->mOldRot.w,
             (double)b->mNewAngVel.x, (double)b->mNewAngVel.y, (double)b->mNewAngVel.z,
             (double)b->mOldAngVel.x, (double)b->mOldAngVel.y, (double)b->mOldAngVel.z,
             (double)b->mTorque.x, (double)b->mTorque.y, (double)b->mTorque.z});
    print_boxes(b->mBoxes);
    std::printf(" %zu", b->mCollisions.size());
    for (ID c : b->mCollisions) std::printf(" %lld", (long long)c);
    std::vector<long long> fs(b->mFollowers.begin(), b->mFollowers.end());
    std::sort(fs.begin(), fs.end());
    std::printf(" %zu", fs.size());
    for (long long f : fs) std::printf(" %lld", f);
    std::printf(" %zu", b->mSensor.balls.size());                          // in the unordered_map's order
    for (auto& kv : b->mSensor.balls) std::printf(" %lld", (long long)kv.first);
    std::printf(" %zd", b->mMiniBalls.GetSize());
    for (MiniBall* m : b->mMiniBalls.items) {
        std::printf(" %lld", (long long)m->mId);
        print_d({(double)m->mRadius, m->mPos.x, m->mPos.y, m->mPos.z});
    }
    std::printf(" %zd", b->mMiniCapsules.GetSize());
    for (MiniCapsule* m : b->mMiniCapsules.items) {
        std::printf(" %lld %lld", (long long)m->mId, (long long)m->mParentBallID);
        print_d({m->mHemisphereA.x, m->mHemisphereA.y, m->mHemisphereA.z, m->mHemisphereB.x, m->mHemisphereB.y,
                 m->mHemisphereB.z, (double)m->mRadius});
    }
    std::printf(" %zd", b->mMiniBoxes.GetSize());
    for (MiniBox* m : b->mMiniBoxes.items) {
        std::printf(" %lld %lld", (long long)m->mId, (long long)m->mParentBallID);
        print_d({m->mCorner.x, m->mCorner.y, m->mCorner.z, m->mLocalX.x, m->mLocalX.y, m->mLocalX.z,
                 m->mLocalY.x, m->mLocalY.y, m->mLocalY.z, m->mLocalZ.x, m->mLocalZ.y, m->mLocalZ.z});
    }
    std::printf("\n");
}

static void print_static(long long cid, int step, int kind, StaticCollidable* s) {
    Box* mb = s->mMyBox;
    std::printf("T %lld %d %d %lld %lld %d %ld %ld %ld %d %d %ld %lld %lld %lld", cid, step, kind, (long long)s->mId,
                (long long)s->mParentBallId, (int)s->isMoribund, s->mEffectStamp, s->mNewBubble, s->mOldBubble,
                (int)s->isInteractive, (int)s->mHandled, mb ? mb->mLevel : -1L, mb ? (long long)mb->mKey.ix : 0LL,
                mb ? (long long)mb->mKey.iy : 0LL, mb ? (long long)mb->mKey.iz : 0LL);
    print_d({s->mNewPos.x, s->mNewPos.y, s->mNewPos.z, s->mOldPos.x, s->mOldPos.y, s->mOldPos.z,
             (double)s->GetBoundingRadius()});
    print_boxes(s->mBoxes);
    std::printf("\n");
}

static void print_state(Ballpark* bp, long long cid, int step) {
    BallIterator it(&bp->mBalls);
    Ball* b;
    while ((b = it++)) print_ball(bp, cid, step, b);
    // the static collidables by id; RemoveCapsule and RemoveOrientedBox read mCapsules[id] and mOrientedBoxes[id]
    // with operator[], which inserts a null entry for an id not present: those entries are counted in P, not printed
    std::vector<std::pair<long long, Capsule*>> caps;
    for (auto& kv : bp->mCapsules) if (kv.second) caps.push_back(std::make_pair((long long)kv.first, kv.second));
    std::sort(caps.begin(), caps.end());
    for (auto& kv : caps) print_static(cid, step, 0, kv.second);
    std::vector<std::pair<long long, OrientedBox*>> obbs;
    for (auto& kv : bp->mOrientedBoxes) if (kv.second) obbs.push_back(std::make_pair((long long)kv.first, kv.second));
    std::sort(obbs.begin(), obbs.end());
    for (auto& kv : obbs) print_static(cid, step, 1, kv.second);
    uint64_t total = 0;
    size_t counts[8];
    for (int l = 0; l < 8; ++l) {
        counts[l] = bp->mPartition->mLevels[l].size();
        for (auto& kv : bp->mPartition->mLevels[l]) total += box_hash(kv.second);
    }
    std::printf("P %lld %d %ld %ld %llu", cid, step, bp->mCurrentTime, bp->mBubbleId, (unsigned long long)total);
    for (int l = 0; l < 8; ++l) std::printf(" %zu", counts[l]);
    std::printf(" %zu", bp->mProximityBalls.size());                            // in the unordered_map's order
    for (auto& kv : bp->mProximityBalls) std::printf(" %lld", (long long)kv.first);
    std::printf(" %zu %lld %d %lld %lld %zu %zu %zu %zu %zu %zu\n", bp->mFreeBalls.size(), (long long)bp->mTime,
                (int)bp->mFirstTime, (long long)bp->mLocalCnt, (long long)bp->mLocalHiCnt, bp->mFormations.size(),
                bp->moribundBalls.size(), bp->moribundCapsules.size(), bp->moribundOrientedBoxes.size(),
                bp->mCapsules.size(), bp->mOrientedBoxes.size());
    PyObject *k, *v;
    Py_ssize_t pos = 0;
    while (PyDict_Next(bp->bubbles, &pos, &k, &v)) {                          // the bubbles, in the dict's order
        std::printf("U %lld %d %lld %zd", cid, step, k->ival, PyDict_Size(v));
        PyObject *k2, *v2;
        Py_ssize_t pos2 = 0;
        while (PyDict_Next(v, &pos2, &k2, &v2)) std::printf(" %lld %lld", k2->ival, v2->ival);
        std::printf("\n");
    }
    const char tags[2] = {'I', 'V'};
    PyObject* dicts[2] = {bp->bubbleInteractives, bp->bubbleKeepAlives};
    for (int t = 0; t < 2; ++t) {
        pos = 0;
        while (PyDict_Next(dicts[t], &pos, &k, &v)) {
            std::printf("%c %lld %d %lld %zd", tags[t], cid, step, k->ival, PySet_Size(v));
            for (long long m : v->members) std::printf(" %lld", m);
            std::printf("\n");
        }
    }
    std::printf("W %lld %d %zd", cid, step, PySet_Size(bp->bubbleKeepAliveBalls));
    for (long long m : bp->bubbleKeepAliveBalls->members) std::printf(" %lld", m);
    std::printf("\n");
}

static void dump_partition(Ballpark* bp, long long cid) {   // every box, sorted by level and key
    for (int l = 0; l < 8; ++l) {
        std::vector<Box*> boxes;
        for (auto& kv : bp->mPartition->mLevels[l]) boxes.push_back(kv.second);
        std::sort(boxes.begin(), boxes.end(), [](Box* a, Box* b) { return a->mKey < b->mKey; });
        for (Box* box : boxes) {
            std::printf("D %lld %ld %lld %lld %lld %ld %d %.17g %.17g %.17g %.17g %.17g %.17g %zu", cid, box->mLevel,
                        (long long)box->mKey.ix, (long long)box->mKey.iy, (long long)box->mKey.iz, box->mBubble,
                        (int)box->mHandled, box->lo.x, box->lo.y, box->lo.z, box->hi.x, box->hi.y, box->hi.z,
                        box->balls.size());
            for (Ball* b : box->balls) std::printf(" %lld", (long long)b->mId);
            std::printf(" %zu", box->mStaticCollidables.size());
            for (StaticCollidable* s : box->mStaticCollidables) std::printf(" %lld", (long long)s->mId);
            std::printf(" %zu", box->mChildren.size());
            for (Box* c : box->mChildren)
                std::printf(" %lld %lld %lld", (long long)c->mKey.ix, (long long)c->mKey.iy, (long long)c->mKey.iz);
            std::printf("\n");
        }
    }
}

// a query's answer: Q case step op a b status n, then the items typed (i<integer> or d<double>)
static void print_q(long long cid, int step, int op, long long a, long long b, int status,
                    const std::vector<std::pair<char, double>>& items, const std::vector<long long>& ints = {}) {
    std::printf("Q %lld %d %d %lld %lld %d %zu", cid, step, op, a, b, status, items.size() + ints.size());
    for (long long i : ints) std::printf(" i%lld", i);
    for (auto& it : items) {
        if (it.first == 'i') std::printf(" i%lld", (long long)it.second);
        else std::printf(" d%.17g", it.second);
    }
    std::printf("\n");
}

static void print_list(long long cid, int step, int op, long long a, long long b, PyObject* res, bool pairs) {
    if (!res) { print_q(cid, step, op, a, b, 0, {}); PyErr_Clear(); return; }
    std::printf("Q %lld %d %d %lld %lld 1 %zu", cid, step, op, a, b, res->items.size() * (pairs ? 2 : 1));
    for (PyObject* o : res->items) {
        if (pairs) std::printf(" d%.17g i%lld", o->items[0]->fval, o->items[1]->ival);
        else std::printf(" i%lld", o->ival);
    }
    std::printf("\n");
}

static PyObject* tuple_of(std::vector<PyObject*> items) { PyObject* t = PyTuple_New(0); t->items = items; return t; }
static PyObject* I(long long v) { return PyLong_FromLongLong(v); }
static PyObject* F(double v) { return PyFloat_FromDouble(v); }

static void apply(Ballpark* bp, const Record& c, long long cid, int s) {
    const ID a = c.a, b = c.b;
    const std::vector<double>& d = c.d;
    auto D = [&d](size_t i) { return i < d.size() ? d[i] : 0.0; };
    const double x = D(0), y = D(1), z = D(2), r = D(3);
    switch (c.op) {
        case 1: bp->GotoPoint(a, x, y, z); break;
        case 2: bp->GotoDirection(a, x, y, z); break;
        case 3: bp->Stop(a); break;
        case 4: bp->FollowBall(a, b, (float)r); break;
        case 5: bp->Orbit(a, b, (float)r); break;
        case 6: bp->WarpTo(a, x, y, z, r, c.w); break;
        case 7: bp->EntityWarpIn(a, x, y, z, c.w); break;
        case 8: bp->SetBallPosition(a, x, y, z); break;
        case 9: bp->SetBallVelocity(a, x, y, z); break;
        case 10: bp->SetBallRadius(a, (float)r); break;
        case 11: bp->SetMaxSpeed(a, (float)r); break;
        case 12: bp->SetBallMass(a, r); break;
        case 13: bp->SetBallAgility(a, (float)r); break;
        case 14: bp->SetSpeedFraction(a, (float)r); break;
        case 15: bp->SetBallFree(a, c.w != 0); break;
        case 16: bp->SetBallMassive(a, c.w != 0); break;
        case 17: bp->SetBallGlobal(a, c.w != 0); break;
        case 18: bp->SetBallInteractive(a, c.w != 0); break;
        case 19: bp->AddProximitySensor(a, (float)r, x, (int)b, c.w != 0); break;
        case 20: bp->RemoveProximitySensor(a); break;
        case 21: bp->SetNotificationRange(a, r); break;
        case 22: bp->SetTargetTracking(a, b, r); break;
        case 23: bp->SetBallHarmonic(a, b, (int)x, (int)y, c.w != 0); break;
        case 24: { Ball* ball = bp->mBalls[a]; if (ball) ball->isCloaked = (uint8_t)c.w; break; }   // the attribute
        case 25: {                                                              // InitializeBubbles(srcId) from Python
            std::vector<PyObject*> args;
            if (a != 0) args.push_back(I(a));
            bp->PyInitializeBubbles(tuple_of(args));
            break;
        }
        case 26: bp->PyCopyBubbles(tuple_of({})); break;
        case 27: {                                      // Actions.clean_up_followers (_actions.py): stop each of them
            PyObject* ids = bp->PyGetRemoteFollowers(tuple_of({}));
            print_list(cid, s, c.op, a, b, ids, false);
            for (PyObject* o : ids->items) bp->Stop(o->ival);
            break;
        }
        case 28: PySet_Add(bp->bubbleKeepAliveBalls, I(a)); break;            // park.bubbleKeepAliveBalls
        case 29: bp->BringOutTheDeads(); break;         // OnTick's first call, before a tick
        case 30: bp->RemoveBall(a, c.w); break;
        case 31: {
            PyObject* res = bp->PyCheckVisibility(tuple_of({I(a), I(b)}));
            print_q(cid, s, c.op, a, b, 1, {}, {res->ival});
            break;
        }
        case 32: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsInRange(tuple_of({I(a), F(r), I(c.w)})), false); break;
        case 33: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsInRange(tuple_of({F(x), F(y), F(z), F(r), I(c.w)})),
                            false); break;
        case 34: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsAndDistInRange(tuple_of({I(a), F(r), I(c.w)})),
                            true); break;
        case 35: {
            PyObject* res = bp->PyGetBubbleAtCoordinates(tuple_of({F(x), F(y), F(z)}));
            print_q(cid, s, c.op, a, b, 1, {}, {res == Py_None ? -999LL : res->ival});
            break;
        }
        case 36: if (!bp->PyMissileFollow(tuple_of({I(a), I(b), I(c.c)}))) PyErr_Clear(); break;
        case 37: if (!bp->PyLaunchMissile(tuple_of({I(a), I(b), I(c.c), I(c.w & 1), I((c.w >> 1) & 1)})))
                     PyErr_Clear();
                 break;
        case 38: {
            std::vector<PyObject*> args = {I(a), I(c.w)};
            if (!d.empty()) args.push_back(F(x));
            if (!bp->PyCloakBall(tuple_of(args))) PyErr_Clear();
            break;
        }
        case 39: bp->PyUncloakBall(tuple_of({I(a)})); break;
        case 40: bp->PyAddCapsule(tuple_of({I(a), I(b), F(D(0)), F(D(1)), F(D(2)), F(D(3)), F(D(4)), F(D(5)),
                                            F(D(6))})); break;
        case 41: {
            std::vector<PyObject*> args = {I(a), I(b)};
            for (size_t i = 0; i < 12; ++i) args.push_back(F(D(i)));
            bp->PyAddOrientedBox(tuple_of(args));
            break;
        }
        case 42: { Ball* ball = bp->mBalls[a];
                   if (ball && !ball->PyAddMiniBall(tuple_of({F(D(0)), F(D(1)), F(D(2)), F(D(3))}))) PyErr_Clear();
                   break; }
        case 43: { Ball* ball = bp->mBalls[a];
                   if (ball && !ball->PyAddMiniCapsule(tuple_of({F(D(0)), F(D(1)), F(D(2)), F(D(3)), F(D(4)), F(D(5)),
                                                                 F(D(6))}))) PyErr_Clear();
                   break; }
        case 44: { Ball* ball = bp->mBalls[a];
                   std::vector<PyObject*> args;
                   for (size_t i = 0; i < 12; ++i) args.push_back(F(D(i)));
                   if (ball && !ball->PyAddMiniBox(tuple_of(args))) PyErr_Clear();
                   break; }
        case 45: bp->RemoveCapsule(a, c.w); break;
        case 46: bp->RemoveOrientedBox(a, c.w); break;
        case 47: bp->PySetBallRotation(tuple_of({I(a), F(D(0)), F(D(1)), F(D(2)), F(D(3))})); break;
        case 48: bp->PySetBallAngularVelocity(tuple_of({I(a), F(D(0)), F(D(1)), F(D(2))})); break;
        case 49: bp->PySetMaxAngularSpeed(tuple_of({I(a), F(D(0))})); break;
        case 50: bp->PySetBallAngularAgility(tuple_of({I(a), F(D(0))})); break;
        case 51: bp->PySetBallTroll(tuple_of({I(a), I(c.w)})); break;
        case 52: bp->PySetBallRigid(tuple_of({I(a)})); break;
        case 53: bp->PySetBoid(tuple_of({I(a), I(c.w)})); break;
        case 54: {                                      // LoadFormations(((name, ((x, y, z), ...)), ...))
            std::vector<PyObject*> forms;
            size_t i = 0;
            const size_t nf = (size_t)D(i++);
            for (size_t f = 0; f < nf; ++f) {
                const size_t nv = (size_t)D(i++);
                std::vector<PyObject*> vecs;
                for (size_t v = 0; v < nv; ++v, i += 3) vecs.push_back(tuple_of({F(D(i)), F(D(i + 1)), F(D(i + 2))}));
                forms.push_back(tuple_of({I((long long)f), tuple_of(vecs)}));
            }
            bp->PyLoadFormations(tuple_of({tuple_of(forms)}));
            break;
        }
        case 55: bp->PySetBallFormation(tuple_of({I(a), I(c.w)})); break;
        case 56: bp->PyFormationFollow(tuple_of({I(a), I(b), I(c.w)})); break;
        case 57: { Ball* ball = bp->mBalls[a];
                   print_q(cid, s, c.op, a, b, ball ? 1 : 0, {}, {ball ? (long long)ball->ReserveFormationSlot() : 0LL});
                   break; }
        case 58: { Ball* ball = bp->mBalls[a]; if (ball) ball->FreeFormationSlot((size_t)(int)c.w); break; }
        case 59: bp->PyAddMushroom(tuple_of({I(a), F(D(0)), F(D(1))})); break;
        case 60: {
            PyObject* res = bp->PyGetAccuracy(tuple_of({I(a), I(b), F(D(0)), F(D(1)), F(D(2)), F(D(3)), F(D(4))}));
            print_q(cid, s, c.op, a, b, 1, {{'d', res->items[0]->fval}, {'d', res->items[1]->fval},
                                           {'d', res->items[2]->fval}});
            break;
        }
        case 61: {                                      // Ball::CalculateRelativeVelocities, flags in w's bits
            Ball* src = bp->mBalls[a];
            Ball* dst = bp->mBalls[b];
            Vector3d relVel(-7777.0, -7777.0, -7777.0);
            double radVel = -7777.0, shortDist = -7777.0, transVel = -7777.0, angVel = -7777.0;
            if (src) src->CalculateRelativeVelocities(dst, (c.w & 1) != 0, (c.w & 2) != 0, (c.w & 4) != 0,
                                                      (c.w & 8) != 0, relVel, radVel, shortDist, transVel, angVel);
            print_q(cid, s, c.op, a, b, src ? 1 : 0, {{'d', relVel.x}, {'d', relVel.y}, {'d', relVel.z},
                                                      {'d', radVel}, {'d', shortDist}, {'d', transVel},
                                                      {'d', angVel}});
            break;
        }
        case 62: bp->OnTick(0, (Be::Time)a, nullptr); break;
        case 63: {                                      // ClientBall::GetShiftedTime(time) on a client's ball
            ClientBall* ball = bp->isMaster ? nullptr : static_cast<ClientBall*>(bp->mBalls[a]);
            print_q(cid, s, c.op, a, b, ball ? 1 : 0, {}, {ball ? (long long)ball->GetShiftedTime((Be::Time)b) : 0LL});
            break;
        }
        case 64: {                                      // Ball::GetRotatedVector (PyGetRotatedVector's float vector)
            Ball* ball = bp->mBalls[a];
            Vector3 v((float)D(0), (float)D(1), (float)D(2));
            if (ball) ball->GetRotatedVector(v);
            print_q(cid, s, c.op, a, b, ball ? 1 : 0, {{'d', (double)v.x}, {'d', (double)v.y}, {'d', (double)v.z}});
            break;
        }
        case 65: bp->PyEvolve(tuple_of({})); break;
        case 66: {
            PyObject* r1 = bp->PyGetCenterDist(tuple_of({I(a), I(b)}));
            PyObject* r2 = bp->PyGetSurfaceDist(tuple_of({I(a), I(b)}));
            print_q(cid, s, c.op, a, b, r1 == Py_None ? 0 : 1,
                    r1 == Py_None ? std::vector<std::pair<char, double>>{}
                                  : std::vector<std::pair<char, double>>{{'d', r1->fval}, {'d', r2->fval}});
            break;
        }
        default: std::exit(6);
    }
}

int main() {
    char tag[8];
    while (std::scanf("%7s", tag) == 1) {
        long long cid; double friction; int tick, steps, flags, master, nr; long start; unsigned seed;
        if (tag[0] != 'C' || std::scanf("%lld %lf %d %d %d %d %ld %u %d", &cid, &friction, &tick, &steps, &flags,
                                        &master, &start, &seed, &nr) != 9)
            return 3;
        g_case = cid;
        g_useNewOrbit = (flags & 1) != 0;               // bit 0 useNewOrbit, bit 1 useIterativeCollision, bit 2
        g_useIterativeCollision = (flags & 2) != 0;     // useDynamicalOrientation, bit 3 its missile exclusion,
        g_useDynamicalOrientation = (flags & 4) != 0;   // bit 4: no Evolve after each step (OnTick or Evolve ops)
        g_disableDynamicalOrientationForMissiles = (flags & 8) != 0;
        const bool manual = (flags & 16) != 0;
        std::srand(seed);                               // floatrand's rand(), for shuffled sensors
        std::vector<Record> recs(nr);
        for (auto& c : recs) {
            if (std::scanf("%7s", tag) != 1) return 4;
            c.kind = tag[0];
            if (c.kind == 'B') {
                int fl[5];
                c.d.assign(13, 0.0);
                if (std::scanf("%d %lld %lf %lf %lf %d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                               &c.step, &c.a, &c.d[0], &c.d[1], &c.d[2], &fl[0], &fl[1], &fl[2], &fl[3], &fl[4],
                               &c.d[3], &c.d[4], &c.d[5], &c.d[6], &c.d[7], &c.d[8], &c.d[9], &c.d[10], &c.d[11],
                               &c.d[12]) != 20)
                    return 4;
                for (int i = 0; i < 5; ++i) c.f[i] = fl[i] != 0;
            } else if (c.kind == 'K') {
                int n;
                if (std::scanf("%d %d %lld %lld %lld %d %d", &c.step, &c.op, &c.a, &c.b, &c.c, &c.w, &n) != 7)
                    return 5;
                c.d.assign((size_t)n, 0.0);
                for (int i = 0; i < n; ++i)
                    if (std::scanf("%lf", &c.d[(size_t)i]) != 1) return 5;
            } else return 5;
        }
        Ballpark* bp = new Ballpark();
        bp->mFriction = friction;
        bp->mTickInterval = tick;
        bp->dt = bp->mTickInterval * 0.001;
        bp->mCurrentTime = start;
        bp->isMaster = master != 0;
        int s = 0;
        try {
            for (s = 0; s <= steps; ++s) {
                for (const auto& c : recs) {
                    if (c.step != s) continue;
                    if (c.kind == 'B') {                // the Python AddBall (PyAddBall), argument by argument
                        std::vector<PyObject*> args = {I(c.a), F(c.d[0]), F(c.d[1]), F(c.d[2]), I(c.f[0]), I(c.f[1]),
                                                       I(c.f[2]), I(c.f[3]), I(c.f[4]), F(c.d[3]), F(c.d[4]),
                                                       F(c.d[5]), F(c.d[6]), F(c.d[7]), F(c.d[8]), F(c.d[9]),
                                                       F(c.d[10])};
                        if (g_useDynamicalOrientation) { args.push_back(F(c.d[11])); args.push_back(F(c.d[12])); }
                        if (!bp->PyAddBall(tuple_of(args))) PyErr_Clear();
                    } else apply(bp, c, cid, s);
                }
                if (s > 0 && !manual) bp->Evolve(0);
                print_state(bp, cid, s);
            }
            dump_partition(bp, cid);
        } catch (const DirectXMathUnavailable&) {
            std::printf("X %lld %d\n", cid, s);         // the scenario needs DirectXMath: abandoned
        }
        g_useDynamicalOrientation = false;              // the destructor's ClearAll runs at the defaults
        g_useIterativeCollision = false;
        delete bp;
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
    end = block_end(src, hits[0].start())              # the pattern may include the opening brace
    kind = name.split()[0]
    if kind == "typedef":                               # `typedef struct X {...} X;`: to the declaration's semicolon
        semi = re.match(r"\s*\w+\s*;", src[end:])
        if not semi:
            raise SystemExit(f"{name}: no name and semicolon after the block")
        end += semi.end()
    elif kind in ("enum", "class", "struct", "array"):    # keep the declaration's semicolon
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


def dependencies(depfile: Path) -> list:
    """The files a `-MD` dependency file lists (its target removed), resolved."""
    text = depfile.read_text().replace("\\\n", " ")
    return [str(Path(p).resolve()) for p in text.split(":", 1)[1].split()]


def dxm_files_read(deps: list, dxm_repo: Path) -> list:
    """The DirectXMath files among a compilation's dependencies, each of which must be a checked header."""
    root = str(dxm_repo.resolve()) + "/"
    read = sorted(p[len(root):] for p in deps if p.startswith(root))
    stray = [p for p in read if p not in DXM_HEADERS]
    if stray:
        raise SystemExit(f"the compiler read DirectXMath files that are not checked: {stray}")
    return read


def main(repo: Path, math_repo: Path, build: Path, dxm_repo: Path | None = None) -> None:
    head, math_head = head_commit(repo), head_commit(math_repo)
    if head != PINNED:
        raise SystemExit(f"destiny is at {head}, not the pinned {PINNED}")
    if math_head != MATH_PINNED:
        raise SystemExit(f"carbonengine/math is at {math_head}, not the pinned {MATH_PINNED}")
    if dxm_repo is not None:                            # DirectXMath: the pinned commit and the checked headers first
        dxm_head = head_commit(dxm_repo)
        if dxm_head != DXM_PINNED:
            raise SystemExit(f"DirectXMath is at {dxm_head}, not the pinned {DXM_PINNED}")
        dxm_sha = {f: sha((dxm_repo / f).read_bytes()) for f in DXM_HEADERS}
        bad = sorted(f for f, s in dxm_sha.items() if s != DXM_HEADERS[f])
        if bad:
            raise SystemExit(f"DirectXMath headers differ from their pinned SHA-256: {bad}")
    files = sorted({f for f, _, _ in BLOCKS} | {f for f, _, _ in LINES} | {"src/Vector3d.h"})
    text = {f: (repo / f).read_text() for f in files}
    blocks = {name: extract_block(text[f], pat, name) for f, name, pat in BLOCKS}
    lines = {name: extract_line(text[f], prefix, name) for f, name, prefix in LINES}
    where = {name: (f, re.search(pat, text[f]).start()) for f, name, pat in BLOCKS}
    math_files = sorted({f for f, _, _ in MATH_STRUCTS} | {f for f, _, _ in MATH_BLOCKS} | {MATH_UNAVAILABLE[0]}
                        | ({f for f, _, _ in MATH_BLOCKS_DXM + MATH_LINES_DXM} if dxm_repo is not None else set()))
    math_text = {f: (math_repo / f).read_text() for f in math_files}
    math_structs, math_dropped, math_kept = {}, {}, {}
    for f, name, pat in MATH_STRUCTS:
        body = extract_block(math_text[f], pat, name)
        keep = MATH_KEEP_DXM.get(name, ()) if dxm_repo is not None else ()

        def dropped(line: str) -> bool:
            return bool(MATH_DROP.search(line)) and line.strip() not in keep
        kept = [line for line in body.splitlines() if not dropped(line)]
        math_dropped[name] = [line for line in body.splitlines() if dropped(line)]
        if keep:
            math_kept[name] = [line for line in body.splitlines() if MATH_DROP.search(line) and not dropped(line)]
            if len(math_kept[name]) != len(keep):
                raise SystemExit(f"{name}: kept {math_kept[name]}, expected {keep}")
        math_structs[name] = "\n".join(kept)
    math_blocks = {name: extract_block(math_text[f], pat, name) for f, name, pat in MATH_BLOCKS}
    unavailable = extract_block(math_text[MATH_UNAVAILABLE[0]], MATH_UNAVAILABLE[2], MATH_UNAVAILABLE[1])
    if "XMQuaternionRotationRollPitchYaw" not in unavailable:
        raise SystemExit("RotationQuaternion(yaw, pitch, roll) no longer calls DirectXMath: extract it instead")
    if dxm_repo is not None:
        math_blocks.update({name: extract_block(math_text[f], pat, name) for f, name, pat in MATH_BLOCKS_DXM})
        dxm_lines = {name: extract_line(math_text[f], prefix, name) for f, name, prefix in MATH_LINES_DXM}

    def piece(name: str, body: str, origin: str) -> str:
        return f"\n// ---- {origin}, {name}, verbatim ----\n{body}\n"

    def from_file(path: str, skip=()) -> str:              # a file's extracted definitions, in source order
        names = sorted((n for f, n, _ in BLOCKS if f == path and n not in skip), key=lambda n: where[n][1])
        return "".join(piece(n, blocks[n], path) for n in names)

    def the_lines(path: str, names) -> str:
        return "".join(piece(n, lines[n], path) for n in names)

    def the_blocks(path: str, names) -> str:
        return "".join(piece(n, blocks[n], path) for n in names)

    math = ("".join(f"\n// ---- carbonengine/math {f}, {n}, verbatim but for the removed DirectXMath/Vector2/Vector4 "
                    f"lines ----\n{math_structs[n]}\n" for f, n, _ in MATH_STRUCTS)
            + "".join(f"\n// ---- carbonengine/math {f}, {n}, verbatim ----\n{math_blocks[n]}\n"
                      for f, n, _ in MATH_BLOCKS + (MATH_BLOCKS_DXM if dxm_repo is not None else [])))
    if dxm_repo is None:
        dxm_head_part, rotation = "", MATH_STUB
    else:                                               # DirectXMath's header, then the function verbatim
        dxm_head_part = DXM_PRELUDE + "".join(
            f"\n// ---- carbonengine/math {f}, {n}, verbatim ----\n{dxm_lines[n]}\n" for f, n, _ in MATH_LINES_DXM)
        rotation = (f"\n// ---- carbonengine/math {MATH_UNAVAILABLE[0]}, {MATH_UNAVAILABLE[1]}, verbatim (DirectXMath's "
                    f"scalar path) ----\n{unavailable}\n")
    declarations = ["class BoxSortComparer", "class BallSortComparer", "class StaticCollidableSortComparer"]
    source = (PRELUDE_A + dxm_head_part + math + rotation
              + the_blocks("src/IDstConstants.h", ("enum DSTBALLMODE", "enum DSTCONSTANTS"))
              + the_lines("src/Ball.h", ("typedef VectorOfBalls", "typedef ID", "typedef VectorOfIDs",
                                         "typedef SetOfIDs", "MAX_ALIGN_TICKS"))
              + piece("AU", lines["AU"], "src/Ballpark.h")
              + "\n// ---- src/Vector3d.h, verbatim ----\n" + text["src/Vector3d.h"]
              + the_blocks("src/Quaternion.h", ("typedef struct QuaternionD",))
              + the_lines("src/Settings.h", ("g_collisionMaxIterationsDefault", "g_useIterativeCollisionDefault",
                                             "g_useDynamicalOrientationDefault",
                                             "g_disableDynamicalOrientationForMissilesDefault",
                                             "g_useNewOrbitDefault"))
              + "".join(piece(n, blocks[n], "src/SortedSets.h") for n in declarations)
              + the_lines("src/SortedSets.h", ("typedef SortedSetOfBoxes", "typedef SortedSetOfBalls",
                                               "typedef SortedSetOfStaticCollidables"))
              + the_lines("src/Partition.h", ("typedef DictOfBalls", "typedef VectorOfStaticCollidables",
                                              "typedef DictEntry", "typedef VectorOfBoxes",
                                              "typedef BoxVectorInsertor"))
              + the_blocks("src/Partition.h", ("class BoxAllocator", "class Key", "struct KeyHash"))
              + piece("typedef MapOfBoxes", lines["typedef MapOfBoxes"], "src/Partition.h")
              + piece("class Partition", blocks["class Partition"], "src/Partition.h")
              + piece("class Partitionable", blocks["class Partitionable"], "src/Partitionable.h")
              + piece("class Box", blocks["class Box"], "src/Box.h")
              + piece("class StaticCollidable", blocks["class StaticCollidable"], "src/StaticCollidable.h")
              + piece("class MapOfLongLongs", blocks["class MapOfLongLongs"], "src/MapOfLongLongs.h")
              + piece("class ProximitySensor", blocks["class ProximitySensor"], "src/Ball.h")
              + the_blocks("src/CollisionStructures.h", ("struct CollisionImpact", "struct CollisionItem",
                                                         "struct BallLocation"))
              + the_lines("src/Ballpark.h", ("typedef DictOfFreeBalls", "typedef SetOfBalls", "typedef SetOfCapsules",
                                             "typedef SetOfOrientedBoxes", "typedef VectorOfVectors",
                                             "typedef VectorOfCapsules"))
              + piece("class Triangle", blocks["class Triangle"], "src/Triangle.h")
              + piece("class Planed", blocks["class Planed"], "src/Plane.h")
              + piece("class AABB", blocks["class AABB"], "src/AABB.h")
              + the_blocks("src/BoxShape.h", ("enum BoxCorner", "enum BoxSide", "class RectangleShape",
                                              "class BoxShape"))
              + the_lines("src/Collision.h", ("declaration GetPointOnBallThatFirstIntersectsPlane",
                                              "declaration IntersectSegmentPlane", "declaration IntersectSegmentSphere",
                                              "declaration IntersectSphereTriangle", "declaration Quadratic",
                                              "declaration CollideTwoSpheres"))
              + piece("class MiniBall", blocks["class MiniBall"], "src/MiniBall.h")
              + piece("TYPEDEF_BLUECLASS(MiniBall)", lines["TYPEDEF_BLUECLASS(MiniBall)"], "src/MiniBall.h")
              + piece("class MiniBox", blocks["class MiniBox"], "src/MiniBox.h")
              + piece("TYPEDEF_BLUECLASS(MiniBox)", lines["TYPEDEF_BLUECLASS(MiniBox)"], "src/MiniBox.h")
              + piece("class MiniCapsule", blocks["class MiniCapsule"], "src/MiniCapsule.h")
              + piece("TYPEDEF_BLUECLASS(MiniCapsule)", lines["TYPEDEF_BLUECLASS(MiniCapsule)"], "src/MiniCapsule.h")
              + PRELUDE_B.replace("__COLLISION_BALL_PROPERTIES__", piece(
                  "class CollisionBallProperties", blocks["class CollisionBallProperties"],
                  "src/CollisionBallProperties.h"))
              + piece("class Capsule", blocks["class Capsule"], "src/Capsule.h")
              + piece("TYPEDEF_BLUECLASS(Capsule)", lines["TYPEDEF_BLUECLASS(Capsule)"], "src/Capsule.h")
              + piece("class OrientedBox", blocks["class OrientedBox"], "src/OrientedBox.h")
              + piece("TYPEDEF_BLUECLASS(OrientedBox)", lines["TYPEDEF_BLUECLASS(OrientedBox)"], "src/OrientedBox.h")
              + from_file("src/Collision.cpp")
              + from_file("src/Triangle.cpp")
              + from_file("src/Plane.cpp")
              + from_file("src/AABB.cpp")
              + from_file("src/BoxShape.cpp")
              + the_lines("src/Partition.cpp", ("collisionCase1, collisionCase3",))
              + from_file("src/Partition.cpp")
              + from_file("src/Box.cpp")
              + from_file("src/Partitionable.cpp")
              + from_file("src/CollisionStructures.cpp")
              + from_file("src/CollisionBallProperties.cpp")
              + from_file("src/StaticCollidable.cpp")
              + from_file("src/MiniBall.cpp")
              + from_file("src/MiniBox.cpp")
              + from_file("src/MiniCapsule.cpp")
              + from_file("src/Capsule.cpp")
              + from_file("src/OrientedBox.cpp")
              + the_lines("src/Ballpark.cpp", ("Timer_ResolveBubbleConflicts", "Timer_HandleProximities",
                                               "Timer_InsertBallInBoxes", "Timer_Gradient", "Timer_CheckVisibility",
                                               "Timer_CopyBubbles", "Timer_AdditionsAndDeletions",
                                               "Timer_GetBallIdsInRange", "Timer_WarpDistance",
                                               "Timer_CalculateYawPitchRoll", "Timer_WriteBallsToStream", "Plus",
                                               "Zero", "Minus", "TICK_EVOLVE", "SIGNUM (Ballpark.cpp)",
                                               "SIGNUMF (Ballpark.cpp)", "ABS (Ballpark.cpp)", "_PI_ (Ballpark.cpp)",
                                               "s_ballNotInParkCallback", "ballparkCounter",
                                               "WARP_FACTOR_TO_AU_PER_SECOND", "WARP_FACTOR_TO_ACCELERATION",
                                               "WARP_FACTOR_TO_DECELERATION", "followModes", "ORBITAL_PRECESSION"))
              + "\nnamespace {   // as in Ballpark.cpp, for compilers other than MSVC\n"
              + piece("_finite", blocks["_finite"], "src/Ballpark.cpp") + "}\n"
              + from_file("src/Ballpark.cpp", skip=("_finite",))
              + from_file("src/Thunkers.cpp")
              + "\n#undef ABS   // Ball.cpp has its own\n#undef SIGNUM\n"
              + the_lines("src/Ball.cpp", ("ABS (Ball.cpp)", "MIN (Ball.cpp)", "SIGNUM (Ball.cpp)", "_HALFPI_",
                                           "_2PI_", "ballCount", "declaration Modulo2pi"))
              + from_file("src/Ball.cpp")
              + PRELUDE_C + DRIVER)
    build.mkdir(parents=True, exist_ok=True)
    (build / "harness.cpp").write_text(source)
    flags = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-fsigned-char"]
    if dxm_repo is not None:                            # the scalar path, the snapshot's headers, the harness's sal.h
        (build / "dxm").mkdir(exist_ok=True)
        (build / "dxm" / "sal.h").write_text(SAL_STUB)
        flags += [DXM_DEFINE, "-I", str(dxm_repo / "Inc"), "-I", str(build / "dxm")]
    deps = ["-MD", "-MF", str(build / "harness.d")] if dxm_repo is not None else []
    cmd = flags + deps + ["-o", str(build / "harness"), str(build / "harness.cpp")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"g++ failed with exit status {res.returncode}")
    gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    manifest = {"destiny_commit": head, "math_commit": math_head, "compiler": gxx, "command": cmd,
                "compiler_warnings": [w for w in res.stderr.splitlines() if "warning:" in w],
                "source_sha256": {f: sha(t) for f, t in text.items()},
                "math_source_sha256": {f: sha(t) for f, t in math_text.items()},
                "extracted_sha256": {**{n: sha(b) for n, b in blocks.items()}, **{n: sha(s) for n, s in lines.items()}},
                "math_extracted_sha256": {**{n: sha(b) for n, b in math_blocks.items()},
                                          **{n: sha(s) for n, s in math_structs.items()}},
                "math_removed_lines": math_dropped,
                "math_unavailable": {"name": MATH_UNAVAILABLE[1], "file": MATH_UNAVAILABLE[0],
                                     "sha256": sha(unavailable), "body": unavailable},
                "extracted_lines": dict(lines),
                "harness_cpp_sha256": sha(source),
                "binary_sha256": sha((build / "harness").read_bytes())}
    if dxm_repo is not None:                            # RotationQuaternion(yaw, pitch, roll) is compiled, not stubbed
        manifest["math_extracted_sha256"].update({MATH_UNAVAILABLE[1]: sha(unavailable),
                                                  **{n: sha(s) for n, s in dxm_lines.items()}})
        manifest["math_rotation_ypr"] = manifest.pop("math_unavailable")
        probe = PROBE_HEAD + dxm_head_part + math + rotation + PROBE_MAIN
        (build / "dxm_probe.cpp").write_text(probe)
        probe_cmd = flags + ["-MD", "-MF", str(build / "dxm_probe.d"), "-o", str(build / "dxm_probe"),
                             str(build / "dxm_probe.cpp")]
        pres = subprocess.run(probe_cmd, capture_output=True, text=True)
        if pres.returncode != 0:
            sys.stderr.write(pres.stderr)
            raise SystemExit(f"g++ failed on dxm_probe with exit status {pres.returncode}")
        manifest.update({
            "directxmath_commit": dxm_head, "directxmath_header_sha256": dxm_sha, "directxmath_define": DXM_DEFINE,
            "directxmath_files_read": dxm_files_read(dependencies(build / "harness.d"), dxm_repo),
            "probe_directxmath_files_read": dxm_files_read(dependencies(build / "dxm_probe.d"), dxm_repo),
            "sal_stub_sha256": sha(SAL_STUB), "math_kept_lines": math_kept,
            "probe_command": probe_cmd, "probe_compiler_warnings": [w for w in pres.stderr.splitlines()
                                                                     if "warning:" in w],
            "probe_cpp_sha256": sha(probe), "probe_binary_sha256": sha((build / "dxm_probe").read_bytes())})
    (build / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("extracted_lines", "math_removed_lines", "math_unavailable", "math_rotation_ypr")},
                     indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None)
