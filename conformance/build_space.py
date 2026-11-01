# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Build the compiled oracle of the space layer: Destiny's space partition, bubbles,
proximity sensors, visibility, range queries, collisions (gradient and iterative) and ball removal, around the tick of
slice 1.

    python3 -I conformance/build_space.py DESTINY_REPO BUILD_DIR

As build_modes.py (whose source, binary and manifest it leaves alone), Destiny's own code is taken from the pinned
source (carbonengine/destiny at 114e89fa, MIT, (c) CCP Games) and never retyped: functions, classes and enums by brace
matching (comments and literals skipped), and whole lines (macros, constants, typedefs, globals) by exact match.
Extracted here, beside slice 1's tick and modes: the partition (`Partition`, `Box`, `BoxAllocator`, `Key`, `KeyHash`,
the sorted-set comparers, `Partitionable`, `Ball::InsertInBoxes`, `Ballpark::InsertInBoxes`, `InsertBallInBoxes`);
bubbles (`AddToBubble`, `InitializeBubbles`, `UpdateBallBubble`, `CopyBubbles`, the interactive and keep-alive
bookkeeping, `InDeadBubble`, `Box::Traverse`, `Box::NearbyBubbles`); `AddBall` and the property setters; the
collision gradient (`Gradient`, `Potential`, `CollideTwoSpheres`, `Quadratic`) and the iterative method
(`CalculateIterativeCollisionResponses` and its helpers, `CollisionBallProperties`, the collision structures); the
proximity sensors (`HandleProximities`, `Ball::CheckForProximity` and its three parts, `AddProximitySensor`,
`UpdateProximityStatus`, `SetNotificationRange`, `SetTargetTracking`); `CheckVisibility`; ball removal (`RemoveBall`,
`BringOutTheDeads`, `BringOutDeadBalls`); the Python entry points `GetBallIdsInRange`, `PyInitializeBubbles`,
`PyGetRemoteFollowers`, `PyCheckVisibility` and `PyGetBubbleAtCoordinates` (Thunkers.cpp); and the ballpark's
constructor and destructor (its defaults), `ClearAll` and `ClearBubbles`.

The prelude supplies what is not Destiny's simulation code: a minimal stand-in for the CPython objects the extracted
code touches (ints, insertion-ordered dicts as CPython's, sets, lists, tuples, `PyArg_ParseTuple`), the `MapOfBalls`
and `BallIterator` containers over such a dict, the `Ball` and `Ballpark` member lists with Destiny's types (Ball.h,
Ballpark.h), the switches at Settings.h's defaults (dynamical orientation off; useNewOrbit and the iterative collision
method set per case), and stubs that abort if reached: missiles, formations, trolls, mushrooms, dynamical orientation,
static collidables (capsules, oriented boxes, miniballs) and the Blue tick registration. Events (`PostEvent`,
`SendEvent`) are printed. A driver reads scenarios on stdin and prints after every tick every ball's state, its boxes,
its collision list and its sensor list, the partition (a checksum per tick, a full dump at the end), the bubble
dictionaries, and the answers to queries.

The build directory receives the source, the binary and a manifest (the SHA-256 of every source file read and every
extracted piece, the extracted lines themselves, the compiler, its command and warnings). Compiled with g++ -O2
-std=c++17 -ffp-contract=off -fno-fast-math, as build.py and build_modes.py. The repository is read, never executed.
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
# (and its semicolon for classes, structs and enums)
BLOCKS = [
    # ---- declarations ----
    ("src/IDstConstants.h", "enum DSTBALLMODE", r"enum DSTBALLMODE\b"),
    ("src/IDstConstants.h", "enum DSTCONSTANTS", r"enum DSTCONSTANTS\b"),
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
    # ---- CollisionStructures.cpp, CollisionBallProperties.cpp: the iterative method's store (COL-714 to COL-718) ----
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
    # ---- definitions: Collision.cpp ----
    ("src/Collision.cpp", "CollideTwoSpheres", r"double CollideTwoSpheres\("),
    ("src/Collision.cpp", "Quadratic", r"bool Quadratic\(double& v1"),
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
    # ---- Partitionable.cpp, StaticCollidable.cpp ----
    ("src/Partitionable.cpp", "Partitionable::Partitionable", r"Partitionable::Partitionable\(\):"),
    ("src/StaticCollidable.cpp", "StaticCollidable::StaticCollidable", r"StaticCollidable::StaticCollidable\(\):"),
    ("src/StaticCollidable.cpp", "StaticCollidable::ClearBoxes", r"void StaticCollidable::ClearBoxes\(\)"),
    ("src/StaticCollidable.cpp", "StaticCollidable::DeleteFromBoxes", r"void StaticCollidable::DeleteFromBoxes\(\)"),
    ("src/StaticCollidable.cpp", "StaticCollidable::InsertInBox", r"void StaticCollidable::InsertInBox\("),
    ("src/StaticCollidable.cpp", "StaticCollidable::GetInflatedRadius",
     r"double StaticCollidable::GetInflatedRadius\("),
    ("src/StaticCollidable.cpp", "StaticCollidableSortComparer::operator()",
     r"bool StaticCollidableSortComparer::operator \(\)"),
    # ---- Ballpark.cpp (slice 1's pieces, then this slice's) ----
    ("src/Ballpark.cpp", "_finite", r"inline float _finite\( double x \)"),
    ("src/Ballpark.cpp", "DynamicalOrientationDisabledForBall", r"bool DynamicalOrientationDisabledForBall\("),
    ("src/Ballpark.cpp", "Ballpark::Ballpark", r"Ballpark::Ballpark\(IRoot\* lockobj\) :"),
    ("src/Ballpark.cpp", "Ballpark::CalculateIterativeCollisionResponses",
     r"void Ballpark::CalculateIterativeCollisionResponses\(\)"),
    ("src/Ballpark.cpp", "Ballpark::~Ballpark", r"Ballpark::~Ballpark\(\)"),
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
    ("src/Ballpark.cpp", "Ballpark::CalculateBallPositionVelocity", r"void Ballpark::CalculateBallPositionVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::Gradient", r"void Ballpark::Gradient\(Ball \*ball\)"),
    ("src/Ballpark.cpp", "Ballpark::Potential", r"void Ballpark::Potential\(Ball \*me"),
    ("src/Ballpark.cpp", "Ballpark::Start", r"void Ballpark::Start\(\)"),
    ("src/Ballpark.cpp", "Ballpark::Pause", r"void Ballpark::Pause\(\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertInBoxes", r"void Ballpark::InsertInBoxes\(Partitionable \*p\)"),
    ("src/Ballpark.cpp", "Ballpark::InsertBallInBoxes", r"void Ballpark::InsertBallInBoxes\(Ball \*ball\)"),
    ("src/Ballpark.cpp", "Ballpark::BallExists", r"Ball \*Ballpark::BallExists\("),
    ("src/Ballpark.cpp", "Ballpark::GetIDForCollisionObject", r"ID Ballpark::GetIDForCollisionObject\("),
    ("src/Ballpark.cpp", "Ballpark::AddBall", r"Ball \* Ballpark::AddBall\("),
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
    ("src/Ballpark.cpp", "Ballpark::SetBallMass", r"void Ballpark::SetBallMass\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallRadius(ID)", r"void Ballpark::SetBallRadius\(\s*const ID& srcId,"),
    ("src/Ballpark.cpp", "Ballpark::SetBallRadius(Ball*)", r"void Ballpark::SetBallRadius\(\s*Ball \*ball,"),
    ("src/Ballpark.cpp", "Ballpark::SetMaxSpeed", r"void Ballpark::SetMaxSpeed\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallPosition", r"void Ballpark::SetBallPosition\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallVelocity", r"void Ballpark::SetBallVelocity\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallHarmonic", r"void Ballpark::SetBallHarmonic\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallFree", r"void Ballpark::SetBallFree\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallAgility", r"void Ballpark::SetBallAgility\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallTimeFactor", r"void Ballpark::SetBallTimeFactor\("),
    ("src/Ballpark.cpp", "Ballpark::SetSpeedFraction", r"void Ballpark::SetSpeedFraction\("),
    ("src/Ballpark.cpp", "Ballpark::SetNotificationRange", r"void Ballpark::SetNotificationRange\("),
    ("src/Ballpark.cpp", "Ballpark::SetTargetTracking", r"void Ballpark::SetTargetTracking\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallMassive", r"void Ballpark::SetBallMassive\("),
    ("src/Ballpark.cpp", "Ballpark::SetBallGlobal", r"void Ballpark::SetBallGlobal\("),
    ("src/Ballpark.cpp", "Ballpark::AddProximitySensor", r"void Ballpark::AddProximitySensor\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveProximitySensor", r"void Ballpark::RemoveProximitySensor\("),
    ("src/Ballpark.cpp", "Ballpark::CheckVisibility", r"ID Ballpark::CheckVisibility\("),
    ("src/Ballpark.cpp", "Ballpark::StopAllFollowers", r"void Ballpark::StopAllFollowers\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveBall", r"void Ballpark::RemoveBall\(\s*const ID& srcId,"),
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
    ("src/Ballpark.cpp", "Ballpark::SetBallInteractive", r"void Ballpark::SetBallInteractive\("),
    ("src/Ballpark.cpp", "Ballpark::DecreaseInteractiveCnt", r"void Ballpark::DecreaseInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::UpdateInteractiveCnt", r"void Ballpark::UpdateInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::IncreaseInteractiveCnt", r"void Ballpark::IncreaseInteractiveCnt\("),
    ("src/Ballpark.cpp", "Ballpark::AddToBubble", r"void Ballpark::AddToBubble\("),
    ("src/Ballpark.cpp", "Ballpark::InitializeBubbles", r"void Ballpark::InitializeBubbles\(\)"),
    ("src/Ballpark.cpp", "Ballpark::UpdateKeepAliveBalls", r"void Ballpark::UpdateKeepAliveBalls\("),
    ("src/Ballpark.cpp", "Ballpark::AddKeepAliveBall", r"void Ballpark::AddKeepAliveBall\("),
    ("src/Ballpark.cpp", "Ballpark::RemoveKeepAliveBall", r"void Ballpark::RemoveKeepAliveBall\("),
    # ---- Thunkers.cpp: the Python entry points this slice compares ----
    ("src/Thunkers.cpp", "Ballpark::PyInitializeBubbles", r"PyObject\* Ballpark::PyInitializeBubbles\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetRemoteFollowers", r"PyObject\* Ballpark::PyGetRemoteFollowers\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBallIdsInRange", r"PyObject\* Ballpark::PyGetBallIdsInRange\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBallIdsAndDistInRange", r"PyObject\* Ballpark::PyGetBallIdsAndDistInRange\("),
    ("src/Thunkers.cpp", "Ballpark::GetBallIdsInRange", r"PyObject\* Ballpark::GetBallIdsInRange\(\s*PyObject\* args,"),
    ("src/Thunkers.cpp", "Ballpark::BallIsValidForIteration", r"bool Ballpark::BallIsValidForIteration\("),
    ("src/Thunkers.cpp", "Ballpark::GetNextValidBallInBubble", r"int Ballpark::GetNextValidBallInBubble\("),
    ("src/Thunkers.cpp", "Ballpark::PyCheckVisibility", r"PyObject\* Ballpark::PyCheckVisibility\("),
    ("src/Thunkers.cpp", "Ballpark::PyGetBubbleAtCoordinates", r"PyObject\* Ballpark::PyGetBubbleAtCoordinates\("),
    ("src/Thunkers.cpp", "Ballpark::PyCopyBubbles", r"PyObject\* Ballpark::PyCopyBubbles\("),
    # ---- Ball.cpp ----
    ("src/Ball.cpp", "floatrand", r"double floatrand\(double a,double b\)"),
    ("src/Ball.cpp", "Modulo2pi", r"void Modulo2pi\(double& a,double& b\)\s*\{"),
    ("src/Ball.cpp", "Ball::Ball", r"Ball::Ball\(IRoot\* lockobj\) :"),
    ("src/Ball.cpp", "Ball::~Ball", r"Ball::~Ball\(\)"),
    ("src/Ball.cpp", "Ball::ClearBoxes", r"void Ball::ClearBoxes\(\)"),
    ("src/Ball.cpp", "Ball::AddProximitySensor", r"long Ball::AddProximitySensor\("),
    ("src/Ball.cpp", "Ball::UpdateProximityStatus", r"void Ball::UpdateProximityStatus\(\)"),
    ("src/Ball.cpp", "Ball::RemoveProximitySensor", r"void Ball::RemoveProximitySensor\(\)"),
    ("src/Ball.cpp", "Ball::CalculateYawPitchRoll", r"void Ball::CalculateYawPitchRoll\(bool snap\)"),
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
    ("src/Ball.cpp", "Ball::AddMiniBalls", r"void Ball::AddMiniBalls\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniBalls", r"void Ball::RemoveMiniBalls\(\)"),
    ("src/Ball.cpp", "Ball::AddMiniCapsules", r"void Ball::AddMiniCapsules\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniCapsules", r"void Ball::RemoveMiniCapsules\(\)"),
    ("src/Ball.cpp", "Ball::AddMiniBoxes", r"void Ball::AddMiniBoxes\(\)"),
    ("src/Ball.cpp", "Ball::RemoveMiniBoxes", r"void Ball::RemoveMiniBoxes\(\)"),
    ("src/Ball.cpp", "Ball::IsWarpish", r"bool Ball::IsWarpish\(\)"),
    ("src/Ball.cpp", "Ball::IsAligningForWarp", r"bool Ball::IsAligningForWarp\(\)"),
    ("src/Ball.cpp", "Ball::IsWarping", r"bool Ball::IsWarping\(\)"),
    ("src/Ball.cpp", "Ball::IsAlignedForWarp", r"bool Ball::IsAlignedForWarp\(\)"),
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
    ("src/Collision.h", "declaration Quadratic",
     "bool Quadratic(double& v1, double& v2, double a, double b, double c);"),
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
    ("src/Ballpark.cpp", "ABS (Ballpark.cpp)", "#define ABS(X)"),
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
]

PRELUDE_A = r"""// Harness prelude: what the extracted Destiny code touches, with its types.
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
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace Be { typedef int64_t Time; }                   // a 64-bit time stamp
struct Vector3 {                                        // Blue's float vector (angular state; never read back here)
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
#define CCP_LOGWARN(...) ((void)0)
#define CCP_ASSERT(x) ((void)0)
#define CCP_STATS_ZONE(x)
#define TASKLET(A)
#define CCP_INT64_FORMAT "lld"
#define PARENTLOCK(x) x()                               // Blue's list locks: the lists below are empty stand-ins
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
static int PySequence_DelSlice(PyObject* l, Py_ssize_t a, Py_ssize_t b) {
    l->items.erase(l->items.begin() + a, l->items.begin() + b);
    return 0;
}
static PyObject* PyTuple_New(Py_ssize_t n) { return PyList_New(n); }
#define PyTuple_SET_ITEM(t, i, v) ((t)->items[(size_t)(i)] = (v))
static PyObject g_pyExcRuntimeError(7), g_pyExcTypeError(7);
static PyObject* PyExc_RuntimeError = &g_pyExcRuntimeError;
static PyObject* PyExc_TypeError = &g_pyExcTypeError;
static int g_pyErr = 0;                                 // the error indicator
static void PyErr_Clear() { g_pyErr = 0; }
static PyObject* PyErr_Occurred() { return g_pyErr ? PyExc_RuntimeError : nullptr; }
static PyObject* PyErr_Format(PyObject*, const char*, ...) { g_pyErr = 1; return nullptr; }
static void PyErr_SetString(PyObject*, const char*) { g_pyErr = 1; }
static void PyErr_Print() { g_pyErr = 0; }
// PyArg_ParseTuple for the formats the extracted thunkers use (L, i, b, d, f and '|'), with CPython's checks: the
// count of arguments, and an integer where L, i or b stands (a float there is a TypeError)
static int PyArg_ParseTuple(PyObject* args, const char* fmt, ...) {
    size_t required = 0, total = 0;
    bool optional = false;
    for (const char* f = fmt; *f; ++f) {
        if (*f == '|') { optional = true; continue; }
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
        if (*f == '|') continue;
        PyObject* o = args->items[i++];
        switch (*f) {
            case 'L': if (o->kind != 1) ok = 0; else *va_arg(ap, long long*) = o->ival; break;
            case 'i': if (o->kind != 1) ok = 0; else *va_arg(ap, int*) = (int)o->ival; break;
            case 'b': if (o->kind != 1) ok = 0; else *va_arg(ap, char*) = (char)o->ival; break;
            case 'd': *va_arg(ap, double*) = o->kind == 1 ? (double)o->ival : o->fval; break;
            case 'f': *va_arg(ap, float*) = (float)(o->kind == 1 ? (double)o->ival : o->fval); break;
            default: ok = 0;
        }
    }
    va_end(ap);
    if (!ok) g_pyErr = 1;
    return ok;
}

struct IRoot {};                                        // Blue's interfaces: empty here
struct INotify : public IRoot {};
struct IEveBallpark {};
struct IBall { void Lock() {} void Unlock() {} };
class Box;
class Ball;
class StaticCollidable;
class Key;
class Partition;
class Ballpark;
class Capsule;
class OrientedBox;
struct MiniBall { int64_t mId; };                       // the minis: never created in this slice
struct MiniCapsule { int64_t mId; };
struct MiniBox { int64_t mId; };
struct MiniListStub {                                   // Blue's lists of minis, always empty here
    ssize_t GetSize() { return 0; }
    void* GetAt(ssize_t) { std::abort(); }
    void Remove(int) { std::abort(); }
};
typedef MiniListStub MiniBallList;
typedef MiniListStub MiniBoxList;
typedef MiniListStub MiniCapsuleList;
struct PyOSStub {                                       // Python's event calls: recorded and printed
    bool PostEvent(IEveBallpark* park, const char* name, const char* method, const char* fmt, ...);
    bool SendEvent(IBall* ball, const char* name, const char* method, void* kw, const char* fmt, ...);
    bool SendEvent(IEveBallpark* park, const char* name, const char* method, void* kw, const char* fmt, ...);
    bool SendEvent(IRoot* root, const char* name, const char* method, void* kw, const char* fmt, ...);
    void PyError() {}
};
static PyOSStub g_pyos;
static PyOSStub* const PyOS = &g_pyos;
struct BeOSStub {                                       // Blue's tick registration: never reached (Start is not called)
    void RegisterForTicks(void*, void*) { std::abort(); }
    void UnregisterForTicks(void*, void*) { std::abort(); }
};
static BeOSStub g_beos;
static BeOSStub* const BeOS = &g_beos;
"""

PRELUDE_B = r"""
// ---- prelude, continued: the switches at Settings.h's defaults (useNewOrbit is set per case) ----
static size_t g_collisionMaxIterations = g_collisionMaxIterationsDefault;
static bool g_useIterativeCollision = g_useIterativeCollisionDefault;
static bool g_useDynamicalOrientation = g_useDynamicalOrientationDefault;
static bool g_disableDynamicalOrientationForMissiles = g_disableDynamicalOrientationForMissilesDefault;
static bool g_useNewOrbit = g_useNewOrbitDefault;
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
inline Matrix RotationMatrix(const Vector3&, float) { std::abort(); }   // the iterative method's DO-only branches
inline Vector3 operator+(const Vector3&, const Vector3&) { std::abort(); }
inline Vector3& operator+=(Vector3&, const Vector3&) { std::abort(); }
inline Vector3& operator*=(Vector3&, float) { std::abort(); }

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
    void SnapOrientation() { std::abort(); }            // dynamical orientation: never reached
    virtual void CalculateYawPitchRoll(bool snap=false);
    void DispatchCollisions() { std::abort(); }         // missiles only: never reached
    void UpdateProximityStatus();
    void FreeFormationSlot(size_t) { std::abort(); }    // formations: never reached
    void SetMode(DSTBALLMODE mode);
    void AddMiniBalls();
    void RemoveMiniBalls();
    void AddActualMiniball(MiniBall*) { std::abort(); }  // static collidables: never reached (no minis)
    void AddMiniBoxes();
    void RemoveMiniBoxes();
    void AddActualMiniBox(MiniBox*) { std::abort(); }
    void AddMiniCapsules();
    void RemoveMiniCapsules();
    void AddActualMinicapsule(MiniCapsule*) { std::abort(); }
    void GetRotatedVector(Vector3&) { std::abort(); }   // missiles, formations: never reached
    bool IsWarpish();
    bool IsAligningForWarp();
    bool IsWarping();
    bool IsAlignedForWarp();
};
class Capsule : public StaticCollidable {};             // never created here: BringOutDeadCapsules finds none
typedef Ball OBall;                                     // Blue's object classes: a client's ball is a plain Ball here
typedef Ball OClientBall;
typedef Ball ClientBall;

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
    void Gradient(Ball* b);
    void Potential(Ball *me, Ball *other, int recurionsDepth = 0);
    void InsertInBoxes(Partitionable *Partitionable);
    void InsertBallInBoxes(Ball *ball);
    Ball *BallExists(const ID& id);
    ID GetIDForCollisionObject(ID objectId);
    Ball* AddBall(const ID& objectId, double mass, float radius, float maxVel, bool isFree, bool isGlobal,
                  bool isMassive, bool isInteractive, bool isSpaceJunk, double x, double y, double z, double vx,
                  double vy, double vz, float agility, float speedFraction);
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
    void SetBallMass(const ID& srcId, double mass);
    void SetBallRadius(const ID& srcId, float radius);
    void SetBallRadius(Ball *ball, float radius);
    void SetMaxSpeed(const ID& srcId, float speed);
    void SetBallPosition(const ID& srcId, double x, double y, double z);
    void SetBallVelocity(const ID& srcId, double vx, double vy, double vz);
    void SetBallHarmonic(const ID& srcId, int64_t harmonic, int corporationID, int allianceID, bool field);
    void SetBallFree(const ID& srcId, bool flag);
    void SetBallAgility(const ID& srcId, float agility);
    void SetSpeedFraction(const ID& srcId, float fraction);
    void SetNotificationRange(const ID& srcId, double notificationRange);
    void SetTargetTracking(const ID& srcID, const ID& targetID, double trackingRange);
    void SetBallMassive(const ID& srcId, bool flag);
    void SetBallGlobal(const ID& srcId, bool flag);
    void AddProximitySensor(const ID& srcId, float range, double period, int shuffle, bool onlyInteractives);
    void RemoveProximitySensor(const ID& srcId);
    ID CheckVisibility(const ID& srcId, const ID& dstId);
    void HandleProximities();
    void ClearBubbles();
    void ClearAll();
    void UpdateBallBubble(const ID& srcID);
    void CopyBubbles();
    bool InDeadBubble(Ball *ball);
    size_t GetInteractiveCnt(Ball *ball);
    void SetBallInteractive(const ID& srcId, bool flag);
    void IncreaseInteractiveCnt(Partitionable *p, long inBubble);
    void DecreaseInteractiveCnt(Partitionable *p, long inBubble);
    void UpdateInteractiveCnt(Partitionable *p, long oldBubble, long newBubble);
    void AddKeepAliveBall(PyObject *ballId, long inBubble);
    void RemoveKeepAliveBall(PyObject *ballId, long inBubble);
    void UpdateKeepAliveBalls(Partitionable *p, long oldBubble, long newBubble);
    void AddToBubble(Partitionable *partitionable);
    void InitializeBubbles();
    PyObject* PyInitializeBubbles(PyObject* args);
    PyObject* PyGetRemoteFollowers(PyObject* args);
    PyObject* PyGetBallIdsInRange(PyObject* args);
    PyObject* PyGetBallIdsAndDistInRange(PyObject* args);
    PyObject* GetBallIdsInRange(PyObject* args, bool includeDist);
    bool BallIsValidForIteration(ID ballID, Ball*& ball, const Ball* refBall);
    int GetNextValidBallInBubble(PyObject* theBubble, Py_ssize_t& pos, PyObject*& key, PyObject*& value,
                                 Ball*& otherBall, const Ball* refBall);
    PyObject* PyCheckVisibility(PyObject* args);
    PyObject* PyGetBubbleAtCoordinates(PyObject* args);
    PyObject* PyCopyBubbles(PyObject* args);
    void RemoveBall(const ID& srcId, int delay = 0);
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
    // outside this slice: abort if reached
    bool TrollReady(Ball*) { std::abort(); }
    void PetrifyTroll(Ball*) { std::abort(); }
    void RemoveCapsule(const ID&, int delay = 0) { std::abort(); }
    void RemoveOrientedBox(const ID&, int delay = 0) { std::abort(); }
    Vector3d EvolveMissile(Ball*) { std::abort(); }
    Vector3d EvolveFormation(Ball*) { std::abort(); }
    Vector3 GotoTorqueBreaks(const Ball*, const Vector3&) { std::abort(); }
    float GotoRollBreaks(const Ball*, const Vector3&) { std::abort(); }
    void ApplyTorque(Quaternion&, Vector3&, const Vector3&, const float, const float, const float) { std::abort(); }
    void CalculateBallRotationVelocity(Ball*, double, Quaternion&, Vector3&) { std::abort(); }
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
    Ball* b = static_cast<Ball*>(static_cast<INotify*>(root));
    std::printf("E %lld %ld %s %lld", g_case, b->mPark ? b->mPark->mCurrentTime : -1L, method, (long long)b->mId);
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
// ---- driver: scenarios on stdin; after every tick every ball's state, the partition, the bubble
// dictionaries; every event and every query's answer, on stdout ----
struct Record {                                         // B: AddBall; K: a command or a query
    char kind; int step, op, w; long long a, b;
    double d[11];                                       // B: mass radius maxVel x y z vx vy vz agility sf; K: x y z r
    bool f[5];                                          // B: free global massive interactive junk
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
    h = mix(h, (uint64_t)box->mChildren.size());
    for (Box* c : box->mChildren) {
        h = mix(h, (uint64_t)c->mKey.ix);
        h = mix(h, (uint64_t)c->mKey.iy);
        h = mix(h, (uint64_t)c->mKey.iz);
    }
    h = mix(h, box->mParent ? (uint64_t)box->mParent->mLevel : 99ULL);
    return h;
}

static void print_state(Ballpark* bp, long long cid, int step) {
    BallIterator it(&bp->mBalls);
    Ball* b;
    while ((b = it++)) {
        Box* mb = b->mMyBox;
        long long active = 0;
        for (int i = 0; i < 27; ++i) if (b->mActiveBoxes[i]) active |= (1LL << i);
        std::printf("S %lld %d %lld %d %ld %d %d %lld %lld %lld %d %d %d %d %d %d %d %ld %ld %d %d %d %lld %d %d %d %d "
                    "%lld %ld %lld %lld %lld %lld %d %d %d",
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
                    (int)b->mAllianceID, (int)b->mSensor.onlyInteractives);
        const double d[] = {b->mNewPos.x, b->mNewPos.y, b->mNewPos.z, b->mNewVel.x, b->mNewVel.y, b->mNewVel.z,
                            b->mOldPos.x, b->mOldPos.y, b->mOldPos.z, b->mOldVel.x, b->mOldVel.y, b->mOldVel.z,
                            b->mGoto.x, b->mGoto.y, b->mGoto.z, b->mLastG.x, b->mLastG.y, b->mLastG.z,
                            b->mLastC.x, b->mLastC.y, b->mLastC.z, b->mMass, (double)b->mRadius,
                            (double)b->mMaxVel, (double)b->mAgility, (double)b->mSpeedFraction,
                            (double)b->mFollowRange, b->mLastCollision, b->mTimeFactor, b->mNotificationRange,
                            b->mTrackingRange, (double)b->mSensor.range, b->mSensor.period, b->mSensor.elapsed};
        for (double x : d) std::printf(" %.17g", x);
        std::printf(" %zu", b->mBoxes.size());
        for (Box* box : b->mBoxes)
            std::printf(" %ld %lld %lld %lld", box->mLevel, (long long)box->mKey.ix, (long long)box->mKey.iy,
                        (long long)box->mKey.iz);
        std::printf(" %zu", b->mCollisions.size());
        for (ID c : b->mCollisions) std::printf(" %lld", (long long)c);
        std::vector<long long> fs(b->mFollowers.begin(), b->mFollowers.end());
        std::sort(fs.begin(), fs.end());
        std::printf(" %zu", fs.size());
        for (long long f : fs) std::printf(" %lld", f);
        std::printf(" %zu", b->mSensor.balls.size());                          // in the unordered_map's order
        for (auto& kv : b->mSensor.balls) std::printf(" %lld", (long long)kv.first);
        std::printf("\n");
    }
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
    std::printf(" %zu", bp->mFreeBalls.size());
    std::printf("\n");
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
            std::printf(" %zu", box->mChildren.size());
            for (Box* c : box->mChildren)
                std::printf(" %lld %lld %lld", (long long)c->mKey.ix, (long long)c->mKey.iy, (long long)c->mKey.iz);
            std::printf("\n");
        }
    }
}

static void print_list(long long cid, int step, int op, long long a, long long b, PyObject* res, bool pairs) {
    if (!res) { std::printf("Q %lld %d %d %lld %lld 0 0\n", cid, step, op, a, b); PyErr_Clear(); return; }
    std::printf("Q %lld %d %d %lld %lld 1 %zu", cid, step, op, a, b, res->items.size());
    for (PyObject* o : res->items) {
        if (pairs) std::printf(" %.17g %lld", o->items[0]->fval, o->items[1]->ival);
        else std::printf(" %lld", o->ival);
    }
    std::printf("\n");
}

static PyObject* tuple_of(std::vector<PyObject*> items) { PyObject* t = PyTuple_New(0); t->items = items; return t; }

static void apply(Ballpark* bp, const Record& c, long long cid, int s) {
    const ID a = c.a, b = c.b;
    const double x = c.d[0], y = c.d[1], z = c.d[2], r = c.d[3];
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
            if (a != 0) args.push_back(PyLong_FromLongLong(a));
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
        case 28: PySet_Add(bp->bubbleKeepAliveBalls, PyLong_FromLongLong(a)); break;   // park.bubbleKeepAliveBalls
        case 29: bp->BringOutTheDeads(); break;         // OnTick's first call, before a tick
        case 30: bp->RemoveBall(a, c.w); break;
        case 31: {
            PyObject* r = bp->PyCheckVisibility(tuple_of({PyLong_FromLongLong(a), PyLong_FromLongLong(b)}));
            std::printf("Q %lld %d %d %lld %lld 1 1 %lld\n", cid, s, c.op, a, b, r->ival);
            break;
        }
        case 32: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsInRange(tuple_of({PyLong_FromLongLong(a),
                            PyFloat_FromDouble(r), PyLong_FromLongLong(c.w)})), false); break;
        case 33: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsInRange(tuple_of({PyFloat_FromDouble(x),
                            PyFloat_FromDouble(y), PyFloat_FromDouble(z), PyFloat_FromDouble(r),
                            PyLong_FromLongLong(c.w)})), false); break;
        case 34: print_list(cid, s, c.op, a, b, bp->PyGetBallIdsAndDistInRange(tuple_of({PyLong_FromLongLong(a),
                            PyFloat_FromDouble(r), PyLong_FromLongLong(c.w)})), true); break;
        case 35: {
            PyObject* r = bp->PyGetBubbleAtCoordinates(tuple_of({PyFloat_FromDouble(x), PyFloat_FromDouble(y),
                                                                 PyFloat_FromDouble(z)}));
            std::printf("Q %lld %d %d %lld %lld 1 1 %lld\n", cid, s, c.op, a, b, r == Py_None ? -999LL : r->ival);
            break;
        }
        default: std::exit(6);
    }
}

int main() {
    char tag[8];
    while (std::scanf("%7s", tag) == 1) {
        long long cid; double friction; int tick, steps, newOrbit, master, nr; long start; unsigned seed;
        if (tag[0] != 'C' || std::scanf("%lld %lf %d %d %d %d %ld %u %d", &cid, &friction, &tick, &steps, &newOrbit,
                                        &master, &start, &seed, &nr) != 9)
            return 3;
        g_case = cid;
        g_useNewOrbit = (newOrbit & 1) != 0;            // bit 0: useNewOrbit; bit 1: useIterativeCollision
        g_useIterativeCollision = (newOrbit & 2) != 0;
        std::srand(seed);                               // floatrand's rand(), for shuffled sensors
        std::vector<Record> recs(nr);
        for (auto& c : recs) {
            if (std::scanf("%7s", tag) != 1) return 4;
            c.kind = tag[0];
            if (c.kind == 'B') {
                int fl[5];
                if (std::scanf("%d %lld %lf %lf %lf %d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf", &c.step, &c.a,
                               &c.d[0], &c.d[1], &c.d[2], &fl[0], &fl[1], &fl[2], &fl[3], &fl[4], &c.d[3], &c.d[4],
                               &c.d[5], &c.d[6], &c.d[7], &c.d[8], &c.d[9], &c.d[10]) != 18)
                    return 4;
                for (int i = 0; i < 5; ++i) c.f[i] = fl[i] != 0;
            } else if (c.kind == 'K') {
                if (std::scanf("%d %d %lld %lld %d %lf %lf %lf %lf", &c.step, &c.op, &c.a, &c.b, &c.w, &c.d[0],
                               &c.d[1], &c.d[2], &c.d[3]) != 9)
                    return 5;
            } else return 5;
        }
        Ballpark* bp = new Ballpark();
        bp->mFriction = friction;
        bp->mTickInterval = tick;
        bp->dt = bp->mTickInterval * 0.001;
        bp->mCurrentTime = start;
        bp->isMaster = master != 0;
        std::set<Ball*> owned;
        for (int s = 0; s <= steps; ++s) {
            for (const auto& c : recs) {
                if (c.step != s) continue;
                if (c.kind == 'B') {                    // AddBall as the Python thunker calls it
                    owned.insert(bp->AddBall(c.a, c.d[0], (float)c.d[1], (float)c.d[2], c.f[0], c.f[1], c.f[2],
                                             c.f[3], c.f[4], c.d[3], c.d[4], c.d[5], c.d[6], c.d[7], c.d[8],
                                             (float)c.d[9], (float)c.d[10]));
                } else apply(bp, c, cid, s);
            }
            if (s > 0) bp->Evolve(0);
            print_state(bp, cid, s);
        }
        dump_partition(bp, cid);
        delete bp;
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
    end = block_end(src, hits[0].start())              # the pattern may include the opening brace
    if name.split()[0] in ("enum", "class", "struct"):    # keep the declaration's semicolon
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

    def piece(name: str, body: str, origin: str) -> str:
        return f"\n// ---- {origin}, {name}, verbatim ----\n{body}\n"

    def from_file(path: str, skip=()) -> str:              # a file's extracted definitions, in source order
        names = sorted((n for f, n, _ in BLOCKS if f == path and n not in skip), key=lambda n: where[n][1])
        return "".join(piece(n, blocks[n], path) for n in names)

    def the_lines(path: str, names) -> str:
        return "".join(piece(n, lines[n], path) for n in names)

    declarations = ["class BoxSortComparer", "class BallSortComparer", "class StaticCollidableSortComparer"]
    source = (PRELUDE_A
              + piece("enum DSTBALLMODE", blocks["enum DSTBALLMODE"], "src/IDstConstants.h")
              + piece("enum DSTCONSTANTS", blocks["enum DSTCONSTANTS"], "src/IDstConstants.h")
              + the_lines("src/Ball.h", ("typedef VectorOfBalls", "typedef ID", "typedef VectorOfIDs",
                                         "typedef SetOfIDs", "MAX_ALIGN_TICKS"))
              + piece("AU", lines["AU"], "src/Ballpark.h")
              + "\n// ---- src/Vector3d.h, verbatim ----\n" + text["src/Vector3d.h"]
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
              + piece("class BoxAllocator", blocks["class BoxAllocator"], "src/Partition.h")
              + piece("class Key", blocks["class Key"], "src/Partition.h")
              + piece("struct KeyHash", blocks["struct KeyHash"], "src/Partition.h")
              + piece("typedef MapOfBoxes", lines["typedef MapOfBoxes"], "src/Partition.h")
              + piece("class Partition", blocks["class Partition"], "src/Partition.h")
              + piece("class Partitionable", blocks["class Partitionable"], "src/Partitionable.h")
              + piece("class Box", blocks["class Box"], "src/Box.h")
              + piece("class StaticCollidable", blocks["class StaticCollidable"], "src/StaticCollidable.h")
              + piece("class MapOfLongLongs", blocks["class MapOfLongLongs"], "src/MapOfLongLongs.h")
              + piece("class ProximitySensor", blocks["class ProximitySensor"], "src/Ball.h")
              + "".join(piece(n, blocks[n], "src/CollisionStructures.h") for n in (
                  "struct CollisionImpact", "struct CollisionItem", "struct BallLocation"))
              + the_lines("src/Ballpark.h", ("typedef DictOfFreeBalls", "typedef SetOfBalls", "typedef SetOfCapsules",
                                             "typedef SetOfOrientedBoxes", "typedef VectorOfVectors",
                                             "typedef VectorOfCapsules"))
              + PRELUDE_B.replace("__COLLISION_BALL_PROPERTIES__", piece(
                  "class CollisionBallProperties", blocks["class CollisionBallProperties"],
                  "src/CollisionBallProperties.h"))
              + the_lines("src/Collision.h", ("declaration Quadratic",))
              + from_file("src/Collision.cpp")
              + the_lines("src/Partition.cpp", ("collisionCase1, collisionCase3",))
              + from_file("src/Partition.cpp")
              + from_file("src/Box.cpp")
              + from_file("src/Partitionable.cpp")
              + from_file("src/CollisionStructures.cpp")
              + from_file("src/CollisionBallProperties.cpp")
              + from_file("src/StaticCollidable.cpp")
              + the_lines("src/Ballpark.cpp", ("Timer_ResolveBubbleConflicts", "Timer_HandleProximities",
                                               "Timer_InsertBallInBoxes", "Timer_Gradient", "Timer_CheckVisibility",
                                               "Timer_CopyBubbles", "Timer_AdditionsAndDeletions",
                                               "Timer_GetBallIdsInRange", "Timer_WarpDistance",
                                               "Timer_CalculateYawPitchRoll", "Timer_WriteBallsToStream", "Plus",
                                               "Zero", "Minus", "TICK_EVOLVE", "SIGNUM (Ballpark.cpp)",
                                               "ABS (Ballpark.cpp)", "s_ballNotInParkCallback", "ballparkCounter",
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
    cmd = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-o", str(build / "harness"),
           str(build / "harness.cpp")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"g++ failed with exit status {res.returncode}")
    gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    manifest = {"destiny_commit": head, "compiler": gxx, "command": cmd,
                "compiler_warnings": [w for w in res.stderr.splitlines() if "warning:" in w],
                "source_sha256": {f: sha(t) for f, t in text.items()},
                "extracted_sha256": {**{n: sha(b) for n, b in blocks.items()}, **{n: sha(s) for n, s in lines.items()}},
                "extracted_lines": dict(lines),
                "harness_cpp_sha256": sha(source),
                "binary_sha256": sha((build / "harness").read_bytes())}
    (build / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "extracted_lines"}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
