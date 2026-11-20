-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.SensorScan

@[expose] public section
namespace ExactBallpark.Space.Uncloak

structure Observer where
  bubble : Int
  registered : Bool
  moribund : Bool
  hasProximity : Bool
  containsShip : Bool

/-- The observer fields are read after any master-side sensor deregistration. -/
def retained (shipBubble : Int) (warping : Bool) (o : Observer) :=
  if warping then o.containsShip else
  if o.registered && o.bubble == shipBubble && !o.moribund && o.hasProximity then false
  else o.containsShip

def uncloak (master : Bool) (cloakMode : Nat) (warpMode : Bool) (effectStamp : Int)
    (massive sensorActive : Bool) :=
  let activeWarp := warpMode && decide (0 ≤ effectStamp)
  ((0 : Nat), if activeWarp then massive else true,
    if master && cloakMode == 1 then false else sensorActive)

theorem warp_and_sensor (master warpMode massive sensorActive : Bool) (mode : Nat) (stamp : Int) :
    let (cloak, mass, sensor) := uncloak master mode warpMode stamp massive sensorActive
    cloak = 0 ∧ (mass = false ↔ warpMode ∧ 0 ≤ stamp ∧ massive = false) ∧
    (sensor = true ↔ sensorActive ∧ (master = false ∨ mode ≠ 1)) := by
  simp only [uncloak]
  cases warpMode <;> cases master <;> cases massive <;> cases sensorActive <;> simp

/-- An existing contact disappears exactly from qualifying observers, including during alignment.
The next complete scan can emit a fresh enter at unchanged separation. -/
theorem erase_and_reenter (owner ship shipBubble : Int) (warping : Bool) (o : Observer)
    (prior : o.containsShip = true) (range radius : Float32) (target : SensorScan.Candidate)
    (id : target.id = ship) (other : ship ≠ owner) (live : target.moribund = false)
    (inside : SensorScan.inside range radius target = true) :
    let removed := warping = false ∧ o.registered ∧ o.bubble = shipBubble ∧ o.moribund = false ∧ o.hasProximity
    (retained shipBubble warping o = false ↔ removed) ∧
    SensorScan.scan owner ship range radius (retained shipBubble warping o) [target] =
      (true, if removed then [true] else []) := by
  dsimp only
  have erased : retained shipBubble warping o = false ↔
      warping = false ∧ o.registered ∧ o.bubble = shipBubble ∧ o.moribund = false ∧ o.hasProximity := by
    cases warping <;> simp [retained, prior, and_assoc]
  refine ⟨erased, ?_⟩
  subst ship
  have sample := SensorScan.present_target owner range radius [] [] target
    (retained shipBubble warping o) (by simp) (by simp)
  simp only [← erased]
  cases h : retained shipBubble warping o <;> simpa [other, live, inside, h] using sample

end ExactBallpark.Space.Uncloak
