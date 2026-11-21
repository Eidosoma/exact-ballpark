-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.ReboundTick

@[expose] public section
namespace ExactBallpark.Collision.ReboundAccuracy
open Movement.PositionEnclosure ReboundFlight ReboundTick

/-- Left x, left vx, right x, right vx at the end of the mathematical tick. -/
noncomputable def idealEnd (c : Scenario) : Fin 4 → ℝ :=
  ![c.after.1.pos (1/2) 0,c.after.1.vel (1/2) 0,c.after.2.pos (1/2) 0,c.after.2.vel (1/2) 0]

def enclosures (c : Scenario) : Fin 4 → Bounds :=
  let q : Bounds := ⟨qLo,qHi⟩
  let w := (Bounds.point 1).sub q
  ![(Bounds.point 2).add (w.scale c.leftKick), (Bounds.point 2).add (q.scale c.leftKick),
    (Bounds.point 1).add (w.scale c.rightKick), (Bounds.point (-2)).add (q.scale c.rightKick)]

theorem ideal_endpoint_enclosed (c : Scenario) (i : Fin 4) :
    (enclosures c i).Contains (idealEnd c i) := by
  let q : Bounds := ⟨qLo,qHi⟩
  have hq : q.Contains (Real.exp (-(1/2:ℝ))) := half_decay
  have hw := Bounds.sub_contains (Bounds.point_contains 1) hq
  obtain ⟨pa,pb,va,vb⟩ := remainder c (1/2)
  unfold idealEnd
  rw [pa,pb,va,vb]
  rw [show (1:ℝ)+2*(1/2) = 2 by norm_num, show (2:ℝ)-2*(1/2) = 1 by norm_num]
  fin_cases i
  · simpa [enclosures, axis, q, mul_comm] using
      Bounds.add_contains (Bounds.point_contains 2) (Bounds.scale_contains c.leftKick hw)
  · simpa [enclosures, axis, q, mul_comm] using
      Bounds.add_contains (Bounds.point_contains 2) (Bounds.scale_contains c.leftKick hq)
  · simpa [enclosures, axis, q, mul_comm] using
      Bounds.add_contains (Bounds.point_contains 1) (Bounds.scale_contains c.rightKick hw)
  · simpa [enclosures, axis, q, mul_comm] using
      Bounds.add_contains (Bounds.point_contains (-2)) (Bounds.scale_contains c.rightKick hq)

def component (s : SideWords × SideWords) : Fin 4 → UInt64 :=
  ![s.1.pos.1,s.1.vel.1,s.2.pos.1,s.2.vel.1]

def tolerance : ℚ := 1/10^15

def withinBudget (b : Bounds) (word : UInt64) : Bool :=
  match finiteValue (Float.ofBits word) with
  | none => false
  | some x => decide (x-tolerance ≤ b.lo ∧ b.hi ≤ x+tolerance)

/-- Decode each actual end word of the replay. The real position error is at most
10⁻¹⁵ m and the real speed-component error at most 10⁻¹⁵ m/s in these three ticks. -/
theorem endpoint_error (c : Scenario) {s : SideWords × SideWords} (ok : replay c = some s)
    (i : Fin 4) : ∃ x : ℚ, finiteValue (Float.ofBits (component s i)) = some x ∧
      |(x:ℝ) - idealEnd c i| ≤ (tolerance:ℝ) := by
  rw [replay_words] at ok
  obtain rfl := Option.some.inj ok
  have checks : ∀ j : Fin 4, withinBudget (enclosures c j) (component (recorded c) j) = true := by
    cases c <;> decide +kernel
  have accepted := checks i
  unfold withinBudget at accepted
  split at accepted
  next => contradiction
  next x hx =>
    have limits := of_decide_eq_true accepted
    have lo : (x:ℝ) - tolerance ≤ ((enclosures c i).lo:ℝ) := by exact_mod_cast limits.1
    have hi : ((enclosures c i).hi:ℝ) ≤ (x:ℝ) + tolerance := by exact_mod_cast limits.2
    obtain ⟨left,right⟩ := ideal_endpoint_enclosed c i
    exact ⟨x,hx,abs_le.mpr ⟨by linarith, by linarith⟩⟩

end ExactBallpark.Collision.ReboundAccuracy
