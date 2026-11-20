-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Proximity
public import ExactBallparkProofs.Numerics.Agility

@[expose] public section
namespace ExactBallpark.Space.SensorScan
open ExactBallpark.Numerics

/-- Per-candidate inputs after spatial collection, with the scalar squared distance supplied. -/
structure Candidate where
  id : Int
  moribund : Bool
  radius : Float32
  distanceSq : Float

def inside (range ownRadius : Float32) (o : Candidate) : Bool :=
  let threshold := widen (narrow (widen (narrow (widen range + widen o.radius)) + widen ownRadius))
  !(decide (o.distanceSq > threshold * threshold))

/-- Project the candidate loop onto one target's membership and ordered event flags. -/
def visits (owner target : Int) (range radius : Float32) (member : Bool) :
    List Candidate → Bool × List Bool
  | [] => (member, [])
  | o :: rest =>
    if o.id ≠ target ∨ o.id = owner ∨ o.moribund then visits owner target range radius member rest
    else
      let (current, event) := edge member (inside range radius o)
      let (final, later) := visits owner target range radius current rest
      (final, event.toList ++ later)

/-- The vanished-target pass runs before the candidate loop. Assumes a complete,
uninterrupted due scan and stable candidates throughout both passes. -/
def scan (owner target : Int) (range radius : Float32) (member : Bool) (nearby : List Candidate) :=
  if member && target != owner && !(nearby.any (fun o => o.id == target)) then
    let (final, later) := visits owner target range radius false nearby
    (final, false :: later)
  else visits owner target range radius member nearby

private theorem skip_unrelated (owner target : Int) (range radius : Float32)
    (xs ys : List Candidate) (member : Bool) (absent : ∀ o ∈ xs, o.id ≠ target) :
    visits owner target range radius member (xs ++ ys) = visits owner target range radius member ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    have hx := absent x (by simp)
    simpa [visits, hx] using ih (fun o ho => absent o (by simp [ho]))

theorem vanished_target (owner target : Int) (range radius : Float32) (xs : List Candidate)
    (member : Bool) (absent : ∀ o ∈ xs, o.id ≠ target) :
    scan owner target range radius member xs =
      (if target = owner then member else false, if member ∧ target ≠ owner then [false] else []) := by
  have miss : xs.any (fun o => o.id == target) = false := by simpa using absent
  have skip (m : Bool) := skip_unrelated owner target range radius xs [] m absent
  cases member <;> by_cases h : target = owner <;> simp_all [scan, visits]

theorem present_target (owner : Int) (range radius : Float32) (before after : List Candidate)
    (o : Candidate) (old : Bool) (left : ∀ x ∈ before, x.id ≠ o.id)
    (right : ∀ x ∈ after, x.id ≠ o.id) :
    scan owner o.id range radius old (before ++ o :: after) =
      if o.id = owner ∨ o.moribund then (old, [])
      else (inside range radius o, if old = inside range radius o then [] else [inside range radius o]) := by
  simp only [scan, List.any_append, List.any_cons, beq_self_eq_true, Bool.true_or, Bool.or_true,
    Bool.not_true, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  rw [skip_unrelated _ _ _ _ _ _ _ left]
  have skip (m : Bool) : visits owner o.id range radius m after = (m, []) := by
    simpa [visits] using skip_unrelated owner o.id range radius after [] m right
  by_cases h : o.id = owner ∨ o.moribund
  · simp [visits, h, skip]
  · simp [visits, h, skip, crossing]
    split <;> rfl

/-- Trace of consecutive due / waiting decisions in the real clock model. -/
noncomputable def polls (elapsed dt period : ℝ) : ℕ → List Bool × ℝ
  | 0 => ([], elapsed)
  | n+1 =>
    let (due, next) := sensorClock elapsed dt period
    let (rest, final) := polls next dt period n
    (due :: rest, final)

/-- The first check occurs on the first sample at or above the period and discards overshoot. -/
theorem first_due_poll (n : ℕ) (elapsed dt period : ℝ) (step : 0 ≤ dt)
    (waiting : elapsed + n*dt < period) (due : period ≤ elapsed + (n+1)*dt) :
    polls elapsed dt period (n+1) = (List.replicate n false ++ [true], 0) := by
  induction n generalizing elapsed with
  | zero =>
    have reached : period ≤ elapsed + dt := by simpa using due
    simp [polls, sensorClock, not_lt.mpr reached]
  | succ n ih =>
    have hn : 0 ≤ (n : ℝ) * dt := mul_nonneg (Nat.cast_nonneg _) step
    push_cast at waiting due
    have early : elapsed + dt < period := by nlinarith
    have rest := ih (elapsed+dt) (by linarith only [waiting]) (by linarith only [due])
    simpa [polls, sensorClock, early, List.replicate_succ] using
      congrArg (fun r : List Bool × ℝ => (false :: r.1, r.2)) rest

end ExactBallpark.Space.SensorScan
