-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import Mathlib.Tactic.NormNum

@[expose] public noncomputable section

namespace ExactBallpark.Warp
open ExactBallpark.Movement

/-- Fields observed by the WarpTo decision. Projected from one existing ball.
minRange holds the decoded value of the follow_id bitcast. -/
structure CommandState where
  pos : Vec3
  target : Vec3
  warping : Bool
  stamp : ℤ
  minRange : ℝ
  factor : ℤ

inductive Decision where | goto | keep | align deriving DecidableEq

def warpDecision (s : CommandState) (dst : Vec3) : Decision := by
  classical
  exact if ‖s.pos - dst‖ ^ 2 < 10000000000 then .goto
    else if s.warping ∧ dst = s.target then .keep else .align

def warpTo (gotoPoint : CommandState → Vec3 → CommandState) (s : CommandState) (dst : Vec3) (minRange : ℝ) (factor : ℤ) :=
  match warpDecision s dst with
  | .goto => gotoPoint s dst
  | .keep => s
  | .align => {s with target := dst, warping := true, stamp := -1,
                      minRange := max 0 minRange, factor := max 1 factor}

theorem short_iff (s : CommandState) (dst : Vec3) :
    warpDecision s dst = .goto ↔ ‖s.pos - dst‖ < 100000 := by
  classical
  have h := sq_lt_sq₀ (norm_nonneg (s.pos - dst)) (show (0 : ℝ) ≤ 100000 by norm_num)
  norm_num at h
  rw [← h]
  unfold warpDecision
  split_ifs <;> simp_all

theorem repeated_warp (gotoPoint : CommandState → Vec3 → CommandState) (s : CommandState) (minRange : ℝ) (factor : ℤ)
    (h : 100000 ≤ ‖s.pos - s.target‖) (hw : s.warping = true) :
    warpTo gotoPoint s s.target minRange factor = s := by
  have far : ¬ ‖s.pos - s.target‖ ^ 2 < 10000000000 := by nlinarith
  simp [warpTo, warpDecision, far, hw]

theorem long_starts_alignment (gotoPoint : CommandState → Vec3 → CommandState) (s : CommandState) (dst : Vec3) (mr : ℝ) (f : ℤ)
    (hd : 100000 ≤ ‖s.pos - dst‖) (fresh : ¬ (s.warping = true ∧ dst = s.target)) :
    let next := warpTo gotoPoint s dst mr f
    next.target = dst ∧ next.warping = true ∧ next.stamp = -1 ∧
      next.minRange = max 0 mr ∧ next.factor = max 1 f := by
  have far : ¬ ‖s.pos - dst‖ ^ 2 < 10000000000 := by nlinarith
  simp [warpTo, warpDecision, far, fresh]

def aligned (inWarp : Bool) (stamp : ℤ) (cosine speedSq maxSpeed : ℝ) : Prop :=
  inWarp = true ∧ stamp < 0 ∧
    ((|1 - cosine| < 5764607523034235 / 576460752303423488 ∧
      speedSq > (9 / 16) * maxSpeed * maxSpeed) ∨ 180 < |stamp|)

/-- stamp starts at -1 and is decremented on each unsuccessful alignment call. -/
theorem alignment_after_waits (n : ℕ) (cosine speed maxSpeed : ℝ)
    (hs : 0 ≤ speed) (hm : 0 ≤ maxSpeed) :
    aligned true (-1 - n) cosine (speed ^ 2) maxSpeed ↔
    (|1 - cosine| < 5764607523034235 / 576460752303423488 ∧ 3 / 4 * maxSpeed < speed) ∨
    180 ≤ n := by
  have t : 180 < |(-1 : ℤ) - n| ↔ 180 ≤ n := by
    rw [abs_of_nonpos (by omega)]; omega
  have v : (9 / 16 : ℝ) * maxSpeed * maxSpeed < speed ^ 2 ↔ 3 / 4 * maxSpeed < speed := by
    have h := (sq_lt_sq₀ (show 0 ≤ (3/4 : ℝ) * maxSpeed by positivity) hs)
    convert h using 2; ring
  simp [aligned, t, v, show (-1 : ℤ) - n < 0 by omega]

/-- RealWarp offsets the original destination toward the ship, even when the offset
exceeds the original distance and the new destination lies behind the start. -/
def exitPoint (pos dst : Vec3) (mr : ℝ) :=
  let delta := pos - dst
  dst + mr • (if ‖delta‖ = 0 then delta else (1 / ‖delta‖) • delta)

theorem exit_distance (pos dst : Vec3) (mr : ℝ) (h : pos ≠ dst) (hm : 0 ≤ mr) :
    ‖exitPoint pos dst mr - dst‖ = mr := by
  have nz : ‖pos - dst‖ ≠ 0 := norm_ne_zero_iff.mpr (sub_ne_zero.mpr h)
  simp [exitPoint, nz, norm_smul, Real.norm_eq_abs, abs_of_nonneg hm]

end ExactBallpark.Warp
