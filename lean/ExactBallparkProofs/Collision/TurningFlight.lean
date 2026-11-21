-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.CurveClearance
public import ExactBallparkProofs.Movement.CoastWitness

@[expose] public section
namespace ExactBallpark.Collision.TurningFlight
open ExactBallpark.Movement DragChord CurveClearance Set

noncomputable def xy (x y : ℝ) : Vec3 := WithLp.toLp 2 ![x,y,0]

noncomputable def turn : Flight := ⟨0, xy 10 0, xy 0 10, 1, 1⟩
private noncomputable def fixed (y : ℝ) : Flight := ⟨xy 4 y, 0, 0, 1, 1⟩

def cached : Float := Float.ofBits 0x3fd78b56362cef38
def origin : FloatPoint := fun _ => 0
def obstacle (y : Float) : FloatPoint := ![4,y,0]

/-- One full tick of the translated scalar integrator. The cache supplies exp(−1). -/
def finish : FloatPoint := fun i =>
  (Reference.binary64 id 0 (![10,0,0] i) (![0,10,0] i) 1 1 cached 1 1).1

def movingBudget : Budget := ⟨15, 0, 1 / 10^12⟩
def fixedBudget : Budget := ⟨0, 0, 0⟩
def clearMargin : ℚ := 5469138 / 10^6 - 1/2 - 15/8 - 1/10^12

private def endPoint : RatPoint :=
  ![7117044782983805 / 1125899906842624, 4141954285442435 / 1125899906842624, 0]

private theorem turn_pos (t : ℝ) :
    turn.pos t = xy (10*(1-Real.exp (-t))) (10*t - 10*(1-Real.exp (-t))) := by
  ext i
  fin_cases i <;> simp [turn, Flight.pos, xy, position, terminal, decay] <;> ring

private theorem exp_bounds :
    36787944117144232/10^17 < Real.exp (-1) ∧ Real.exp (-1) < 36787944117144233/10^17 := by
  obtain ⟨h₁,h₂⟩ := abs_sub_le_iff.mp Real.exp_one_near_20
  rw [Real.exp_neg, inv_eq_one_div]
  constructor
  · apply (lt_div_iff₀ (Real.exp_pos _)).mpr
    linarith
  · apply (div_lt_iff₀ (Real.exp_pos _)).mpr
    linarith

private theorem half_overlap : dist (turn.pos (1/2)) (xy 4 1) < 1/10 := by
  let q := Real.exp (-(1/2))
  have q₀ : 0 ≤ q := (Real.exp_pos _).le
  have square : q^2 = Real.exp (-1) := by
    rw [sq, ← Real.exp_add]; norm_num
  have lower : 6065/10000 < q := (sq_lt_sq₀ (by norm_num) q₀).mp (by
    rw [square]; linarith [exp_bounds.1])
  have upper : q < 6066/10000 := (sq_lt_sq₀ q₀ (by norm_num)).mp (by
    rw [square]; linarith [exp_bounds.2])
  rw [turn_pos, dist_eq_norm, ← sq_lt_sq₀ (norm_nonneg _) (by norm_num),
    EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  change (10*(1-q)-4)^2 + (10*(1/2)-10*(1-q)-1)^2 + (0-0:ℝ)^2 < (1/10)^2
  nlinarith [mul_pos (show 0 < 7/100 - (10*q-6) by linarith)
    (show 0 < 7/100 + (10*q-6) by linarith)]

private theorem ideal_chord_clear {u : ℝ} (hu : u ∈ Icc 0 1) :
    1/2 < dist (lerp turn.start (turn.pos 1) u) (xy 4 1) := by
  let q := Real.exp (-1)
  let dx := u*(10*(1-q))-4
  let dy := u*(10-10*(1-q))-1
  have line : dx-2*dy ≤ -2 := by
    have hq : 10-30*q ≤ 0 := by dsimp [q]; linarith [exp_bounds.1]
    dsimp [dx,dy]; nlinarith [mul_nonpos_of_nonneg_of_nonpos hu.1 hq]
  rw [turn_pos, dist_eq_norm, ← sq_lt_sq₀ (by norm_num) (norm_nonneg _),
    EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  simp only [turn, lerp, smul_zero, zero_add, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
  change (1/2:ℝ)^2 < (u*(10*(1-q))-4)^2 + (u*(10*1-10*(1-q))-1)^2 + (u*0-0)^2
  simp only [mul_one, mul_zero, sub_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, add_zero]
  change (1/2:ℝ)^2 < dx^2 + dy^2
  nlinarith [sq_nonneg (2*dx+dy), mul_nonneg (show 0 ≤ -(dx-2*dy)-2 by linarith)
    (show 0 ≤ -(dx-2*dy)+2 by linarith)]

/-- Even with exact ordinary endpoints the chord misses. Inside the tick the real
trajectory passes within ten centimetres of an obstacle with combined radius half a metre. -/
theorem turning_overlap :
    (∀ u ∈ Icc (0:ℝ) 1, 1/2 < dist (lerp turn.start (turn.pos 1) u) (xy 4 1)) ∧
    dist (turn.pos (1/2)) (xy 4 1) < 1/10 ∧
    ∀ t ∈ Icc (0:ℝ) 1, ¬ turn.friction < threshold * turn.mass * t := by
  refine ⟨fun _ hu => ideal_chord_clear hu, half_overlap, fun t ht => ?_⟩
  norm_num [turn, threshold]
  linarith [ht.2]

open Scalar in
def gotoWords : Option (List UInt64) := do
  let ship : Spawn := ⟨1, 0.25, 10, zeros, vec3 10 0 0, 1, 1⟩
  let park ← run ⟨1, 1⟩ CoastWitness.coastExp {}
    [.edit 1 (.add ship), .edit 1 (.gotoPoint (vec3 0 100 0)), .tick]
  let b ← park.balls[(1 : Int)]?
  return [b.timeFactor.toBits, b.lastG.x.toBits, b.lastG.y.toBits, b.lastG.z.toBits,
    b.pos.x.toBits, b.pos.y.toBits, b.pos.z.toBits]

set_option maxRecDepth 32768 in
set_option maxHeartbeats 3000000 in
/-- The modeled add, GOTO and full tick produce the supplied cache, applied thrust
and all three endpoint words used by the collision replay. -/
theorem goto_sweep_inputs : gotoWords =
    some [cached.toBits, 0, (10 : Float).toBits, 0, (finish 0).toBits, (finish 1).toBits, (finish 2).toBits] := by
  decide +kernel

set_option maxRecDepth 16384 in
set_option maxHeartbeats 2000000 in
/-- Both engine sweeps return −1. Only the lower obstacle admits a continuous clearance certificate. -/
theorem turning_words :
    (finish 0).toBits = 0x401948ea1e23ea7d ∧ (finish 1).toBits = 0x400d6e2bc3b82b06 ∧
    (finish 2).toBits = 0 ∧
    (contact64 origin finish (obstacle 1) (obstacle 1) 0.5).toBits = 0xbff0000000000000 ∧
    (contact64 origin finish (obstacle (-4)) (obstacle (-4)) 0.5).toBits = 0xbff0000000000000 ∧
    check origin finish (obstacle 1) (obstacle 1) 0.5 1 movingBudget fixedBudget = none ∧
    check origin finish (obstacle (-4)) (obstacle (-4)) 0.5 1 movingBudget fixedBudget = some clearMargin := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> decide +kernel

private theorem turn_error : dist (turn.pos 1) (asVec endPoint) ≤ 1/10^12 := by
  rw [turn_pos]
  have h := position_error_from_axes (xy (10*(1-Real.exp (-1))) (10*1-10*(1-Real.exp (-1))))
    (asVec endPoint) (fun _ => 1/10^13) (by
      intro i
      fin_cases i <;> norm_num [xy, asVec, realPoint, endPoint, abs_le] <;> constructor <;>
        linarith [exp_bounds.1,exp_bounds.2])
  linarith

private theorem turn_budget : movingBudget.covers turn 0 (asVec endPoint) 1 := by
  refine ⟨?_, by simp [turn, movingBudget], ?_⟩
  · change ‖turn.forceField turn.initialVelocity‖ ≤ (15:ℝ)
    rw [← sq_le_sq₀ (norm_nonneg _) (by norm_num), EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
    norm_num [turn, xy, Flight.forceField]
  · simpa [movingBudget] using turn_error

private theorem fixed_pos (y t : ℝ) : (fixed y).pos t = xy 4 y := by
  ext i
  simp [fixed, Flight.pos, position, terminal]

set_option maxRecDepth 16384 in
/-- Discharge both bodies' curvature and endpoint premises for the clear control.
The gap between their surfaces exceeds three metres throughout this ordinary tick. -/
theorem turning_clearance : 3 < clearMargin ∧
    ∀ u ∈ Icc (0:ℝ) 1, (1/2 : ℝ) + clearMargin ≤ dist (turn.pos u) (xy 4 (-4)) := by
  have passed := turning_words.2.2.2.2.2.2
  obtain ⟨_, _, s, dec, safe⟩ := certified_clearance passed
  let control : ExactChord := ⟨0, endPoint, ![4,-4,0], ![4,-4,0], 1/2⟩
  have decoded : decodeChord origin finish (obstacle (-4)) (obstacle (-4)) 0.5 = some control := by
    unfold decodeChord
    rw [show decodePoint origin = some (0 : RatPoint) by decide +kernel,
      show decodePoint finish = some endPoint by decide +kernel,
      show decodePoint (obstacle (-4)) = some (![4,-4,0] : RatPoint) by decide +kernel,
      show finiteValue 0.5 = some (1/2) by decide +kernel]
    rfl
  have eq := Option.some.inj (dec.symm.trans decoded)
  subst s
  have obs : asVec control.q₀ = xy 4 (-4) := by ext i; fin_cases i <;> norm_num [control, asVec, realPoint, xy]
  have stationary : fixedBudget.covers (fixed (-4)) (asVec control.q₀) (asVec control.q₁) 1 := by
    change fixedBudget.covers (fixed (-4)) (asVec control.q₀) (asVec control.q₀) 1
    unfold Budget.covers
    rw [fixed_pos, obs]
    simp [fixedBudget, Flight.initialAccel, Flight.forceField, fixed]
  have start : asVec control.p₀ = 0 := by ext i; simp [control, asVec, realPoint]
  norm_num only [Rat.cast_one] at safe
  have clear := safe turn (fixed (-4)) (by constructor <;> norm_num [turn]) (by constructor <;> norm_num [fixed])
    (by rw [start]; exact turn_budget) stationary
  refine ⟨by norm_num [clearMargin], fun u hu => ?_⟩
  simpa [control, fixed_pos] using clear u hu

end ExactBallpark.Collision.TurningFlight
