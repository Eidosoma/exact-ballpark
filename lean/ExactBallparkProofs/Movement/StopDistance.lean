-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Modes
public import Mathlib.Analysis.SpecificLimits.Basic

@[expose] public noncomputable section
namespace ExactBallpark.Movement.StopDistance
open Filter Topology

/-- On the vertical axis STOP prepares velocity before integrating; horizontal axes coast. -/
def damping (vertical : Bool) : ℝ := if vertical then stopVertical 1 else 1

def step (vertical : Bool) (m k t : ℝ) (s : ℝ × ℝ) : ℝ × ℝ :=
  let v := if vertical then stopVertical s.2 else s.2
  (position s.1 v 0 m k t, velocity v 0 m k t)

def gain (vertical : Bool) (m k t : ℝ) := decay m k t * damping vertical
def travel (vertical : Bool) (m k t : ℝ) := m / k * (1 - decay m k t) * damping vertical

private theorem one_step (b : Bool) (m k t : ℝ) (s : ℝ × ℝ) :
    step b m k t s = (s.1 + travel b m k t * s.2, gain b m k t * s.2) := by
  apply Prod.ext <;> cases b <;> simp [step, damping, travel, gain, position, velocity, terminal, stopVertical] <;> ring

theorem axis_velocity (b : Bool) (m k t : ℝ) (s : ℝ × ℝ) (n : ℕ) :
    ((step b m k t)^[n] s).2 = gain b m k t ^ n * s.2 := by
  induction n with
  | zero => simp
  | succ n ih => rw [Function.iterate_succ_apply', one_step]; simp only [ih, pow_succ]; ring

theorem axis_position (b : Bool) (m k t : ℝ) (s : ℝ × ℝ) (n : ℕ) (hg : gain b m k t ≠ 1) :
    ((step b m k t)^[n] s).1 =
      s.1 + travel b m k t * s.2 * (1 - gain b m k t ^ n) / (1 - gain b m k t) := by
  have den := sub_ne_zero.mpr (Ne.symm hg)
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', one_step, ih, axis_velocity, pow_succ]
    field_simp
    ring

/-- The horizontal asymptote is p + m v/k. Vertical damping shortens the distance
travelled for a positive initial speed. Both axes use the ordinary position branch. -/
theorem rest_point {m k t : ℝ} (hm : 0 < m) (hk : 0 < k) (ht : 0 < t) (p v : ℝ) :
    let rest := fun b => p + travel b m k t * v / (1 - gain b m k t)
    (∀ b, Tendsto (fun n : ℕ => ((step b m k t)^[n] (p,v)).1) atTop (𝓝 (rest b))) ∧
    rest false = p + m / k * v ∧ (0 < v → rest true < rest false) := by
  have hq := (decay_bounds hm hk ht.le).1
  have qlt : decay m k t < 1 := Real.exp_lt_one_iff.mpr
    (mul_neg_of_neg_of_pos (div_neg_of_neg_of_pos (neg_lt_zero.mpr hk) hm) ht)
  have hd : 0 < damping true ∧ damping true < 1 := by norm_num [damping,stopVertical]
  have gains (b : Bool) : 0 ≤ gain b m k t ∧ gain b m k t < 1 := by
    cases b
    · simpa [gain,damping] using And.intro hq.le qlt
    · exact ⟨(mul_pos hq hd.1).le, (mul_lt_of_lt_one_right hq hd.2).trans qlt⟩
  dsimp only
  constructor
  · intro b
    have lim := tendsto_pow_atTop_nhds_zero_of_lt_one (gains b).1 (gains b).2
    simp_rw [axis_position b m k t (p,v) _ (gains b).2.ne]
    simpa using (tendsto_const_nhds (x := p)).add (((tendsto_const_nhds (x := travel b m k t * v)).mul
      ((tendsto_const_nhds (x := (1:ℝ))).sub lim)).div_const (1 - gain b m k t))
  have horiz : p + travel false m k t * v / (1 - gain false m k t) = p + m/k*v := by
    simp [travel,gain,damping]
    field_simp [sub_ne_zero.mpr qlt.ne']
  refine ⟨horiz,fun hv => ?_⟩
  rw [horiz, add_lt_add_iff_left, div_lt_iff₀ (sub_pos.mpr (gains true).2)]
  unfold travel gain
  nlinarith only [mul_pos (mul_pos (div_pos hm hk) hv) (sub_pos.mpr hd.2)]

/-- Loss in twice the kinetic energy splits into the vertical preparation and
isotropic drag on the already-prepared velocity. -/
theorem stop_loss_budget (m k t x y z : ℝ) (hm : 0 < m) (hk : 0 < k) (ht : 0 ≤ t) :
    let q := decay m k t
    let d := damping true
    let before := x^2+y^2+z^2
    let prepared := x^2+(d*y)^2+z^2
    let after := (step false m k t (0,x)).2^2 + (step true m k t (0,y)).2^2 +
      (step false m k t (0,z)).2^2
    m*(before-after) = m*(1-d^2)*y^2 + m*(1-q^2)*prepared ∧
      0 ≤ m*(1-d^2)*y^2 ∧ 0 ≤ m*(1-q^2)*prepared := by
  have hq := decay_bounds hm hk ht
  have hd : 0 ≤ 1 - damping true ^ 2 := by norm_num [damping,stopVertical]
  have drag : 0 ≤ 1-decay m k t^2 := by nlinarith [mul_nonneg (sub_nonneg.mpr hq.2) (by linarith : 0 ≤ 1 + decay m k t)]
  dsimp only
  refine ⟨?_,by positivity,by positivity⟩
  simp only [one_step,gain]
  have : damping false = 1 := rfl
  rw [this]; ring

/-- Dividing an interval in two applies a second vertical preparation. -/
theorem split_stop (b : Bool) (m k s t : ℝ) (v : ℝ) :
    (step b m k t (step b m k s (0,v))).2 = damping b * (step b m k (s+t) (0,v)).2 ∧
    (v ≠ 0 → ((step b m k t (step b m k s (0,v))).2 = (step b m k (s+t) (0,v)).2 ↔ b = false)) := by
  simp only [one_step,gain,decay,mul_add,Real.exp_add]
  constructor
  · ring
  intro hv
  cases b
  · simp [damping]; ring
  · have hd : damping true ≠ 0 ∧ damping true ≠ 1 := by norm_num [damping,stopVertical]
    simp only [Bool.true_eq_false, iff_false]
    intro eq
    have h : Real.exp (-k/m*s)*Real.exp (-k/m*t)*v*damping true*(damping true-1) = 0 := by
      linarith only [eq, show Real.exp (-k/m*t)*damping true*(Real.exp (-k/m*s)*damping true*v) -
        (Real.exp (-k/m*s)*Real.exp (-k/m*t)*damping true*v) =
        Real.exp (-k/m*s)*Real.exp (-k/m*t)*v*damping true*(damping true-1) by ring]
    exact (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (Real.exp_ne_zero _) (Real.exp_ne_zero _)) hv) hd.1)
      (sub_ne_zero.mpr hd.2)) h

end ExactBallpark.Movement.StopDistance
