-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.ContactExamples
public import ExactBallparkProofs.Collision.Impact
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic

@[expose] public section
namespace ExactBallpark.Collision.ReboundFlight
open Movement DragChord PositionEnclosure Set
open scoped InnerProductSpace

/-- The right body's agility is the reciprocal of its base mass. Both effective
masses are one. The left GOTO speed is maxVel times its constructor speed fraction. -/
inductive Scenario where
  | equalMass | unequalMass | limited
  deriving Repr, DecidableEq

def Scenario.rightMass : Scenario → Nat
  | .unequalMass => 2
  | _ => 1

def Scenario.leftFraction : Scenario → Nat
  | .limited => 4
  | _ => 1

/-- Signed changes in x velocity after AddImpact in the real model. -/
def Scenario.leftKick : Scenario → ℚ
  | .equalMass => -4
  | .unequalMass => -16/3
  | .limited => -25/16

def Scenario.rightKick : Scenario → ℚ
  | .unequalMass => 8/3
  | _ => 4

noncomputable section

def axis (x : ℝ) : Vec3 := EuclideanSpace.single 0 x

private theorem axis_add (x y : ℝ) : axis x + axis y = axis (x+y) := (PiLp.single_add ..).symm
private theorem axis_sub (x y : ℝ) : axis x - axis y = axis (x-y) := by
  ext i; fin_cases i <;> simp [axis]
private theorem axis_neg (x : ℝ) : -axis x = axis (-x) := by
  ext i; fin_cases i <;> simp [axis]
private theorem axis_smul (a x : ℝ) : a • axis x = axis (a*x) := by
  ext i; fin_cases i <;> simp [axis]
private theorem axis_norm (x : ℝ) : ‖axis x‖ = |x| := PiLp.norm_single ..
private theorem axis_inner (x y : ℝ) : ⟪axis x, axis y⟫_ℝ = x*y := by
  simp [axis, EuclideanSpace.inner_single_left]
private theorem axis_dist (x y : ℝ) : dist (axis x) (axis y) = |x-y| := by
  rw [dist_eq_norm, axis_sub, axis_norm]

def flight (p v a : ℝ) : Flight := ⟨axis p, axis v, axis a, 1, 1⟩
def left := flight 0 2 2
def right := flight 3 (-2) (-2)

/-- GOTO sustains the initial speeds. The different base mass in the unequal
case is exactly offset by its agility, and the ordinary branch holds throughout. -/
theorem ordinary_inputs (c : Scenario) {t : ℝ} (ht : t ∈ Icc 0 1) :
    (c.rightMass:ℝ) * (1/c.rightMass) = 1 ∧
    (2/(c.leftFraction:ℝ)) * c.leftFraction = 2 ∧ ¬ (1:ℝ) < threshold * 1 * t := by
  refine ⟨by cases c <;> norm_num [Scenario.rightMass],
    by cases c <;> norm_num [Scenario.leftFraction], ?_⟩
  have h := mul_le_mul_of_nonneg_left ht.2 (by norm_num [threshold] : 0 ≤ threshold)
  simp only [mul_one] at h ⊢
  exact not_lt.mpr (h.trans (by norm_num [threshold]))

private theorem axial_pos (p v a t : ℝ) : (flight p v a).pos t =
    axis (p + a*t + (1-Real.exp (-t))*(v-a)) := by
  ext i
  fin_cases i <;> simp [flight, Flight.pos, position, terminal, decay, axis, PiLp.single_apply]

private theorem axial_vel (p v a t : ℝ) : (flight p v a).vel t =
    axis (a + Real.exp (-t)*(v-a)) := by
  ext i
  fin_cases i <;> simp [flight, Flight.vel, velocity3, velocity, terminal, decay, axis, PiLp.single_apply]

theorem incoming (t : ℝ) :
    left.pos t = axis (2*t) ∧ right.pos t = axis (3-2*t) ∧
    left.vel t = axis 2 ∧ right.vel t = axis (-2) := by
  simp [left, right, axial_pos, axial_vel, sub_eq_add_neg]

/-- Connect the dynamics on either side of the impact to the existing fixture. -/
theorem incoming_distance (t : ℝ) : ContactExamples.crossing.distance t = dist (left.pos t) (right.pos t) := by
  have straight_eq (x v : ℚ) : (ContactExamples.straight x v).flight = flight x v v := by
    unfold ContactExamples.straight Body.flight flight
    congr 1 <;> first | (ext i; fin_cases i <;> simp [CurveClearance.asVec, realPoint, axis]) | norm_num
  change dist ((ContactExamples.straight 0 2).flight.pos t) ((ContactExamples.straight 3 (-2)).flight.pos t) = _
  rw [straight_eq, straight_eq]
  norm_num [left, right]

theorem exact_first_contact : FirstContact.First ContactExamples.crossing.distance 1 1 (1/2) := by
  have gap (t : ℝ) : ContactExamples.crossing.distance t = |2*t - (3-2*t)| := by
    rw [incoming_distance, (incoming t).1, (incoming t).2.1, axis_dist]
  refine ⟨by constructor <;> norm_num, by rw [gap]; norm_num, ?_⟩
  intro s hs
  rw [gap, abs_of_nonpos (by linarith [hs.2] : 2*s-(3-2*s) ≤ 0)]
  linarith [hs.2]

/-- The contact normal points from the right centre to the left centre. The
elastic exchange consumes base masses. The limiter acts on each change separately. -/
def Scenario.raw (c : Scenario) : Vec3 × Vec3 :=
  let u := left.vel (1/2)
  let v := right.vel (1/2)
  let d := left.pos (1/2) - right.pos (1/2)
  let normal := ‖d‖⁻¹ • d
  let out := pairVelocity 1 c.rightMass u v normal
  (out.1-u, out.2-v)

def Scenario.kicks (c : Scenario) : Vec3 × Vec3 :=
  (limitImpact c.raw.1 (2 / c.leftFraction), limitImpact c.raw.2 2)

theorem raw_impulse (c : Scenario) : c.raw =
    (axis (-8*(c.rightMass:ℝ)/(1+c.rightMass)), axis (8/(1+c.rightMass))) := by
  obtain ⟨p,q,u,v⟩ := incoming (1/2)
  cases c <;> norm_num [Scenario.raw, p, q, u, v, pairVelocity, Scenario.rightMass,
    axis_add, axis_sub, axis_neg, axis_smul, axis_norm, axis_inner]

theorem applied_impulse (c : Scenario) : c.kicks = (axis c.leftKick, axis c.rightKick) := by
  rw [Scenario.kicks, raw_impulse]
  cases c <;> norm_num [limitImpact, Scenario.rightMass, Scenario.leftFraction, Scenario.leftKick,
    Scenario.rightKick, axis_norm, axis_smul]

/-- Mass-weighted changes cancel in both unrestricted exchanges. The one-sided
limiter gives the pair a net change of +39/16 kg m/s along x at the impact. -/
theorem momentum_change (c : Scenario) :
    c.kicks.1 + (c.rightMass:ℝ) • c.kicks.2 =
      axis (if c = .limited then 39/16 else 0) := by
  rw [applied_impulse]
  cases c <;> norm_num [Scenario.leftKick, Scenario.rightKick, Scenario.rightMass, axis_smul, axis_add]

/-- Positive-time impulses keep the contact position and ongoing thrust. -/
def restart (f : Flight) (t : ℝ) (kick : Vec3) : Flight :=
  { f with start := f.pos t, initialVelocity := f.vel t + kick }

theorem restart_state (f : Flight) (t : ℝ) (kick : Vec3) :
    (restart f t kick).pos 0 = f.pos t ∧ (restart f t kick).vel 0 = f.vel t + kick := by
  simp [restart]

def Scenario.after (c : Scenario) : Flight × Flight :=
  (restart left (1/2) c.kicks.1, restart right (1/2) c.kicks.2)

private theorem after_flights (c : Scenario) : c.after =
    (flight 1 (2+c.leftKick) 2, flight 2 (-2+c.rightKick) (-2)) := by
  obtain ⟨p,q,u,v⟩ := incoming (1/2)
  simp only [Scenario.after, restart, applied_impulse, p, q, u, v, axis_add]
  norm_num [left, right, flight]

/-- All positions and velocities for elapsed time s after impact, including the tick endpoint. -/
theorem remainder (c : Scenario) (s : ℝ) :
    c.after.1.pos s = axis (1+2*s+(1-Real.exp (-s))*c.leftKick) ∧
    c.after.2.pos s = axis (2-2*s+(1-Real.exp (-s))*c.rightKick) ∧
    c.after.1.vel s = axis (2+Real.exp (-s)*c.leftKick) ∧
    c.after.2.vel s = axis (-2+Real.exp (-s)*c.rightKick) := by
  rw [after_flights]
  simp only [axial_pos, axial_vel]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> congr 1 <;> ring

def qLo : ℚ := 6065306597126334 / 10^16
def qHi : ℚ := 6065306597126335 / 10^16

/-- Fourth power of the checked enclosure for exp(−1/8). This bound uses no host exp word. -/
theorem half_decay : (qLo:ℝ) ≤ Real.exp (-(1/2:ℝ)) ∧ Real.exp (-(1/2:ℝ)) ≤ (qHi:ℝ) := by
  let b := expBounds (1/8)
  have hb := decay_enclosed (1/8) (by norm_num)
  have small : b.Contains (Real.exp (-(1/8:ℝ))) := by simpa [b] using hb
  have h0 : (0:ℝ) ≤ b.lo := by exact_mod_cast (show 0 ≤ b.lo by decide +kernel)
  have lo : (qLo:ℝ) ≤ (b.lo:ℝ)^4 := by exact_mod_cast (show qLo ≤ b.lo^4 by decide +kernel)
  have hi : (b.hi:ℝ)^4 ≤ (qHi:ℝ) := by exact_mod_cast (show b.hi^4 ≤ qHi by decide +kernel)
  rw [show -(1/2:ℝ) = (4:ℕ)*(-(1/8:ℝ)) by norm_num, Real.exp_nat_mul]
  exact ⟨lo.trans (pow_le_pow_left₀ h0 small.1 _), (pow_le_pow_left₀ (Real.exp_pos _).le small.2 _).trans hi⟩

/-- A strict endpoint gap suffices along this line when the relative kick is
nonnegative. Convexity of exp bounds the whole interval, even when relative
velocity changes sign before the end. -/
theorem positive_gap (rate : ℝ) {H s gain speed : ℝ} (hH : 0 < H) (hs : s ∈ Ioc 0 H)
    (gain_nonneg : 0 ≤ gain) (endpoint : speed*H < gain*(1-Real.exp (-(rate*H)))) :
    speed*s < gain*(1-Real.exp (-(rate*s))) := by
  let u := s/H
  have hu : 0 < u := div_pos hs.1 hH
  have hu' : u ≤ 1 := (div_le_one hH).mpr hs.2
  have cv := convexOn_exp.2 (mem_univ (-(rate*H))) (mem_univ 0) hu.le (sub_nonneg.mpr hu')
    (show u + (1-u) = 1 by ring)
  simp only [smul_eq_mul, mul_zero, add_zero, Real.exp_zero, mul_one] at cv
  have hs' : u*H = s := div_mul_cancel₀ s hH.ne'
  rw [mul_neg, show u*(rate*H) = rate*s by rw [← hs']; ring] at cv
  have part := mul_le_mul_of_nonneg_left
    (show u*(1-Real.exp (-(rate*H))) ≤ 1-Real.exp (-(rate*s)) by linarith) gain_nonneg
  have strict := mul_lt_mul_of_pos_left endpoint hu
  calc
    speed*s = u*(speed*H) := by rw [← hs']; ring
    _ < u*(gain*(1-Real.exp (-(rate*H)))) := strict
    _ ≤ gain*(1-Real.exp (-(rate*s))) := by linarith only [part]

/-- Two collinear ordinary flights start in contact with the right centre ahead.
For shared positive drag parameters, a nonnegative relative velocity excess
over the terminal relative velocity and a positive signed endpoint gap certify
the entire open remainder. A final distance alone would lose the ordering. -/
theorem shared_drag_separation {p q u v a b m k r H s : ℝ} (hm : 0 < m) (hk : 0 < k)
    (hH : 0 < H) (hs : s ∈ Ioc 0 H) (contact : q-p = r)
    (gain : 0 ≤ (v-terminal m k b)-(u-terminal m k a))
    (endpoint : r < position q v b m k H - position p u a m k H) :
    r < position q v b m k s - position p u a m k s := by
  let speed := terminal m k a - terminal m k b
  let gain' := m/k*((v-terminal m k b)-(u-terminal m k a))
  have gap (t : ℝ) : position q v b m k t - position p u a m k t =
      r-speed*t + gain'*(1-Real.exp (-(k/m*t))) := by
    rw [← contact]; unfold position decay speed gain'; rw [neg_div, neg_mul]; ring
  rw [gap] at endpoint ⊢
  have h := positive_gap (k/m) hH hs (show 0 ≤ gain' from mul_nonneg (div_pos hm hk).le gain)
    (show speed*H < gain'*(1-Real.exp (-(k/m*H))) by linarith)
  linarith

theorem no_recontact (c : Scenario) {s : ℝ} (hs : s ∈ Ioc 0 (1/2)) :
    1 < dist (c.after.1.pos s) (c.after.2.pos s) := by
  obtain ⟨p,q,_,_⟩ := remainder c s
  rw [p, q, axis_dist]
  have cap : Real.exp (-(1/2:ℝ)) ≤ 5/8 := half_decay.2.trans (by norm_num [qHi])
  have gain : 0 ≤ (c.rightKick:ℝ)-c.leftKick := by
    cases c <;> norm_num [Scenario.rightKick, Scenario.leftKick]
  have ends : 4*(1/2:ℝ) < ((c.rightKick:ℝ)-c.leftKick)*(1-Real.exp (-(1/2:ℝ))) := by
    cases c <;> norm_num [Scenario.rightKick, Scenario.leftKick] <;> linarith
  have clear := shared_drag_separation (p := 1) (q := 2) (u := 2+c.leftKick) (v := -2+c.rightKick)
    (a := 2) (b := -2) (m := 1) (k := 1) (r := 1) (by norm_num) (by norm_num)
    (by norm_num) hs (by norm_num) (by simpa [terminal] using gain) (by
      norm_num [position, terminal, decay] at ends ⊢
      linarith only [ends])
  simp only [position, terminal, decay, div_one, neg_mul, one_mul, add_sub_cancel_left] at clear
  exact lt_of_lt_of_le (by linarith only [clear] : 1 < -((1+2*s+(1-Real.exp (-s))*(c.leftKick:ℝ))-
    (2-2*s+(1-Real.exp (-s))*(c.rightKick:ℝ)))) (neg_le_abs _)

/-- The joined trajectories touch exactly once in the closed tick, including
both endpoints and all intermediate times. -/
def Scenario.distance (c : Scenario) (t : ℝ) : ℝ :=
  if t ≤ 1/2 then dist (left.pos t) (right.pos t)
  else dist (c.after.1.pos (t-1/2)) (c.after.2.pos (t-1/2))

theorem only_contact (c : Scenario) {t : ℝ} (ht : t ∈ Icc 0 1) :
    c.distance t ≤ 1 ↔ t = 1/2 := by
  rcases lt_trichotomy t (1/2) with h | rfl | h
  · rw [Scenario.distance, ite_eq_left h.le, ← incoming_distance]
    exact iff_of_false (exact_first_contact.2.2 t ⟨ht.1,h⟩).not_ge h.ne
  · simpa [Scenario.distance, ← incoming_distance] using exact_first_contact.2.1
  · rw [Scenario.distance, ite_eq_right h.not_ge]
    exact iff_of_false (no_recontact c ⟨by linarith, by linarith [ht.2]⟩).not_ge h.ne'

theorem limited_closing_at_end : (Scenario.limited.after.2.vel (1/2)) 0 <
    (Scenario.limited.after.1.vel (1/2)) 0 := by
  obtain ⟨_,_,a,b⟩ := remainder Scenario.limited (1/2)
  rw [a,b]
  simp only [axis, PiLp.single_eq_same]
  norm_num [Scenario.rightKick, Scenario.leftKick]
  linarith [half_decay.2, show (qHi:ℝ) < 5/8 by norm_num [qHi]]

end
end ExactBallpark.Collision.ReboundFlight
