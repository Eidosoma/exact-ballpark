-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.SweepCertificate

/-! Exact bounds on the earlier root for an initially clear, moving sweep with
a nonnegative discriminant. Rational comparisons work even when the root is irrational. -/
@[expose] public section
namespace ExactBallpark.Collision

noncomputable def entryRoot (a b c : ℝ) := (-b - Real.sqrt (b*b - 4*a*c)) * (1/2) / a

/-- At or before the first root, the slope is nonpositive and the gap is nonnegative. -/
abbrev beforeEntry (a b c t : ℚ) := 2*a*t + b ≤ 0 ∧ 0 ≤ a*t^2 + b*t + c

/-- On the far side of the vertex the first root has already passed. -/
abbrev afterEntry (a b c t : ℚ) := 0 ≤ 2*a*t + b ∨ a*t^2 + b*t + c ≤ 0

theorem entry_comparisons (a b c t : ℚ) (ha : 0 < a) (hd : 0 ≤ b*b - 4*a*c) :
    (beforeEntry a b c t ↔ (t : ℝ) ≤ entryRoot a b c) ∧
    (afterEntry a b c t ↔ entryRoot a b c ≤ (t : ℝ)) := by
  have pos : 0 < (a : ℝ) := by exact_mod_cast ha
  let D : ℝ := (b : ℝ)*b - 4*a*c
  have dnonneg : 0 ≤ D := by dsimp [D]; exact_mod_cast hd
  let slope : ℝ := 2*(a : ℝ)*t + b
  let poly : ℝ := (a : ℝ)*t^2 + b*t + c
  have eqn : slope^2 - D = 4*(a : ℝ)*poly := by dsimp [slope,D,poly]; ring
  have sq := Real.sq_sqrt dnonneg
  have rootNonneg := Real.sqrt_nonneg D
  have bef : beforeEntry a b c t ↔ slope ≤ 0 ∧ 0 ≤ poly := by
    dsimp [slope,poly,beforeEntry]; exact_mod_cast Iff.rfl
  have aft : afterEntry a b c t ↔ 0 ≤ slope ∨ poly ≤ 0 := by
    dsimp [slope,poly,afterEntry]; exact_mod_cast Iff.rfl
  have leftBound : (t : ℝ) ≤ entryRoot a b c ↔ slope ≤ -Real.sqrt D := by
    unfold entryRoot
    rw [le_div_iff₀ pos]
    dsimp [slope,D]; constructor <;> intro h <;> linarith
  have rightBound : entryRoot a b c ≤ (t : ℝ) ↔ -Real.sqrt D ≤ slope := by
    unfold entryRoot
    rw [div_le_iff₀ pos]
    dsimp [slope,D]; constructor <;> intro h <;> linarith
  rw [bef,aft,leftBound,rightBound]
  have scale : 0 < 4*(a : ℝ) := by positivity
  constructor
  · constructor
    · rintro ⟨hs,hp⟩
      by_contra! bad
      nlinarith [mul_nonneg scale.le hp, mul_pos (by linarith : 0 < Real.sqrt D - slope)
        (by linarith : 0 < Real.sqrt D + slope)]
    · intro h
      refine ⟨by linarith, nonneg_of_mul_nonneg_right (a := 4*(a : ℝ)) ?_ scale⟩
      nlinarith [mul_nonneg (by linarith : 0 ≤ -slope - Real.sqrt D)
        (by linarith : 0 ≤ -slope + Real.sqrt D)]
  · constructor
    · rintro (hs | hp)
      · linarith
      · by_contra! bad
        nlinarith [mul_nonpos_of_nonneg_of_nonpos scale.le hp,
          mul_pos (by linarith : 0 < -slope - Real.sqrt D) (by linarith : 0 < -slope + Real.sqrt D)]
    · intro h
      by_cases hs : 0 ≤ slope
      · exact Or.inl hs
      · right
        apply nonpos_of_mul_nonpos_right (a := 4*(a : ℝ)) _ scale
        nlinarith [mul_nonneg (by linarith : 0 ≤ Real.sqrt D + slope)
          (by linarith : 0 ≤ Real.sqrt D - slope)]

structure TimingResult where
  inTick : Bool
  roundedTime : ℚ
  deriving DecidableEq

/-- The supplied bracket [lo,hi] concerns the exact entry time, including times
outside the tick. The decoded binary64 return retains the solver's −1 sentinel. -/
def inspectTiming (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) (lo hi : ℚ) : Option TimingResult := do
  let s ← decodeChord p₀ p₁ q₀ q₁ r
  let (a,b,c) := exactCoefficients s
  if 0 < a ∧ 0 < c ∧ 0 ≤ b*b - 4*a*c ∧ beforeEntry a b c lo ∧ afterEntry a b c hi then
    let rounded ← finiteValue (contact64 p₀ p₁ q₀ q₁ r)
    return ⟨decide (beforeEntry a b c 0 ∧ afterEntry a b c 1), rounded⟩
  else none

private theorem decoded_time (s : ExactChord) :
    let (a,b,c) := exactCoefficients s
    contactTime (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀) (realPoint s.q₁) s.radius =
      quadraticTime a b c := by
  dsimp [exactCoefficients, contactTime, dot, realPoint]
  push_cast
  rfl

theorem timing_sound (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) (lo hi : ℚ) (result : TimingResult)
    (h : inspectTiming p₀ p₁ q₀ q₁ r lo hi = some result) :
    finiteValue (contact64 p₀ p₁ q₀ q₁ r) = some result.roundedTime ∧
    ∃ s, decodeChord p₀ p₁ q₀ q₁ r = some s ∧
      let (a,b,c) := exactCoefficients s
      let entry := entryRoot a b c
      (lo : ℝ) ≤ entry ∧ entry ≤ (hi : ℝ) ∧
      (result.inTick = true ↔ 0 ≤ entry ∧ entry ≤ 1) ∧
      contactTime (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀) (realPoint s.q₁) s.radius =
        (if result.inTick then entry else -1) := by
  unfold inspectTiming at h
  obtain ⟨s, decoded, h⟩ := Option.bind_eq_some_iff.mp h
  generalize coeff : exactCoefficients s = abc at h
  obtain ⟨a,b,c⟩ := abc
  dsimp only at h
  split at h
  next good =>
    obtain ⟨ha,hc,hd,hl,hu⟩ := good
    obtain ⟨rounded, observed, h⟩ := Option.bind_eq_some_iff.mp h
    cases h
    refine ⟨observed, s, decoded, ?_⟩
    rw [coeff]
    dsimp only
    have inTick : (beforeEntry a b c 0 ∧ afterEntry a b c 1) ↔
        0 ≤ entryRoot a b c ∧ entryRoot a b c ≤ 1 := by
      simpa using and_congr (entry_comparisons a b c 0 ha hd).1 (entry_comparisons a b c 1 ha hd).2
    refine ⟨(entry_comparisons a b c lo ha hd).1.mp hl, (entry_comparisons a b c hi ha hd).2.mp hu,
      (decide_eq_true_iff).trans inTick, ?_⟩
    have model := decoded_time s
    rw [coeff] at model
    rw [model]
    have pos : 0 < (a : ℝ) := by exact_mod_cast ha
    have clear : 0 < (c : ℝ) := by exact_mod_cast hc
    have disc : 0 ≤ (b : ℝ)*b - 4*a*c := by exact_mod_cast hd
    simp only [quadraticTime, not_le.mpr clear, pos.ne', false_and, ↓reduceIte, not_lt.mpr disc]
    change (if 0 ≤ entryRoot a b c ∧ entryRoot a b c ≤ 1 then _ else _) = _
    simp only [decide_eq_true_eq, inTick, entryRoot]
    split
    next hroot => exact ite_eq_left hroot
    next hroot => exact ite_eq_right hroot
  next => contradiction

/-- Axis-aligned sweep past a stationary unit sphere used by the timing witnesses. -/
def axisTiming (start finish : Float) (lo hi : ℚ) :=
  inspectTiming (horizontalPoint start 0) (horizontalPoint finish 0) (fun _ => 0) (fun _ => 0) 1 lo hi

set_option maxRecDepth 8192 in
theorem tick_boundary_examples :
    axisTiming 2 (-1048574) (1/1048576) (1/1048576) = some ⟨true, 1/1048576⟩ ∧
    axisTiming 2 1 1 1 = some ⟨true, 1⟩ := by decide +kernel

set_option maxRecDepth 8192 in
/-- The exact first root lies strictly past the tick. Rounding the displacement from 4
to the successor of 1 gives −3, so the binary64 solver accepts the endpoint. -/
theorem late_root_rounded_hit :
    let first : ℚ := 3 / (3 - 1 / 2^52)
    1 < first ∧ axisTiming 4 (Float.ofBits 0x3ff0000000000001) first first =
      some ⟨false, 1⟩ := by decide +kernel

/-- At exact tangency the quadratic gap is positive at every time except its vertex. -/
theorem tangent_contact_iff {a b c t : ℝ} (ha : 0 < a) (hd : b*b - 4*a*c = 0) :
    a*t^2 + b*t + c ≤ 0 ↔ t = -b / (2*a) := by
  have identity : 4*a*(a*t^2 + b*t + c) = (2*a*t + b)^2 := by nlinarith [hd]
  rw [eq_div_iff (by positivity : 2*a ≠ 0)]
  constructor
  · intro touch
    have bound := mul_nonpos_of_nonneg_of_nonpos (by positivity : 0 ≤ 4*a) touch
    rw [identity] at bound
    have zero := sq_eq_zero_iff.mp (le_antisymm bound (sq_nonneg _))
    linarith
  · intro vertex
    have zero : 2*a*t + b = 0 := by linarith
    rw [zero, zero_pow (by decide : 2 ≠ 0)] at identity
    exact nonpos_of_mul_nonpos_right identity.le (by positivity)

set_option maxRecDepth 8192 in
/-- A six-metre chord tangent to a sphere of combined radius 2^28. The exact
discriminant is zero while the rounded discriminant is negative. -/
theorem tangent_rounded_miss :
    let r : Float := 268435456
    let p := horizontalPoint (-3) r
    let q := horizontalPoint 3 r
    let z : FloatPoint := fun _ => 0
    (decodeChord p q z z r).map exactCoefficients = some (36, -36, 9) ∧
    inspectTiming p q z z r (1/2) (1/2) = some ⟨true, -1⟩ ∧
    (sweepQuadratic64 p q z z r).bind (fun (a,b,c) => finiteValue (discriminant64 a b c)) =
      some (-1008) := by decide +kernel

set_option maxRecDepth 8192 in
theorem tangent_tick_boundaries :
    let probe := fun x => inspectTiming (horizontalPoint (-3) 1) (horizontalPoint x 1)
      (fun _ => 0) (fun _ => 0) 1
    probe 0 1 1 = some ⟨true, 1⟩ ∧ probe (-1) (3/2) (3/2) = some ⟨false, -1⟩ := by decide +kernel

end ExactBallpark.Collision
