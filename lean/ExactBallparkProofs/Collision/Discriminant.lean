-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.FiniteSweep

/-! Finite-arithmetic certificate for the discriminant branch of COL-705.
The caller supplies absolute error budgets for the three quadratic coefficients.
All certificate arithmetic after the binary64 evaluation uses exact rationals. -/
@[expose] public section
namespace ExactBallpark.Collision

/-- Exact dyadic value of a finite word, with both signed zeros mapped to zero. -/
def finiteValue (x : Float) : Option ℚ :=
  match x.toModel.unpack with
  | .zero _ => some 0
  | .finite sign m e _ =>
    let v := (m : ℚ) * (2 : ℚ)^e
    some (if sign = .negative then -v else v)
  | _ => none

theorem finite_negative_iff (x : Float) (q : ℚ) (h : finiteValue x = some q) :
    x < Float.ofBits 0 ↔ q < 0 := by
  change x.toModel.unpack.lt (.zero .positive) = true ↔ _
  unfold finiteValue at h
  generalize hv : x.toModel.unpack = value at h ⊢
  cases value with
  | zero sign => cases sign <;> simp_all [Float.Model.UnpackedFloat.lt, Float.Model.UnpackedFloat.compare]
  | infinity sign => simp at h
  | notANumber => simp at h
  | finite sign m e hm =>
    have pos : 0 < (m : ℚ) * (2 : ℚ)^e := mul_pos (by exact_mod_cast hm) (zpow_pos (by decide) _)
    cases sign <;> simp [Float.Model.UnpackedFloat.lt, Float.Model.UnpackedFloat.compare] at h ⊢
    · rw [← h]; exact neg_lt_zero.mpr pos
    · rw [← h]; exact pos.le

/-- Product uncertainty around x₀*y₀ when the two inputs lie within ex and ey. -/
def productRadius (x₀ y₀ ex ey : ℚ) := ex*(|y₀|+ey) + |x₀| * ey

private theorem product_error {x y x₀ y₀ ex ey : ℚ} (hx : |x-x₀| ≤ ex) (hy : |y-y₀| ≤ ey) :
    |x*y - x₀*y₀| ≤ productRadius x₀ y₀ ex ey := by
  have yBound : |y| ≤ |y₀| + ey := calc
    |y| = |y₀ + (y-y₀)| := by congr 1; ring
    _ ≤ |y₀| + |y-y₀| := abs_add_le ..
    _ ≤ _ := by linarith
  rw [show x*y - x₀*y₀ = (x-x₀)*y + x₀*(y-y₀) by ring]
  apply (abs_add_le ..).trans
  rw [abs_mul, abs_mul]
  exact add_le_add (mul_le_mul hx yBound (abs_nonneg _) ((abs_nonneg _).trans hx))
    (mul_le_mul_of_nonneg_left hy (abs_nonneg _))

/-- d is the decoded result of the ordered binary64 discriminant arithmetic.
Its difference from b₀²−4a₀c₀ includes rounding in both multiplications and subtraction. -/
def discriminantRadius (a₀ b₀ c₀ d ea eb ec : ℚ) :=
  productRadius b₀ b₀ eb eb + 4 * productRadius a₀ c₀ ea ec + |b₀*b₀ - 4*a₀*c₀ - d|

theorem discriminant_error {a b c a₀ b₀ c₀ d ea eb ec : ℚ}
    (ha : |a-a₀| ≤ ea) (hb : |b-b₀| ≤ eb) (hc : |c-c₀| ≤ ec) :
    |b*b - 4*a*c - d| ≤ discriminantRadius a₀ b₀ c₀ d ea eb ec := by
  let e₁ := b*b - b₀*b₀
  let e₂ := a*c - a₀*c₀
  let e₃ := b₀*b₀ - 4*a₀*c₀ - d
  have eqn : b*b - 4*a*c - d = (e₁ + (-4)*e₂) + e₃ := by dsimp [e₁,e₂,e₃]; ring
  rw [eqn]
  apply (abs_add_le ..).trans
  apply add_le_add_left
  apply (abs_add_le ..).trans
  rw [abs_mul, show |(-4 : ℚ)| = 4 by decide]
  exact add_le_add (product_error hb hb) (mul_le_mul_of_nonneg_left (product_error ha hc) (by decide))

inductive DiscSign where | negative | positive deriving DecidableEq, Repr

/-- A strict margin excludes tangency. Unknown includes a nonfinite intermediate
result, a negative input budget, or a coefficient box too wide for this bound. -/
def certifyDiscriminant (a b c : Float) (ea eb ec : ℚ) : Option DiscSign := do
  if ea < 0 ∨ eb < 0 ∨ ec < 0 then none else
  let a₀ ← finiteValue a
  let b₀ ← finiteValue b
  let c₀ ← finiteValue c
  let d ← finiteValue (discriminant64 a b c)
  let margin := discriminantRadius a₀ b₀ c₀ d ea eb ec
  if d < -margin then some .negative else if margin < d then some .positive else none

/-- The reported sign holds both for the engine's comparison with zero and for
every exact quadratic inside the supplied coefficient box. -/
theorem certified_sign {af bf cf : Float} {a₀ b₀ c₀ a b c ea eb ec : ℚ} {sign : DiscSign}
    (da : finiteValue af = some a₀) (db : finiteValue bf = some b₀) (dc : finiteValue cf = some c₀)
    (ha : |a-a₀| ≤ ea) (hb : |b-b₀| ≤ eb) (hc : |c-c₀| ≤ ec)
    (cert : certifyDiscriminant af bf cf ea eb ec = some sign) :
    match sign with
    | .negative => discriminant64 af bf cf < Float.ofBits 0 ∧ b*b - 4*a*c < 0
    | .positive => ¬ discriminant64 af bf cf < Float.ofBits 0 ∧ 0 < b*b - 4*a*c := by
  have allowed : ¬ (ea < 0 ∨ eb < 0 ∨ ec < 0) := by
    have := (abs_nonneg _).trans ha
    have := (abs_nonneg _).trans hb
    have := (abs_nonneg _).trans hc
    grind
  cases dd : finiteValue (discriminant64 af bf cf) with
  | none => simp [certifyDiscriminant, allowed, da, db, dc, dd] at cert
  | some d =>
    have bounds := abs_le.mp (discriminant_error (d := d) ha hb hc)
    have nonneg := (abs_nonneg _).trans (discriminant_error (d := d) ha hb hc)
    have cmp := finite_negative_iff _ _ dd
    simp [certifyDiscriminant, allowed, da, db, dc, dd] at cert
    split at cert
    next h => cases cert; exact ⟨cmp.mpr (by linarith), by linarith⟩
    next =>
      split at cert
      next h => cases cert; exact ⟨fun hn => by linarith [cmp.mp hn], by linarith⟩
      next => contradiction

/-- For a negative certificate, the solver returns its miss sentinel and the
associated exact gap polynomial stays positive throughout the real sweep. -/
theorem certified_miss (p₀ p₁ q₀ q₁ : FloatPoint) (r af bf cf : Float)
    (formed : sweepQuadratic64 p₀ p₁ q₀ q₁ r = some (af, bf, cf))
    {a₀ b₀ c₀ a b c ea eb ec : ℚ}
    (da : finiteValue af = some a₀) (db : finiteValue bf = some b₀) (dc : finiteValue cf = some c₀)
    (ha : |a-a₀| ≤ ea) (hb : |b-b₀| ≤ eb) (hc : |c-c₀| ≤ ec) (moving : 0 < a)
    (cert : certifyDiscriminant af bf cf ea eb ec = some .negative) :
    (contact64 p₀ p₁ q₀ q₁ r).toBits = 0xbff0000000000000 ∧
    ∀ t : ℝ, 0 < (a : ℝ)*t^2 + b*t + c := by
  obtain ⟨guard, neg⟩ := certified_sign da db dc ha hb hc cert
  constructor
  · have guard' : discriminant64 af bf cf < 0 := guard
    simp [contact64, formed, guard']
    rfl
  · intro t
    have ha' : 0 < (a : ℝ) := by exact_mod_cast moving
    have hd : (b : ℝ)*b - 4*a*c < 0 := by exact_mod_cast neg
    have sq : 4*(a : ℝ)*(a*t^2 + b*t + c) =
        (2*a*t + b)^2 - (b*b - 4*a*c) := by ring
    have hp : 0 < 4*(a : ℝ)*(a*t^2 + b*t + c) := by rw [sq]; exact sub_pos_of_lt (hd.trans_le (sq_nonneg _))
    exact pos_of_mul_pos_right hp (by positivity)

end ExactBallpark.Collision
