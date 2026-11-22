-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Impact

@[expose] public noncomputable section
namespace ExactBallpark.Collision.LimitedPair
open Movement
open scoped InnerProductSpace

/-- Real interpretation of the per-body AddImpact multiplier for a nonzero normal kick. -/
def retain (kick cap : ℝ) :=
  let fraction := 25*cap*cap / kick^2
  if fraction < 1 then fraction else 1

theorem limit_normal (n : Vec3) (unit : ‖n‖ = 1) (kick cap : ℝ) :
    limitImpact (kick • n) cap = retain kick cap • (kick • n) := by
  simp only [limitImpact,retain,norm_smul,Real.norm_eq_abs,unit,mul_one,sq_abs]
  split <;> simp

def impulse (m₁ m₂ u v : ℝ) := m₁*m₂*(-2*(u-v))/(m₁+m₂)

/-- u and v are signed velocities on the first body's outward normal. -/
def applyFractions (m₁ m₂ u v α β : ℝ) : ℝ × ℝ :=
  let j := impulse m₁ m₂ u v
  (u + α*(j/m₁), v - β*(j/m₂))

def limited (m₁ m₂ u v cap₁ cap₂ : ℝ) :=
  let j := impulse m₁ m₂ u v
  applyFractions m₁ m₂ u v (retain (j/m₁) cap₁) (retain (-(j/m₂)) cap₂)

theorem from_pair_velocity {m₁ m₂ : ℝ} (_h₁ : 0 < m₁) (_h₂ : 0 < m₂)
    (n : Vec3) (unit : ‖n‖ = 1) {u v : ℝ} (closing : u < v) (c₁ c₂ : ℝ) :
    let raw := pairVelocity m₁ m₂ (u • n) (v • n) n
    let out := limited m₁ m₂ u v c₁ c₂
    (u • n + limitImpact (raw.1-u • n) c₁, v • n + limitImpact (raw.2-v • n) c₂) =
      (out.1 • n,out.2 • n) := by
  have rel : ⟪v • n-u • n,-n⟫_ℝ = u-v := by
    rw [inner_neg_right,inner_sub_left,real_inner_smul_left,real_inner_smul_left,real_inner_self_eq_norm_sq,unit]; ring
  have hit : ¬ (-2*(u-v) < 0) := by linarith
  simp only [pairVelocity,rel,hit,ite_false,add_sub_cancel_left,limited,applyFractions,impulse,
    smul_neg,← neg_smul]
  rw [limit_normal _ unit, limit_normal _ unit]
  simp only [smul_smul, ← add_smul]
  congr 2
  ring

/-- Both masses see the same raw impulse magnitude. Momentum after their
independent caps is conserved exactly when the two retained fractions agree. -/
theorem momentum_iff {m₁ m₂ u v : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂) (closing : u < v) (α β : ℝ) :
    let out := applyFractions m₁ m₂ u v α β
    m₁*out.1 + m₂*out.2 = m₁*u + m₂*v ↔ α = β := by
  have jpos : 0 < impulse m₁ m₂ u v := by
    unfold impulse; exact div_pos (mul_pos (mul_pos h₁ h₂) (by linarith)) (add_pos h₁ h₂)
  have diff : m₁*(applyFractions m₁ m₂ u v α β).1 + m₂*(applyFractions m₁ m₂ u v α β).2 -
      (m₁*u + m₂*v) = (α-β)*impulse m₁ m₂ u v := by dsimp [applyFractions]; field_simp; ring
  dsimp only
  rw [← sub_eq_zero, diff,mul_eq_zero,or_iff_left jpos.ne',sub_eq_zero]

theorem separates_iff {m₁ m₂ u v : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂) (closing : u < v) (α β : ℝ) :
    let out := applyFractions m₁ m₂ u v α β
    out.2 ≤ out.1 ↔ m₁+m₂ ≤ 2*(m₂*α+m₁*β) := by
  have gap : (applyFractions m₁ m₂ u v α β).1 - (applyFractions m₁ m₂ u v α β).2 =
      ((v-u)/(m₁+m₂))*(2*(m₂*α+m₁*β)-(m₁+m₂)) := by
    dsimp [applyFractions,impulse]; field_simp; ring
  dsimp only
  rw [← sub_nonneg, gap, mul_nonneg_iff_of_pos_left (div_pos (sub_pos.mpr closing) (add_pos h₁ h₂)),sub_nonneg]

def energy (m₁ m₂ u v : ℝ) := (m₁*u^2+m₂*v^2)/2

theorem common_fraction_loss {m₁ m₂ : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂)
    (u v α : ℝ) (lo : 0 ≤ α) (hi : α ≤ 1) :
    let out := applyFractions m₁ m₂ u v α α
    let loss := 2*(m₁*m₂/(m₁+m₂))*α*(1-α)*(v-u)^2
    energy m₁ m₂ u v - energy m₁ m₂ out.1 out.2 = loss ∧ 0 ≤ loss := by
  refine ⟨?_,by exact mul_nonneg (mul_nonneg (mul_nonneg (by positivity) lo) (sub_nonneg.mpr hi)) (sq_nonneg _)⟩
  dsimp [applyFractions,impulse,energy]; field_simp; ring

def witness : ℚ × ℚ × ℚ × ℚ := (-(195/2),-100,-(15/2),5625/8)

/-- An approaching pair with the same masses and different max speeds. Only
the first kick is limited. Energy is measured in the given inertial frame. -/
theorem asymmetric_witness :
    let out := limited 1 1 (-100) (-90) 1 2
    out = ((witness.1 : ℝ),(witness.2.1 : ℝ)) ∧ out.1+out.2-(-190) = witness.2.2.1 ∧
      energy 1 1 out.1 out.2 - energy 1 1 (-100) (-90) = witness.2.2.2 := by
  norm_num [limited,applyFractions,retain,impulse,energy,witness]

end ExactBallpark.Collision.LimitedPair
