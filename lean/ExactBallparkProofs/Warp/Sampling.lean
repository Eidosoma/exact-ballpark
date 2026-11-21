-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Profile
public import ExactBallparkProofs.Movement.Bounds

@[expose] public noncomputable section
namespace ExactBallpark.Warp.Sampling
open ExactBallpark.Movement

def reportedSpeed (a t : ℝ) (v : Vec3) :=
  if ‖v‖ > a * Real.exp (a*t) then ‖v‖ else a * Real.exp (a*t)

/-- The continuity guard can keep the incoming speed even as the scalar
position follows its exponential acceleration curve. -/
theorem acceleration_sample (a t : ℝ) (v : Vec3) :
    HasDerivAt (accelDistance a) (a*Real.exp (a*t)) t ∧
    (HasDerivAt (accelDistance a) (reportedSpeed a t v) t ↔ ‖v‖ ≤ a*Real.exp (a*t)) := by
  have d : HasDerivAt (accelDistance a) (a*Real.exp (a*t)) t := by
    change HasDerivAt (fun x => Real.exp (a*x)) _ t
    simpa [mul_comm] using ((hasDerivAt_id t).const_mul a).exp
  refine ⟨d, ?_⟩
  by_cases h : ‖v‖ ≤ a*Real.exp (a*t)
  · simp [reportedSpeed,not_lt.mpr h,h,d]
  · simp only [reportedSpeed,lt_of_not_ge h,ite_true,h,iff_false]
    exact fun q => h (q.unique d).le

/-- For samples in the extra second admitted by the deceleration branch,
the signed remaining coordinate can move past the nominal exit. -/
theorem deceleration_extension {a b s δ : ℝ} (hb : 0 < b) (hs : 0 < s)
    (start : 0 ≤ δ) (stop : δ < 1) (tc : ℝ) :
    let u := decelTime b s + δ
    let rest := (s/a+s*tc+(s/b-1)) - decelDistance a b s tc u
    u < decelTime b s + 1 ∧ rest = Real.exp (-b*δ)-1 ∧
      s*Real.exp (-b*u) = b*Real.exp (-b*δ) ∧
      -1 < rest ∧ rest ≤ 0 ∧ (rest < 0 ↔ 0 < δ) := by
  have arg : -b*(decelTime b s + δ) = -Real.log (s/b) + -b*δ := by
    unfold decelTime; field_simp; ring
  have speed : s * Real.exp (-b*(decelTime b s+δ)) = b*Real.exp (-b*δ) := by
    rw [arg,Real.exp_add,Real.exp_neg,Real.exp_log (div_pos hs hb)]; field_simp
  have coord : (s/a+s*tc+(s/b-1))-decelDistance a b s tc (decelTime b s+δ) =
      Real.exp (-b*δ)-1 := by
    unfold decelDistance
    calc _ = (s*Real.exp (-b*(decelTime b s+δ)))/b-1 := by ring
         _ = _ := by rw [speed]; field_simp
  dsimp only
  rw [coord]
  refine ⟨by linarith,rfl,speed,by linarith [Real.exp_pos (-b*δ)],?_,?_⟩
  · have := Real.exp_le_one_iff.mpr (show -b*δ ≤ 0 by nlinarith); linarith
  · rw [sub_neg,Real.exp_lt_one_iff]; constructor <;> intro h <;> nlinarith

def sight (dst p : Vec3) : Vec3 := ‖dst-p‖⁻¹ • (dst-p)
def place (dst p : Vec3) (rest : ℝ) := dst - rest • sight dst p

/-- The next sample recomputes its direction from the current position.
The sign of the remainder determines the entire next sight vector. -/
theorem placement_ray (dst p : Vec3) (h : dst ≠ p) (rest : ℝ) :
    ‖place dst p rest - dst‖ = |rest| ∧
    sight dst (place dst p rest) =
      if 0 < rest then sight dst p else if rest = 0 then 0 else -sight dst p := by
  have nz := norm_ne_zero_iff.mpr (sub_ne_zero.mpr h)
  have unit : ‖sight dst p‖ = 1 := by
    simp [sight,norm_smul,nz]
  have disp : dst - place dst p rest = rest • sight dst p := by simp [place]
  refine ⟨by rw [norm_sub_rev,disp,norm_smul,unit,Real.norm_eq_abs,mul_one],?_⟩
  change ‖dst - place dst p rest‖⁻¹ • (dst-place dst p rest) = _
  rw [disp,norm_smul,unit,Real.norm_eq_abs,mul_one,smul_smul]
  rcases lt_trichotomy rest 0 with neg | eq | pos
  · simp [neg.not_gt,ne_of_lt neg,abs_of_neg neg]
  · simp [eq]
  · simp [pos,abs_of_pos pos,pos.ne']

end ExactBallpark.Warp.Sampling
