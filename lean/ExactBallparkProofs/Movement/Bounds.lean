-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Integrator
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Tactic.Linarith

@[expose] public section

namespace ExactBallpark.Movement

noncomputable section

abbrev Vec3 := EuclideanSpace ℝ (Fin 3)

theorem decay_bounds {m k t : ℝ} (hm : 0 < m) (hk : 0 < k) (ht : 0 ≤ t) :
    0 < decay m k t ∧ decay m k t ≤ 1 := by
  exact ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr
    (mul_nonpos_of_nonpos_of_nonneg (div_nonpos_of_nonpos_of_nonneg (by linarith) hm.le) ht)⟩

/-- Apply exactly the scalar closed form to each of the three source components. -/
def velocity3 (v a : Vec3) (m k t : ℝ) : Vec3 :=
  WithLp.toLp 2 (fun i => velocity (v i) (a i) m k t)

def terminal3 (a : Vec3) (m k : ℝ) : Vec3 := (m / k) • a

private theorem velocity3_sub (v a : Vec3) (m k t : ℝ) :
    velocity3 v a m k t - terminal3 a m k =
    decay m k t • (v - terminal3 a m k) := by
  ext i
  change terminal m k (a i) + _ * _ - (m / k) * a i = _ * (v i - (m / k) * a i)
  dsimp [terminal]
  ring

theorem velocity_contraction (v a : Vec3) (m k t : ℝ) :
    ‖velocity3 v a m k t - terminal3 a m k‖ = decay m k t * ‖v - terminal3 a m k‖ := by
  rw [velocity3_sub, norm_smul, Real.norm_eq_abs, decay, abs_of_pos (Real.exp_pos _)]

theorem speed_envelope {v a : Vec3} {m k t S : ℝ} (hm : 0 < m) (hk : 0 < k) (ht : 0 ≤ t)
    (hv : ‖v‖ ≤ S) (hu : ‖terminal3 a m k‖ ≤ S) : ‖velocity3 v a m k t‖ ≤ S := by
  let q := decay m k t
  have hq : 0 < q ∧ q ≤ 1 := decay_bounds hm hk ht
  have hvec : velocity3 v a m k t = q • v + (1 - q) • terminal3 a m k := by
    ext i
    change terminal m k (a i) + q * (v i - terminal m k (a i)) =
      q * v i + (1 - q) * (m / k * a i)
    dsimp [terminal]
    ring
  rw [hvec]
  calc
    _ ≤ ‖q • v‖ + ‖(1 - q) • terminal3 a m k‖ := norm_add_le ..
    _ = q * ‖v‖ + (1 - q) * ‖terminal3 a m k‖ := by
      simp only [norm_smul, Real.norm_eq_abs, abs_of_pos hq.1, abs_of_nonneg (sub_nonneg.mpr hq.2)]
    _ ≤ q * S + (1 - q) * S := add_le_add (mul_le_mul_of_nonneg_left hv hq.1.le)
      (mul_le_mul_of_nonneg_left hu (sub_nonneg.mpr hq.2))
    _ = S := by ring

open Filter Topology in
theorem velocity_tends_to_terminal {m k : ℝ} (hm : 0 < m) (hk : 0 < k) (v a : ℝ) :
    Tendsto (velocity v a m k) atTop (𝓝 (terminal m k a)) := by
  have hq : Tendsto (decay m k) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_id.const_mul_atTop_of_neg (div_neg_of_neg_of_pos (neg_neg_of_pos hk) hm))
  unfold velocity
  simpa using (tendsto_const_nhds (x := terminal m k a)).add (hq.mul_const (v - terminal m k a))

end
end ExactBallpark.Movement
