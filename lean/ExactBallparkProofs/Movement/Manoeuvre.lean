-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Energy

@[expose] public noncomputable section
namespace ExactBallpark.Movement
open scoped InnerProductSpace

/-- Acceleration is fixed inside an interval and may change in any direction at its end. -/
structure Burn where
  acceleration : Vec3
  duration : NNReal

structure WorkAccount where
  velocity : Vec3
  thrustWork : ℝ
  dragLoss : ℝ

def applyBurn (m k : ℝ) (s : WorkAccount) (b : Burn) : WorkAccount :=
  let v := velocity3 s.velocity b.acceleration m k
  ⟨v b.duration,
   s.thrustWork + ∫ t in (0:ℝ)..b.duration, m * ⟪v t, b.acceleration⟫_ℝ,
   s.dragLoss + ∫ t in (0:ℝ)..b.duration, k * ‖v t‖^2⟩

/-- Work is accumulated independently from the endpoint energies. Velocity is
passed to the next burn, including at a change in thrust direction. -/
theorem manoeuvre_work_energy {m k : ℝ} (hm : 0 < m) (hk : 0 < k) (v : Vec3) (burns : List Burn) :
    let result := burns.foldl (applyBurn m k) ⟨v, 0, 0⟩
    kineticEnergy m result.velocity - kineticEnergy m v = result.thrustWork - result.dragLoss ∧
      0 ≤ result.dragLoss := by
  suffices ∀ s : WorkAccount, let r := burns.foldl (applyBurn m k) s
      kineticEnergy m r.velocity - kineticEnergy m s.velocity =
        (r.thrustWork-s.thrustWork) - (r.dragLoss-s.dragLoss) ∧ s.dragLoss ≤ r.dragLoss by
    simpa using this ⟨v,0,0⟩
  induction burns with
  | nil => intro s; simp
  | cons b bs ih =>
    intro s
    let vel := velocity3 s.velocity b.acceleration m k
    have continuous : Continuous vel := continuous_iff_continuousAt.mpr
      (fun t => (velocity3_hasDerivAt s.velocity b.acceleration hm hk t).continuousAt)
    have work := drag_work_energy s.velocity b.acceleration hm hk 0 b.duration
    have start : vel 0 = s.velocity := by
      ext i
      simp [vel, velocity3, velocity, decay]
    change kineticEnergy m (vel b.duration) - kineticEnergy m (vel 0) = _ at work
    have thrustInt : IntervalIntegrable (fun t => m * ⟪vel t, b.acceleration⟫_ℝ)
        MeasureTheory.volume 0 b.duration :=
      (continuous_const.mul (continuous.inner continuous_const)).intervalIntegrable ..
    have dragInt : IntervalIntegrable (fun t => k * ‖vel t‖^2)
        MeasureTheory.volume 0 b.duration :=
      (continuous_const.mul (continuous.norm.pow 2)).intervalIntegrable ..
    rw [start, intervalIntegral.integral_sub thrustInt dragInt] at work
    have loss : 0 ≤ ∫ t in (0:ℝ)..b.duration, k * ‖vel t‖^2 :=
      intervalIntegral.integral_nonneg b.duration.2 (fun _ _ => mul_nonneg hk.le (sq_nonneg _))
    obtain ⟨tailWork,tailLoss⟩ := ih (applyBurn m k s b)
    simp only [List.foldl_cons]
    change kineticEnergy m _ - kineticEnergy m (vel b.duration) =
      (_ - (s.thrustWork + _)) - (_ - (s.dragLoss + _)) at tailWork
    change s.dragLoss + _ ≤ _ at tailLoss
    constructor <;> linarith

end ExactBallpark.Movement
