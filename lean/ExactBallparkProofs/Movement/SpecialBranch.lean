-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Integrator

@[expose] public section

namespace ExactBallpark.Movement

noncomputable section

/-- Exact real value of Python's binary64 literal 1e-10. -/
def threshold : ℝ := 7737125245533627 / 77371252455336267181195264

def specialPosition (p v a m k t : ℝ) :=
  let k2 := k * k
  let invM := 1 / m
  (m * (a * (-k * t) + k * (a * t + v * k * invM * t)) + p * k2) * (1 / k2)

def integrate (p v a m k dt cached t : ℝ) : ℝ × ℝ :=
  if t = 0 then (p, v) else
    let q := stepFactor m k dt cached t
    (if k < threshold * m * t then specialPosition p v a m k t
     else regularPosition p v a m k q t, sourceVelocity v a m k q)

theorem integrate_zero (p v a m k dt cached : ℝ) :
    integrate p v a m k dt cached 0 = (p, v) := by simp [integrate]

theorem integrate_regular {m k dt cached t : ℝ} (hk : k ≠ 0)
    (hguard : ¬ k < threshold * m * t) (hcache : t = dt → cached = decay m k dt)
    (p v a : ℝ) :
    integrate p v a m k dt cached t = (position p v a m k t, velocity v a m k t) := by
  by_cases ht : t = 0
  · subst t
    exact (integrate_zero ..).trans (flow_zero ..).symm
  · simpa only [integrate, ite_eq_right ht, ite_eq_right hguard, stepFactor_eq hcache] using
      source_regular hk p v a m t

theorem special_position_eq {m k : ℝ} (hm : m ≠ 0) (hk : k ≠ 0) (p v a t : ℝ) :
    specialPosition p v a m k t = p + v * t := by
  dsimp [specialPosition]
  field_simp
  ring

theorem integrate_special {m k dt cached t : ℝ} (hm : m ≠ 0) (hk : k ≠ 0) (ht : t ≠ 0)
    (hguard : k < threshold * m * t) (p v a : ℝ) :
    integrate p v a m k dt cached t =
    (p + v * t, sourceVelocity v a m k (stepFactor m k dt cached t)) := by
  simp only [integrate, ite_eq_right ht, ite_eq_left hguard, special_position_eq hm hk]

/-- Both position branches share the same velocity update when the tick cache is valid. -/
theorem integrate_velocity {m k dt cached t : ℝ} (hk : k ≠ 0)
    (hcache : t = dt → cached = decay m k dt) (p v a : ℝ) :
    (integrate p v a m k dt cached t).2 = velocity v a m k t := by
  by_cases ht : t = 0
  · subst t
    rw [integrate_zero]
    simp [velocity, decay]
  · simpa only [integrate, ite_eq_right ht, stepFactor_eq hcache] using
      congrArg Prod.snd (source_regular hk p v a m t)

end
end ExactBallpark.Movement
