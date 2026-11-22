-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Profile

@[expose] public noncomputable section
namespace ExactBallpark.Warp.LengthScale

abbrev Profile := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

/-- L is the numerical length of one metre. Durations are always in seconds. -/
def inUnits (L d requested a b : ℝ) : Profile :=
  let s := min requested ((d+L)*a*b/(a+b))
  let tc := d/s - 1/a - 1/b + L/s
  (Real.log (s/(L*a))/a, tc, Real.log (s/(L*b))/b, s/a, s*tc, s/b-L, a, b, s)

def convert (c : ℝ) : Profile → Profile
  | (ta, tc, td, da, dc, dd, a, b, s) => (ta, tc, td, c*da, c*dc, c*dd, a, b, c*s)

theorem source_metres (factor d : ℝ) :
    inUnits 1 d (requestedSpeed factor) (acceleration factor) (deceleration factor) =
      Reference.constants factor d := by simp [inUnits, Reference.constants, requestedSpeed,
        acceleration, deceleration, milli]

/-- All nine outputs in either peak-speed branch under a change of length unit. -/
theorem unit_conversion (factor d : ℝ) {c : ℝ} (hc : 0 < c) :
    inUnits c (c*d) (c*requestedSpeed factor) (acceleration factor) (deceleration factor) =
      convert c (Reference.constants factor d) := by
  rw [← source_metres]
  generalize requestedSpeed factor = r
  generalize acceleration factor = a
  generalize deceleration factor = b
  let s := min r ((d+1)*a*b/(a+b))
  have scaled : min (c*r) ((c*d+c)*a*b/(a+b)) = c*s := by
    rw [show (c*d+c)*a*b/(a+b) = c*((d+1)*a*b/(a+b)) by ring]
    exact (mul_min_of_nonneg _ _ hc.le).symm
  simp only [inUnits, convert, scaled, one_mul]
  have unitOver : c/(c*s) = 1/s := by simpa only [mul_one] using mul_div_mul_left 1 s hc.ne'
  simp only [← show s = min r ((d+1)*a*b/(a+b)) from rfl, mul_div_mul_left _ _ hc.ne', unitOver]
  repeat' apply Prod.ext
  all_goals first | rfl | ring

end ExactBallpark.Warp.LengthScale
