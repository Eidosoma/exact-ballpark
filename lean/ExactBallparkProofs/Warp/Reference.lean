-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-! Generated from modes.py by scripts/warp_reference.py. -/
@[expose] public noncomputable section

namespace ExactBallpark.Warp.Reference

def constants (warp_factor warp_distance : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  let s : ℝ := ((warp_factor * (1152921504606847 / 1152921504606846976)) * 149597870700)
  let acc : ℝ := (warp_factor * (1152921504606847 / 1152921504606846976))
  let dec : ℝ := (min (warp_factor * (6148914691236517 / 18446744073709551616)) 2)
  let s : ℝ := (min s ((((warp_distance + 1) * acc) * dec) / (acc + dec)))
  let t_a : ℝ := ((Real.log (s / acc)) / acc)
  let t_c : ℝ := ((((warp_distance / s) - (1 / acc)) - (1 / dec)) + (1 / s))
  let t_d : ℝ := ((Real.log (s / dec)) / dec)
  (t_a, t_c, t_d, (s / acc), (s * t_c), ((s / dec) - 1), acc, dec, s)

end ExactBallpark.Warp.Reference
