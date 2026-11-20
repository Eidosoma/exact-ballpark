-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Commands
public import ExactBallparkProofs.Movement.SpecialBranch

@[expose] public noncomputable section
namespace ExactBallpark.Warp
open ExactBallpark.Movement

/-- (Next effect stamp, requested thrust). The activation branch starts warp proper at now. -/
def alignStep (stamp now : ℤ) (ready : Bool) (gotoThrust : ℤ → Vec3) : ℤ × Vec3 :=
  if 0 ≤ stamp then (stamp, 0) else if ready then (now, 0) else (stamp-1, gotoThrust (stamp-1))

theorem waiting_ticks (n : ℕ) (stamp now : ℤ) (h : stamp < 0) (g : ℤ → Vec3) :
    (fun s => (alignStep s now false g).1)^[n] stamp = stamp - n ∧
    (alignStep (stamp - n) now false g).2 = g (stamp - n - 1) := by
  constructor
  · induction n with
    | zero => simp
    | succ n ih =>
      rw [Function.iterate_succ_apply', ih]
      simp [alignStep, show ¬ (n : ℤ) ≤ stamp by omega]
      omega
  · simp [alignStep, show ¬ (n : ℤ) ≤ stamp by omega]

def backdate (now : ℤ) := max (now-5) 0

theorem entity_elapsed (now : ℤ) (h : 0 ≤ now) :
    now - backdate now = min 5 now ∧ 0 ≤ backdate now := by
  unfold backdate
  omega

def afterWarp (stopped : Bool) (p v a m k dt cached : ℝ) :=
  if stopped then integrate p v a m k dt cached dt else (p, v)

theorem exit_regular (p v a m k dt : ℝ) (hk : k ≠ 0) (hg : ¬ k < threshold*m*dt) :
    afterWarp true p v a m k dt (decay m k dt) = (position p v a m k dt, velocity v a m k dt) :=
  integrate_regular hk hg (fun _ => rfl) p v a

theorem exit_special (p v a m k dt cached : ℝ) (hm : m ≠ 0) (hk : k ≠ 0)
    (ht : dt ≠ 0) (hg : k < threshold*m*dt) :
    (afterWarp true p v a m k dt cached).1 = p + v*dt :=
  congrArg Prod.fst (integrate_special (dt := dt) (cached := cached) hm hk ht hg p v a)

end ExactBallpark.Warp
