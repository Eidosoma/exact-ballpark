-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Data.Int.Basic
public import Mathlib.Tactic.Ring

/-! Integer clock and observable call order of FullBallpark.on_tick, in 100 ns units.
An evolve event carries its simulation timestamp. Handler side effects are outside this projection. -/
@[expose] public section

namespace ExactBallpark.Time

inductive Call where
  | clean | pre | evolve (stamp : ℤ) | post
  deriving DecidableEq

/-- One iteration consumes exactly one interval of the backlog. -/
def rounds (interval : ℕ) (now delta : ℤ) : ℕ → List Call × ℤ
  | 0 => ([], delta)
  | n + 1 =>
    let (later, rem) := rounds interval now (delta - interval) n
    (.pre :: .evolve (now - delta) :: .post :: later, rem)

theorem rounds_remainder (i : ℕ) (now delta : ℤ) (n : ℕ) :
    (rounds i now delta n).2 = delta - n * i := by
  induction n generalizing delta with
  | zero => simp [rounds]
  | succ n ih => simp only [rounds, ih, Nat.cast_succ]; ring

theorem rounds_trace (i : ℕ) (now delta : ℤ) (n : ℕ) :
    (rounds i now delta n).1 = (List.range n).flatMap (fun j =>
      [.pre, .evolve (now - delta + j * i), .post]) := by
  induction n generalizing delta with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ_eq_map]
    simp only [rounds, ih, List.flatMap_cons, List.flatMap_map, List.cons_append, List.nil_append,
      Nat.cast_zero, zero_mul, add_zero]
    congr 3
    apply congrArg (List.flatMap · (List.range n))
    funext j
    simp only [Nat.cast_succ]
    congr 3; ring

structure TickResult where
  calls : List Call
  time : ℤ
  first : Bool
  deriving DecidableEq

def onTick (i : ℕ) (now last : ℤ) (first master : Bool) : TickResult :=
  let delta := now - last
  if delta < (i : ℤ) then ⟨[.clean], last, first⟩ else
  let due := if master && decide (delta / i > 5) then (i : ℤ) else delta
  if first then ⟨[.clean, .evolve now], now, false⟩ else
  let (calls, rem) := rounds i now due (due / i).toNat
  ⟨.clean :: calls, now - rem, false⟩

theorem early (i : ℕ) (now last : ℤ) (first master : Bool) (h : now - last < i) :
    onTick i now last first master = ⟨[.clean], last, first⟩ := by simp [onTick, h]

theorem first_due {i : ℕ} (_hi : 0 < i) (now last : ℤ) (master : Bool) (h : (i : ℤ) ≤ now - last) :
    onTick i now last true master = ⟨[.clean, .evolve now], now, false⟩ := by
  simp [onTick, not_lt.mpr h]

theorem master_resync {i : ℕ} (hi : 0 < i) (now last : ℤ) (h : (i : ℤ) ≤ now - last)
    (backlog : 5 < (now - last) / i) :
    onTick i now last false true = ⟨[.clean, .pre, .evolve (now - i), .post], now, false⟩ := by
  have hz : (i : ℤ) ≠ 0 := by omega
  simp [onTick, not_lt.mpr h, backlog, Int.ediv_self hz, rounds]

theorem ordinary_tick {i : ℕ} (hi : 0 < i) (now last : ℤ) (master : Bool)
    (h : (i : ℤ) ≤ now - last) (normal : master = false ∨ (now - last) / i ≤ 5) :
    let n := ((now - last) / i).toNat
    onTick i now last false master =
      ⟨.clean :: (List.range n).flatMap (fun (j : ℕ) => [.pre, .evolve (last + (j : ℤ) * i), .post]),
       now - (now - last) % i, false⟩ := by
  have hd : 0 ≤ now - last := by omega
  have hn : 0 ≤ (now - last) / i := Int.ediv_nonneg hd (by omega)
  have keep : (master && decide (5 < (now - last) / i)) = false := by
    rcases normal with hm | hn <;> simp [*, not_lt.mpr]
  simp only [onTick, ite_eq_right (not_lt.mpr h), keep, Bool.false_eq_true,
    reduceIte, rounds_trace, rounds_remainder, Int.toNat_of_nonneg hn, sub_sub_cancel]
  rw [Int.emod_def, mul_comm ((now - last) / i)]

def renderTime (stamp : ℤ) (millis : ℕ) := stamp - 2 * millis * 10000

theorem render_two_ticks (stamp : ℤ) (millis : ℕ) :
    renderTime (stamp + millis * 10000 * 2) millis = stamp := by unfold renderTime; ring

end ExactBallpark.Time
