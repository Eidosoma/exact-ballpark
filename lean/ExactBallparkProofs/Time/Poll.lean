-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.Tick

@[expose] public section
namespace ExactBallpark.Time

/-- For every monotone poll the stored time is within one interval of the poll. -/
theorem clock_window {i : ℕ} (hi : 0 < i) (now last : ℤ) (ordered : last ≤ now)
    (first master : Bool) :
    let stamp := (onTick i now last first master).time
    last ≤ stamp ∧ stamp ≤ now ∧ now - stamp < i := by
  by_cases earlyPoll : now-last < i
  · rw [early i now last first master earlyPoll]
    exact ⟨le_rfl, ordered, earlyPoll⟩
  have due := le_of_not_gt earlyPoll
  cases first with
  | true => rw [first_due hi now last master due]; simp; omega
  | false =>
    by_cases normal : master = false ∨ (now-last)/i ≤ 5
    · rw [ordinary_tick hi now last master due normal]
      have := Int.emod_nonneg (now-last) (show (i : ℤ) ≠ 0 by omega)
      have := Int.emod_lt_of_pos (now-last) (show (0 : ℤ) < i by omega)
      dsimp
      omega
    · have hm : master = true := by cases master <;> simp_all
      subst master
      rw [master_resync hi now last due (by simp_all)]
      dsimp
      omega

/-- At five whole intervals the master processes five steps. At six it drops the backlog,
even when the poll has a fractional interval left over. -/
theorem master_boundary {i : ℕ} (hi : 0 < i) (last r : ℤ) (hr : 0 ≤ r) (small : r < i) :
    let now₅ := last + 5*i + r
    let now₆ := last + 6*i + r
    onTick i now₅ last false true =
      ⟨.clean :: (List.range 5).flatMap (fun j : ℕ => [.pre, .evolve (last + j*i), .post]),
       last + 5*i, false⟩ ∧
    onTick i now₆ last false true = ⟨[.clean,.pre,.evolve (now₆-i),.post],now₆,false⟩ := by
  dsimp only
  have hi' : (0 : ℤ) < i := by omega
  have quotient (k : ℤ) : (last+k*i+r-last)/i = k := by
    rw [show last+k*i+r-last = r+k*i by ring, Int.add_mul_ediv_right _ _ hi'.ne',
      Int.ediv_eq_zero_of_lt hr small, zero_add]
  have rem : (last+5*i+r-last)%i = r := by
    rw [show last+5*i+r-last = r+5*i by ring, Int.add_mul_emod_self_right, Int.emod_eq_of_lt hr small]
  rw [ordinary_tick hi _ _ true (by omega) (Or.inr (by rw [quotient])),
    master_resync hi _ _ (by omega) (by rw [quotient]; decide), quotient, rem]
  simp

/-- Once a non-first client has polled, the next evolve is due exactly one interval
after its stored timestamp. Earlier repeat polls only clean up. -/
theorem next_client_deadline {i : ℕ} (hi : 0 < i) (last : ℤ) :
    (∀ t : ℤ, last ≤ t → t < last+i →
      onTick i t last false false = ⟨[.clean], last, false⟩) ∧
    onTick i (last+i) last false false = ⟨[.clean,.pre,.evolve last,.post],last+i,false⟩ := by
  constructor
  · intro t _ h
    exact early i t last false false (by omega)
  · rw [ordinary_tick hi _ _ false (by omega) (Or.inl rfl)]
    simp [Int.ediv_self (show (i : ℤ) ≠ 0 by omega), List.range_succ]

end ExactBallpark.Time
