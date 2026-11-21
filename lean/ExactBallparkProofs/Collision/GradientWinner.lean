-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Collision.Selection
public import Mathlib.Data.List.MinMax
public import Mathlib.Order.Prod.Lex.Basic

@[expose] public noncomputable section
namespace ExactBallpark.Collision.GradientWinner
open ExactBallpark.Movement

structure Hit where
  id : Int
  time : ℝ
  acceleration : Vec3

def scaled (h : Hit) := (7656119366529843 / 9007199254740992 : ℝ) • h.acceleration

def key (h : Hit) : ℝ ×ₗ ℝ := toLex (h.time, ‖scaled h‖^2)

/-- The ordinary ball-candidate branch after geometry and eligibility have
succeeded. The initial empty accumulator corresponds to last_collision = -1
and a list of nonnegative hits. -/
def keep (previous : Option Hit) (h : Hit) : Option Hit :=
  match previous with
  | none => some h
  | some p =>
    if h.time ≥ p.time then
      if h.time = p.time then
        if ‖scaled h‖^2 > ‖scaled p‖^2 then some h else some p
      else some h
    else some p

/-- argmax retains the first maximizing candidate, including ties with equal
scaled norms and different vector directions. -/
theorem gradient_winner (hits : List Hit) :
    hits.foldl keep none = hits.argmax key := by
  induction hits using List.reverseRecOn with
  | nil => rfl
  | append_singleton xs x ih =>
    rw [List.foldl_append,List.foldl_cons,List.foldl_nil,ih,List.argmax_concat]
    cases prev : List.argmax key xs with
    | none => rfl
    | some y =>
      change keep (some y) x = if key y < key x then some x else some y
      simp only [keep,key,Prod.Lex.toLex_lt_toLex]
      split_ifs <;> grind

end ExactBallpark.Collision.GradientWinner
