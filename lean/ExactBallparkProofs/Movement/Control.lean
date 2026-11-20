-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.SplitIfs

/-! Discrete mode projection of ModeBallpark's commands. A follow link names a
live target. The target-side follower set and formation slot allocator are external. -/
@[expose] public section

namespace ExactBallpark.Control

inductive Mode where
  | goto | follow | stop | warp | orbit | missile | mushroom | boid | troll | miniball | field | rigid | formation
  deriving DecidableEq, Repr

def hasFollow : Mode → Bool
  | .follow | .orbit | .missile | .formation => true
  | _ => false

inductive ModeEvent where | exitWarp | changeMode (old new : Mode)
  deriving DecidableEq

def events (old new : Mode) : List ModeEvent :=
  if old = new then [] else
    (if new ≠ .warp ∧ old = .warp then [.exitWarp] else []) ++ [.changeMode old new]

theorem exit_iff (old new : Mode) :
    ModeEvent.exitWarp ∈ events old new ↔ old = .warp ∧ new ≠ .warp := by
  unfold events; split_ifs <;> simp_all [and_comm]

theorem exiting_order {new : Mode} (h : new ≠ .warp) :
    events .warp new = [.exitWarp, .changeMode .warp new] := by simp [events, h, Ne.symm h]

structure Following where
  mode : Mode
  target : Option Int
  owner : Int
  stamp : Int
  deriving DecidableEq

def stop (s : Following) : Following :=
  if hasFollow s.mode then ⟨.stop, none, 0, 0⟩ else {s with mode := .stop}

theorem stop_clears (s : Following) (h : s.target ≠ none → hasFollow s.mode) :
    (stop s).mode = .stop ∧ (stop s).target = none ∧ stop (stop s) = stop s := by
  unfold stop
  split
  · simp [hasFollow]
  next ht =>
    have h₀ : s.target = none := by by_contra bad; exact ht (h bad)
    simp [hasFollow, h₀]

/-- State observed by the command, including the initial STOP transition. -/
noncomputable def goto {Point : Type} (s : Following) (dst : Point) (speed : ℝ) :=
  ({stop s with mode := .goto}, dst, if speed = 0 then 1 else speed)

/-- The speed default depends only on the zero test. -/
theorem goto_cancels {Point : Type} (s : Following) (dst : Point) (speed : ℝ)
    (h : s.target ≠ none → hasFollow s.mode) :
    (goto s dst speed).1.target = none ∧ (goto s dst speed).2.1 = dst ∧
    ((goto s dst speed).2.2 = 1 ↔ speed = 0 ∨ speed = 1) := by
  simp only [goto, (stop_clears s h).2.1, true_and]
  split <;> simp_all

def canFollow (finiteRange sourcePresent targetPresent dead cloaked : Bool)
    (src dst srcBubble dstBubble : Int) : Bool :=
  if !finiteRange then false else if !sourcePresent || src == dst then false else
    if !targetPresent || dead || srcBubble != dstBubble || cloaked then false else true

theorem follow_iff (f s t dead cloak : Bool) (src dst sb db : Int) :
    canFollow f s t dead cloak src dst sb db = true ↔
    f = true ∧ s = true ∧ src ≠ dst ∧ t = true ∧ dead = false ∧ sb = db ∧ cloak = false := by
  simp [canFollow, and_assoc]

inductive Release where | discardStale | flyOn (clearMassive : Bool) | stop
  deriving DecidableEq

def release (idMatches ptrMatches interactive : Bool) (mode : Mode) : Release :=
  if !idMatches || !ptrMatches then .discardStale else
  if mode = .missile || (interactive && mode == .orbit) then .flyOn (mode == .missile) else .stop

theorem fly_on_iff (a b interactive : Bool) (mode : Mode) :
    (∃ flag, release a b interactive mode = .flyOn flag) ↔
    a = true ∧ b = true ∧ (mode = .missile ∨ interactive = true ∧ mode = .orbit) := by
  cases a <;> cases b <;> cases interactive <;> simp [release]
  all_goals split_ifs <;> simp_all
  exact or_comm.mp (em _)

end ExactBallpark.Control
