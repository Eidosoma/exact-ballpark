-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarTrace
public import FloatLib.Floats.Formats.BinaryInterchange.Dyadic.Core
public import Mathlib.Analysis.Complex.ExponentialBounds

@[expose] public section
namespace ExactBallpark.Movement.CoastWitness
open Scalar

/-- The only host elementary-function input for this run is exp(-1). -/
def coastExp (x : Float) : Float :=
  if x.toBits = 0xbff0000000000000 then Float.ofBits 0x3fd78b56362cef38 else Float.nan

def finiteValue (x : Float) : Option Rat :=
  (FloatLib.Floats.Formats.BinaryInterchange.Model.unpackedToDyadic? x.toModel.unpack).map (·.toRat)

def config : Config := ⟨1, 1⟩
def initial : Spawn := ⟨1, 1, 100, zeros, vec3 100 0 0, 1, 1⟩
def flight : List Action := .edit 1 (.add initial) :: List.replicate 5 .tick

/-- A horizontal STOP flight starting at x = 0 and vx = 100. The constructor computes
the cache from the supplied exp. Every position and velocity comes from the three-pass tick. -/
def endpoint (expFn : Float → Float) : Option (Rat × Rat) := do
  let s ← run config expFn {} flight
  let b ← s.balls[(1 : Int)]?
  return (← finiteValue b.pos.x, ← finiteValue b.vel.x)

set_option maxRecDepth 32768 in
set_option maxHeartbeats 6000000 in
private theorem endpoint_value :
    endpoint coastExp = some (3494730165450135 / 35184372088832, 6069003118864693 / 9007199254740992) := by
  decide +kernel

private theorem single_exp_input (expFn : Float → Float)
    (word : expFn (Float.ofBits 0xbff0000000000000) = Float.ofBits 0x3fd78b56362cef38) :
    run config expFn {} flight = run config coastExp {} flight := by
  have mass : (initialBall initial).mass * (initialBall initial).agility = (1 : Float) := by decide +kernel
  have guard : ¬ (1 : Float) ≤ Float.ofBits 0 := by decide +kernel
  have argument : -config.friction * config.dt / (1 : Float) = Float.ofBits 0xbff0000000000000 := by decide +kernel
  have born : spawn config expFn initial = spawn config coastExp initial := by
    simp only [spawn, refresh, Source.decay, mass, ite_eq_right guard, argument, word]
    rfl
  have created : advance config expFn {} (.edit 1 (.add initial)) =
      advance config coastExp {} (.edit 1 (.add initial)) := by
    have edits : editBall config expFn (.add initial) = editBall config coastExp (.add initial) :=
      funext fun _ => congrArg some born
    simp only [advance, edits]
  have sameTick (s : Park) : advance config expFn s .tick = advance config coastExp s .tick := by
    have ordered : (config.dt != config.dt) = false := by decide +kernel
    have coords : Source.coordinates config expFn = Source.coordinates config coastExp := by
      funext b i
      simp only [Source.coordinates, Reference.binary64, ordered, Bool.false_eq_true, ite_false]
    simp only [advance, Source.tick, coords]
  have coasting (n : Nat) (s : Park) :
      execute config expFn s (List.replicate n .tick) = execute config coastExp s (List.replicate n .tick) := by
    induction n generalizing s with
    | zero => rfl
    | succ n ih =>
      rw [List.replicate_succ]
      simp only [execute]
      split
      · rw [sameTick]
        exact ih _
      · rfl
  have allowed : configAllowed config = true := by decide +kernel
  have first (f : Float → Float) (rest : List Action) : run config f {} (.edit 1 (.add initial) :: rest) =
      execute config f (advance config f {} (.edit 1 (.add initial))) rest := by
    simp only [run, ite_eq_left allowed, execute, actionAllowed, editAllowed, ite_true]
  let rest := List.replicate 5 Action.tick
  exact (first expFn rest).trans <|
    (congrArg (fun s => execute config expFn s rest) created).trans <|
    (coasting 5 _).trans (first coastExp rest).symm

/-- Rational brackets on the ideal five-second speed follow from mathlib's certified
bound on exp(1). Both endpoint errors are measured against the continuous drag solution. -/
theorem five_second_coast (expFn : Float → Float)
    (word : expFn (Float.ofBits 0xbff0000000000000) = Float.ofBits 0x3fd78b56362cef38) :
    ∃ p v : ℚ, endpoint expFn = some (p, v) ∧
      |(p : ℝ) - position 0 100 0 1 1 5| < 1 / (10 : ℝ)^14 ∧
      |(v : ℝ) - velocity 100 0 1 1 5| < 1 / (10 : ℝ)^15 := by
  have result : endpoint expFn = endpoint coastExp := by
    unfold endpoint
    rw [single_exp_input expFn word]
  refine ⟨_, _, result.trans endpoint_value, ?_⟩
  let lo : ℝ := 27182818284590452353 / 10^19
  let hi : ℝ := 27182818284590452354 / 10^19
  obtain ⟨upper, lower⟩ := abs_sub_le_iff.mp Real.exp_one_near_20
  have el : lo ≤ Real.exp 1 := by dsimp [lo]; linarith
  have eu : Real.exp 1 ≤ hi := by dsimp [hi]; linarith
  have pos : 0 < lo := by dsimp [lo]; norm_num
  have exactSpeed : 100 * Real.exp (-5) = 100 / (Real.exp 1)^5 := by
    rw [Real.exp_neg, ← Real.exp_nat_mul]
    norm_num [div_eq_mul_inv]
  have l : 100 / hi^5 ≤ 100 * Real.exp (-5) := by
    rw [exactSpeed]
    exact div_le_div_of_nonneg_left (by norm_num) (by positivity) (pow_le_pow_left₀ (Real.exp_pos _).le eu _)
  have u : 100 * Real.exp (-5) ≤ 100 / lo^5 := by
    rw [exactSpeed]
    exact div_le_div_of_nonneg_left (by norm_num) (pow_pos pos _) (pow_le_pow_left₀ pos.le el _)
  norm_num [lo, hi] at l u
  norm_num [position, velocity, terminal, decay, abs_lt]
  exact ⟨⟨by linarith only [l], by linarith only [u]⟩,
         ⟨by linarith only [u], by linarith only [l]⟩⟩

end ExactBallpark.Movement.CoastWitness
