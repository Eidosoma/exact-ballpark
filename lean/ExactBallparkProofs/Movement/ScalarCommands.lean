-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarReference
public import ExactBallparkProofs.Numerics.Clamp

@[expose] public section
namespace ExactBallpark.Movement.Scalar

def refresh (cfg : Config) (expFn : Float → Float) (b : Ball) : Ball :=
  { b with timeFactor := Source.decay cfg expFn b }

def initialBall (s : Spawn) : Ball :=
  let agility := stored s.agility
  {
    mass := Numerics.clampMass s.mass
    radius := Numerics.clampMass (stored s.radius)
    maxVel := Numerics.clampMass (stored s.maxVel)
    pos := s.pos, vel := s.vel
    agility := if agility ≤ Float.ofBits 0 then 1 else agility
    speedFraction := Numerics.clampMass (stored s.speedFraction)
  }

def spawn (cfg : Config) (expFn : Float → Float) (s : Spawn) : Ball :=
  refresh cfg expFn (initialBall s)

def gotoPoint (b : Ball) (v : Vec) : Ball :=
  if (v 0).isFinite && (v 1).isFinite && (v 2).isFinite then
    { b with mode := .goto, target := v,
             speedFraction := if b.speedFraction == Float.ofBits 0 then 1 else b.speedFraction }
  else b

/-- Commands observe a completed API call. Adding an existing ID replaces every field. -/
def editBall (cfg : Config) (expFn : Float → Float) (command : Edit) (old : Option Ball) :
    Option Ball :=
  match command with
  | .add s => some (spawn cfg expFn s)
  | .mass x => old.map fun b =>
    if x ≤ Float.ofBits 0 then b else refresh cfg expFn { b with mass := x }
  | .agility x => old.map fun b =>
    let a := stored x
    if a ≤ Float.ofBits 0 then b else refresh cfg expFn { b with agility := a }
  | .maxVelocity x => old.map fun b =>
    let v := stored x
    if v < Float.ofBits 0 then b else { b with maxVel := v }
  | .radius x => old.map fun b =>
    let r := stored x
    if r < Float.ofBits 0 then b else { b with radius := r }
  | .speedFraction x => old.map fun b =>
    let s := stored x
    if !s.isFinite then b else
      { b with speedFraction := if s < Float.ofBits 0 then 0 else if s > 1 then 1 else s }
  | .stop => old.map fun b => { b with mode := .stop }
  | .gotoPoint v => old.map fun b => gotoPoint b v
  | .gotoDirection v => old.map fun b => gotoPoint b (Source.directionTarget b v)

/-- A finite CPython setter argument whose binary32 conversion overflows raises
before its guard. The constructor instead saturates such arguments. -/
def setterFits (x : Float) : Bool := !(x.isFinite && (Numerics.narrow x).isInf)

def editAllowed (s : Park) (id : Int) (command : Edit) : Bool :=
  match command with
  | .add _ => true
  | .agility x | .maxVelocity x | .radius x | .speedFraction x =>
    s.balls.contains id && setterFits x
  | _ => s.balls.contains id

/-- A sufficient execution domain for fixed-configuration, collision-free scalar parks. -/
def configAllowed (cfg : Config) : Bool :=
  cfg.dt.isFinite && decide (Float.ofBits 0 ≤ cfg.dt) &&
  cfg.friction.isFinite && decide (Float.ofBits 0 < cfg.friction) &&
  decide (Float.ofBits 0 < cfg.friction * cfg.friction)

def actionAllowed (s : Park) : Action → Bool
  | .edit id command => editAllowed s id command
  | .tick => s.balls.toList.all fun (_, b) =>
    match b.mode with
    | .stop => true
    | .goto => b.mass * b.agility != Float.ofBits 0

/-- Structural equality here compares stored Float values, including the signs of zeros.
The Python correspondence probes use the engine's two-NaNs-equal convention. -/
def CacheValid (cfg : Config) (expFn : Float → Float) (b : Ball) : Prop :=
  b.timeFactor = Source.decay cfg expFn b

private theorem refresh_valid (cfg : Config) (expFn : Float → Float) (b : Ball) :
    CacheValid cfg expFn (refresh cfg expFn b) := by rfl

private theorem goto_valid {cfg : Config} {expFn : Float → Float} {b : Ball}
    (h : CacheValid cfg expFn b) (v : Vec) : CacheValid cfg expFn (gotoPoint b v) := by
  unfold gotoPoint
  split <;> exact h

theorem edit_preserves_cache (cfg : Config) (expFn : Float → Float) (command : Edit)
    (old : Option Ball) (valid : ∀ b ∈ old, CacheValid cfg expFn b) :
    ∀ b ∈ editBall cfg expFn command old, CacheValid cfg expFn b := by
  cases command with
  | add s => simpa [editBall, spawn] using refresh_valid cfg expFn _
  | mass x | agility x =>
    cases old with
    | none => simp [editBall]
    | some b =>
      have h := valid b rfl
      simp only [editBall, Option.map_some, Option.mem_some_iff]
      intro _ rfl
      split <;> first | exact h | apply refresh_valid
  | maxVelocity x | radius x | speedFraction x =>
    cases old with
    | none => simp [editBall]
    | some b =>
      have h := valid b rfl
      simp only [editBall, Option.map_some, Option.mem_some_iff]
      intro _ rfl
      split <;> exact h
  | stop | gotoPoint _ | gotoDirection _ =>
    cases old with
    | none => simp [editBall]
    | some b =>
      have h := valid b rfl
      simp only [editBall, Option.map_some, Option.mem_some_iff]
      intro _ rfl
      first | exact h | apply goto_valid h

end ExactBallpark.Movement.Scalar
