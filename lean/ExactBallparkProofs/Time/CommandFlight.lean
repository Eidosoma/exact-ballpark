-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarTrace
public import ExactBallparkProofs.Time.Snapshots
public import Mathlib.Logic.Function.Iterate

@[expose] public section
namespace ExactBallpark.Time.CommandFlight
open ExactBallpark.Movement.Scalar

def command (b : Ball) (direction : Vec) := gotoPoint b (Source.directionTarget b direction)

structure Frame where
  ball : Ball
  sent : List Int

/-- One accepted GotoDirection, no other actions, starting after session startup.
The flushed event retains the producer's stamp. The parent call precedes Evolve. -/
def server (cfg : Config) (initial : Ball) (direction : Vec) (issued : Int) (delay : Nat) : Nat → Frame
  | 0 => ⟨initial,[]⟩
  | n+1 =>
    let prior := server cfg initial direction issued delay n
    let batch := if n = delay then enqueue [] (issued+n) else []
    let ready := batch.foldl (fun b _ => command b direction) prior.ball
    ⟨localTick cfg ready,prior.sent ++ batch⟩

theorem goto_first_commit (cfg : Config) (b : Ball) (direction : Vec) (issued : Int) (delay : Nat) :
    (∀ n ≤ delay, server cfg b direction issued delay n = ⟨(localTick cfg)^[n] b,[]⟩) ∧
    server cfg b direction issued delay (delay+1) =
      ⟨localTick cfg (command ((localTick cfg)^[delay] b) direction),[issued+delay]⟩ := by
  have before : ∀ n ≤ delay, server cfg b direction issued delay n = ⟨(localTick cfg)^[n] b,[]⟩ := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro time
      rw [server, ih (by omega)]
      simp [show n ≠ delay by omega, Function.iterate_succ_apply']
  exact ⟨before, by simp [server,before delay (le_refl _),enqueue]⟩

/-- Restore the chosen stream, replay to the command stamp, apply the parent GOTO,
then compare after the same number of subsequent movement ticks. -/
def lateFlight (cfg : Config) (streams : List Snapshots.Snapshot) (readBall : Nat → Ball)
    (predicted : Ball) (now desired : Int) (direction : Vec) (after : Nat) : Option Ball := do
  let (snap,steps) ← Snapshots.synchronize streams now desired
  let base := (snap.map (fun s => readBall s.token)).getD predicted
  return (localTick cfg)^[after] (command ((localTick cfg)^[steps] base) direction)

theorem rewind_goto_same_stamp (cfg : Config) (initial predicted : Ball) (direction : Vec)
    (before later : List Snapshots.Snapshot) (snap : Snapshots.Snapshot) (readBall : Nat → Ball)
    (origin now : Int) (pre gap after : Nat) (stamp : snap.stamp = origin + pre)
    (retained : readBall snap.token = (localTick cfg)^[pre] initial)
    (late : origin + pre + gap < now) (future : ∀ s ∈ later, origin + pre + gap < s.stamp) :
    lateFlight cfg (before ++ snap :: later) readBall predicted now (origin+pre+gap) direction after =
      some ((localTick cfg)^[after] (command ((localTick cfg)^[pre+gap] initial) direction)) := by
  have chosen := (Snapshots.rewind_stream before later snap now (origin+pre+gap) late (by omega) future).1
  have steps : (origin + (pre:Int) + gap - snap.stamp).toNat = gap := by omega
  simp [lateFlight, chosen, steps, retained, Nat.add_comm pre gap, Function.iterate_add_apply]

def sessionBall : Ball :=
  {mass := 2000000, radius := 40, maxVel := 300, pos := zeros, vel := zeros,
   agility := 1, speedFraction := 1, timeFactor := Float.ofBits 0x3fe368b2fc6f960a}

def sessionFrame (n : Nat) := server ⟨1000000,1⟩ sessionBall (vec3 1 0 0) 3 1 n

set_option maxRecDepth 16384 in
theorem goto_moves_at_five :
    (sessionFrame 1).sent = [] ∧ ((sessionFrame 1).ball.pos 0).toBits = 0 ∧
    (sessionFrame 2).sent = [4] ∧ (sessionFrame 2).ball.mode = .goto ∧
    ((sessionFrame 2).ball.pos 0).toBits = 0x404ff58dfe961e9f ∧
    ((sessionFrame 2).ball.vel 0).toBits = 0x405d829c805a7859 := by decide +kernel

end ExactBallpark.Time.CommandFlight
