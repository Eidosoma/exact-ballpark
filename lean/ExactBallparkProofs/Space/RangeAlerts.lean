-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Proximity
public import ExactBallparkProofs.Movement.Control

@[expose] public section
namespace ExactBallpark.Space.RangeAlerts
open Control

/-- Inputs at CheckForProximity after the moribund guard. Each squared distance
is supplied by its own source lensq. Stored radii and follow range are widened. -/
structure Sample where
  mode : Mode
  notify : Float
  radius : Float
  otherRadius : Float
  followRange : Float
  gotoSq : Float
  followSq : Float
  trackRange : Float
  target : Bool
  trackSq : Float

def notification (s : Sample) : Option Bool := Id.run do
  if s.notify == 0 then return none
  if s.mode == .goto then
    let lim := s.notify+s.radius
    return some (decide (s.gotoSq < lim*lim))
  if s.mode == .follow || s.mode == .missile || s.mode == .orbit then
    let hi := s.otherRadius+s.followRange+s.notify+s.radius
    let lower := hi-2*s.notify
    let lo := if lower < 0 then 0 else lower
    return some (decide (s.followSq < hi*hi) && decide (s.followSq > lo*lo))
  return some false

def targetTracking (s : Sample) : Option Bool :=
  if s.trackRange ≥ Float.ofBits 0x3ee4f8b588e368f1 && s.target then
    let lim := s.trackRange+s.radius
    some (decide (s.trackSq < lim*lim))
  else none

inductive Alert where | range (inside : Bool) | track (inside : Bool)
  deriving DecidableEq, Repr

def transition (old : Bool) : Option Bool → Bool × Option Bool
  | none => (old,none)
  | some inside => edge old inside

abbrev Flags := Bool × Bool

def poll (old : Flags) (s : Sample) : Flags × List Alert :=
  let (r,er) := transition old.1 (notification s)
  let (t,et) := transition old.2 (targetTracking s)
  ((r,t),er.toList.map .range ++ et.toList.map .track)

def replay (old : Flags) : List Sample → List Alert
  | [] => []
  | s :: ss => let (next,events) := poll old s; events ++ replay next ss

def changes (old : Bool) : List (Option Bool) → List Bool
  | [] => []
  | x :: xs => let (state,event) := transition old x; event.toList ++ changes state xs

def rangeEvent : Alert → Option Bool | .range b => some b | .track _ => none
def trackEvent : Alert → Option Bool | .track b => some b | .range _ => none

theorem trace_channels (old : Flags) (ss : List Sample) :
    (replay old ss).filterMap rangeEvent = changes old.1 (ss.map notification) ∧
    (replay old ss).filterMap trackEvent = changes old.2 (ss.map targetTracking) := by
  induction ss generalizing old with
  | nil => simp [replay,changes]
  | cons s ss ih =>
    obtain ⟨r,er⟩ := ih (poll old s).1
    simp only [replay,List.filterMap_append,List.map_cons,changes,r,er]
    cases hr : transition old.1 (notification s) with | mk nr e =>
      cases ht : transition old.2 (targetTracking s) with | mk nt f =>
        cases e <;> cases f <;> simp [poll,hr,ht,List.filterMap_cons,rangeEvent,trackEvent]

def observed (prev : Bool) (x : Option Bool) := x.getD prev

/-- An odd number of notifications is exactly a change in membership, even with
disabled polls interleaved. Setters that reset a flag start a fresh history. -/
theorem edge_parity (old : Bool) (xs : List (Option Bool)) :
    xs.foldl observed old =
      if (changes old xs).length % 2 = 0 then old else !old := by
  induction xs generalizing old with
  | nil => simp [changes]
  | cons x xs ih =>
    cases x with
    | none => cases old <;> simp [changes,transition,observed,ih]
    | some x =>
      cases x <;> cases old <;> simp [changes,transition,edge,observed,List.foldl_cons,ih] <;>
        apply Bool.eq_iff_iff.mpr <;> simp <;> omega

def setNotification (value : Float) (oldRange : Float) (flags : Flags) : Float × Flags :=
  if value < 0 then (oldRange,flags) else (value,(false,flags.2))

/-- Even setting the same range again re-arms the enter callback. -/
theorem notification_rearms (s : Sample) (old : Flags) (accept : ¬ s.notify < 0)
    (inside : notification s = some true) :
    let updated := setNotification s.notify s.notify old
    updated.1 = s.notify ∧ updated.2.2 = old.2 ∧
      ((poll updated.2 s).2.filterMap rangeEvent) = [true] := by
  simp only [setNotification,ite_eq_right accept,poll,inside,transition,edge]
  cases h : (transition old.2 (targetTracking s)).2 <;> simp [rangeEvent]

theorem stop_exit_once (s : Sample) (mode : s.mode = .stop) (enabled : (s.notify == 0) = false)
    (tracked : Bool) :
    ((replay (true,tracked) [s,s]).filterMap rangeEvent) = [false] := by
  rw [(trace_channels ..).1]
  have no : notification s = some false := by simp [notification,mode,enabled]
  simp [changes,no,transition,edge]

/-- The predecessor and successor of 11, the two strict follow boundaries 2
and 12, a disabled tracking sample, and a clamped lower boundary at the centre. -/
def witnessSamples : List Sample := Id.run do
  let base : Sample := ⟨.goto,10,1,2,4,0,0,10,true,0⟩
  let g := fun (d : Float) => {base with gotoSq := d*d, trackSq := d*d}
  let f := fun (d : Float) => {base with mode := .follow, notify := 5, followSq := d*d, target := false}
  let centre := {base with
    mode := .orbit
    otherRadius := 0
    followRange := 0
    followSq := 0
    trackRange := Float.ofBits 0x3ee4f8b588e368f0}
  return [g (Float.ofBits 0x4025ffffffffffff),g 11,g (Float.ofBits 0x4026000000000001),
    f 2,f 3,f 12,centre,{base with mode := .stop,trackRange := Float.ofBits 0x3ee4f8b588e368f1}]

theorem boundary_words :
    replay (false,false) witnessSamples =
      [.range true,.track true,.range false,.track false,.range true,.range false,.track true] ∧
    (witnessSamples.map notification) = [some true,some false,some false,some false,some true,
      some false,some false,some false] := by decide +kernel

end ExactBallpark.Space.RangeAlerts
