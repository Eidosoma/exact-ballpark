-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.ContactCertificate
public import ExactBallparkProofs.Collision.TurningFlight

@[expose] public section
namespace ExactBallpark.Collision.ContactExamples
open Set Movement PositionEnclosure DragChord FirstContact ContactCertificate TurningFlight

def turnBody : Body := ⟨0, ![10,0,0], ![0,10,0], 1, 1⟩
def fixed (x y : ℚ) : Body := ⟨![x,y,0], 0, 0, 1, 1⟩
def turning (y : ℚ) : Scene := ⟨turnBody, fixed 4 y, 1/2, 1⟩
def lower : ℚ := 435661 / 10^6
def upper : ℚ := 435662 / 10^6

private theorem turning_distance (y : ℚ) (t : ℝ) :
    (turning y).distance t = dist (turn.pos t) (xy 4 y) := by
  have ship : turnBody.flight = turn := by
    unfold turnBody Body.flight turn
    congr 1 <;> first | (ext i; fin_cases i <;> norm_num [CurveClearance.asVec, realPoint, xy]) | norm_num
  have stationary : (fixed 4 y).flight.pos t = xy 4 y := by
    ext i; fin_cases i <;> simp [fixed, Body.flight, Flight.pos, position, terminal,
      CurveClearance.asVec, realPoint, xy]
  change dist (turnBody.flight.pos t) ((fixed 4 y).flight.pos t) = _
  rw [ship, stationary]

set_option maxRecDepth 32768 in
set_option maxHeartbeats 4000000 in
/-- Exact rational evaluations prove the whole prefix clear and the upper sample
inside the sphere. No root finder or evaluated host exponential enters the proof. -/
theorem turning_window : ∃ t ∈ Icc (lower:ℝ) upper,
    First (fun u => dist (turn.pos u) (xy 4 1)) (1/2) 1 t := by
  have ok : verify (turning 1) 0 (.window lower upper) = .window lower upper := by decide +kernel
  have sound := verified_result (turning 1) 0 (.window lower upper)
  rw [ok] at sound
  change ∃ t ∈ Icc (lower:ℝ) upper, First (turning 1).distance (↑(1/2:ℚ)) (↑(1:ℚ)) t at sound
  simpa only [funext (turning_distance 1), Rat.cast_one, Rat.cast_div, Rat.cast_ofNat] using sound

/-- The original lower obstacle stays at least 3.5 m from the turning ship's
surface, using the state-defined path directly. -/
theorem clear_control {t : ℝ} (ht : t ∈ Icc 0 1) :
    (1/2:ℝ) + 7/2 ≤ dist (turn.pos t) (xy 4 (-4)) := by
  have ok : (turning (-4)).margin = some (7/2) := by decide +kernel
  have sound := whole_interval_margin _ (by decide +kernel) ok (t := t)
    (by simpa [turning] using ht)
  rw [turning_distance] at sound
  norm_num [turning] at sound ⊢
  exact sound

/-- Both bodies turn, with drag times of one and two seconds respectively. -/
def twoTurns : Scene :=
  ⟨turnBody, ⟨![4,1,0], ![-1/2,-1/4,0], ![0,-1,0], 2, 1⟩, 1/2, 1⟩

set_option maxRecDepth 32768 in
theorem two_turning_bodies : ∃ t ∈ Icc (404889/10^6:ℝ) (404890/10^6),
    First twoTurns.distance (1/2) 1 t := by
  let l : ℚ := 404889/10^6
  let h : ℚ := 404890/10^6
  have ok : verify twoTurns 0 (.window l h) = .window l h := by decide +kernel
  have sound := verified_result twoTurns 0 (.window l h)
  rw [ok] at sound
  change ∃ t ∈ Icc (l:ℝ) h, First twoTurns.distance (↑(1/2:ℚ)) (↑(1:ℚ)) t at sound
  simpa [l,h] using sound

/-- Mass and friction are one, so the applied acceleration exactly balances drag. -/
def straight (x vx : ℚ) : Body := ⟨![x,0,0], ![vx,0,0], ![vx,0,0], 1, 1⟩
def crossing : Scene := ⟨straight 0 2, straight 3 (-2), 1, 1⟩

theorem moving_pair_window : ∃ t ∈ Icc (499999/10^6:ℝ) (1/2),
    First crossing.distance 1 1 t := by
  let l : ℚ := 499999/10^6
  let h : ℚ := 1/2
  have ok : verify crossing 0 (.window l h) = .window l h := by decide +kernel
  have sound := verified_result crossing 0 (.window l h)
  rw [ok] at sound
  change ∃ t ∈ Icc (l:ℝ) h, First crossing.distance (↑(1:ℚ)) (↑(1:ℚ)) t at sound
  simpa [l,h] using sound

/-- Both spheres already overlap at 0.6 s. Treating that sample as evidence for a
first hit after 0.55 s fails because the checked prefix contains the earlier hit. -/
theorem late_window_rejected : crossing.overlap (3/5) = true ∧
    verify crossing 3 (.window (11/20) (3/5)) = .unresolved := by decide +kernel

/-- The stored partial sweep runs from 0.5 to 1 second, with a local clock starting
at zero at x=0. The two contact times are compared on that same clock. -/
def candidate (x : ℚ) : Scene := ⟨straight 0 16, fixed x 0, 1, 1/2⟩

theorem obstacle_order {near far : ℝ} (hn : First (candidate 3).distance 1 (1/2) near)
    (hf : First (candidate 5).distance 1 (1/2) far) : 1/2 + near < 1/2 + far := by
  have window (x l h : ℚ) (ok : verify (candidate x) 0 (.window l h) = .window l h) :
      ∃ t ∈ Icc (l:ℝ) h, First (candidate x).distance 1 (1/2) t := by
    have sound := verified_result (candidate x) 0 (.window l h)
    rw [ok] at sound
    change ∃ t ∈ Icc (l:ℝ) h, First (candidate x).distance (↑(1:ℚ)) (↑(1/2:ℚ)) t at sound
    simpa using sound
  have a := window 3 (124999/10^6) (1/8) (by decide +kernel)
  have b := window 5 (249999/10^6) (1/4) (by decide +kernel)
  exact add_lt_add_right (disjoint_first_contacts a b (by norm_num) hn hf) _

end ExactBallpark.Collision.ContactExamples
