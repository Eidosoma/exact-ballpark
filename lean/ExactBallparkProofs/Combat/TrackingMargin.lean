-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Combat.Execution
public import ExactBallparkProofs.Collision.SweepCertificate

@[expose] public section
namespace ExactBallpark.Combat.TrackingMargin
open ExactBallpark.Collision
set_option exponentiation.threshold 2048

def dotQ (a b : RatPoint) := a 0*b 0+a 1*b 1+a 2*b 2

/-- Evaluate each component of v × r with the supplied difference of products. -/
def crossWith {α : Type} (det : α → α → α → α → α) (v r : Fin 3 → α) : Fin 3 → α :=
  ![det (v 1) (r 2) (v 2) (r 1), det (v 2) (r 0) (v 0) (r 2), det (v 0) (r 1) (v 1) (r 0)]

/-- Angular speed squared from the exact coordinates decoded from finite words. -/
def crossAngularSq (v r : RatPoint) :=
  let c := crossWith (fun a b x y => a*b-x*y) v r
  dotQ c c / (dotQ r r)^2

/-- On a nonzero sightline the exact projection formula agrees with the squared cross
product, and vanishes precisely when every component of the cross product vanishes. -/
theorem cross_angular_sq (v r : RatPoint) (sight : dotQ r r ≠ 0) :
    let c := crossWith (fun a b x y => a*b-x*y) v r
    (dotQ v v - (dotQ v r)^2 / dotQ r r) / dotQ r r = crossAngularSq v r ∧
      (crossAngularSq v r = 0 ↔ c 0 = 0 ∧ c 1 = 0 ∧ c 2 = 0) := by
  let c := crossWith (fun a b x y => a*b-x*y) v r
  have identity : dotQ c c = dotQ v v * dotQ r r - (dotQ v r)^2 := by
    simp [c,crossWith,dotQ]
    ring
  change _ = dotQ c c / (dotQ r r)^2 ∧ (dotQ c c / (dotQ r r)^2 = 0 ↔ _)
  constructor
  · rw [identity]
    field_simp [sight]
  · rw [div_eq_zero_iff, or_iff_left (pow_ne_zero 2 sight)]
    change dotQ c c = 0 ↔ c 0 = 0 ∧ c 1 = 0 ∧ c 2 = 0
    simp only [dotQ, ← sq,
      add_eq_zero_iff_of_nonneg (add_nonneg (sq_nonneg (c 0)) (sq_nonneg (c 1))) (sq_nonneg (c 2)),
      add_eq_zero_iff_of_nonneg (sq_nonneg (c 0)) (sq_nonneg (c 1)),
      sq_eq_zero_iff, and_assoc]

/-- Candidate binary64 control. Each fused operation rounds once in Lean's Float model. -/
def compensatedDet (a b c d : Float) : Float :=
  let cd := c*d
  let main := a.toModel.fma b.toModel (-cd).toModel
  let residual := (-c).toModel.fma d.toModel cd.toModel
  Float.ofBits (main + residual).toBits

set_option maxRecDepth 16384 in
/-- Separately rounded products erase all three cross components in the recorded replay.
The fused control recovers the exact components and the decoded geometry gives positive speed squared. -/
theorem radial_cross_words :
    let v := radialVelocity
    let r := radialSightline
    let plain := crossWith (fun a b x y => a*b-x*y) v r
    let fused := crossWith compensatedDet v r
    let components := fun (p : FloatPoint) => ((p 0).toBits, (p 1).toBits, (p 2).toBits)
    components plain = (0,0,0) ∧
    (do let vq ← decodePoint v
        let rq ← decodePoint r
        let recovered ← decodePoint fused
        let c := crossWith (fun a b x y => a*b-x*y) vq rq
        return (decide (∀ i, recovered i = c i), c 0, c 1, c 2, crossAngularSq vq rq)) =
      some (true, 1/2^55, 0, -(1/2^57), 17/(2^114*169^2)) := by decide +kernel

structure Margin where
  sidewaysSq : ℚ
  threshold : ℚ
  deriving DecidableEq, Repr

/-- For a finite rounded operation, the usual relative budget plus half the
smallest subnormal covers gradual underflow. The certificate checks both final
roundings on the actual encoded words. -/
def roundBudget (x : ℚ) := |x| / 2^53 + 1/2^1075

/-- Express the threshold in squared velocity units. It depends on the input
vectors and rounded squared speeds and distance, before the final subtraction
and division. Their observed residuals are checked independently of their sign. -/
def inspect (v r : FloatPoint) : Option Margin := do
  let vq ← decodePoint v
  let rq ← decodePoint r
  let rr := dotQ rq rq
  if rr ≤ 0 then none else
  let vv := dotQ vq vq
  let along := (dotQ vq rq)^2/rr
  let d := (dot64 r r).sqrt
  let radial := dot64 v r / d
  let total := dot64 v v
  let projected := radial*radial
  let head ← finiteValue total
  let proj ← finiteValue projected
  let den ← finiteValue (d*d)
  if den ≤ 0 then none else
  let diff ← finiteValue (total-projected)
  let speed ← finiteValue ((total-projected)/(d*d))
  let subErr := roundBudget (head-proj)
  let divErr := (|head-proj|+subErr)/2^53+den/2^1075
  if |diff-(head-proj)| ≤ subErr ∧ |den*speed-diff| ≤ divErr then
    some ⟨vv-along,|head-vv| + |proj-along| + subErr+divErr⟩
  else none

/-- This bound applies to the actual input of GetAccuracy's square root. A
positive margin also bounds its magnitude after rescaling by the stored d². -/
theorem sideways_margin (v r : FloatPoint) (m : Margin) (res track sig : Float)
    (ok : inspect v r = some m) :
    let speed := (trackingTerms v r res track sig).1
    ∃ den y : ℚ, finiteValue (((dot64 r r).sqrt)*((dot64 r r).sqrt)) = some den ∧
      finiteValue speed = some y ∧ 0 < den ∧ |den*y-m.sidewaysSq| ≤ m.threshold ∧
      (m.threshold ≤ m.sidewaysSq → ¬ speed < Float.ofBits 0) := by
  unfold inspect at ok
  obtain ⟨vq,dv,ok⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨rq,dr,ok⟩ := Option.bind_eq_some_iff.mp ok
  dsimp only at ok
  split at ok
  next => contradiction
  next =>
    obtain ⟨head,dh,ok⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨proj,dp,ok⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨den,dd,ok⟩ := Option.bind_eq_some_iff.mp ok
    split at ok
    next => contradiction
    next hd =>
      obtain ⟨diff,ds,ok⟩ := Option.bind_eq_some_iff.mp ok
      obtain ⟨y,dy,ok⟩ := Option.bind_eq_some_iff.mp ok
      split at ok
      next residuals =>
        cases ok
        let vv := dotQ vq vq
        let along := (dotQ vq rq)^2/dotQ rq rq
        let subErr := roundBudget (head-proj)
        let divErr := (|head-proj|+subErr)/2^53+den/2^1075
        have bound : |den*y-(vv-along)| ≤ |head-vv| + |proj-along| + subErr+divErr := by
          obtain ⟨e1,e2⟩ := residuals
          change |diff-(head-proj)| ≤ subErr at e1
          change |den*y-diff| ≤ divErr at e2
          rw [abs_le] at e1 e2 ⊢
          constructor <;> linarith only [e1.1,e1.2,e2.1,e2.2,le_abs_self (head-vv),neg_le_abs (head-vv),
            le_abs_self (proj-along),neg_le_abs (proj-along)]
        refine ⟨den,y,dd,dy,lt_of_not_ge hd,bound,?_⟩
        change |head-vv| + |proj-along| + subErr+divErr ≤ vv-along → _
        intro safe
        have nonneg : 0 ≤ den*y := by linarith only [safe,(abs_le.mp bound).1]
        exact fun bad => (not_lt.mpr (nonneg_of_mul_nonneg_right nonneg (lt_of_not_ge hd)))
          ((finite_negative_iff _ _ dy).mp bad)
      next => contradiction

end ExactBallpark.Combat.TrackingMargin
