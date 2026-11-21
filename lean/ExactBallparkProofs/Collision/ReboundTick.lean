-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.ReboundFlight
public import ExactBallparkProofs.Collision.FiniteImpact
public import ExactBallparkProofs.Collision.ImpactHistory
public import ExactBallparkProofs.Movement.ScalarCommands

@[expose] public section
namespace ExactBallpark.Collision.ReboundTick
open Movement History
open ReboundFlight (Scenario)

def cfg : Scalar.Config := ⟨1,1⟩

/-- The full-tick cache and the partial-step host libm word for all three cases. -/
def expWord (x : Float) : Float := Float.ofBits (match x.toBits with
  | 0xbff0000000000000 => 0x3fd78b56362cef38
  | 0xbfe0000000000000 => 0x3fe368b2fc6f960a
  | _ => 0x7ff8000000000000)

def bodies (c : Scenario) : Scalar.Ball × Scalar.Ball :=
  let m := c.rightMass.toUInt64.toFloat
  let f := c.leftFraction.toUInt64.toFloat
  let make := fun (sp : Scalar.Spawn) (target : Float) =>
    Scalar.Source.prepare cfg (Scalar.gotoPoint (Scalar.spawn cfg expWord sp) (Scalar.vec3 target 0 0))
  (make ⟨1,0.5,2/f,Scalar.zeros,Scalar.vec3 2 0 0,1,f⟩ 64,
   make ⟨m,0.5,2,Scalar.vec3 3 0 0,Scalar.vec3 (-2) 0 0,1/m,1⟩ (-61))

def advance (b : Scalar.Ball) (motion : Motion) (t : Float) : Motion :=
  let atAxis := fun i : Fin 3 => Reference.binary64 expWord (motion.pos i) (motion.vel i)
    (b.lastG i + 0) (b.mass*b.agility) cfg.friction b.timeFactor t cfg.dt
  ⟨fun i => (atAxis i).1, fun i => (atAxis i).2⟩

def initial (b : Scalar.Ball) : Motion := ⟨b.pos,b.vel⟩

def path (b : Scalar.Ball) : Path :=
  let m := initial b
  ⟨[⟨0,-1,m⟩,⟨cfg.dt,-1,advance b m cfg.dt⟩],0,cfg.dt⟩

/-- Positive-time, noncoincident, translational branch of CalculateCollisionImpactFree. -/
def rawChanges (a b : Scalar.Ball) (p q : Motion) : Option (FloatPoint × FloatPoint) := do
  let delta := diff64 p.pos q.pos
  let len := (dot64 delta delta).sqrt
  guard (len > 0)
  let n := fun i => delta i * (1 / len)
  let other := fun i => -(n i)
  let speed := (-2) * dot64 (diff64 q.vel p.vel) other
  guard (speed ≥ 0)
  let j := a.mass * b.mass * speed / (a.mass + b.mass)
  return (fun i => n i * (j/a.mass), fun i => other i * (j/b.mass))

abbrev PointWords := UInt64 × UInt64 × UInt64

structure SideWords where
  cache : UInt64
  time : UInt64
  thrust : PointWords
  incomingPos : PointWords
  incomingVel : PointWords
  raw : PointWords
  applied : PointWords
  contactPos : PointWords
  outgoingVel : PointWords
  pos : PointWords
  vel : PointWords
  deriving DecidableEq, Repr

def triple (p : FloatPoint) : PointWords := ((p 0).toBits, (p 1).toBits, (p 2).toBits)

def finishBall (b : Scalar.Ball) (p : Path) (hit : Impact) : Option Path := do
  let (recorded, invalid) := insert (advance b) (initial b) p hit
  guard invalid.isEmpty
  finish (advance b) cfg.dt 22 recorded

/-- Require exactly one stored contact with the partner, both sentinel endpoints
and agreement of the committed state with the calculated history endpoint. -/
def sampleSide (b : Scalar.Ball) (p : Path) (partner : Int) (time : Float)
    (incoming : Motion) (raw applied : FloatPoint) : Option SideWords := do
  let [start,hit,last] := p.locations | none
  guard (start.time == 0 && start.collider == -1 && hit.collider == partner && hit.time == time &&
    last.time == cfg.dt && last.collider == -1)
  let endpoint ← position (advance b) p.locations cfg.dt
  guard (triple endpoint.pos == triple last.motion.pos && triple endpoint.vel == triple last.motion.vel)
  return ⟨b.timeFactor.toBits,hit.time.toBits,triple b.lastG,triple incoming.pos,triple incoming.vel,
    triple raw,triple applied,triple hit.motion.pos,triple hit.motion.vel,triple endpoint.pos,triple endpoint.vel⟩

/-- Ordinary iterative tick projected to the only eligible pair (IDs 1 and 2).
The second search starts on the touching surfaces and tests the source's
duplicate guard. A further accepted event or an unexpected history fails the replay. -/
def replay (c : Scenario) : Option (SideWords × SideWords) := do
  let (a,b) := bodies c
  let ap := path a
  let bp := path b
  let atTime := fun (ball : Scalar.Ball) (p : Path) => position (advance ball) p.locations
  let a0 ← atTime a ap ap.startTime
  let a1 ← atTime a ap ap.endTime
  let b0 ← atTime b bp bp.startTime
  let b1 ← atTime b bp bp.endTime
  let radius := a.radius + b.radius
  let u := contact64 a0.pos a1.pos b0.pos b1.pos radius
  guard (0 < u && u < 1 && !ignoreCollision ap.locations 2 u)
  let t := ap.startTime + (ap.endTime-ap.startTime)*u
  let p ← atTime a ap t
  let q ← atTime b bp t
  let (rawA,rawB) ← rawChanges a b p q
  let da := limitImpact64 rawA (ExactBallpark.Numerics.narrow a.maxVel)
  let db := limitImpact64 rawB (ExactBallpark.Numerics.narrow b.maxVel)
  let bp' ← finishBall b bp ⟨t,1,db⟩
  let ap' ← finishBall a ap ⟨t,2,da⟩
  let s := ap'.startTime
  let e := min ap'.endTime bp'.endTime
  guard (s < e)
  let as' ← atTime a ap' s
  let ae ← atTime a ap' e
  let bs ← atTime b bp' s
  let be ← atTime b bp' e
  let retry := contact64 as'.pos ae.pos bs.pos be.pos radius
  guard (retry == 0 && ignoreCollision ap'.locations 2 retry)
  return (← sampleSide a ap' 2 t p rawA da, ← sampleSide b bp' 1 t q rawB db)

abbrev EndWords := UInt64 × UInt64 × UInt64 × UInt64

/-- Final x, vx for ID 1, then x, vx for ID 2. -/
def recordedEnd : Scenario → EndWords
  | .equalMass => (0x3fdb4597e37cb050,0xbfdb4597e37cb050,0x4004974d039069f6,0x3fdb4597e37cb050)
  | .unequalMass => (0xbfb937809811a900,0xbff3c1dd4bd43ac4,0x400064de026046a4,0xbfd87c4568578a78)
  | .limited => (0x3ff629cbd5372d38,0x3ff0d6342ac8d2c8,0x4004974d039069f6,0x3fdb4597e37cb050)

def recorded (c : Scenario) : SideWords × SideWords := Id.run do
  let (rawA,rawB,appliedA,postA,postB) : UInt64 × UInt64 × UInt64 × UInt64 × UInt64 := match c with
    | .equalMass => (0xc010000000000000,0x4010000000000000,0xc010000000000000,0xc000000000000000,0x4000000000000000)
    | .unequalMass => (0xc015555555555555,0x4005555555555555,0xc015555555555555,0xc00aaaaaaaaaaaaa,0x3fe5555555555554)
    | .limited => (0xc010000000000000,0x4010000000000000,0xbff9000000000000,0x3fdc000000000000,0x4000000000000000)
  let (xa,va,xb,vb) := recordedEnd c
  let make := fun (g p raw applied post x v zero : UInt64) =>
    SideWords.mk 0x3fd78b56362cef38 0x3fe0000000000000 (g,0,0) (p,0,0) (g,0,0)
      (raw,zero,zero) (applied,zero,zero) (p,0,0) (post,0,0) (x,0,0) (v,0,0)
  return (make 0x4000000000000000 0x3ff0000000000000 rawA appliedA postA xa va 0,
    make 0xc000000000000000 0x4000000000000000 rawB rawB postB xb vb 0x8000000000000000)

set_option maxRecDepth 65536 in
set_option maxHeartbeats 8000000 in
theorem replay_words (c : Scenario) : replay c = some (recorded c) := by
  cases c <;> decide +kernel

end ExactBallpark.Collision.ReboundTick
