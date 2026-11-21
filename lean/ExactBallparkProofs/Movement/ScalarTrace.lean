-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarCommands
public import ExactBallparkProofs.Movement.TensorLane

@[expose] public section
namespace ExactBallpark.Movement.Scalar

/-- A single ball advances immediately using its cached factor and the independent
tensor-lane arithmetic. A zero-duration tick still prepares the mode. -/
def localTick (cfg : Config) (b : Ball) : Ball :=
  let pre := Source.prepare cfg b
  let coord := fun i =>
    if cfg.dt == Float.ofBits 0 then (pre.pos i, pre.vel i)
    else TensorLane.atFactor (pre.pos i) (pre.vel i) (pre.lastG i + Float.ofBits 0)
      (pre.mass * pre.agility) cfg.friction pre.timeFactor cfg.dt
  Source.commit pre (Motion.ofFn coord)

private theorem cached_coordinate (cfg : Config) (expFn : Float → Float)
    (ordered : (cfg.dt != cfg.dt) = false) (pre : Ball) (i : Fin 3) :
    Source.coordinates cfg expFn pre i =
      if cfg.dt == Float.ofBits 0 then (pre.pos i, pre.vel i)
      else TensorLane.atFactor (pre.pos i) (pre.vel i) (pre.lastG i + Float.ofBits 0)
        (pre.mass * pre.agility) cfg.friction pre.timeFactor cfg.dt := by
  have component (v : Vec) (j : Fin 3) :
      (Vec.ofFn fun n => v n + Float.ofBits 0) j = v j + Float.ofBits 0 := by
    fin_cases j <;> rfl
  unfold Source.coordinates
  rw [component]
  cases zero : cfg.dt == Float.ofBits 0 <;>
    simp [Reference.binary64, TensorLane.atFactor, ordered, zero]

theorem tick_lookup (cfg : Config) (expFn : Float → Float) (ordered : (cfg.dt != cfg.dt) = false)
    (s : Park) (id : Int) :
    (Source.tick cfg expFn s).balls[id]? = s.balls[id]?.map (localTick cfg) := by
  simp only [Source.tick, Std.TreeMap.getElem?_map]
  cases found : s.balls[id]? with
  | none => rfl
  | some b =>
    let pre := Source.prepare cfg b
    have coords := funext (cached_coordinate cfg expFn ordered pre)
    exact congrArg (fun f => some (Source.commit pre (Motion.ofFn f))) coords

theorem tick_order (cfg : Config) (expFn : Float → Float) (s : Park) :
    (Source.tick cfg expFn s).balls.keys = s.balls.keys ∧
    s.balls.keys.Pairwise (fun a b => compare a b = .lt) := by
  exact ⟨by simp [Source.tick], s.balls.ordered_keys⟩

def advance (cfg : Config) (expFn : Float → Float) (s : Park) : Action → Park
  | .tick => Source.tick cfg expFn s
  | .edit id cmd => { s with balls := s.balls.alter id (editBall cfg expFn cmd) }

def execute (cfg : Config) (expFn : Float → Float) (s : Park) : List Action → Option Park
  | [] => some s
  | a :: rest => if actionAllowed s a then execute cfg expFn (advance cfg expFn s a) rest else none

/-- none marks a trace outside the stated correspondence domain. Inputs are binary64
values at the API boundary. Configuration stays fixed for the entire trace. -/
def run (cfg : Config) (expFn : Float → Float) (s : Park) (actions : List Action) : Option Park :=
  if configAllowed cfg then execute cfg expFn s actions else none

/-- Statements of the domain conditions in the successive states visited by the commands. -/
def Admitted (cfg : Config) (expFn : Float → Float) (s : Park) : List Action → Prop
  | [] => True
  | a :: rest => actionAllowed s a = true ∧ Admitted cfg expFn (advance cfg expFn s a) rest

theorem run_accepts_iff (cfg : Config) (expFn : Float → Float) (s : Park) (actions : List Action) :
    (run cfg expFn s actions).isSome = true ↔ configAllowed cfg = true ∧ Admitted cfg expFn s actions := by
  have successful (xs : List Action) (p : Park) :
      (execute cfg expFn p xs).isSome = true ↔ Admitted cfg expFn p xs := by
    induction xs generalizing p with
    | nil => simp [execute, Admitted]
    | cons a rest ih => cases h : actionAllowed p a <;> simp [execute, Admitted, h, ih]
  cases h : configAllowed cfg <;> simp [run, h, successful]

inductive BallAction where
  | tick
  | edit (cmd : Edit)

/-- Every tick and exactly the commands addressed to the observed ID. -/
def forBall (id : Int) : Action → Option BallAction
  | .tick => some .tick
  | .edit target cmd => if target = id then some (.edit cmd) else none

def localStep (cfg : Config) (expFn : Float → Float) (b : Option Ball) : BallAction → Option Ball
  | .tick => b.map (localTick cfg)
  | .edit cmd => editBall cfg expFn cmd b

def ticks : List Action → Nat
  | [] => 0
  | .tick :: rest => 1 + ticks rest
  | .edit _ _ :: rest => ticks rest

private theorem finite_ordered {t : Float} (finite : t.isFinite = true) : (t != t) = false := by
  change t.toModel.unpack.isFinite = true at finite
  change (!t.toModel.unpack.beq t.toModel.unpack) = false
  cases u : t.toModel.unpack with
  | notANumber | infinity _ => simp [u, Float.Model.UnpackedFloat.isFinite] at finite
  | zero _ => rfl
  | finite s m e h =>
    cases s <;> simp [Float.Model.UnpackedFloat.beq, Float.Model.UnpackedFloat.compare]

private theorem advance_lookup (cfg : Config) (expFn : Float → Float) (ordered : (cfg.dt != cfg.dt) = false)
    (s : Park) (id : Int) (a : Action) :
    (advance cfg expFn s a).balls[id]? =
      (forBall id a).elim s.balls[id]? (localStep cfg expFn s.balls[id]?) := by
  cases a with
  | tick => exact tick_lookup cfg expFn ordered s id
  | edit target cmd =>
    by_cases h : target = id
    · subst target
      simp [advance, forBall, localStep]
    · simp [advance, forBall, h, Std.TreeMap.getElem?_alter]

private theorem execute_refines (cfg : Config) (expFn : Float → Float)
    (ordered : (cfg.dt != cfg.dt) = false) (s out : Park) (actions : List Action)
    (done : execute cfg expFn s actions = some out) :
    out.clock = s.clock + ticks actions ∧ ∀ id : Int,
      out.balls[id]? = (actions.filterMap (forBall id)).foldl (localStep cfg expFn) s.balls[id]? := by
  induction actions generalizing s with
  | nil =>
    simp [execute] at done
    subst out
    simp [ticks]
  | cons a rest ih =>
    simp only [execute] at done
    split at done
    · obtain ⟨clock, balls⟩ := ih _ done
      constructor
      · cases a <;> simpa [advance, Source.tick, ticks, Nat.add_assoc] using clock
      · intro id
        rw [balls, advance_lookup cfg expFn ordered]
        cases sel : forBall id a <;> simp [sel]
    · contradiction

/-- Equality of the entire ball record and clock for any ID and any finite admitted command
sequence, starting in any modeled park. Choosing a prefix gives intermediate observations. -/
theorem trace_refines (cfg : Config) (expFn : Float → Float) (s out : Park) (actions : List Action)
    (done : run cfg expFn s actions = some out) :
    out.clock = s.clock + ticks actions ∧ ∀ id : Int,
      out.balls[id]? = (actions.filterMap (forBall id)).foldl (localStep cfg expFn) s.balls[id]? := by
  unfold run at done
  split at done
  · rename_i allowed
    have finite : cfg.dt.isFinite = true := by
      simp only [configAllowed, Bool.and_eq_true] at allowed
      exact allowed.1.1.1.1
    exact execute_refines cfg expFn (finite_ordered finite) s out actions done
  · contradiction

private theorem execute_append (cfg : Config) (expFn : Float → Float) (s : Park) (xs ys : List Action) :
    execute cfg expFn s (xs ++ ys) =
      (execute cfg expFn s xs).bind (fun mid => execute cfg expFn mid ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons a rest ih => cases h : actionAllowed s a <;> simp [execute, h, ih]

/-- Resuming at any split produces the same result, including rejection of an inadmissible suffix. -/
theorem trace_append (cfg : Config) (expFn : Float → Float) (s : Park) (xs ys : List Action) :
    run cfg expFn s (xs ++ ys) = (run cfg expFn s xs).bind (fun mid => run cfg expFn mid ys) := by
  cases h : configAllowed cfg <;> simp [run, h, execute_append]

private theorem local_tick_valid {cfg : Config} {expFn : Float → Float} {b : Ball}
    (h : CacheValid cfg expFn b) : CacheValid cfg expFn (localTick cfg b) := by
  cases mode : b.mode <;> simpa [CacheValid, localTick, Source.commit, Source.prepare, mode, Source.decay]
    using h

private theorem local_cache (cfg : Config) (expFn : Float → Float) (events : List BallAction)
    (old : Option Ball) (valid : ∀ b ∈ old, CacheValid cfg expFn b) :
    ∀ b ∈ events.foldl (localStep cfg expFn) old, CacheValid cfg expFn b := by
  induction events generalizing old with
  | nil => exact valid
  | cons event rest ih =>
    apply ih
    cases event with
    | edit cmd => exact edit_preserves_cache cfg expFn cmd old valid
    | tick =>
      cases old with
      | none => simp [localStep]
      | some b =>
        simp only [localStep, Option.map_some, Option.mem_some_iff]
        intro _ rfl
        exact local_tick_valid (valid b rfl)

def ValidPark (cfg : Config) (expFn : Float → Float) (s : Park) : Prop :=
  ∀ (id : Int) (b : Ball), b ∈ s.balls[id]? → CacheValid cfg expFn b

theorem trace_preserves_cache (cfg : Config) (expFn : Float → Float) (s out : Park)
    (actions : List Action) (valid : ValidPark cfg expFn s)
    (done : run cfg expFn s actions = some out) : ValidPark cfg expFn out := by
  intro id b
  rw [(trace_refines cfg expFn s out actions done).2 id]
  exact local_cache cfg expFn _ _ (valid id) b

theorem cache_from_empty (cfg : Config) (expFn : Float → Float) (out : Park) (actions : List Action)
    (done : run cfg expFn {} actions = some out) : ValidPark cfg expFn out := by
  apply trace_preserves_cache cfg expFn {} out actions _ done
  intro id b
  simp

end ExactBallpark.Movement.Scalar
