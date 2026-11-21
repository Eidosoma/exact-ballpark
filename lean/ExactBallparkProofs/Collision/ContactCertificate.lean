-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.FirstContact
public import ExactBallparkProofs.Movement.PositionEnclosure

@[expose] public section
namespace ExactBallpark.Collision.ContactCertificate
open Set Movement PositionEnclosure FirstContact

structure Scene where
  a : Body
  b : Body
  radius : ℚ
  duration : ℚ

/-- Exact encoding of the binary64 near-zero-friction guard in the scalar source. -/
def guardCoefficient : ℚ := 7737125245533627 / 77371252455336267181195264

def Ordinary (b : Body) (T : ℚ) : Prop :=
  b.Valid ∧ guardCoefficient * b.mass * T ≤ b.friction

def Scene.Admitted (s : Scene) : Prop :=
  0 ≤ s.radius ∧ 0 ≤ s.duration ∧ Ordinary s.a s.duration ∧ Ordinary s.b s.duration

instance (b : Body) (T : ℚ) : Decidable (Ordinary b T) := inferInstanceAs (Decidable (_ ∧ _))
instance (s : Scene) : Decidable s.Admitted := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem ordinary_throughout {b : Body} {T : ℚ} (hb : Ordinary b T) {t : ℝ} (ht : t ≤ T) :
    ¬ (b.friction : ℝ) < threshold * b.mass * t := by
  have h : threshold * b.mass * (T:ℝ) ≤ (b.friction:ℝ) := by
    have hc : (guardCoefficient:ℝ) = threshold := by norm_num [guardCoefficient, threshold]
    rw [← hc]
    exact_mod_cast hb.2
  have hm : (0:ℝ) ≤ b.mass := by exact_mod_cast hb.1.1.le
  exact not_lt.mpr ((mul_le_mul_of_nonneg_left ht (mul_nonneg (by norm_num [threshold]) hm)).trans h)

noncomputable def Scene.distance (s : Scene) (t : ℝ) := dist (s.a.flight.pos t) (s.b.flight.pos t)

private theorem continuous_distance (s : Scene) (ha : s.Admitted) : Continuous s.distance :=
  (continuous_iff_continuousAt.mpr (fun t =>
    (DragChord.position_derivative _ (s.a.valid_flight ha.2.2.1.1) t).continuousAt)).dist
    (continuous_iff_continuousAt.mpr (fun t =>
      (DragChord.position_derivative _ (s.b.valid_flight ha.2.2.2.1) t).continuousAt))

def axisMin (b : Bounds) : ℚ := max 0 (max b.lo (-b.hi))
def axisMax (b : Bounds) : ℚ := max (-b.lo) b.hi

private theorem axis_distance {b : Bounds} {x : ℝ} (h : b.Contains x) :
    (axisMin b:ℝ) ≤ |x| ∧ |x| ≤ (axisMax b:ℝ) := by
  dsimp [axisMin, axisMax]; push_cast
  exact ⟨max_le (abs_nonneg x) (max_le (h.1.trans (le_abs_self _))
    ((neg_le_neg h.2).trans (neg_le_abs _))),
    abs_le.mpr ⟨by linarith [le_max_left (-(b.lo:ℝ)) (b.hi:ℝ), h.1],
      h.2.trans (le_max_right _ _)⟩⟩

def Scene.squaredBounds (s : Scene) (l h : ℚ) : Bounds :=
  let p := s.a.box l h
  let q := s.b.box l h
  let gap := fun i => (p i).sub (q i)
  let total := fun (f : Bounds → ℚ) => (f (gap 0))^2 + (f (gap 1))^2 + (f (gap 2))^2
  ⟨total axisMin, total axisMax⟩

theorem squared_distance_enclosed (s : Scene) (ha : s.Admitted) {l h : ℚ} (hl : 0 ≤ l)
    {t : ℝ} (ht : t ∈ Icc (l:ℝ) h) : (s.squaredBounds l h).Contains (s.distance t ^ 2) := by
  let gap := fun i => (s.a.box l h i).sub (s.b.box l h i)
  let x := fun i => s.a.flight.pos t i - s.b.flight.pos t i
  have each (i : Fin 3) : ((axisMin (gap i):ℝ))^2 ≤ (x i)^2 ∧
      (x i)^2 ≤ ((axisMax (gap i):ℝ))^2 := by
    have habs := axis_distance (Bounds.sub_contains
      (position_enclosed _ ha.2.2.1.1 hl ht i) (position_enclosed _ ha.2.2.2.1 hl ht i))
    have lower : (0:ℝ) ≤ axisMin (gap i) := by exact_mod_cast (le_max_left 0 _ : 0 ≤ axisMin (gap i))
    exact ⟨(sq_le_sq₀ lower (abs_nonneg (x i))).mpr habs.1 |>.trans_eq (sq_abs _),
      (sq_abs (x i)).symm.le.trans ((sq_le_sq₀ (abs_nonneg _) ((abs_nonneg _).trans habs.2)).mpr habs.2)⟩
  dsimp only [Bounds.Contains, Scene.squaredBounds]
  push_cast
  rw [Scene.distance, dist_eq_norm, EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  exact ⟨add_le_add (add_le_add (each 0).1 (each 1).1) (each 2).1,
    add_le_add (add_le_add (each 0).2 (each 1).2) (each 2).2⟩

def Scene.separated (s : Scene) (l h : ℚ) : Bool := decide (s.radius^2 < (s.squaredBounds l h).lo)
def Scene.overlap (s : Scene) (t : ℚ) : Bool := decide ((s.squaredBounds t t).hi ≤ s.radius^2)

/-- A whole-interval enclosure sometimes also gives a distance in metres.
Recheck the square before using the integer square-root approximation. -/
def Scene.margin (s : Scene) : Option ℚ :=
  let sq := (s.squaredBounds 0 s.duration).lo
  let lower := CurveClearance.lowerMetres sq
  if 0 ≤ lower ∧ lower^2 ≤ sq ∧ s.radius < lower then some (lower-s.radius) else none

theorem whole_interval_margin (s : Scene) (ha : s.Admitted) {m : ℚ} (ok : s.margin = some m)
    {t : ℝ} (ht : t ∈ Icc 0 (s.duration:ℝ)) : (s.radius:ℝ) + m ≤ s.distance t := by
  dsimp only [Scene.margin] at ok
  split at ok
  next h =>
    obtain ⟨h0, hsq, _⟩ := h
    obtain rfl := Option.some.inj ok
    have bound := (squared_distance_enclosed s ha (l := 0) le_rfl (by simpa using ht)).1
    have hs : (CurveClearance.lowerMetres (s.squaredBounds 0 s.duration).lo : ℝ)^2 ≤
        (s.squaredBounds 0 s.duration).lo := by
      exact_mod_cast hsq
    have lower := (sq_le_sq₀ (by exact_mod_cast h0) dist_nonneg).mp (hs.trans bound)
    change (CurveClearance.lowerMetres (s.squaredBounds 0 s.duration).lo : ℝ) ≤ s.distance t at lower
    push_cast
    linarith
  next => contradiction

private theorem separated_sound (s : Scene) (ha : s.Admitted) {l h : ℚ} (hl : 0 ≤ l)
    (accepted : s.separated l h = true) {t : ℝ} (ht : t ∈ Icc (l:ℝ) h) :
    (s.radius:ℝ) < s.distance t := by
  have hb : (s.radius:ℝ)^2 < ((s.squaredBounds l h).lo:ℝ) := by
    exact_mod_cast (of_decide_eq_true accepted : s.radius^2 < (s.squaredBounds l h).lo)
  have hh := hb.trans_le (squared_distance_enclosed s ha hl ht).1
  exact (sq_lt_sq₀ (by exact_mod_cast ha.1) dist_nonneg).mp hh

private theorem overlap_sound (s : Scene) (ha : s.Admitted) {t : ℚ} (ht : 0 ≤ t)
    (accepted : s.overlap t = true) : s.distance t ≤ (s.radius:ℝ) := by
  have hb : ((s.squaredBounds t t).hi:ℝ) ≤ (s.radius:ℝ)^2 := by
    exact_mod_cast (of_decide_eq_true accepted : (s.squaredBounds t t).hi ≤ s.radius^2)
  have hh := (squared_distance_enclosed s ha ht (show (t:ℝ) ∈ Icc (t:ℝ) t from ⟨le_rfl,le_rfl⟩)).2
  exact (sq_le_sq₀ dist_nonneg (by exact_mod_cast ha.1)).mp (hh.trans hb)

/-- Both halves must be certified. Adjacent slabs share their endpoints. -/
def cover (s : Scene) (l h : ℚ) (depth : Nat) : Bool :=
  if s.separated l h then true else
    match depth with
    | 0 => false
    | n+1 =>
      let mid := (l+h)/2
      cover s l mid n && cover s mid h n

theorem prefix_clear (s : Scene) (ha : s.Admitted) {depth : Nat} {l h : ℚ}
    (hl : 0 ≤ l) (order : l ≤ h) (accepted : cover s l h depth = true) :
    ∀ t ∈ Icc (l:ℝ) h, (s.radius:ℝ) < s.distance t := by
  induction depth generalizing l h with
  | zero =>
    simp [cover] at accepted
    exact fun _ ht => separated_sound s ha hl accepted ht
  | succ n ih =>
    by_cases sep : s.separated l h = true
    · exact fun _ ht => separated_sound s ha hl sep ht
    · simp [cover, sep] at accepted
      have left := ih hl (show l ≤ (l+h)/2 by linarith) accepted.1
      have right := ih (show 0 ≤ (l+h)/2 by linarith) (show (l+h)/2 ≤ h by linarith) accepted.2
      intro t ht
      by_cases side : t ≤ (((l+h)/2:ℚ):ℝ)
      · exact left t ⟨ht.1, side⟩
      · exact right t ⟨le_of_not_ge side, ht.2⟩

inductive Result where
  | clear
  | window (lo hi : ℚ)
  | unresolved
  deriving Repr, DecidableEq

/-- Only checked certificates become answers. A positive witness at hi is
insufficient on its own to claim a late first contact. -/
def verify (s : Scene) (depth : Nat) (proposal : Result) : Result :=
  if s.Admitted then
    match proposal with
    | .clear => if cover s 0 s.duration depth then .clear else .unresolved
    | .window l h =>
      if 0 ≤ l && l ≤ h && h ≤ s.duration && (l == 0 || cover s 0 l depth) && s.overlap h then
        .window l h else .unresolved
    | .unresolved => .unresolved
  else .unresolved

def Result.Sound (s : Scene) : Result → Prop
  | .clear => ∀ t ∈ Icc 0 (s.duration:ℝ), (s.radius:ℝ) < s.distance t
  | .window l h => ∃ t ∈ Icc (l:ℝ) h, First s.distance s.radius s.duration t
  | .unresolved => True

theorem verified_result (s : Scene) (depth : Nat) (proposal : Result) :
    (verify s depth proposal).Sound s := by
  by_cases ha : s.Admitted
  · cases proposal with
    | unresolved => simp [verify, ha, Result.Sound]
    | clear =>
      simp only [verify, ite_eq_left ha]
      split
      next h => simpa [Result.Sound] using prefix_clear s ha (l := 0) le_rfl ha.2.1 h
      next => trivial
    | window l h =>
      simp only [verify, ite_eq_left ha]
      split
      next check =>
        simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq] at check
        obtain ⟨⟨⟨⟨hl, lh⟩, hT⟩, pref⟩, hit⟩ := check
        apply first_in_window (continuous_distance s ha).continuousOn
          (by exact_mod_cast hl) (by exact_mod_cast lh) (by exact_mod_cast hT) _
          (overlap_sound s ha (hl.trans lh) hit)
        intro t ht
        rcases pref with zero | yes
        · subst l; simp at ht
        · exact prefix_clear s ha le_rfl hl yes t ⟨by simpa using ht.1, ht.2.le⟩
      next => trivial
  · simp [verify, ha, Result.Sound]

/-- Search the grid left to right, pruning whole clear slabs. An undecided leaf
ends the search because it can hide an earlier grazing contact. -/
def propose (s : Scene) (step : ℚ) (i j : Nat) (fuel : Nat) : Result :=
  let l := (i:ℚ)*step
  let h := (j:ℚ)*step
  if s.separated l h then .clear
  else if j ≤ i+1 then
    if s.overlap h then .window l h else .unresolved
  else match fuel with
    | 0 => .unresolved
    | n+1 =>
      let mid := (i+j)/2
      match propose s step i mid n with
      | .clear => propose s step mid j n
      | answer => answer

def classify (s : Scene) (cells : Nat) : Result :=
  if ¬ s.Admitted ∨ cells = 0 then .unresolved
  else
    let depth := cells.log2 + 1
    let guess := if s.overlap 0 then Result.window 0 0
      else propose s (s.duration/cells) 0 cells depth
    verify s depth guess

theorem classify_sound (s : Scene) (cells : Nat) : (classify s cells).Sound s := by
  unfold classify
  split
  · trivial
  · exact verified_result _ _ _

end ExactBallpark.Collision.ContactCertificate
