-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Control
public import ExactBallparkProofs.Movement.Guidance

@[expose] public section
namespace ExactBallpark.Collision
open ExactBallpark.Control ExactBallpark.Movement

inductive Search where | gradient | iterative | skip deriving DecidableEq

def method (enabled free massive live : Bool) : Search :=
  if !free || !live || !massive then .skip else if enabled then .iterative else .gradient

theorem gradient_iff (enabled free massive live : Bool) :
    method enabled free massive live = .gradient ↔ enabled = false ∧ free ∧ massive ∧ live := by
  cases enabled <;> cases free <;> cases massive <;> cases live <;> decide

structure Body where
  id : ℤ
  owner : ℤ
  target : ℤ
  mode : Mode
  bubble : ℤ
  global : Bool
  massive : Bool
  cloak : Bool
  junk : Bool
  deriving DecidableEq

/-- Candidate relation after broad phase. Ordinary missile target lookup bypasses its filters. -/
def candidate (master : Bool) (b o : Body) : Bool :=
  if b.mode = .missile ∧ 0 ≤ b.owner then o.id == b.target && o.bubble == b.bubble else
  !o.cloak && o.massive && (!master || o.bubble == b.bubble || o.global) &&
    (b.mode == .missile || o.mode != .missile)

theorem missile_target (master : Bool) (b o : Body) (h : b.mode = .missile) (owner : 0 ≤ b.owner) :
    candidate master b o ↔ o.id = b.target ∧ o.bubble = b.bubble := by simp [candidate, h, owner]

theorem ordinary_candidate (master : Bool) (b o : Body) (h : b.mode ≠ .missile) :
    candidate master b o ↔ o.cloak = false ∧ o.massive ∧
      (master = false ∨ o.bubble = b.bubble ∨ o.global) ∧ o.mode ≠ .missile := by
  simp [candidate, h, and_assoc, or_assoc]

def ignore (b o : Body) : Bool :=
  (b.mode == .missile && (o.id == b.owner || (o.mode == .miniball && o.owner == b.owner))) ||
  ((o.mode == .missile || o.mode == .mushroom) && b.id == o.owner) ||
  (o.mode == .missile && b.id == o.target) || (o.junk && !b.junk)

theorem owner_junk_skip (b o : Body) :
    (b.mode = .missile ∧ o.id = b.owner ∨ o.junk = true ∧ b.junk = false) → ignore b o := by
  rintro (⟨hm, hi⟩ | ⟨hj, hb⟩) <;> simp_all [ignore]

def freePair (mode : Mode) (id other : ℤ) := other > id ∨ mode = .missile

theorem unique_pair (a b : ℤ) (h : a ≠ b) (m n : Mode) (hm : m ≠ .missile) (hn : n ≠ .missile) :
    freePair m a b ↔ ¬freePair n b a := by simp [freePair, hm, hn]; omega

/-- Number of passes actually entered, given the next-frontier nonemptiness for each pass. -/
def passes (following : ℕ → Bool) (start : ℕ) : ℕ → ℕ
  | 0 => 0
  | budget+1 => 1 + if following start then passes following (start+1) budget else 0

private theorem passes_bound (f : ℕ → Bool) (s n : ℕ) : passes f s n ≤ n := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    have := ih (s+1)
    simp only [passes]; split <;> omega

theorem iteration_cap (f : ℕ → Bool) :
    1 ≤ passes f 0 20 ∧ passes f 0 20 ≤ 20 ∧ (f 0 = false → passes f 0 20 = 1) := by
  refine ⟨?_, passes_bound .., ?_⟩
  · change 1 ≤ 1 + _
    omega
  · intro h; simp [passes, h]

def orphan (bubble : ℤ) (owner : Option (ℤ × Bool)) :=
  match owner with
  | none => true
  | some (ob, cloak) => (ob != bubble && bubble != -1) || cloak

theorem orphan_with_owner (b ob : ℤ) (c : Bool) :
    orphan b (some (ob,c)) ↔ c = true ∨ (ob ≠ b ∧ b ≠ -1) := by
  simp [orphan, or_comm]

/-- Existing collision IDs include any accepted iterative locations in their recorded order. -/
def dispatch (mode : Mode) (moved master : Bool) (id bubble : ℤ)
    (owner : Option (ℤ × Bool)) (collisions : List ℤ) : List ℤ :=
  if !moved || mode != .missile then [] else
    (if master && orphan bubble owner then collisions ++ [id] else collisions)

theorem dispatch_moving (master : Bool) (id b : ℤ) (owner : Option (ℤ × Bool)) (cs : List ℤ) :
    dispatch .missile true master id b owner cs =
      cs ++ (if master && orphan b owner then [id] else []) := by
  simp [dispatch]; split <;> simp_all

theorem dispatch_silent (m : Mode) (moved master : Bool) (id b : ℤ)
    (owner : Option (ℤ × Bool)) (cs : List ℤ) (h : m ≠ .missile ∨ moved = false) :
    dispatch m moved master id b owner cs = [] := by
  rcases h with hm | hm <;> simp [dispatch, hm]

noncomputable def keepResponse (skip : Bool) (s last : ℝ) (a previous : Vec3) :=
  let scaled := (7656119366529843 / 9007199254740992 : ℝ) • a
  if !skip && decide (s ≥ last) then
    if s = last ∧ ¬ ‖scaled‖^2 > ‖previous‖^2 then previous else scaled
  else previous

theorem response_tie (t : ℝ) (a prev : Vec3) :
    ‖keepResponse false t t a prev‖ =
    max ‖prev‖ ‖(7656119366529843 / 9007199254740992 : ℝ) • a‖ := by
  simp only [keepResponse, ge_iff_le, le_refl, decide_true, Bool.not_false, Bool.true_and,
    ↓reduceIte, true_and]
  split
  next h => rw [max_eq_left]; nlinarith [norm_nonneg prev, norm_nonneg ((7656119366529843 / 9007199254740992 : ℝ) • a)]
  next h => push Not at h; rw [max_eq_right]; nlinarith [norm_nonneg prev, norm_nonneg ((7656119366529843 / 9007199254740992 : ℝ) • a)]

theorem lighter_boost (m₁ m₂ : ℝ) (h : 0 < m₁) : 25 < m₂ / m₁ ↔ 25 * m₁ < m₂ := lt_div_iff₀ h

end ExactBallpark.Collision
