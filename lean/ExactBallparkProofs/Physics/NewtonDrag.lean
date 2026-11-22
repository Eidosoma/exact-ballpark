-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.DragChord
public import Physlib.ClassicalMechanics.PointParticle.NewtonianSystem.Basic

@[expose] public noncomputable section
namespace ExactBallpark.Physics.NewtonDrag
open ClassicalMechanics ClassicalMechanics.ReferenceFrame ClassicalMechanics.PointParticle
open Movement Movement.DragChord
open scoped Classical

variable (frame : ReferenceFrame 3) [Fact frame.IsInertial] (f : Flight) (valid : f.Valid)

/-- Express engine displacements on the fixed axes of the inertial frame. -/
def coordinates := (Vector.contDispEquiv (frame := frame) 0).symm

include valid in
private theorem position_rate (t : ℝ) :
    HasDerivAt (fun s => coordinates frame (f.pos s)) (coordinates frame (f.vel t)) t :=
  (coordinates frame).toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t (position_derivative f valid t)

include valid in
private theorem velocity_rate (t : ℝ) :
    HasDerivAt (fun s => coordinates frame (f.vel s)) (coordinates frame (f.forceField (f.vel t))) t :=
  (coordinates frame).toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t
    (velocity3_hasDerivAt f.initialVelocity f.applied valid.1 valid.2 t)

def particle : frame.Particle where
  mass := ⟨f.mass, valid.1⟩
  pos t := coordinates frame (f.pos t)
  pos_twice_differentiable _ := by
    refine ⟨fun t => (position_rate frame f valid t).differentiableAt, ?_⟩
    rw [show deriv (fun s => coordinates frame (f.pos s)) = (fun s => coordinates frame (f.vel s)) from
      funext (fun t => (position_rate frame f valid t).deriv)]
    exact fun t => (velocity_rate frame f valid t).differentiableAt

private theorem particle_velocity :
    (particle frame f valid).velocity = fun t => coordinates frame (f.vel t) :=
  funext fun t => (position_rate frame f valid t).deriv

/-- The translated ordinary position is the trajectory of this Physlib particle. -/
theorem source_particle_position {dt cached t : ℝ}
    (ordinary : ¬ f.friction < threshold * f.mass * t)
    (cache : t = dt → cached = decay f.mass f.friction dt) :
    coordinates frame (WithLp.toLp 2 fun i : Fin 3 =>
      (Reference.real (f.start i) (f.initialVelocity i) (f.applied i)
        f.mass f.friction cached t dt).1) = (particle frame f valid).pos t := by
  rw [extracted_position f valid ordinary cache]
  rfl

/-- The applied force minus drag equals mass times the second derivative of position. -/
theorem force_balance (t : ℝ) :
    coordinates frame (f.mass • f.applied) - f.friction • (particle frame f valid).velocity t =
      (particle frame f valid).mass.val • (particle frame f valid).acceleration t := by
  change _ = f.mass • deriv (particle frame f valid).velocity t
  rw [particle_velocity, (velocity_rate frame f valid t).deriv]
  rw [← map_smul, ← map_sub, ← map_smul]
  congr 1
  rw [Flight.forceField, smul_sub, smul_smul, mul_comm f.mass, div_mul_cancel₀ _ valid.1.ne']

/-- One particle, a prescribed force and a separate velocity-dependent drag force. -/
def system : NewtonianSystem 3 := by
  let ball := particle frame f valid
  let balls : Multiset frame.Particle := {ball}
  let id : balls := ⟨ball, ⟨0, by simp [balls]⟩⟩
  let applied : frame.Force balls := ⟨fun _ => coordinates frame (f.mass • f.applied), id⟩
  let drag : frame.Force balls := ⟨fun t => -(f.friction • ball.velocity t), id⟩
  let forces : Multiset (frame.Force balls) := {applied, drag}
  have one : Subsingleton balls := Fintype.card_le_one_iff_subsingleton.mp (by simp [balls])
  refine { frame := frame, particles := balls, internalForces := 0, externalForces := forces,
           newton_third_law := by simp, newton_second_law := ?_ }
  intro p
  obtain rfl := one.elim p id
  funext t
  simp only [netForce, Multiset.map_zero, one.elim _ id, Finset.filter_true]
  change ((Finset.univ : Finset forces).val.map (fun x => x.1.value t)).sum = _
  rw [Multiset.map_univ forces (fun force => force.value t)]
  change (forces.map (fun force => force.value t)).sum = ball.mass.val • ball.acceleration t
  simpa [forces, applied, drag, id, ball, sub_eq_add_neg] using force_balance frame f valid t

/-- Physlib counts exactly the moving particle and its two external forces. -/
theorem system_inventory :
    (system frame f valid).particles = ({particle frame f valid} : Multiset frame.Particle) ∧
    (system frame f valid).internalForces.card = 0 ∧
    (system frame f valid).externalForces.card = 2 := by
  exact ⟨rfl, rfl, rfl⟩

/-- For balanced forces the source trajectory is uniform in both position branches,
even when the supplied decay factor differs from the exact exponential. -/
theorem balanced_source_step {m k a v : ℝ} (hm : 0 < m) (hk : 0 < k)
    (balance : m * a = k * v) (p cached t dt : ℝ) :
    Reference.real p v a m k cached t dt = (p + v * t, v) := by
  rw [extracted_integrator]
  by_cases ht : t = 0
  · simp [ht, integrate_zero]
  simp only [integrate, ite_eq_right ht]
  have ha : a = k * v / m := (eq_div_iff hm.ne').2 (by simpa [mul_comm] using balance)
  subst a
  split
  · rw [special_position_eq hm.ne' hk.ne']
    congr 1
    unfold sourceVelocity
    field_simp
    ring
  · apply Prod.ext <;> dsimp [regularPosition, sourceVelocity] <;> field_simp <;> ring

end ExactBallpark.Physics.NewtonDrag
