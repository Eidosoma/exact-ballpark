-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Movement.Correspondence

@[expose] public section
namespace ExactBallpark.Movement.TensorLane

/-- The cached full-tick expression in _integrate_dt, also used with a selected
factor by _integrate_t. Both candidate positions are computed before selection. -/
def atFactor (p v a m k f t : Float) : Float × Float :=
  let k2 := k*k
  let one := Float.ofBits 0x3ff0000000000000
  let regular := (m*(m*a*(f-one) + k*(a*t+v-f*v)) + p*k2)*(one/k2)
  let special := (m*(a*(-k*t) + k*(a*t+v*k*(one/m)*t)) + p*k2)*(one/k2)
  let small := k < Float.ofBits 0x3ddb7cdfd9d7bdbb*m*t
  let pn := if small then special else regular
  let vn := (m*a-(m*a-v*k)*f)*(one/k)
  (pn,vn)

/-- One coordinate of the variable-duration PyTorch lane. -/
def step (p v a m k cache t dt partialFactor : Float) : Float × Float :=
  let f := if t != dt then partialFactor else cache
  let (pn,vn) := atFactor p v a m k f t
  let zero := t == Float.ofBits 0
  (if zero then p else pn, if zero then v else vn)

/-- The cached word can be arbitrary. A partial tick only needs its own supplied
factor to agree. Signed zero and the special position formula use the source masks. -/
theorem coordinate_words (expFn : Float → Float) (p v a m k cache t dt partialFactor : Float)
    (factor : (t == Float.ofBits 0) = false → (t != dt) = true →
      partialFactor = expFn ((-k/m)*t)) :
    let out := step p v a m k cache t dt partialFactor
    let ref := Reference.binary64 expFn p v a m k cache t dt
    let full := atFactor p v a m k cache dt
    let fullRef := Reference.binary64 expFn p v a m k cache dt dt
    (out.1.toBits,out.2.toBits) = (ref.1.toBits,ref.2.toBits) ∧
    ((dt == Float.ofBits 0) = false → (dt != dt) = false →
      (full.1.toBits,full.2.toBits) = (fullRef.1.toBits,fullRef.2.toBits)) := by
  constructor
  · cases hz : t == Float.ofBits 0 <;> cases hf : t != dt <;>
      simp [step,atFactor,Reference.binary64,hz,hf]
    simp [factor hz hf]
  · intro nonzero ordered
    simp [atFactor,Reference.binary64,nonzero,ordered]

end ExactBallpark.Movement.TensorLane
