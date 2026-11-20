-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

@[expose] public section
namespace ExactBallpark.Numerics

/-- Five state flags followed by the box, sphere and capsule presence bits. -/
def stateFlags (free global massive interactive junk box sphere capsule : Bool) : UInt8 :=
  (if free then 1 else 0) ||| (if global then 2 else 0) ||| (if massive then 4 else 0) |||
  (if interactive then 8 else 0) ||| (if junk then 16 else 0) ||| (if box then 32 else 0) |||
  (if sphere then 64 else 0) ||| (if capsule then 128 else 0)

theorem flags_roundtrip : ∀ free global massive interactive junk box sphere capsule : Bool,
    let flags := stateFlags free global massive interactive junk box sphere capsule
    ((flags &&& 1 != 0), (flags &&& 2 != 0), (flags &&& 4 != 0), (flags &&& 8 != 0),
      (flags &&& 16 != 0), (flags &&& 32 != 0), (flags &&& 64 != 0), (flags &&& 128 != 0)) =
    (free, global, massive, interactive, junk, box, sphere, capsule) := by
  intro f g m i j b s c
  cases f <;> cases g <;> cases m <;> cases i <;> cases j <;> cases b <;> cases s <;> cases c <;> rfl

end ExactBallpark.Numerics
