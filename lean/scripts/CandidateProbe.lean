-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Space.CandidateOrder

open ExactBallpark.Space.Filing ExactBallpark.Space.CandidateOrder

private def side : Nat → Option Side
  | 0 => some .low | 1 => some .stay | 2 => some .high | _ => none

private def body (input : String) : Option (Nat × Footprint) := do
  let [id,level,x,y,z,i,j,k,r,loX,hiX,pX,loY,hiY,pY,loZ,hiZ,pZ] ←
    (input.splitOn ",").mapM String.toNat? | none
  let faceAt := fun lo hi p => face (Float.ofBits lo.toUInt64) (Float.ofBits hi.toUInt64)
    (Float.ofBits r.toUInt64) (Float.ofBits p.toUInt64)
  let f : Footprint := ⟨(x,y,z),level,faceAt loX hiX pX,faceAt loY hiY pY,faceAt loZ hiZ pZ⟩
  guard (f.x == (← side i) && f.y == (← side j) && f.z == (← side k))
  return (id,f)

private def csv (xs : List Nat) := String.intercalate "," (xs.map toString)

def main (worlds : List String) : IO Unit := do
  for world in worlds do
    let some all := (world.splitOn ";").mapM body | throw (IO.userError "Invalid filing")
    for (id,query) in all do
      if h : query.level < 8 then
        let others := all.filter (fun b => b.1 != id)
        let s := occurrences query h others
        let relatedIds := (others.filter (fun b => related query b.2)).map Prod.fst
        let minima := (batchWinners (s.image pack)).sort
        let keys := minima.map (fun x => (ofLex x).1)
        IO.println s!"{id}|{csv relatedIds}|{csv (scalarOrder s)}|{csv (batchOrder s)}|{csv keys}"
      else throw (IO.userError "Invalid filing level")
