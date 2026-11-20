-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import Lean
import ExactBallparkProofs

open Lean Elab Command

private structure Claim where
  moduleName : String
  theorems : Array String
  deriving FromJson

private structure Registry where
  claims : Array Claim
  deriving FromJson

/- Run after building the library, so names, declaration kinds, modules and axiom
closures come from the imported Lean environment. -/
run_cmd do
  let json ← IO.ofExcept <| Json.parse (← IO.FS.readFile "claims.json")
  let registry : Registry ← IO.ofExcept <| fromJson? json
  let env ← getEnv
  let allowed := #[`propext, `Quot.sound, `Classical.choice]
  for path in ← (System.FilePath.mk "ExactBallparkProofs").walkDir do
    if path.extension == some "lean" then
      let modName := (String.intercalate "." (path.withExtension "").components).toName
      unless (env.getModuleIdx? modName).isSome do
        throwError "Proof module {modName} is missing from the root import graph"
  -- Include library theorems even before there is a claim page entry for them.
  for (name, info) in env.constants do
    match info with
    | .thmInfo _ | .axiomInfo _ => pure ()
    | _ => continue
    let some idx := env.getModuleIdxFor? name | continue
    unless (`ExactBallparkProofs).isPrefixOf env.header.moduleNames[idx]! do continue
    if info matches .axiomInfo _ then throwError "Project introduces axiom {name}"
    for ax in ← collectAxioms name do
      unless allowed.contains ax do
        throwError "{name} depends on disallowed axiom {ax}"
  for claim in registry.claims do
    for theoremName in claim.theorems do
      let name := theoremName.toName
      unless (env.find? name).any (· matches .thmInfo _) do
        throwError "{name} is missing or is not a theorem"
      let some idx := env.getModuleIdxFor? name | throwError "{name} has no module"
      let actual := env.header.moduleNames[idx]!
      unless (`ExactBallparkProofs).isPrefixOf actual && actual == claim.moduleName.toName do
        throwError "{name} belongs to {actual}, expected {claim.moduleName}"
      let axioms ← collectAxioms name
      IO.println <| Json.compress <| Json.mkObj [
        ("theorem", toJson theoremName), ("axioms", toJson (axioms.map toString))]
