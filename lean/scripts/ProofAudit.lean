-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Lean

open Lean Elab Command

namespace ProofAudit

private structure Claim where
  moduleName : String
  theorems : Array String
  deriving FromJson

private structure Registry where
  claims : Array Claim
  deriving FromJson

/-- Audit the imported modules under one library root, including unregistered theorems. -/
public def checkLibrary (root : Name) (dir : System.FilePath) : CommandElabM Unit := do
  let env ← getEnv
  for path in ← dir.walkDir do
    if path.extension == some "lean" then
      let suffix := (path.withExtension "").components.drop dir.components.length
      let modName := suffix.foldl Name.str root
      unless (env.getModuleIdx? modName).isSome do
        throwError "Proof module {modName} is missing from the root import graph"
  let allowed := #[`propext, `Quot.sound, `Classical.choice]
  for (name, info) in env.constants do
    match info with
    | .thmInfo _ | .axiomInfo _ => pure ()
    | _ => continue
    let some idx := env.getModuleIdxFor? name | continue
    unless root.isPrefixOf env.header.moduleNames[idx]! do continue
    if info matches .axiomInfo _ then throwError "Project introduces axiom {name}"
    for ax in ← collectAxioms name do
      unless allowed.contains ax do
        throwError "{name} depends on disallowed axiom {ax}"

/-- Resolve each recorded name to a theorem in the specified library and module. -/
public def registered (root : Name) (file : System.FilePath) : CommandElabM (Array Json) := do
  let json ← IO.ofExcept <| Json.parse (← IO.FS.readFile file)
  let registry : Registry ← IO.ofExcept <| fromJson? json
  let env ← getEnv
  let mut seen : NameSet := {}
  let mut rows := #[]
  for claim in registry.claims do
    for theoremName in claim.theorems do
      let name := theoremName.toName
      if seen.contains name then throwError "Duplicate registered theorem {name}"
      seen := seen.insert name
      unless (env.find? name).any (· matches .thmInfo _) do
        throwError "{name} is missing or is not a theorem"
      let some idx := env.getModuleIdxFor? name | throwError "{name} has no module"
      let actual := env.header.moduleNames[idx]!
      unless root.isPrefixOf actual && actual == claim.moduleName.toName do
        throwError "{name} belongs to {actual}, expected {claim.moduleName}"
      let axioms ← collectAxioms name
      rows := rows.push <| Json.mkObj [
        ("theorem", toJson theoremName), ("axioms", toJson (axioms.map toString))]
  if rows.isEmpty then throwError "Empty proof registry {file}"
  return rows

end ProofAudit
