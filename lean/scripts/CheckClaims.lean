-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ProofAudit
import ExactBallparkProofs

open Lean Elab Command

/- Run after building the library, so names, declaration kinds, modules and axiom
closures come from the imported Lean environment. -/
run_cmd do
  let env ← getEnv
  if env.header.moduleNames.any (`RealPhysics).isPrefixOf then
    throwError "The engine library imports experimental physics"
  ProofAudit.checkLibrary `ExactBallparkProofs "ExactBallparkProofs"
  for row in ← ProofAudit.registered `ExactBallparkProofs "claims.json" do
    IO.println row.compress
