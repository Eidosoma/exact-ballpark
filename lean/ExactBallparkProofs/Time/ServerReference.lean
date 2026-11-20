-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

/-! Generated from server.py by scripts/server_reference.py. Lists contain calls in execution order. -/
@[expose] public section
namespace ExactBallpark.Time.Reference

def preTick : List String := ["self._finalize_tick", "self._increment_timestamp", "self._flush_history", "self._pre_update_bubbles", "self._update_bubbles", "self._post_update_bubbles"]

def removeBall (isGlobal : Bool) : List String :=
  (if isGlobal then ["self._notify_remove_global_ball"] else []) ++ ["self._park._parent_RemoveBall"]

end ExactBallpark.Time.Reference
