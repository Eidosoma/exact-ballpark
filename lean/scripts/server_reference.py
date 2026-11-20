# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Extract ordered calls in two scalar server entry points. Reject unknown statements."""
import ast
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/exact_ballpark/server.py"
DEST = Path(__file__).resolve().parents[1] / "ExactBallparkProofs/Time/ServerReference.lean"


def calls(body, arg):
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise ValueError(f"Unsupported call statement {ast.unparse(stmt)}")
        call = stmt.value
        if (not isinstance(call.func, ast.Attribute) or ast.unparse(call.func.value) not in ("self", "self._park")
                or [ast.unparse(a) for a in call.args] != arg or call.keywords):
            raise ValueError(f"Review call receiver and arguments in {ast.unparse(call)}")
        result.append(ast.unparse(call.func))
    return json.dumps(result)


def extract():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    def method(cls, name, params):
        owner, = (c for c in module.body if isinstance(c, ast.ClassDef) and c.name == cls)
        fn, = (f for f in owner.body if isinstance(f, ast.FunctionDef) and f.name == name)
        if ast.unparse(fn.args) != params or fn.decorator_list:
            raise ValueError(f"Review signature of {cls}.{name}")
        return fn.body[1:] if ast.get_docstring(fn) is not None else fn.body

    tick = calls(method("BaseTicker", "tick", "self"), [])
    lookup, branch, remove = method("Actions", "remove_ball", "self, src_id")
    if (ast.unparse(lookup) != "ball = self._park.balls[src_id]" or not isinstance(branch, ast.If)
            or ast.unparse(branch.test) != "ball.isGlobal" or branch.orelse):
        raise ValueError("Review removal lookup and notification guard")
    notification = calls(branch.body, ["src_id"])
    removal = calls([remove], ["src_id"])
    return f"""-- SPDX-License-Identifier: MIT\n-- Copyright (c) 2026 Eidosoma\nmodule

/-! Generated from server.py by scripts/server_reference.py. Lists contain calls in execution order. -/
@[expose] public section
namespace ExactBallpark.Time.Reference

def preTick : List String := {tick}

def removeBall (isGlobal : Bool) : List String :=
  (if isGlobal then {notification} else []) ++ {removal}

end ExactBallpark.Time.Reference
"""


if __name__ == "__main__":
    DEST.write_text(extract(), encoding="utf-8")
