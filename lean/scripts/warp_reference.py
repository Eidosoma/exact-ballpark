# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Extract the straight-line real interpretation of ModeBallpark.warp_constants."""
import ast
import operator
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src/exact_ballpark/modes.py"
DEST = Path(__file__).resolve().parents[1] / "ExactBallparkProofs/Warp/Reference.lean"
OPS = {ast.Add: ("+", operator.add), ast.Sub: ("-", operator.sub),
       ast.Mult: ("*", operator.mul), ast.Div: ("/", operator.truediv)}


def literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)][1](literal(node.left), literal(node.right))
    raise ValueError(f"Unsupported module constant: {ast.unparse(node)}")


def number(value):
    n, d = value.as_integer_ratio()
    return str(n) if d == 1 else f"({n} / {d})"


def extract():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls, = (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ModeBallpark")
    func, = (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "warp_constants")
    params = [p.arg for p in func.args.args]
    if (params != ["warp_factor", "warp_distance"] or [ast.unparse(d) for d in func.decorator_list] != ["staticmethod"]
            or func.args.posonlyargs or func.args.kwonlyargs or func.args.kwarg or func.args.vararg or func.args.defaults):
        raise ValueError("Review warp_constants signature and its source correspondence")
    constants = {s.targets[0].id: s.value for s in module.body if isinstance(s, ast.Assign)
                 and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)}
    names = set(params)

    def expr(node):
        if isinstance(node, ast.Name):
            return node.id if node.id in names else number(literal(constants[node.id]))
        if isinstance(node, ast.Constant):
            return number(literal(node))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return f"({expr(node.left)} {OPS[type(node.op)][0]} {expr(node.right)})"
        if isinstance(node, ast.Call) and not node.keywords:
            name = ast.unparse(node.func)
            if (name, len(node.args)) in (("min", 2), ("math.log", 1)):
                fn = "min" if name == "min" else "Real.log"
                return "(" + " ".join([fn, *map(expr, node.args)]) + ")"
        raise ValueError(f"Unsupported warp_constants expression: {ast.unparse(node)}")

    body = func.body[1:] if ast.get_docstring(func) is not None else func.body
    result = body[-1]
    if not isinstance(result, ast.Return) or not isinstance(result.value, ast.Tuple) or len(result.value.elts) != 9:
        raise ValueError("Expected nine warp constants")
    lines = ["-- SPDX-License-Identifier: MIT\n-- Copyright (c) 2026 Eidosoma\nmodule", "", "public import Mathlib.Analysis.SpecialFunctions.Log.Basic", "",
             "/-! Generated from modes.py by scripts/warp_reference.py. -/",
             "@[expose] public noncomputable section", "", "namespace ExactBallpark.Warp.Reference", "",
             "def constants (" + " ".join(params) + " : ℝ) : " + " × ".join(["ℝ"] * 9) + " :="]
    for stmt in body[:-1]:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise ValueError(f"Unsupported warp_constants statement: {ast.unparse(stmt)}")
        name = stmt.targets[0].id
        lines.append(f"  let {name} : ℝ := {expr(stmt.value)}")
        names.add(name)
    lines += ["  (" + ", ".join(map(expr, result.value.elts)) + ")", "", "end ExactBallpark.Warp.Reference", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    DEST.write_text(extract(), encoding="utf-8")
