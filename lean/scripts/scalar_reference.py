# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Extract the successful scalar GOTO/STOP tick and cache calculations from ref.py."""
import ast
import struct
import textwrap
from pathlib import Path

from integrator_reference import SOURCE

DEST = Path(__file__).resolve().parents[1] / "ExactBallparkProofs/Movement/ScalarReference.lean"


def capture(fn, args, pattern):
    """Match the ordered statements, with EXPR_ names marking translated expressions."""
    if [a.arg for a in fn.args.args] != args.split() or fn.decorator_list or any(
            (fn.args.posonlyargs, fn.args.kwonlyargs, fn.args.defaults, fn.args.vararg, fn.args.kwarg)):
        raise ValueError(f"Review {fn.name} signature")
    actual = fn.body[1:] if ast.get_docstring(fn, clean=False) is not None else fn.body
    fields = {}

    def matches(expected, value):
        if isinstance(expected, ast.Name) and expected.id.startswith("EXPR_"):
            if not isinstance(value, ast.expr) or expected.id in fields:
                return False
            fields[expected.id] = value
            return True
        if isinstance(expected, ast.AST):
            return type(expected) is type(value) and all(
                matches(field, getattr(value, name)) for name, field in ast.iter_fields(expected))
        if isinstance(expected, list):
            return len(expected) == len(value) and all(matches(x, y) for x, y in zip(expected, value))
        return type(expected) is type(value) and expected == value

    if not matches(ast.parse(textwrap.dedent(pattern)).body, actual):
        raise ValueError(f"Review {fn.name} control flow, state access or calls")
    return fields


def expr(node, bindings):
    name = ast.unparse(node)
    if name in bindings:
        return bindings[name]
    if isinstance(node, ast.Constant) and type(node.value) is float:
        bits = int.from_bytes(struct.pack(">d", node.value), "big")
        return f"(Float.ofBits 0x{bits:016x})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{expr(node.operand, bindings)})"
    if isinstance(node, ast.BinOp):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
        if op:
            return f"({expr(node.left, bindings)} {op} {expr(node.right, bindings)})"
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = {ast.Eq: "==", ast.Lt: "<", ast.LtE: "≤"}.get(type(node.ops[0]))
        if op:
            return f"({expr(node.left, bindings)} {op} {expr(node.comparators[0], bindings)})"
    if isinstance(node, ast.IfExp):
        return f"(if {expr(node.test, bindings)} then {expr(node.body, bindings)} else {expr(node.orelse, bindings)})"
    if isinstance(node, ast.Subscript) and ast.unparse(node.slice) in ("0", "1", "2", "i"):
        return f"({expr(node.value, bindings)} {ast.unparse(node.slice)})"
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        fn = {"math.exp": "expFn", "math.sqrt": "Float.sqrt", "_normalized": "normalized", "float": ""}.get(
            ast.unparse(node.func))
        if fn is not None:
            value = expr(node.args[0], bindings)
            return f"({fn} {value})" if fn else value
    if isinstance(node, ast.List) and len(node.elts) == 3:
        return "(vec3 " + " ".join(expr(x, bindings) for x in node.elts) + ")"
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)) and len(node.generators) == 1:
        gen = node.generators[0]
        if isinstance(gen.target, ast.Name) and not gen.ifs and not gen.is_async:
            var, source = gen.target.id, ast.unparse(gen.iter)
            if (var, source) == ("i", "range(3)"):
                return f"(Vec.ofFn fun i => {expr(node.elt, bindings)})"
            if (var, source) in (("c", "unit"), ("c", "b.last_g")):
                return f"(Vec.ofFn fun i => {expr(node.elt, bindings | {var: f'({bindings[source]} i)'})})"
    raise ValueError(f"Unsupported scalar expression {name}")


def extract():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    normalized, = (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_normalized")
    park, = (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Ballpark")
    methods = {n.name: n for n in park.body if isinstance(n, ast.FunctionDef)}
    constants = {ast.unparse(n.targets[0]): n.value for n in module.body if isinstance(n, ast.Assign)}
    if ast.literal_eval(constants["(STOP, GOTO)"]) != (2, 0):
        raise ValueError("Review scalar mode IDs")
    fields = capture(normalized, "v", """
        norm = EXPR_norm
        if EXPR_zero:
            return list(v)
        inv = EXPR_inv
        return EXPR_normalized
    """)
    fields.update(capture(methods["_refresh_time_factor"], "self b", """
        m = EXPR_mass
        b.time_factor = EXPR_factor
    """))
    fields.update(capture(methods["goto_direction"], "self bid x y z", """
        b = self.balls[bid]
        d = _normalized([float(x), float(y), float(z)])
        self.goto_point(bid, *EXPR_target)
    """))
    fields.update(capture(methods["goto_thrust"], "self b", """
        a = EXPR_delta
        length2 = EXPR_length2
        dist = EXPR_dist
        max_thrust = EXPR_force
        unit = EXPR_unit
        if EXPR_guard:
            coff = EXPR_coff
            return EXPR_taper
        return EXPR_full
    """))
    fields.update(capture(methods["evolve"], "self", """
        order = sorted(self.balls)
        for bid in order:
            b = self.balls[bid]
            b.last_g = [0.0, 0.0, 0.0]
            if b.mode == GOTO:
                b.last_g = self.goto_thrust(b)
            elif b.mode == STOP:
                b.vel[1] = EXPR_damped
        new = {}
        for bid in order:
            b = self.balls[bid]
            a = EXPR_accel
            new[bid] = self.integrate(b.pos, b.vel, a, EXPR_effective, self.friction, b.time_factor, self.dt)
        for bid in order:
            self.balls[bid].pos, self.balls[bid].vel = new[bid]
        self.current_time += 1
    """))
    bindings = {n: n for n in "v norm inv m d a length2 dist unit coff".split()}
    bindings.update({"self.dt": "cfg.dt", "self.friction": "cfg.friction", "b.goto": "b.target",
                     "b.pos": "b.pos", "b.vel": "b.vel", "b.last_g": "b.lastG", "b.mass": "b.mass",
                     "b.agility": "b.agility", "b.max_vel": "b.maxVel", "b.speed_fraction": "b.speedFraction",
                     "max_thrust": "maxThrust", "STOP_DAMP": expr(constants["STOP_DAMP"], {})})
    e = {key.removeprefix("EXPR_"): expr(node, bindings) for key, node in fields.items()}
    return f"""-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarState
public import ExactBallparkProofs.Movement.Reference

/-! Generated by scripts/scalar_reference.py from the scalar reference. -/
@[expose] public section
namespace ExactBallpark.Movement.Scalar.Source

def normalized (v : Vec) : Vec :=
  let norm := {e['norm']}
  if {e['zero']} then v else
  let inv := {e['inv']}
  {e['normalized']}

def decay (cfg : Config) (expFn : Float → Float) (b : Ball) : Float :=
  let m := {e['mass']}
  {e['factor']}

def directionTarget (b : Ball) (direction : Vec) : Vec :=
  let d := normalized direction
  {e['target']}

def thrust (cfg : Config) (b : Ball) : Vec :=
  let a : Vec := {e['delta']}
  let length2 := {e['length2']}
  let dist := {e['dist']}
  let maxThrust := {e['force']}
  let unit := {e['unit']}
  if {e['guard']} then
    let coff := {e['coff']}
    {e['taper']}
  else {e['full']}

def prepare (cfg : Config) (b : Ball) : Ball :=
  let b := {{ b with lastG := zeros }}
  match b.mode with
  | .goto => {{ b with lastG := thrust cfg b }}
  | .stop => {{ b with vel := vec3 (b.vel 0) {e['damped']} (b.vel 2) }}

def coordinates (cfg : Config) (expFn : Float → Float) (b : Ball) : Fin 3 → Float × Float :=
  let a : Vec := {e['accel']}
  fun i => Reference.binary64 expFn (b.pos i) (b.vel i) (a i) {e['effective']}
    cfg.friction b.timeFactor cfg.dt cfg.dt

def commit (b : Ball) (next : Motion) : Ball :=
  {{ b with pos := next.pos, vel := next.vel }}

/-- Each map represents a pass over the same ordered IDs. The commit reads the staged result by ID. -/
def tick (cfg : Config) (expFn : Float → Float) (s : Park) : Park :=
  let prepared := s.balls.map fun _ b => prepare cfg b
  let next := prepared.map fun _ b => Motion.ofFn (coordinates cfg expFn b)
  let balls := prepared.map fun id b => (next[id]?).elim b (commit b)
  ⟨balls, s.clock + 1⟩

end ExactBallpark.Movement.Scalar.Source
"""


if __name__ == "__main__":
    DEST.write_text(extract(), encoding="utf-8")
