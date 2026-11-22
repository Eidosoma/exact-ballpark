# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Read and check the pinned source revisions used by the CPU replays."""
import ast
import hashlib
import json
import re
import subprocess


# Each citation must cover its operation at the pinned line, even without a Destiny clone.
DESTINY_OPERATIONS = {
    "Capsule motion cutoff": ("src/Capsule.cpp", 146, "if (a < 1e-10)"),
    "Client tail append": ("python/destiny/net/client/_util.py", 68, "time_list[tl_idx]], wait_for_bubble]"),
    "Client admission": ("python/destiny/net/client/_ticker.py", 74, "if entry_time < self._latest_set_state_time:"),
    "Client wait guard": ("python/destiny/net/client/_ticker.py", 95, "if wait_for_bubble:"),
    "Update order": ("python/destiny/net/client/_baseticker.py", 57, "self._report_update_if_bad(state)"),
    "Mixed-stamp diagnostic": ("python/destiny/net/client/_baseticker.py", 117, "self._error_handler.on_fatal_desync()"),
    "Application callback": ("python/destiny/net/client/_interface.py", 104, "@abc.abstractmethod"),
    "Merge rationale": ("python/destiny/net/client/_util.py", 23, "indefinitely blocking subsequent updates."),
    "Batch count per tick": ("python/destiny/net/server/_parkupdatebatcher.py", 60, "each client should expect per tick"),
    "Server stamp check": ("python/destiny/net/server/_parkupdatebatcher.py", 128,
                           "if entry[0] != self._park.currentTime and entry[0] > 0:"),
    "Server’s per-recipient wait flag": ("python/destiny/net/server/_parkupdatebatcher.py", 108,
                                        "wait_for_bubble = client_id in clients_waiting_for_bubble"),
    "Tracking subtraction": ("src/Ballpark.cpp", 5207, "combined.LengthSq() - radialSpeed*radialSpeed"),
    "Fixed-ball guard": ("src/Ballpark.cpp", 2158, "timeOfImpact < collisionItem.timeOfImpact"),
    "Static-shape comparison": ("src/Ballpark.cpp", 2117, "timeOfImpact < collisionItem.timeOfImpact"),
    "Free-ball comparison": ("src/Ballpark.cpp", 2272, "toi < collisionItem.timeOfImpact"),
    "History cutoff": ("src/CollisionBallProperties.cpp", 220, "m_locations[index].time + m_fuzzyDelta < time"),
    "Tick-end motion lookup": ("src/Ballpark.cpp", 2738, "collisionIter->second.CalculatePosition(position, velocity, timeStepFraction)"),
    "Free-ball loop": ("src/Ballpark.cpp", 2275, "ballProp.IgnoreCollision(other->mId, timeOfImpact)"),
    "Sphere sweep": ("src/Collision.cpp", 108, "double CollideTwoSpheres("),
    "Dot product": ("src/Vector3d.h", 49, "x*v.x + y*v.y + z*v.z"),
    "Root interval test": ("src/Collision.cpp", 127, "s2 >= 0.0 && s2 <= 1.0"),
    "Coefficient construction": ("src/Collision.cpp", 123, "p0q0_2 - collRadius * collRadius"),
    "Quadratic miss guard": ("src/Collision.cpp", 151, "if (det < 0.0)"),
    "Maximum speed field": ("src/Ball.h", 126, "float mMaxVel;"),
    "Impact speed limiter": ("src/Ballpark.cpp", 2600, "25 * ball->mMaxVel * ball->mMaxVel / speed2"),
    "Angular impulse limiter": ("src/Ballpark.cpp", 2610, "4 * ball->mMaxAngVel * ball->mMaxAngVel / omega2"),
    "Integrator": ("src/Ballpark.cpp", 767, "a * (-k * t) + k * (a * t + v * k / m * t)"),
    "Vector division": ("src/Vector3d.h", 46, "c = 1.0/c"),
    "Warp cruise duration": ("src/Ballpark.cpp", 4253, "cruiseDuration = (warpDistance / warpSpeed"),
    "Destiny’s Taylor substitution": ("src/Ballpark.cpp", 763, "if (k < 1e-10 * m * t)"),
    "Destiny shrink comment and loop": (
        "src/Ballpark.cpp", 6720, "i = mFormations[ball->mFormationID].size() ; i < newLength"
    ),
    "Destiny STOP": ("src/Ballpark.cpp", 4632, "ball->mFollowPtr->FreeFormationSlot(ball->mEffectStamp)"),
}


def checkout_destiny(repo, pin, destination):
    """Give a harness its own ordinary .git directory and the exact source tree it expects."""
    subprocess.run(["git", "clone", "--quiet", "--local", "--shared", "--no-checkout", "--",
                    repo.resolve(), destination], check=True)
    subprocess.run(["git", "-C", destination, "checkout", "--quiet", "--detach", pin], check=True)


def check(log, root, destiny_pin, destiny_repo=None):
    registry = json.loads((root / "lean/claims.json").read_text(encoding="utf-8"))
    engine_pin = registry["revision"]
    engine = "https://github.com/Eidosoma/exact-ballpark/blob/"
    destiny = "https://github.com/carbonengine/destiny/blob/"
    has_engine_blob = (root / ".git").exists() and subprocess.run(
        ["git", "-C", root, "cat-file", "-e", f"{engine_pin}^{{commit}}"], stderr=subprocess.DEVNULL
    ).returncode == 0
    files, scopes = {}, {}
    for label, url in re.findall(r"\[([^\[\]]+)\]\(([^)]+)\)", log):
        if label in DESTINY_OPERATIONS and not url.startswith(destiny):
            raise SystemExit(f"Expected a pinned Destiny source for {label}: {url}")
        if "#" not in url and label not in DESTINY_OPERATIONS:
            continue
        target, _, anchor = url.partition("#")
        if target.startswith("../src/"):
            raise SystemExit(f"Use the reviewed commit for the source citation {url}")
        if not target.startswith((engine, destiny)):
            continue
        local = target.startswith(engine)
        pin, path = target.removeprefix(engine if local else destiny).split("/", 1)
        span = re.fullmatch(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", anchor)
        allowed_path = path.startswith("src/") or (not local and path.startswith(
            ("python/destiny/net/client/", "python/destiny/net/server/")))
        if pin != (engine_pin if local else destiny_pin) or not allowed_path or span is None:
            raise SystemExit(f"Invalid pinned source citation {url}")
        start, end = int(span[1]), int(span[2] or span[1])
        if start > end or (local and path not in registry["sources"]):
            raise SystemExit(f"Invalid source span or unregistered file {url}")
        if not local:
            operation = DESTINY_OPERATIONS.get(label)
            if operation is None:
                raise SystemExit(f"Unregistered Destiny citation {label}: {url}")
            expected_path, line, snippet = operation
            if path != expected_path or not start <= line <= end:
                raise SystemExit(f"Destiny citation for {label} must include {expected_path}:{line}: {url}")
            if destiny_repo is None:
                continue
        key = (pin, path)
        if key not in files:
            if local and not has_engine_blob:
                # An exported source archive can still check the registered working snapshot.
                data = (root / path).read_bytes()
                if hashlib.sha256(data).hexdigest() != registry["sources"][path]:
                    raise SystemExit(f"Source archive differs from the registered {path}")
            else:
                data = subprocess.check_output(["git", "-C", root if local else destiny_repo,
                                                "show", f"{pin}:{path}"])
            source = data.decode("utf-8")
            files[key] = source.splitlines()
            if local:
                body = ast.parse(source).body
                scopes[key] = {
                    f"{cls.name}.{fn.name}": (fn.lineno, fn.end_lineno)
                    for cls in body if isinstance(cls, ast.ClassDef)
                    for fn in cls.body if isinstance(fn, ast.FunctionDef)
                }
                scopes[key].update({fn.name: (fn.lineno, fn.end_lineno)
                                    for fn in body if isinstance(fn, ast.FunctionDef)})
        if end > len(files[key]):
            raise SystemExit(f"Source citation exceeds file length {url}")
        if local:
            scope = scopes[key].get(label.strip("`"))
            if scope is None or not (scope[0] <= start <= end <= scope[1]):
                raise SystemExit(f"Source citation falls outside {label}: {url}")
        elif snippet not in files[key][line - 1]:
            raise SystemExit(f"Expected {snippet!r} for {label} at {path}:{line} in {pin}")
