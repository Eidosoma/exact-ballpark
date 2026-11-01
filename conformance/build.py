# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Build the compiled-oracle harness of conformance row 2.

    python3 -I conformance/build.py DESTINY_REPO BUILD_DIR

Destiny's own functions are extracted by brace matching from the pinned source (carbonengine/destiny at 114e89fa,
MIT, (c) CCP Games), never retyped: `Ballpark::Integrate`, `Ballpark::GotoThrust`, `Ballpark::EvolveStop`,
`Ballpark::SetBallTimeFactor`, and `src/Vector3d.h` whole. A minimal prelude supplies the members they touch (Ball's
fields with Destiny's types; Ballpark's dt and mFriction; the global dynamical-orientation switch, off) and a driver
evolves one ball per case in Destiny's order (mode thrust, then integration, then commit). Compiled with the base
environment's g++ with `-O2 -ffp-contract=off` (no fused multiply-add, as the reference cannot fuse). The build
directory receives the source, the binary and a manifest with the SHA-256 of every extracted piece and the compiler
version; nothing is installed. The repository is read, never executed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PINNED = "114e89fa584f0e1a81eb0a679aabf41af8870072"
FUNCS = ["void Ballpark::Integrate(", "Vector3d Ballpark::GotoThrust(", "void Ballpark::EvolveStop(",
         "void Ballpark::SetBallTimeFactor("]

PRELUDE = r"""// Harness prelude: the members Destiny's extracted functions touch, with Destiny's types (Ball.h).
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
using std::exp; using std::sqrt;
struct Vector3 {                             // only referenced by Vector3d.h's conversions; unused here
    float x, y, z;
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float a, float b, float c) : x(a), y(b), z(c) {}
};
"""

BALLPARK = r"""
static bool g_useDynamicalOrientation = false;   // Destiny's default (MOV-710)
struct Ball {
    Vector3d mNewPos, mNewVel, mGoto;
    double mMass = 0.0, mTimeFactor = 0.0;
    float mMaxVel = 0.0f, mAgility = 1.0f, mSpeedFraction = 0.0f;
};
struct Ballpark {
    double dt = 1.0;
    double mFriction = 1.0e6;
    int mTickInterval = 1000;
    void Integrate(Vector3d& p, Vector3d& v, const Vector3d& a, double m, double k, double timeFactor, double t);
    Vector3d GotoThrust(const Ball *ball, const Vector3d& target, bool missile);
    void EvolveStop(Ball* ball);
    void SetBallTimeFactor(Ball* ball);
};
"""

DRIVER = r"""
// Driver: one ball per input line, Destiny's evolve order for GOTO (0) and STOP (2); %.17g output.
int main() {
    long id; double friction; int tick, steps, mode; double mass, agility, maxvel, sf;
    double px, py, pz, vx, vy, vz, gx, gy, gz;
    while (std::scanf("%ld %lf %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &id, &friction, &tick,
                      &steps, &mode, &mass, &agility, &maxvel, &sf, &px, &py, &pz, &vx, &vy, &vz, &gx, &gy, &gz) == 18) {
        Ballpark bp; bp.mFriction = friction; bp.mTickInterval = tick; bp.dt = tick * 0.001;
        Ball b; b.mMass = mass; b.mAgility = (float)agility; b.mMaxVel = (float)maxvel; b.mSpeedFraction = (float)sf;
        b.mNewPos = Vector3d(px, py, pz); b.mNewVel = Vector3d(vx, vy, vz); b.mGoto = Vector3d(gx, gy, gz);
        bp.SetBallTimeFactor(&b);
        for (int s = 1; s <= steps; ++s) {
            Vector3d a(0.0, 0.0, 0.0), c(0.0, 0.0, 0.0);
            if (mode == 0) a = bp.GotoThrust(&b, b.mGoto, false); else bp.EvolveStop(&b);
            Vector3d p = b.mNewPos, v = b.mNewVel;
            bp.Integrate(p, v, a + c, b.mMass * b.mAgility, bp.mFriction, b.mTimeFactor, bp.dt);
            b.mNewPos = p; b.mNewVel = v;
            std::printf("%ld %d %.17g %.17g %.17g %.17g %.17g %.17g\n", id, s, p.x, p.y, p.z, v.x, v.y, v.z);
        }
    }
    return 0;
}
"""


def extract(src: str, signature: str) -> str:
    i = src.index(signature)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        depth += {"{": 1, "}": -1}.get(src[k], 0)
        if depth == 0:
            return src[i:k + 1]
    raise ValueError(f"unbalanced braces after {signature}")


def main(repo: Path, build: Path) -> None:
    head = (repo / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):                         # follow a symbolic HEAD to its commit (loose or packed ref)
        ref = head[5:]
        loose = repo / ".git" / ref
        head = loose.read_text().strip() if loose.is_file() else next(
            line.split()[0] for line in (repo / ".git" / "packed-refs").read_text().splitlines()
            if line.endswith(" " + ref))
    if head != PINNED:
        raise SystemExit(f"destiny is at {head}, not the pinned {PINNED}")
    build.mkdir(parents=True, exist_ok=True)
    ballpark = (repo / "src" / "Ballpark.cpp").read_text()
    vector = (repo / "src" / "Vector3d.h").read_text()
    pieces = {f: extract(ballpark, f) for f in FUNCS}
    source = (PRELUDE + "// ---- src/Vector3d.h, verbatim ----\n" + vector + BALLPARK
              + "".join(f"\n// ---- src/Ballpark.cpp, {f[:-1]}, verbatim ----\n{body}\n" for f, body in pieces.items())
              + DRIVER)
    (build / "harness.cpp").write_text(source)
    cmd = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-o", str(build / "harness"),
           str(build / "harness.cpp")]
    subprocess.run(cmd, check=True)
    gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    manifest = {"destiny_commit": head, "compiler": gxx, "command": cmd,
                "ballpark_cpp_sha256": hashlib.sha256(ballpark.encode()).hexdigest(),
                "vector3d_h_sha256": hashlib.sha256(vector.encode()).hexdigest(),
                "extracted_sha256": {f[:-1]: hashlib.sha256(b.encode()).hexdigest() for f, b in pieces.items()},
                "harness_cpp_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "binary_sha256": hashlib.sha256((build / "harness").read_bytes()).hexdigest()}
    (build / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
