# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Fetch the three upstream repositories at their pinned commits, verify them, and build the harnesses.

    python3 conformance/fetch_upstream.py            # clone (or reuse) and verify
    python3 conformance/fetch_upstream.py --build    # then build the four harnesses

Each repository is cloned into the folder that `conformance/kit.toml` names (or the file `$EXACT_BALLPARK_KIT`
names) and checked out at its pinned commit, detached. It is then verified, and any failure stops the script:
- the commit's tree is the recorded tree (`git rev-parse <commit>^{tree}`): the whole tree at the commit, every file
  and folder, not a list of files;
- the working tree is clean (`git status --porcelain` prints nothing).
A SHA-256 manifest of every checked-out file (`sha256  ./path`, sorted by path) is written beside each clone; the
tests compare the DirectXMath headers a harness read with DirectXMath's manifest. DirectXMath checks out its headers
with CRLF line ends (its `.gitattributes`), so its manifest is of those bytes.

With `--build`, the build scripts beside this file extract destiny's C++ from the clones and compile the harnesses
(`g++`, see README.md) into the configured harness folders.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name: (repository, pinned commit, the commit's tree)
PINS = {
    "destiny": ("https://github.com/carbonengine/destiny.git", "114e89fa584f0e1a81eb0a679aabf41af8870072",
                "c41df8c5d5dc9dc9a2d7768718aece7bfdb9f65a"),
    "math": ("https://github.com/carbonengine/math.git", "8273322093eb1f80168ae60a35891f2d454369f9",
             "d58f521655214f2e4bf3ef6afb0f31687e134d18"),
    "directxmath": ("https://github.com/microsoft/DirectXMath.git", "f5d82574b03417d905fdd2f85bfa9bf7a2f1deac",
                    "293ed06c7912b823d6cd2d775e4b62243c42e530"),
}
# harness: (build script, the repositories it reads, in its arguments' order before the build folder; after it)
BUILDS = {"core": ("build.py", ["destiny"], []), "modes": ("build_modes.py", ["destiny"], []),
          "space": ("build_space.py", ["destiny"], []),
          "full": ("build_full.py", ["destiny", "math"], ["directxmath"])}


class Refused(RuntimeError):
    """A clone that is not the pinned tree, or a dirty one."""


def config_file() -> Path:
    return Path(os.environ.get("EXACT_BALLPARK_KIT") or HERE / "kit.toml")


def paths() -> dict:
    f = config_file()
    with open(f, "rb") as fh:
        table = tomllib.load(fh)["paths"]
    out = {}
    for k, v in table.items():
        p = Path(v).expanduser()
        out[k] = p if p.is_absolute() else f.parent / p
    return out


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def fetch(name: str, folder: Path) -> None:
    """Clone `name` into `folder` (or reuse a clone there) and check out its pinned commit, detached."""
    url, commit, _ = PINS[name]
    if not (folder / ".git").exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(folder)], check=True)
    if subprocess.run(["git", "-C", str(folder), "cat-file", "-e", f"{commit}^{{commit}}"]).returncode != 0:
        subprocess.run(["git", "-C", str(folder), "fetch", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(folder), "-c", "advice.detachedHead=false", "checkout", "--detach", commit],
                   check=True)


def verify(name: str, folder: Path) -> None:
    """The pinned commit checked out, its tree the recorded one, the working tree clean; else Refused."""
    _, commit, tree = PINS[name]
    head = git(folder, "rev-parse", "HEAD").strip()
    if head != commit:
        raise Refused(f"{name}: HEAD is {head}, not the pinned {commit}")
    got = git(folder, "rev-parse", f"{commit}^{{tree}}").strip()
    if got != tree:
        raise Refused(f"{name}: the tree of {commit} is {got}, not the recorded {tree}")
    dirty = git(folder, "status", "--porcelain")
    if dirty.strip():
        raise Refused(f"{name}: the working tree is not clean:\n{dirty}")


def manifest(folder: Path) -> str:
    """`sha256  ./path` of every file under `folder` but .git, sorted by path."""
    rows = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if rel.parts[0] == ".git" or not p.is_file() or p.is_symlink():
            continue
        rows.append((f"./{rel.as_posix()}", hashlib.sha256(p.read_bytes()).hexdigest()))
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(rows))


def build(where: dict) -> None:
    for harness, (script, before, after) in BUILDS.items():
        out = where[f"harness_{harness}"].parent
        out.mkdir(parents=True, exist_ok=True)
        args = [str(where[r]) for r in before] + [str(out)] + [str(where[r]) for r in after]
        subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--build", action="store_true", help="build the harnesses after the checks")
    ap.add_argument("--no-fetch", action="store_true", help="verify existing clones only")
    a = ap.parse_args(argv)
    where = paths()
    for name in PINS:
        if not a.no_fetch:
            fetch(name, where[name])
        verify(name, where[name])
        text = manifest(where[name])
        target = where["directxmath_manifest"] if name == "directxmath" else \
            where[name].parent / f"MANIFEST_{name}_{PINS[name][1][:8]}.sha256"
        target.write_text(text)
        print(f"{name}: {PINS[name][1]} verified (tree {PINS[name][2]}); manifest {target} ({len(text.splitlines())} files)")
    if a.build:
        build(where)
    return 0


if __name__ == "__main__":
    sys.exit(main())
