# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Every kernel library's build key holds nvcc's version: `vec_orient_cuda.nvcc_version` runs `nvcc
--version` once a process for a compiler's path, and each of the ten loaders of exact_ballpark keys its library on it
beside the path, so a new nvcc installed at the same path builds anew instead of reusing a library an older one built.
Checked without compiling, on the CPU: a stand-in nvcc at one path answers --version with a given text and fails any
other call, recording it; each loader's build, run into an empty folder, reaches the compile in a work folder named by
its key (`.{stem}.{pid}`): two versions give two keys, one version the same key in a new process; the version is asked
once a process. (That the new keys change no source and no machine code was checked when they came in: every library
built under them had the same PTX and SASS as before.)"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from exact_ballpark import vec_orient_cuda as K

LOADERS = ("vec_orient_cuda", "vec_collide_cuda", "vec_modes_cuda", "vec_combat_cuda", "vec_composed_cuda",
           "vec_ypr_cuda", "vec_server_cuda", "vec_server_combat_cuda", "vec_server_space_cuda", "vec_space")


def _stand_in(path: Path, version: str, log: Path) -> None:
    path.write_text("#!/bin/bash\n"
                    f'if [ "$1" = "--version" ]; then echo "{version}"; echo "--version" >> "{log}"; exit 0; fi\n'
                    f'echo "$@" >> "{log}"\n'
                    "exit 1\n")
    path.chmod(0o755)


def _key(name: str, tmp: Path, version: str, monkeypatch) -> str:
    """The stem (the variant and the key) the loader's build gives its library under a stand-in nvcc of `version`,
    as a new process (the version cache emptied), read from the work folder of the compile it reaches."""
    nvcc, log = tmp / "nvcc", tmp / "log"
    _stand_in(nvcc, version, log)
    log.write_text("")
    K._NVCC_VERSIONS.clear()
    monkeypatch.setattr(K, "_nvcc", lambda: nvcc)
    collide = importlib.import_module("exact_ballpark.vec_collide_cuda")
    monkeypatch.setattr(collide, "_nvcc", lambda: nvcc)
    mod = importlib.import_module(f"exact_ballpark.{name}")
    out = tmp / f"out_{name}_{len(list(tmp.iterdir()))}"
    with pytest.raises(RuntimeError, match="nvcc failed"):
        mod.build(out)
    calls = [ln.split() for ln in log.read_text().splitlines() if ln != "--version"]
    # a test runner that records dependencies may run a compiler a second time, with its flags and `-M -MF <file>`,
    # to read the compile's headers: such a probe is never a compile
    calls = [c for c in calls if "-M" not in c]
    assert len(calls) == 1, calls
    lib = Path(calls[0][calls[0].index("-o") + 1])
    return lib.parent.name[1:].rsplit(".", 1)[0]


@pytest.mark.parametrize("name", LOADERS)
def test_each_libraries_key_holds_nvccs_version(name, tmp_path, monkeypatch):
    a = _key(name, tmp_path, "Cuda compilation tools, release 13.0, V13.0.88", monkeypatch)
    b = _key(name, tmp_path, "Cuda compilation tools, release 13.1, V13.1.12", monkeypatch)
    a2 = _key(name, tmp_path, "Cuda compilation tools, release 13.0, V13.0.88", monkeypatch)
    assert a != b and a == a2, (a, b, a2)
    assert a.split("_")[-1] != b.split("_")[-1]


def test_nvccs_version_is_asked_once_a_process(tmp_path):
    nvcc, log = tmp_path / "nvcc", tmp_path / "log"
    _stand_in(nvcc, "v1", log)
    log.write_text("")
    K._NVCC_VERSIONS.pop(str(nvcc), None)
    first = K.nvcc_version(nvcc)
    _stand_in(nvcc, "v2", log)                      # a new compiler at the same path, within the process
    assert K.nvcc_version(nvcc) == first == "v1\n"
    assert log.read_text().splitlines() == ["--version"]
    K._NVCC_VERSIONS.pop(str(nvcc), None)


def test_the_real_compilers_version_keys_the_libraries():
    """This port's nvcc: its whole --version output is what the keys hold (the manifests record its last line)."""
    v = K.nvcc_version(K._nvcc())
    assert "Cuda compilation tools" in v and v == K.nvcc_version(K._nvcc())
