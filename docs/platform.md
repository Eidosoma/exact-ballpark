# The qualified platform

exact-ballpark claims exactness on one platform only: the one on which every test in `exactness.md` was run. This
document covers:
- that platform;
- what "qualified" means;
- why exactness depends on the platform;
- how another compiler, C library or GPU can be qualified.

## 1. The qualified platform

**The source.** destiny at commit `114e89fa`, with carbonengine/math `82733220` and DirectXMath `f5d82574` (its
scalar path), compiled by the conformance kit (`provenance.md`).

**The environment.** These commands, run on the CPU on 1 November 2026, report the qualification environment. Site
paths are shortened to `...`, and `nvcc --version`'s build line is left out.

```text
$ uname -m
aarch64
$ ldd --version | head -1
ldd (GNU libc) 2.34
$ rpm -q glibc
glibc-2.34-283.el9_8.aarch64
$ sha256sum /usr/lib64/libm.so.6
b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86  /usr/lib64/libm.so.6
$ g++ --version | head -1
g++ (GCC) 14.3.0
$ sha256sum "$(readlink -f "$(g++ -print-file-name=libstdc++.so.6)")"
ac709fc594acdcdf9b19868b3ad7d7275815018f82d39f028710e3706407022f  .../libstdc++.so.6.0.33
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jul_16_07:31:19_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.48
$ python3 -c "import sys, torch; print(sys.version); print(torch.__version__)"
3.13.5 (main, Oct 22 2026, 14:03:33) [GCC 14.3.0]
2.9.1+cu128
$ grep Model /proc/driver/nvidia/gpus/*/information
Model:           NVIDIA GH200 480GB
```

In short:

| Component | Qualified |
|---|---|
| Operating system and CPU | Linux on aarch64 |
| C library | the GNU C Library 2.34 (`glibc-2.34-283.el9_8.aarch64`); `libm.so.6` with the SHA-256 above |
| C++ compiler and standard library | GCC 14.3.0; `libstdc++.so.6.0.33` with the SHA-256 above |
| CUDA compiler | nvcc 13.0 (V13.0.48), with `cuobjdump` beside it |
| GPU | NVIDIA GH200 480GB, running the kernels' `sm_90` code |
| Python | 3.13.5 (3.13 or later is required: the transcriptions use `math.fma`) |
| PyTorch | 2.9.1 (`2.9.1+cu128`) |

The kernels are compiled by nvcc at first use, outside PyTorch's own build, loaded through `ctypes` and launched on
PyTorch's current stream.

## 2. What "qualified" means

A platform is qualified when the whole test suite passes on it, every comparison exact. All three subsets must pass:
- **`oracle`:** the conformance kit compiles destiny on that platform, and the scalar reference equals it on every
  scenario. The reference's Python server and client layers pass destiny's published tests.
- **`cpu`:** the transcribed C-library functions equal the platform's C library, and the tables read from the library
  equal its bytes. The batched parks equal the scalar reference on the CPU.
- **`gpu`:** the batched parks equal the reference on the GPU. Each kernel equals its park's torch path on every
  tensor after every tick. Each CUDA graph equals the eager ticks.

On a qualified platform the chain closes. destiny compiled there equals the reference, which equals the batched parks,
which equal the kernels and their graphs. Each release names the platform it was qualified on (`CHANGELOG.md`).

## 3. Why exactness depends on the platform

**The C library.**
- destiny's results depend on the exact bits of `exp`, `log`, `sin`, `cos`, `asin`, `atan2`, `pow`, `powf` and the
  `float` functions it calls (`numerics.md`).
- The transcriptions reproduce the GNU C Library 2.34 as compiled for aarch64, including where its compiler fused
  multiply-adds.
- On x86-64, glibc chooses some functions' variants by CPU, with or without fused multiply-adds. The same source
  compiled for another architecture can contain other fusions, or none.
- Another C library, such as a Windows C runtime, differs again.

**The C++ compiler.** The conformance kit compiles destiny with `-O2 -std=c++17 -ffp-contract=off -fno-fast-math`.
Three compiler decisions matter on the qualified platform:
- **No contraction.** No contraction of destiny's own arithmetic is allowed.
- **`sincos`.** g++ fuses `sin(x)` and `cos(x)` of one argument into glibc's `sincos` at six call sites, and the
  reference does the same. A test reads the harnesses' compiled calls, so a compiler that decides otherwise fails it.
- **Plain `char`.** Plain `char` is unsigned on aarch64 but signed on x86 and under MSVC, which destiny targets. The
  kit compiles with `-fsigned-char` where destiny's `char` values are reached.

**DirectXMath.** The quaternion from yaw, pitch and roll is DirectXMath's scalar path (`_XM_NO_INTRINSICS_`), whose
sines and cosines are the C library's.
- On aarch64 the header would otherwise take its NEON path.
- A Windows x64 build of destiny would most likely take its SSE path. There the sines and cosines are DirectXMath's own
  polynomials, and under MSVC 2019 and later they may come from Intel's SVML.
- Such a build can differ in the last place of a turned vector.

**The C++ standard library.** destiny walks its hash maps and sets in the order the standard library gives them. The
port reproduces libstdc++'s order (`numerics.md`, section 6). Another standard library may order them differently.

**The GPU.**
- The kernels are built for `sm_90` and run there.
- The default build also holds `sm_100` binaries and `compute_90` PTX. They have never been run, so they are not
  qualified.
- GPUs older than Hopper (`sm_80`, `sm_86`, `sm_89`) have no build target, and the kernels are not expected to load
  there. That has not been tried.

## 4. How the test suite behaves elsewhere

- **The scalar reference** calls the running C library, through Python's `math` module and `ctypes`. So do a few torch
  paths (`numerics.md`, section 3). The transcriptions and the kernels always compute glibc 2.34 aarch64's functions,
  whatever the platform.
- **The tests that compare the transcriptions with the running C library** run everywhere. On another C library they
  fail wherever it differs: they are how a mismatch shows.
- **The tests that read the C library's tables** skip unless the running `libm.so.6` has the recorded SHA-256.
- **The `oracle` tests** skip when the harnesses are not built. Built with another compiler or C library, they test the
  reference against destiny as built there. That is meaningful, but it is not the qualified platform.
- **The `gpu` tests** skip without a CUDA device and `nvcc`.
- **The space layer's CPU path** is host code compiled by nvcc, so its CPU tests need `nvcc` but no GPU. Its build
  check refuses fused instructions in that host code, and it knows aarch64's fused instructions only.
- **Everything the GPU builds need:**
  - `nvcc`, from `EXACT_BALLPARK_NVCC`, else `$CUDA_HOME/bin/nvcc`, else `PATH`;
  - `cuobjdump`, beside `nvcc`, for the PTX check;
  - a cache folder for the built libraries, `EXACT_BALLPARK_CACHE`, by default the user's cache folder.

## 5. Qualifying another compiler, C library or GPU

**The procedure.**
1. **Fetch and build the conformance kit** (`conformance/README.md`) with the platform's compiler, at the same flags.
2. **Run the three subsets:** `pytest -m oracle`, `pytest -m cpu`, `pytest -m gpu`.
3. **Record the qualification:**
   - the platform's versions and checksums: the C library's `libm.so.6`, libstdc++, the compiler, `nvcc`, the GPU,
     Python and PyTorch;
   - every test's outcome, by test id.

A platform is qualified only when every test passes, with no tolerance.

**What to expect.**
- **Another compiler, with the qualified C library.** All three subsets should pass if the compiler keeps the flags'
  promises and makes the same `sincos` decision. Any failure is a finding about that compiler.
- **Another C library, or glibc on another architecture.**
  - The `oracle` subset still compares the reference with destiny built against that library.
  - The `cpu` tests of the transcriptions fail where that library's functions differ from glibc 2.34 aarch64's.
  - The batched parks and kernels, which keep glibc 2.34 aarch64's bits, then differ from the reference there.
  - Qualifying such a platform needs that library's functions transcribed the same way: from its compiled code, with
    its fused steps and its tables read out of it. This release contains no such transcription.
- **Another GPU.**
  - The kernels must build for it and load. An architecture outside the build targets needs a target added.
  - The PTX check still applies: it pins every operation's rounding in the PTX, and no later stage contracts or
    approximates an instruction so pinned. What a new GPU qualifies is the code generated from that PTX and the
    kernels' behaviour on the device.
  - The `gpu` subset must pass in full.
