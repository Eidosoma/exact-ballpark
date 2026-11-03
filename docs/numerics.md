# The numerical contract

exact-ballpark's three layers (the scalar reference, the batched parks and the CUDA kernels) produce the same bits as
destiny compiled from source on the qualified platform (`platform.md`). This document states the rules that make that
possible:
- where destiny computes in `float` and where in `double`, and how each layer follows it;
- which functions of the C library are transcribed, and how;
- where fused multiply-adds occur;
- how every CUDA build is held to all of it.

`exactness.md` gives the evidence that the rules hold.

## 1. The rules in one list

- **Correctly rounded operations only:** `+`, `-`, `*`, `/` and `sqrt`, each rounded once to nearest. Comparisons and
  integer steps are exact.
- **destiny's order of operations:** every sum left to right as written, every product grouped as written, and the
  same parentheses.
- **`float` where destiny uses `float`, `double` where it uses `double`.**
- **The C library's own algorithms** for every transcendental function destiny calls. They are transcribed from the
  library's compiled code, so the CPU and the GPU return the library's bits.
- **A fused multiply-add exactly where the C library's compiled code has one, and nowhere else.**
- **No compilation that changes arithmetic:** no `torch.compile`, no contraction, no fast-math, no flush to zero, no
  approximate division or square root.
- **No floating-point reductions and no floating-point atomics.** Where an order matters, it is computed with
  integers.

## 2. `float` and `double`

destiny keeps positions, velocities, goto points and thrust in `double` (its `Vector3d`). Several ball properties are
32-bit floats. Radius, maximum speed, agility, speed fraction and follow range are `float` in `Ball.h` (rule MOV-712).
With dynamical orientation, the rotation, the angular velocity, the maximum angular speed and the rotational agility
are floats as well. DirectXMath's quaternion from yaw, pitch and roll is computed in `float`, and so is a mushroom's
growth.

**How each layer holds a `float`.**
- **The scalar reference** holds a `float` value in a Python float, which is a C `double`. It rounds the value to the
  nearest `float` wherever destiny stores or converts one.
- **The batched parks** hold `float` fields in `float64` tensors. Loading a ball and every setter round the value to
  `float`, as destiny stores it.
- **A `float` operation of two floats** is computed as the `double` operation, rounded to `float`. For `+`, `-`, `*`,
  `/` and `sqrt` this equals the native `float` operation, because 53 ≥ 2 × 24 + 2 (Figueroa's condition for double
  rounding to be innocuous).
- **The CUDA kernels** use the native correctly rounded `float` intrinsics (`__fadd_rn`, `__fsub_rn`, `__fmul_rn`,
  `__fdiv_rn`, `__fsqrt_rn`) exactly where the torch code rounds a `double` operation of two floats.

**`double` operations.**
- On the CPU they are plain IEEE operations in Python or torch.
- On the GPU they are the correctly rounded intrinsics `__dadd_rn`, `__dsub_rn`, `__dmul_rn`, `__ddiv_rn` and
  `__dsqrt_rn`.
- Where destiny divides a `Vector3d` by a scalar, the port does what destiny's operator does: a multiplication by the
  reciprocal.

**Divisions in torch.** Every division has a tensor divisor and a tensor dividend. On CUDA, torch divides a tensor by
a host scalar through its reciprocal, and computes `scalar / tensor` as a reciprocal times the scalar. Neither is
correctly rounded, so neither form is used.

**Conversions.**
- **`double` to `float`:** round to nearest.
- **`double` to a 64-bit integer, where destiny casts:** aarch64's conversion. It truncates toward zero, saturates out
  of range, and converts NaN to 0.
- **64-bit integer to `double`:** exact below 2^53.

**Integer steps.** These include the bit manipulations inside the C library's functions and the sort keys of the
collision order. They use 64-bit integers with two's-complement wraparound, as torch's `int64` has it. Where an
intermediate could overflow, the steps are split into 32-bit halves.

**Inputs where Python and C part ways.** Python's `math` module raises an exception where C returns an infinity or a
NaN, for example `math.sin(inf)` or `math.log(0.0)`.
- The transcriptions return C's value.
- A NaN result is a NaN, but not necessarily with the C library's payload.
- `errno` and the floating-point exception flags are not modelled.
- Where the scalar reference itself would raise, the case is outside the reference's domain (`limits.md`).

## 3. The C library's functions

destiny's results depend on the elementary functions of the C library it is linked against. On the qualified platform
that is the GNU C Library 2.34 for aarch64 (`libm.so.6`, SHA-256 in `platform.md`).

Each function below is transcribed from that library's compiled code, operation for operation:
- **Read from the disassembly.** Each step is taken from the disassembly of the pinned `libm.so.6`.
- **Fused exactly where GCC fused.** Every multiply-add that GCC fused when it compiled the library is fused in the
  transcription, with the disassembly's operand order, and no other step is fused.
- **Constants never typed.** Constants and tables are read out of the library by script, at the addresses the code
  loads them from, or are assembled from its immediate operands. The tests hold every table and constant to the
  library's bytes when they run on that library.

| Function | Precision | Where destiny calls it | Algorithm in glibc 2.34 | Fused multiply-adds in the compiled code | CPU transcription | GPU |
|---|---|---|---|---|---|---|
| `exp` | double | the integrator over part of a tick (`Ballpark::Integrate`); the old orbit's radial factor; the new orbit's spiral; the warp's acceleration and deceleration | Arm's optimized-routines (`e_exp.c`) | 8: 7 on the main path, 1 in a special case | `exact_ballpark.vec_exp` (tensors) | the collision, flight-mode and combat kernels |
| `sin`, `cos` | double | the old orbit's precessing normal; the new orbit's spiral angle and default normal | IBM's Accurate Mathematical Library (`s_sin.c`), with `branred` for huge arguments | `__sin`: 50 `fmadd`, 18 `fmsub`, 2 `fnmsub`; `__cos`: 49, 18 and 3. `branred` has none: glibc compiles it without contraction. | `exact_ballpark.dlibm` | the flight-mode kernel and the kernels that include it |
| `sincos` | double | the same six call sites (see below) | the same pieces as `sin` and `cos` | as `sin` and `cos` | `exact_ballpark.dlibm.sincos` | the same |
| `log` | double | the new orbit's spiral; the warp's two logarithms (`SetupWarpConstants`) | Arm's optimized-routines (`e_log.c`) | 22 `fmadd`, 1 `fmsub` | `exact_ballpark.dlibm` | the same |
| `asin`, `atan2` | double | yaw, pitch and roll without dynamical orientation (`Ball::CalculateYawPitchRoll`) | IBM's Accurate Mathematical Library (`e_asin.c`, `e_atan2.c`) | as compiled, each with the disassembly's operands | `exact_ballpark.dlibm_inv` | `exact_ballpark.vec_ypr_cuda` |
| `expf`, `logf`, `sinf`, `cosf`, `pow` | float (`pow`: double) | dynamical orientation; `sinf` and `cosf` also in DirectXMath's quaternion from yaw, pitch and roll; `pow` also in the turret accuracy function | Arm's optimized-routines, computed in `double` and rounded to `float` once | as compiled | `exact_ballpark.vec_libm` (tensors) | the orientation kernel, and every kernel that includes it |
| `acosf`, `atanf`, `atan2f` | float | dynamical orientation | Sun's fdlibm as carried in glibc, computed in `float` | GCC's fused `float` multiply-adds, as compiled | `exact_ballpark.vec_libm` | the same |
| `powf` | float | a mushroom's growth (`Ballpark.cpp:569`, `range*pow(timeFraction, 0.25f)`, the `float` overload) | Arm's optimized-routines (`e_powf.c`) | 9: 6 in `log2_inline`, 3 in `exp2_inline` | `exact_ballpark.dlibm_powf` | the combat kernel |
| `sqrt` | both | throughout | correctly rounded by IEEE 754 | none | IEEE | `__dsqrt_rn`, `__fsqrt_rn` |

**Where the scalar reference gets these functions.** The scalar reference calls the running C library:
- through Python's `math` module for the `double` functions;
- through `ctypes` for `sincos`, the `float` functions and `powf`.

On the qualified platform that is glibc 2.34 itself. The transcriptions are what the batched parks and the kernels
compute. In a few places a torch path calls the running C library on the host instead, among them:
- the yaw, pitch and roll without dynamical orientation, where a park computes them on the host (the default; the
  orientation, flight-mode and combat parks also offer a device path, `ypr="cuda"`);
- the composed park's orbit sines and cosines;
- a mushroom's `powf`.

`platform.md` explains what this means on another platform.

**`sincos`, and why it matters.**
- **What g++ does.** At `-O2`, g++ fuses `sin(x)` and `cos(x)` of one argument into one call of glibc's `sincos`.
  Compiled destiny calls `sincos` twice in each of `DefaultOrbitalNormal`, `EvolveOldStyleOrbit` and
  `EvolveNewStyleOrbit`.
- **Where it differs.** glibc 2.34's `sincos` is not always `sin`. Between 0.855469 and 2.426265 it reaches the same
  table cell by a different path. An exhaustive check of every argument near each of the 202 points where the two
  paths can part (855,638,424 arguments) found exactly two that differ: x = ±0x1.ad1fb54442d18p+0. There `sincos`
  gives the sine 0x1.fd27acee550dbp-1 and `sin` gives 0x1.fd27acee550dcp-1. The cosines never differ. Elsewhere the
  two compute the same operations on the same values. That rests on GCC having compiled the shared code alike in
  both functions, which was checked by sampling (13,082,912,032 arguments on the CPU, 393,740,288 on the GPU), not
  instruction by instruction.
- **What the port does.** The reference and the kernels take those six sines and cosines from `sincos`: the C
  library's on the CPU, and its transcription on the GPU. The one lone `sin` in `CalculateYawPitchRoll` stays `sin`.
- **`sincosf`.** g++ also fuses `sinf` and `cosf` of one argument into `sincosf`, at nine call sites. glibc's
  `sincosf` equals `sinf` and `cosf` on all 4,294,967,296 `float` bit patterns, so the reference keeps separate
  `sinf` and `cosf` there.

**`rand()`.** A proximity sensor added with a shuffle starts its period at `floatrand(0, period)`, which draws from
the C library's `rand()`.
- The reference transcribes glibc's generator: the `TYPE_3` additive feedback table with its front and rear pointers.
- The batched space layer keeps one generator state per world, seeded per world. Its host path and its kernel draw
  as the reference does.

## 4. Fused multiply-adds

A fused multiply-add rounds once, so it gives different bits from a multiplication followed by an addition.
exact-ballpark has one rule: **a fused multiply-add exactly where the C library's compiled code has one, and nowhere
else.** destiny's own code is compiled by the conformance kit with `-ffp-contract=off`, so its own arithmetic has
none.

**On the GPU.** The fused steps are the IEEE intrinsics `__fma_rn` (`double`) and `__fmaf_rn` (`float`), written
explicitly in the source. Everything else is compiled with `-fmad=false`, so nvcc contracts nothing. The build counts
the result in the PTX: every `fma.rn` instruction must be one of the source's explicit calls. For example:
- **the collision kernel:** exactly 9 `fma.rn.f64`, the 8 of `exp` and 1 in a test kernel;
- **the orientation kernel:** 44 `fma.rn.f64` and 47 `fma.rn.f32`, each a call site in the source, with inlined
  copies counted (`pow` 20, `sincosf` 15, `logf` 5, `expf` 3, `acosf` 32, `atanf` 12, `atan2f` 2, and 1 each in the
  test kernel);
- **the flight-mode library:** 110 `fma.rn.f64` and 47 `fma.rn.f32`. Built again with that file's fused calls written
  as a multiplication and an addition, it holds exactly the orientation kernel's 44 and 47, so the other 66 are that
  file's explicit calls and none is a contraction.

**On the CPU, in torch.** torch has no fused multiply-add with a defined rounding, so `exact_ballpark.vec_exp.fma`
emulates one exactly from correctly rounded operations. It is the algorithm of Boldo and Melquiond ("Emulation of FMA
and correctly rounded sums: proved algorithms using rounding to odd", IEEE Transactions on Computers 57(4), 2008,
Algorithm 5):
- Dekker's product with Veltkamp's split gives `uh + ul = a × b` exactly.
- Knuth's TwoSum gives `th + tl = c + uh` exactly.
- `v = tl + ul` is rounded to odd, and the result is `th + v` rounded to nearest.

The emulation is exact while no operand or partial product overflows or underflows. Every multiply-add inside `exp`
stays far from both: `|r| < 0.003` or exactly 0, and the table's scales are normal.

The `float` fused multiply-add, `fmaf`, is computed as the exact product plus the addend rounded to odd in `double`,
then rounded to `float`. That is the correctly rounded result, because 53 ≥ 24 + 2 (the same paper).

**In scalar Python.** `math.fma` (Python 3.13 or later) rounds once.

**The native and emulated operations agree.** The tests hold the device's `__fma_rn` and `__fmaf_rn` to the
emulations and to the C library's `fma` and `fmaf` on hundreds of thousands of triples: exact ties, cancellations,
products' exact errors, near ties and random bit patterns (`exactness.md`).

## 5. How the CUDA builds are held to the contract

The kernels are compiled from source at first use, never shipped prebuilt. Every build goes through the same steps,
and fails if any check fails.

**The flags.** nvcc compiles with:

```text
-O3 -std=c++17 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false
-Xcompiler -fPIC,-ffp-contract=off,-fno-fast-math -shared -Xptxas -v
```

There is no `--use_fast_math`. The host part of each library is compiled with `-ffp-contract=off -fno-fast-math`.

**The targets.**

| Variant | Code generated |
|---|---|
| `native` (the default) | `sm_90` and `sm_100` binaries, and `compute_90` PTX |
| `sm90` | an `sm_90` binary and `compute_90` PTX |
| `ptx` | `compute_90` PTX only |

Only `sm_90` has been run (`platform.md`).

**The source scan.** Before compiling, the build refuses any source that calls a function of CUDA's math library by
name, in its `double` or `float` form:
- among them `exp`, `log`, `sin`, `cos`, `sincos`, `pow`, `atan2`, `asin`, `sqrt`, `fma`, `trunc`, `floor` and
  `ceil`;
- the approximate intrinsics (`__expf`, `__logf`, `__sinf`, `__cosf`, `__powf`, `__fdividef` and others);
- the intrinsics with directed rounding (`__fsqrt_rd`, `__fdiv_ru`, `__fma_rz` and the others).

The kernels' own transcriptions have other names. `trunc` and `ceil` are written as single PTX conversions
(`cvt.rzi`, `cvt.rpi`), which are exact. `std::max` and `std::min` are written out as the ternaries the reference's
`max` and `min` resolve to.

**The PTX check.** After compiling, the build reads the PTX back with `cuobjdump` and lists every floating-point
instruction by form. It refuses the build unless:
- every `add`, `sub`, `mul`, `div`, `sqrt` and `rcp` carries `.rn`, which no later stage contracts or approximates;
- every `fma` is `fma.rn`, and so is one of the source's explicit fused calls;
- nothing is flushed (`.ftz`), approximate (`.approx`, `.full`) or a `mad`;
- there is no transcendental instruction (`ex2`, `lg2`, `sin`, `cos`, `tanh`, `rsqrt`).

Two forms need a note:
- **`rcp.rn`** is how nvcc writes `1.0 / x`. It is the IEEE reciprocal, equal to `div.rn` of 1.0.
- **`min` and `max` without `.ftz` are allowed.** The compiler forms them from clamps with a constant, non-NaN,
  nonzero operand, where they equal the ternaries for every input, NaN included.

**The host check.** The space layer's sequential algorithm is compiled by nvcc into both host functions (its CPU path)
and kernels.
- **Without nvcc,** g++ compiles the host functions alone, from the same source and header. It uses the flags nvcc
  gives its host compiler (`-O3 -std=c++17 -fPIC -ffp-contract=off -fno-fast-math`), with the source's three CUDA
  qualifiers defined empty. Its library is cached under its own name, keyed by the compiler's path and version.
- **Either way,** the build disassembles the host code and refuses any fused instruction in it (on aarch64: `fmadd`,
  `fmsub`, `fnmadd`, `fnmsub`, `fmla`, `fmls`).
- **The two compilers agree.** On the qualified platform, their host code gives the same bits (`devices.md`, section 4).

**Constants are generated, never typed.**
- **The generated header.** Every table and constant a kernel needs is written into a header at build time, as bit
  patterns. The values come from the Python modules that hold them: the C library's tables read out of `libm.so.6`,
  and the reference's own constants and literals.
- **Literals.** The few decimal literals of an expression are carried by name. The build, or a test, checks that the
  Python source still writes each one in the expression named.
- **The header test.** A test reads the header back against the modules one value at a time. A deliberately perturbed
  header must differ in exactly one bit.

**The build cache and its manifest.**
- **The cache.** Each library is built into a cache folder outside the source tree. It is named by the SHA-256 of its
  inputs: the source, the generated header, the flags, the targets and the compiler's path.
- **The manifest.** Beside the library a manifest records the source's and the header's SHA-256, nvcc's version, the
  flags and targets, each kernel's registers and spills, the PTX check's result, the library's SHA-256 and the UTC time
  of the build.
- **Reuse.** A library already built from the same inputs is reused only if its SHA-256 still matches the manifest.

**Builds for planted faults.** The tests compile copies of a source with one deliberate fault each. Those builds
record the source scan and the PTX check without enforcing them, so the fault reaches the comparison. They are the
only builds that do (`exactness.md`).

**Optional features keep the default machine code.** Some features are compiled in only behind a compile-time switch,
among them static collidables, minis, force fields and the composed park's candidates. When such a switch was added,
the default build's PTX and SASS were compared with the build before it and found identical.

## 6. Orders that decide results

Some results depend on an order, not on arithmetic. The port reproduces each order instead of avoiding it.

- **The order of collision contacts.** destiny applies a ball's contacts in the order its spatial partition's walk
  finds them.
  - The batched parks compute an integer sort key for each pair of ball and candidate. The key holds the segment, the
    part and the level or post-order code of the box. Pairs sort lexicographically by (key, candidate id, slot), which
    reproduces the walk.
  - The kernel finds the next contact in that order by rescanning, so it needs no per-thread list.
- **libstdc++'s unordered containers.** The proximity map, each sensor's list and each ball's followers are iterated
  in libstdc++'s order.
  - The reference reproduces libstdc++'s `_Prime_rehash_policy` (the bucket counts, the growth points and the
    re-linking on rehash) in a clean-room implementation, written from a behavioural specification without the
    library's code (`provenance.md`, section 3.5).
  - The table of primes is read out of the library destiny's harness loads (libstdc++ from GCC 14.3.0; `platform.md`),
    never typed.
- **CPython's dictionaries and sets.** destiny's server layer is Python. Its iteration orders are CPython's, and the
  transcription keeps them.
- **Heap addresses.** destiny cleans up removed balls in an order that follows their heap addresses. No port can
  reproduce that order. The reference uses ascending id there (`limits.md`).

## 7. What the tests cannot see

A few choices cannot be observed by any test, because no input can show them. They are kept as the C library's
compiled code has them, and they rest on the disassembly alone.
- **A last-bit error in a `double` entry of `expf`'s table** changes none of the 4,294,967,296 `float` results: the
  final rounding to `float` removes it for every input. The same holds for `pow`'s `double` tables wherever `pow`'s
  result is rounded to `float`.
- **The last fused step of `atan2`'s table polynomial** rounds a term of about 2^-9 of the result's last place, so no
  argument can see whether it is fused.
- **The placement of `powf`'s nine fused steps** is not observable at a mushroom's one call. With all nine unfused,
  the device's `powf` still equals the C library's on all 4,294,967,296 values of `x` at `y = 0.25f`. Exactness there
  holds either way. For other arguments, the placement rests on the disassembly.
- **The high coefficients of the Taylor polynomials in `sin` and `cos`** are too small for an error in their last
  bit to reach any result.
