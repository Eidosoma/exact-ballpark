# Provenance

This document covers:
- what exact-ballpark ports, and from where;
- the commits and checksums of the upstream sources;
- how destiny's code was transcribed and held to destiny itself.

The licences and notices are in `LICENSE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md`.

## 1. What is ported from where

| Upstream | What exact-ballpark takes from it | Pinned commit |
|---|---|---|
| **destiny** (CCP Games), https://github.com/carbonengine/destiny | Its behaviour, from three parts of its source. The C++ engine in `src/`: the ballpark, its tick, flight modes, collisions, spatial partition, bubbles, sensors and combat movement. The Python server and client layers in `python/destiny/net/`. The expected values of its published Python tests. | `114e89fa584f0e1a81eb0a679aabf41af8870072` (committed 2026-10-03T13:49:13+02:00) |
| **carbonengine/math** (CCP Games), https://github.com/carbonengine/math | The `float` vector, quaternion and matrix types and operations that destiny calls. | `8273322093eb1f80168ae60a35891f2d454369f9` (committed 2026-09-28T20:39:43Z) |
| **DirectXMath** (Microsoft), https://github.com/microsoft/DirectXMath | The scalar path of `XMQuaternionRotationRollPitchYaw`, which carbonengine/math's `RotationQuaternion(yaw, pitch, roll)` calls. | `f5d82574b03417d905fdd2f85bfa9bf7a2f1deac` (2026-10-21) |
| **The GNU C Library 2.34**, as built for aarch64 | The elementary functions destiny's results depend on, transcribed from the compiled library (`numerics.md`, section 3). They come from IBM's Accurate Mathematical Library, Arm's optimized-routines and Sun's fdlibm. Also the algorithm of its `rand()` generator. | the library file itself: `libm.so.6` of the package `glibc-2.34-283.el9_8.aarch64`, SHA-256 `b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86` |
| **libstdc++** (GCC 14.3.0) | The behaviour of the bucket policy of its unordered containers (`_Prime_rehash_policy`), which decides the order in which destiny's compiled code walks its hash maps and sets. It is re-implemented clean-room, with none of the library's code (section 3.5). The table of primes is read out of the library. | the library file itself: `libstdc++.so.6.0.33`, SHA-256 `ac709fc594acdcdf9b19868b3ad7d7275815018f82d39f028710e3706407022f` |

**What is never copied.** destiny, carbonengine/math and DirectXMath are not copied into this repository. The
conformance kit fetches them at the pinned commits and checks every file against its SHA-256
(`conformance/README.md`). Nothing from any game's content or data is used.

**Which revisions destiny was built with is not known.**
- destiny's `vcpkg.json` asks for carbon-math 1.2.1 or later and pins no revision.
- carbonengine/math's own dependency list asks only for a DirectXMath release from 2025-04-03 or later.

Every bit-level comparison therefore names the commits above.

## 2. The SHA-256 of the sources used

The conformance kit reads each repository only at its pinned commit and refuses any other. When it builds a harness,
it records the SHA-256 of every source file it reads. The tables below are those records, copied from the kit's manifests.
The four harnesses agree on every file they share.

### 2.1 destiny at `114e89fa` (43 files)

| File | SHA-256 |
|---|---|
| `src/AABB.cpp` | `46a52de07c56060c49665918e74bfcb7ee59504ef0122a35a5467cb256369113` |
| `src/AABB.h` | `30c9e27ac95c8109ae978cbd0d31e3d1da42cfca59af7aa5099606994773f24d` |
| `src/Ball.cpp` | `cac570282018a3703fc5f08b857140fe6f55de2c30b1267a65a1273fbe12aae4` |
| `src/Ball.h` | `c8867329c07557a154438c0e9d32ac874e149afa1297f48f68b94fe3a07f28b3` |
| `src/Ballpark.cpp` | `c8d462d295edbb24567e953bce983db8d0af791fb9aafb9189fb7ad00fb0162a` |
| `src/Ballpark.h` | `2fc7d886c089d9e477a36ec635c86305bcfd5d775051fb366ab55e9b81117ffb` |
| `src/Box.cpp` | `d72b01c8fc00bb5ac0381c7936fa698a57231812995a3becedba0d2efe4e0fd8` |
| `src/Box.h` | `c12df62ee09a04437da7edb073cd86c71d67d4d3863c26fc103117f6ce97cecd` |
| `src/BoxShape.cpp` | `56355ed5b93e83e7c02137f3b4eb6d3e305c24451c684a692297cd4a48c19535` |
| `src/BoxShape.h` | `b79b21412f0d2958f220d0090ff902ef156a29ba6fc323ebb343c47a2d458603` |
| `src/Capsule.cpp` | `9539900224c89599060dfdc3e8d5715d94cab8a380dc4075d49a0994aa2f7dac` |
| `src/Capsule.h` | `42295c4d67f46395a3aeae761e735183e7c368e881b78e0e9373cf20e7267df1` |
| `src/Collision.cpp` | `0c10833be7e92783d04ba7e74873dcca06637c178a7139c80c6695c76dc774fb` |
| `src/Collision.h` | `6ccc852b58dc3558d5a64b24383ed559b1b5008b1e92e434d2b993d2961f5679` |
| `src/CollisionBallProperties.cpp` | `d34bbd39972304abfd5d7f205a281dca01ca5942c2601cc05c68f00a73935097` |
| `src/CollisionBallProperties.h` | `3dac15d8d5999e10b4df0ed0c8c098170eb188ba9037805b1b2cdf27bd5790d2` |
| `src/CollisionStructures.cpp` | `3393b69d26ba10ae1411d49739d3af02b4c92bccbc9a4cd2df7fe915dd847200` |
| `src/CollisionStructures.h` | `cd6b5985a1b40dd4749fe6a321a4d7cbe1fcc37bb7b8ba2db39da2bd4dd80a26` |
| `src/IDstConstants.h` | `90d722e6815d83b183a3903fcd9ffc25f1473aeb876f8615c30f0a601a5e27d8` |
| `src/MapOfLongLongs.h` | `dcdb36c2741f6b11dddb7b7fcd6c263b00c351c12fb33fdbfe339db208efc085` |
| `src/MiniBall.cpp` | `c8f961b90b1e96c2129e714727a77caf1f7056e221f44d16cbb66d07e3a4ed17` |
| `src/MiniBall.h` | `6844d5089b0a69292c210fdd6bed24e57830a8350854520539a1c98502194a1c` |
| `src/MiniBox.cpp` | `52cb5e758e85c03c2db67dfc4122fa262f6eb7087b10260618c18a96704f9968` |
| `src/MiniBox.h` | `8a73c3b12527376ff55fc016f294956a5917708101adaa6fb962d39077586f6a` |
| `src/MiniCapsule.cpp` | `6fa84bb53f117d1db10e4457ad0e21a3fef7ebce3200d28c4d079372c1a001cb` |
| `src/MiniCapsule.h` | `cd8c4d73aa7e99eb816893fb0a6783d046d7b2148d2fbba754715eaf7d3e6b3d` |
| `src/OrientedBox.cpp` | `09e5ddfa96b8056279cb1cdbd6f4f1ad28d2e2dc06fa6a148b6869da46b12933` |
| `src/OrientedBox.h` | `cf9d293a4271f2f83a150643da53a991c6a37ec239399d9cfbf0c3cf631e85df` |
| `src/Partition.cpp` | `ed46c5caaf1e2fd926ddb63b3021a4c5034c6a2b67f67ab2ec2a391798a44677` |
| `src/Partition.h` | `31bfaf15eedd3f5bf73d7f661376a06b363457c0b045f38e35517260bdc68398` |
| `src/Partitionable.cpp` | `39190aeb458e8bea2aae4ea09d53b9f251338bbf738b116d0538ae6c4dc93fea` |
| `src/Partitionable.h` | `f7c8d7c6172877c2b8f1873092634508c7d2b2e53ffd83e955bff20f1ffd3785` |
| `src/Plane.cpp` | `5a3a6502829f6ca6fae81f28c79deb0bbb53664db0e9fad04254b8700a317919` |
| `src/Plane.h` | `b97c0af7c16cd94004c8cb3996c81a20cb4bce9b00feb023764a19d51d7ca730` |
| `src/Quaternion.h` | `a0311a2fbef3d973f6675fcebe718c7be6978aa0a15927a51fc4cf0aa40c64dc` |
| `src/Settings.h` | `8d6fc829a2620c5f7e3efc073485645a48bd9ca9bc183210eca60b2ea5ecfdf8` |
| `src/SortedSets.h` | `8d1482f2204b28d7bc16827b460432708d0ef4e36eb95faeae37c88618326cfc` |
| `src/StaticCollidable.cpp` | `752a22a1adbdca6f60bd2607e4442eb4469729549f9dc8a8612012ab363d374a` |
| `src/StaticCollidable.h` | `a295866dc809a2237f41de304b270ab79d78f733ba4ca498735616d8c9ee7e84` |
| `src/Thunkers.cpp` | `61acb87a22569561e7a608cd7a9a82698145afd2d5a3671365c24372a035ea17` |
| `src/Triangle.cpp` | `2c64bc0eb39abbc4b9f397174b15ba3828684593b8138a3befeb3c7405cf4809` |
| `src/Triangle.h` | `db9d44e7efbe630b717414e02197331e9a6eebf115d96bea867736f1eda395aa` |
| `src/Vector3d.h` | `7d74c8349232c6b31f01f3bbac0f5de2a0fc387c98dfa8449fc71b6035b96238` |

### 2.2 carbonengine/math at `82733220` (8 files)

| File | SHA-256 |
|---|---|
| `include/Matrix.h` | `6b15d8a08ab89caf0ae1830ea5606851725adde5a1467feebad6ee8e4a0f8e1c` |
| `include/Matrix_inline.h` | `61994486b74f52699885b4f221607466e102ade3ddff39b460387790673f5a21` |
| `include/Quaternion.h` | `16df2f185866be9fc75d2809e014cf840e637e1e6b7a9972ff5f7d3a7fd2d759` |
| `include/Quaternion_inline.h` | `5c1b499601b1f35a39154d9986a56ab206c21b9127304e80a7aeb1d421400b32` |
| `include/Requirements.h` | `4e7db91f444567a4a3449d07ae1b6a03513c305ec8df4204478d09f5ad31a7b9` |
| `include/Vector3.h` | `5d06657ff829a49a9069e4aeb286551300f0b00111f86894314fee668793b3af` |
| `include/Vector3_inline.h` | `741990d6e86c64ae6df2e35a25ba56bdbf0efd0231f3e362594d9df6d8d7270c` |
| `src/Quaternion.cpp` | `d1628ac912b7369ece630dd3cdf899733ed2be187f1aa52ac9c50b7ba098934a` |

### 2.3 DirectXMath at `f5d82574` (5 headers)

The build checks these headers against their SHA-256 before compiling. The compiler's dependency list must name no
other DirectXMath file.

| File | SHA-256 |
|---|---|
| `Inc/DirectXMath.h` | `489726b39be07e945cbbdbc19752c80367286ee2652af1da153d62cfce36ce87` |
| `Inc/DirectXMathConvert.inl` | `cb764b2e91b79ae3bcae650b1e15fbc99b507cf316f49c3d32644186bb94f6a5` |
| `Inc/DirectXMathMatrix.inl` | `31fe84d5c869858f5b243b777f7e2f76fdfe57b3dc434844f274d6b7a4c15b6f` |
| `Inc/DirectXMathMisc.inl` | `3f1d7cb555e7a796c75344897f28f21635a3f83bca201a7650dd2b75f111850f` |
| `Inc/DirectXMathVector.inl` | `8353782a3ef72070c526c5293c2e353b5c9e09cfb770112ed1df1185a0cb2e36` |

### 2.4 destiny's Python layers

The scalar server and client layers transcribe these files of destiny's Python package. Every class and method cites
its source by file and line.
- **The server layer, under `python/destiny/net/server/`:**
  - `_actions.py`, lines 14 to 766;
  - `_baseticker.py`, lines 10 to 115;
  - `_ticker.py`, lines 15 to 196;
  - `_bubbleupdater.py`, lines 6 to 54;
  - `_parkupdatebatcher.py`, lines 15 to 359;
  - `_interface.py`, lines 16 to 195.
- **The constant and helpers the server layer uses:**
  - `python/destiny/net/const.py`, lines 6 to 8;
  - `python/destiny/_util/signal.py`, lines 8 to 101;
  - `python/destiny/_util/timing.py`, lines 7 to 31;
  - `python/destiny/_util/settings.py`, lines 6 to 31.
- **The client layer, under `python/destiny/net/client/`:**
  - `_baseticker.py`, lines 12 to 140;
  - `_ticker.py`, lines 14 to 415;
  - `_util.py`, lines 6 to 68;
  - `_interface.py`, lines 6 to 118.
- **The client package's own account of the timing:** `python/destiny/net/__init__.py`, lines 3 to 23.

These files are pinned by the commit. The conformance kit's fetch step checks them, with every other fetched file,
against its manifest of SHA-256.

## 3. How destiny's code was transcribed

### 3.1 The scalar reference

The scalar reference is a transcription of destiny's C++ into Python, function by function, in destiny's order of
operations. Docstrings and comments cite destiny's file and line for each piece, and destiny's rule identifiers
(`rules.md`). The modules are:

| Module | What it transcribes |
|---|---|
| `exact_ballpark.ref` | the integrator, goto and stop |
| `exact_ballpark.modes` | follow, orbit (both laws) and warp, with destiny's tick and phase order |
| `exact_ballpark.space` | the spatial partition, bubbles, proximity sensors, visibility and range queries, both collision methods, and removal |
| `exact_ballpark.full` | missiles, formations, static collidables and minis, cloaking, dynamical orientation and the orientation without it, mushrooms, trolls, the accuracy helpers and OnTick |
| `exact_ballpark.server` | destiny's Python server layer, and the C++ functions it depends on (the bubble lists and the byte streams) |
| `exact_ballpark.client` | destiny's Python client layer |

**Types.**
- A Python float is a C `double`.
- A `float` member or parameter is rounded to `float` wherever destiny stores or converts one.
- The C library's functions are called through Python's `math` module and `ctypes` (`numerics.md`).

**Undefined behaviour.** Where destiny's C++ leaves a result undefined, the reference refuses the call by raising an
error, and scenarios that reach it are counted apart. Examples:
- removing a capsule or an oriented box by an id that is not there (destiny inserts a null entry, which a later add
  dereferences);
- a capsule whose diameter is below half an ulp of its length, which never finishes filing;
- a formation read past the end of the formation list after a smaller `LoadFormations`;
- a formation slot freed out of range (a `std::bitset<16>` reset past its end).

**destiny's Python server and client layers** were read, never imported or run, and transcribed with every loop and
call in its order. The dictionaries' and sets' iteration orders are CPython's, as in destiny. The C++ functions they
depend on are transcribed with them: the bubble additions and deletions, and the streams that write and read balls. The
streams produce the same bytes as the C++ on a little-endian machine.

A few stand-ins are labelled where they are made:
- `blue.MemStream` is a byte buffer;
- `blue.marshal` keeps the marshalled structure itself, not its byte format;
- profiling timers do nothing;
- the process-wide settings are a module-level object;
- events sent to the game's Python side are recorded.

### 3.2 The conformance kit

The kit (`conformance/`) compiles destiny's own code into small test programs (harnesses). Each harness prints the
state of its slice after every step of a scenario (the balls, and where the slice has them the partition, the bubbles,
the events and the answers to queries), so the reference can be compared with destiny itself.

**destiny's code is extracted, never retyped.**
- Functions, classes and enums are extracted from the pinned source by brace matching, with comments and literals
  skipped.
- Macros, constants, typedefs and globals are extracted as whole lines, by exact match.
- The struct declarations of carbonengine/math are extracted with their DirectXMath conversion lines removed. Each
  removed line is recorded.

| Build script | What it compiles |
|---|---|
| `conformance/build.py` | the integrator, goto thrust and stop (`Ballpark::Integrate`, `GotoThrust`, `EvolveStop`, `SetBallTimeFactor`) and `src/Vector3d.h` |
| `conformance/build_modes.py` | destiny's tick (`Ballpark::Evolve`, `EvolveBehaviorForBall`) with follow, orbit and warp and their commands |
| `conformance/build_space.py` | the partition, bubbles, proximity sensors, visibility, range queries, both collision methods and removal |
| `conformance/build_full.py` | missiles, static collidables and minis, cloaking, dynamical orientation and the orientation without it, mushrooms, trolls, formations, the accuracy helpers and OnTick, with carbonengine/math's types and, optionally, DirectXMath |
| `conformance/hashtable_probe.cpp` | not destiny's code: a probe of libstdc++'s unordered containers, compiled with the harnesses' compiler and flags |

**What a harness adds around destiny's code.** A prelude supplies what is not destiny's simulation code:
- the members with destiny's own types (`float` members stay `float`);
- a minimal stand-in for the CPython interface that destiny's bindings call;
- the list and class macros of destiny's binding layer;
- destiny's global switches at their defaults, with the orbit law, the collision method and dynamical orientation set
  per scenario.

Each slice stubs what lies outside it. A driver reads scenarios on standard input. Events are printed. The display
state of a client's ball is not part of the simulation and is not compiled.

**How it is compiled.**

```text
g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math
```

Two additions apply to `build_full.py` only:
- **`-fsigned-char`.** Plain `char` is unsigned on aarch64 Linux but signed on x86 and under MSVC, and that slice
  reaches destiny's `char` values: a formation id of −1, the cloak mode.
- **DirectXMath's scalar path** (`-D_XM_NO_INTRINSICS_`). Linux has no `sal.h`, so the build writes its own: the ten
  annotations DirectXMath uses, defined empty. They carry no meaning for code generation.

A second program, `dxm_probe`, is the quaternion function alone, so the transcription can be held to it on any angles.
Each build directory receives the generated source, the binary and a manifest: the SHA-256 of every source file read
and every extracted piece, the compiler, its command and its warnings.

**Where destiny's tests and its compiled code disagree, the compiled code decides.** Two examples:
- The moving-leader vector of `test_follow.py` differs in the tenth decimal from the code compiled on the qualified
  platform. The published vectors were made on another platform.
- `test_iterative_collision.py`'s zero-length-capsule test measures its distance from a different capsule, so the
  zero-length one is never checked.

### 3.3 The C library's functions

Each function is transcribed from the disassembly of the pinned `libm.so.6`, operation for operation, with every
fused multiply-add the compiler made and no other (`numerics.md`).
- **Constants and tables are read by script.** They are read out of the library at the addresses the code loads them
  from, after checking each loading instruction's text, or are assembled from its immediate operands.
- **Where each value came from is kept.** The address each value was read from is recorded beside it. The one
  exception is `exp`'s table, which is generated from its definition and checked against the 256 words read from the
  library.
- **The tests check the bytes.** When the tests run on that library, they hold every table and constant to its bytes.

### 3.4 The batched parks and the kernels

The batched parks and the CUDA kernels are new code. They run the reference's algorithms on tensors and in CUDA, with
the same operations in the same order. They are not transcribed from destiny directly: they are held to the reference
bit for bit, and in places to destiny's compiled code directly (`exactness.md`).

### 3.5 The hash tables: a clean-room implementation

destiny walks its hash maps and sets in the iteration order of GCC's libstdc++, so the reference and the space kernel
reproduce that order (`numerics.md`, section 6). The code that does so, `StdHashMap` and `StdHashSet` in
`src/exact_ballpark/space.py` and the `umap_*` functions in `src/exact_ballpark/vec_space.cu`, is not derived from
libstdc++'s code. It was written clean-room from a behavioural specification: the containers' rules in words (how a key
is linked into the order, when and how the bucket count grows, and what a rehash, an erase, a clear and a copy do),
worked examples, and the table of bucket counts as data. The specification holds no code, and was written by a
different author from its implementer. The implementer worked from the specification, the rest of this repository and
its tests, and never saw libstdc++'s source or headers, nor any earlier implementation. The table of primes is data,
read out of the compiled library (`platform.md`). Both parts are under the MIT licence. The tests hold them to the
library itself: a probe compiled against libstdc++ (`conformance/hashtable_probe.cpp`) checks the prime table, the
growth policy's answers, and each container's bucket count and order after every operation, on development and
held-out sequences; planted faults show that the tests catch a wrong rule.
