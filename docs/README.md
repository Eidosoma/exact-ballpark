# Documentation

- **`numerics.md`:** the numerical contract. It covers where destiny computes in `float` and where in `double`, which
  C-library functions are transcribed and how, where fused multiply-adds occur, and how the CUDA builds are held to all
  of it.
- **`exactness.md`:** the evidence that the three layers agree. It covers the scenario generators, the held-out seeds,
  the planted faults the tests must catch, the CUDA-graph checks and the volumes compared, feature by feature.
- **`provenance.md`:** what is ported from where. It gives the upstream repositories and their pinned commits, the
  SHA-256 of the sources used, and how destiny's C++ was transcribed and checked.
- **`platform.md`:** the qualified platform, what "qualified" means, and how to qualify another compiler, C library or
  GPU.
- **`limits.md`:** what is not covered, or covered only on some paths, park by park.
- **`rules.md`:** the table of destiny's rules this port is held to. Each rule has its source location in destiny and
  the tests that check it.
