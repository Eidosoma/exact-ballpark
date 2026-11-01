# Changelog

All notable changes are recorded here. Versions follow `0.y.z` until the interface settles. Each release names the
upstream commits it is checked against (destiny, carbonengine/math, DirectXMath) and the platform it is qualified on.

## [Unreleased]

- The first public tree: the package (`src/exact_ballpark`), its tests, the conformance kit, the benchmark harness's
  core and an example.
  - Checked against destiny `114e89fa`, carbonengine/math `82733220` and DirectXMath `f5d82574`; qualified on GCC
    14.3 with glibc 2.34 (aarch64 Linux) and on the NVIDIA GH200 (sm_90).
  - libstdc++'s hash-table order is reproduced by a clean-room implementation (`docs/provenance.md`, section 3.5).
