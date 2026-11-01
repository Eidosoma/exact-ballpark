# The conformance kit

The kit checks the reference against destiny itself.

1. `fetch_upstream.py` clones the three upstream repositories at their pinned commits. It checks that each checkout is
   the commit's recorded tree and is clean, and writes a SHA-256 manifest of every file beside it:
   - destiny `114e89fa584f0e1a81eb0a679aabf41af8870072`;
   - carbonengine/math `8273322093eb1f80168ae60a35891f2d454369f9`;
   - DirectXMath `f5d82574b03417d905fdd2f85bfa9bf7a2f1deac`.
2. The build scripts extract destiny's C++ from the pinned sources by brace matching, never retyping it. They compile
   it into small test harnesses with `g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math`.
3. The `oracle` tests drive each harness and the reference with the same scenarios, and compare every value.

On another compiler or C library the harnesses test the reference against destiny as built there. That is still
meaningful, but it is not the qualified platform.
