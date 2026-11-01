# Tests

Three subsets, selected with pytest markers:

- **`pytest -m cpu`:** the scalar reference and the batched parks on the CPU.
- **`pytest -m oracle`:** destiny compiled from source against the reference. Run the conformance kit first.
- **`pytest -m gpu`:** the CUDA kernels against the batched parks, and their CUDA-graph replays. These need an NVIDIA
  GPU and `nvcc`.

Every comparison is exact equality. Scenario generators are seeded, and each feature also has held-out seeds and
deliberately planted faults that the tests must catch (`docs/exactness.md`).
