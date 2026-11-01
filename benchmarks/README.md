# Benchmarks

Speed probes for the parks: collisions, flight modes, orientation, space and sensors, and the server layer. They
compare the scalar reference, the batched parks on the CPU and the CUDA kernels at several world counts.

No speed is quoted in this repository until it has been measured on an otherwise idle GPU. The few relative
costs in `docs/limits.md` come from a shared GPU and are marked provisional.
