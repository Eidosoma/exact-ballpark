# Examples

- **`flight_modes_on_gpu.py`**: a few thousand worlds of ships following, orbiting and warping, stepped as a CUDA graph.
  The first few worlds (8 by default) are also stepped by the scalar reference, and every ship's position and velocity
  are compared bit for bit after every tick.
