# Benchmarks

Each row is one measurement: the median of its timed windows, the spread of those windows ((max - min) / median) and the median time of a batch tick, which steps every world of the batch once. Worlds is the number of independent worlds in the batch and Ships the number of ships in all of them (missiles, fixed spheres, obstacles and static collidables are not counted as ships). Each rate is given twice: in world-ticks a second (one world stepped one tick) and in ship-ticks a second (one ship stepped one tick). A measurement is clean when no other process was on its GPU and the GPU and the host were idle in the gaps before and after its windows. The scalar reference is pure Python, one park a world, and the CPU tensor path runs with the torch threads each environment shows: their ratios (Against: world-ticks over world-ticks of the same configuration, so the same in ship-ticks) are workload measurements of these implementations, not a claim against an optimised whole-CPU implementation and not an exactness result (the conformance tests are). Only clean measurements are shown.

## Environment

| | |
|---|---|
| GPU | NVIDIA GH200 120GB (driver 610.57.04, 132 SMs, 97871 MiB) |
| Compilers | Cuda compilation tools, release 13.0, V13.0.48; torch 2.9.1+cu128 (CUDA 12.8) |
| Python, C library | Python 3.13.5, glibc 2.34 |
| CPU | aarch64, 288 CPUs, torch threads 1 |
| Source | the engine of this repository's first public tree (0.1.0.dev0) |
| Measured | 2026-11-01T23:38:52Z to 2026-11-02T00:03:49Z |

The numbers were measured on the engine of this repository's first public tree. The later device-contract update changed no measured path: the composed park's collisions-off kernel path, the combat graph's refusal of a composed park and the space layer's CPU build without nvcc are not measured cells.

### Collisions (the gradient method), tensor path (`collide`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| collisions=off,obstacles=6,ships=2 | torch_gpu | 8,192 | 16,384 | 3.77 M | 7.55 M | 0.086 | 2.17 |  |
| collisions=on,obstacles=12,ships=4 | torch_gpu | 1,024 | 4,096 | 51.4 k | 205 k | 0.061 | 19.9 |  |
| collisions=on,obstacles=12,ships=4 | torch_gpu | 8,192 | 32,768 | 306 k | 1.22 M | 0.13 | 26.8 |  |
| collisions=on,obstacles=12,ships=4 | torch_gpu | 65,536 | 262,144 | 662 k | 2.65 M | 0.011 | 98.9 |  |
| collisions=on,obstacles=6,ships=2 | torch_cpu | 256 | 512 | 28.3 k | 56.7 k | 0.012 | 9.03 |  |
| collisions=on,obstacles=6,ships=2 | torch_gpu | 1,024 | 2,048 | 65.7 k | 131 k | 0.079 | 15.6 |  |
| collisions=on,obstacles=6,ships=2 | torch_gpu | 8,192 | 16,384 | 369 k | 738 k | 0.043 | 22.2 |  |
| collisions=on,obstacles=6,ships=2 | torch_gpu | 65,536 | 131,072 | 1.9 M | 3.81 M | 0.055 | 34.4 |  |

### The combat park: missiles and formations (`combat`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| do=1,missiles=4,ships=4 | reference | 16 | 64 | 664 | 2.66 k | 0.0068 | 24.1 |  |
| do=1,missiles=4,ships=4 | torch_cpu | 256 | 1,024 | 1.44 k | 5.78 k | 0.035 | 177 | 2.18x reference |
| do=1,missiles=4,ships=4 | torch_gpu | 1,024 | 4,096 | 2.72 k | 10.9 k | 0.098 | 377 | 4.09x reference |
| do=1,missiles=4,ships=4 | torch_gpu | 8,192 | 32,768 | 22.5 k | 90.1 k | 0.16 | 364 | 33.9x reference |
| do=1,missiles=4,ships=4 | kernel | 1,024 | 4,096 | 696 k | 2.78 M | 0.085 | 1.47 | 1,050x reference |
| do=1,missiles=4,ships=4 | kernel | 8,192 | 32,768 | 5.21 M | 20.9 M | 0.17 | 1.57 | 7,850x reference |
| do=1,missiles=4,ships=4 | kernel | 65,536 | 262,144 | 21.8 M | 87.3 M | 0.12 | 3 | 32,900x reference |
| do=1,missiles=4,ships=4 | graph | 1,024 | 4,096 | 1.07 M | 4.28 M | 0.087 | 0.956 | 1,610x reference |
| do=1,missiles=4,ships=4 | graph | 8,192 | 32,768 | 7.31 M | 29.2 M | 0.15 | 1.12 | 11,000x reference |
| do=1,missiles=4,ships=4 | graph | 65,536 | 262,144 | 23.2 M | 92.7 M | 0.14 | 2.83 | 34,900x reference |

### The composed park: modes, missiles and collisions (`composed`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| do=1,missiles=2,ships=6,spheres=1 | reference | 16 | 96 | 598 | 3.59 k | 0.023 | 26.8 |  |
| do=1,missiles=2,ships=6,spheres=1 | torch_cpu | 256 | 1,536 | 1.32 k | 7.89 k | 0.023 | 195 | 2.2x reference |
| do=1,missiles=2,ships=6,spheres=1 | torch_gpu | 1,024 | 6,144 | 2.47 k | 14.8 k | 0.19 | 414 | 4.14x reference |
| do=1,missiles=2,ships=6,spheres=1 | torch_gpu | 8,192 | 49,152 | 19.8 k | 119 k | 0.15 | 415 | 33.1x reference |
| do=1,missiles=2,ships=6,spheres=1 | kernel | 1,024 | 6,144 | 522 k | 3.13 M | 0.069 | 1.96 | 873x reference |
| do=1,missiles=2,ships=6,spheres=1 | kernel | 8,192 | 49,152 | 3.89 M | 23.3 M | 0.17 | 2.11 | 6,510x reference |
| do=1,missiles=2,ships=6,spheres=1 | kernel | 65,536 | 393,216 | 13.5 M | 80.9 M | 0.076 | 4.86 | 22,600x reference |
| do=1,missiles=2,ships=6,spheres=1 | graph | 1,024 | 6,144 | 974 k | 5.84 M | 0.082 | 1.05 | 1,630x reference |
| do=1,missiles=2,ships=6,spheres=1 | graph | 8,192 | 49,152 | 6 M | 36 M | 0.11 | 1.36 | 10,000x reference |
| do=1,missiles=2,ships=6,spheres=1 | graph | 65,536 | 393,216 | 14.8 M | 89 M | 0.023 | 4.42 | 24,800x reference |

### A tick under a per-world mask (`masked`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| base=graph,fraction=0.125,park=combat | masked | 4,096 | 16,384 | 2.93 M | 11.7 M | 0.033 | 1.4 | 0.718x unmasked |
| base=graph,fraction=0.125,park=combat | masked | 65,536 | 262,144 | 18.7 M | 74.7 M | 0.081 | 3.51 | 0.767x unmasked |
| base=graph,fraction=0.125,park=composed | masked | 4,096 | 24,576 | 2.34 M | 14 M | 0.076 | 1.75 | 0.671x unmasked |
| base=graph,fraction=0.125,park=composed | masked | 65,536 | 393,216 | 10.3 M | 61.7 M | 0.017 | 6.37 | 0.694x unmasked |
| base=graph,fraction=0.125,park=modes | masked | 4,096 | 16,384 | 10.6 M | 42.4 M | 0.046 | 0.386 | 0.461x unmasked |
| base=graph,fraction=0.125,park=modes | masked | 65,536 | 262,144 | 25.8 M | 103 M | 0.15 | 2.54 | 0.751x unmasked |
| base=graph,fraction=0.125,park=orient | masked | 4,096 | 4,096 | 10.1 M | 10.1 M | 0.018 | 0.404 | 0.915x unmasked |
| base=graph,fraction=0.125,park=orient | masked | 65,536 | 65,536 | 70.4 M | 70.4 M | 0.05 | 0.93 | 0.918x unmasked |
| base=graph,fraction=0.5,park=combat | masked | 4,096 | 16,384 | 2.94 M | 11.8 M | 0.023 | 1.39 | 0.721x unmasked |
| base=graph,fraction=0.5,park=combat | masked | 65,536 | 262,144 | 19.3 M | 77.1 M | 0.045 | 3.4 | 0.791x unmasked |
| base=graph,fraction=0.5,park=composed | masked | 4,096 | 24,576 | 2.38 M | 14.3 M | 0.041 | 1.72 | 0.682x unmasked |
| base=graph,fraction=0.5,park=composed | masked | 65,536 | 393,216 | 11.3 M | 67.6 M | 0.052 | 5.81 | 0.76x unmasked |
| base=graph,fraction=0.5,park=modes | masked | 4,096 | 16,384 | 10.7 M | 42.6 M | 0.028 | 0.384 | 0.463x unmasked |
| base=graph,fraction=0.5,park=modes | masked | 65,536 | 262,144 | 26.5 M | 106 M | 0.13 | 2.47 | 0.773x unmasked |
| base=graph,fraction=0.5,park=orient | masked | 4,096 | 4,096 | 10.6 M | 10.6 M | 0.014 | 0.386 | 0.957x unmasked |
| base=graph,fraction=0.5,park=orient | masked | 65,536 | 65,536 | 71.4 M | 71.4 M | 0.058 | 0.917 | 0.931x unmasked |
| base=graph,fraction=1.0,park=combat | masked | 4,096 | 16,384 | 2.94 M | 11.8 M | 0.066 | 1.39 | 0.72x unmasked |
| base=graph,fraction=1.0,park=combat | masked | 65,536 | 262,144 | 18.8 M | 75.3 M | 0.13 | 3.48 | 0.773x unmasked |
| base=graph,fraction=1.0,park=composed | masked | 4,096 | 24,576 | 2.41 M | 14.4 M | 0.054 | 1.7 | 0.69x unmasked |
| base=graph,fraction=1.0,park=composed | masked | 65,536 | 393,216 | 11.4 M | 68.5 M | 0.034 | 5.74 | 0.769x unmasked |
| base=graph,fraction=1.0,park=modes | masked | 4,096 | 16,384 | 10.5 M | 41.9 M | 0.034 | 0.391 | 0.456x unmasked |
| base=graph,fraction=1.0,park=modes | masked | 65,536 | 262,144 | 27.4 M | 110 M | 0.11 | 2.39 | 0.798x unmasked |
| base=graph,fraction=1.0,park=orient | masked | 4,096 | 4,096 | 10.6 M | 10.6 M | 0.017 | 0.387 | 0.955x unmasked |
| base=graph,fraction=1.0,park=orient | masked | 65,536 | 65,536 | 74.2 M | 74.2 M | 0.036 | 0.883 | 0.967x unmasked |
| base=graph,fraction=off,park=combat | graph | 4,096 | 16,384 | 4.08 M | 16.3 M | 0.043 | 1 |  |
| base=graph,fraction=off,park=combat | graph | 65,536 | 262,144 | 24.3 M | 97.4 M | 0.09 | 2.69 |  |
| base=graph,fraction=off,park=composed | graph | 4,096 | 24,576 | 3.49 M | 20.9 M | 0.03 | 1.17 |  |
| base=graph,fraction=off,park=composed | graph | 65,536 | 393,216 | 14.8 M | 89 M | 0.014 | 4.42 |  |
| base=graph,fraction=off,park=modes | graph | 4,096 | 16,384 | 23 M | 92 M | 0.064 | 0.178 |  |
| base=graph,fraction=off,park=modes | graph | 65,536 | 262,144 | 34.3 M | 137 M | 0.14 | 1.91 |  |
| base=graph,fraction=off,park=orient | graph | 4,096 | 4,096 | 11.1 M | 11.1 M | 0.013 | 0.37 |  |
| base=graph,fraction=off,park=orient | graph | 65,536 | 65,536 | 76.8 M | 76.8 M | 0.02 | 0.854 |  |
| base=kernel,fraction=0.125,park=collide | masked | 4,096 | 16,384 | 8.52 M | 34.1 M | 0.17 | 0.481 | 0.867x unmasked |
| base=kernel,fraction=0.125,park=collide | masked | 65,536 | 262,144 | 36.4 M | 145 M | 0.066 | 1.8 | 1.04x unmasked |
| base=kernel,fraction=0.125,park=combat | masked | 4,096 | 16,384 | 1.25 M | 5.01 M | 0.075 | 3.27 | 0.451x unmasked |
| base=kernel,fraction=0.125,park=combat | masked | 65,536 | 262,144 | 14.6 M | 58.4 M | 0.083 | 4.49 | 0.68x unmasked |
| base=kernel,fraction=0.125,park=composed | masked | 4,096 | 24,576 | 1.01 M | 6.06 M | 0.037 | 4.06 | 0.482x unmasked |
| base=kernel,fraction=0.125,park=composed | masked | 65,536 | 393,216 | 9.35 M | 56.1 M | 0.037 | 7.01 | 0.674x unmasked |
| base=kernel,fraction=0.125,park=modes | masked | 4,096 | 16,384 | 2.51 M | 10 M | 0.099 | 1.63 | 0.127x unmasked |
| base=kernel,fraction=0.125,park=modes | masked | 65,536 | 262,144 | 20.7 M | 82.6 M | 0.085 | 3.17 | 0.601x unmasked |
| base=kernel,fraction=0.125,park=orient | masked | 4,096 | 4,096 | 3.4 M | 3.4 M | 0.049 | 1.2 | 0.755x unmasked |
| base=kernel,fraction=0.125,park=orient | masked | 65,536 | 65,536 | 41.6 M | 41.6 M | 0.081 | 1.58 | 0.825x unmasked |
| base=kernel,fraction=0.5,park=collide | masked | 4,096 | 16,384 | 8.68 M | 34.7 M | 0.074 | 0.472 | 0.882x unmasked |
| base=kernel,fraction=0.5,park=collide | masked | 65,536 | 262,144 | 36.1 M | 145 M | 0.13 | 1.81 | 1.03x unmasked |
| base=kernel,fraction=0.5,park=combat | masked | 4,096 | 16,384 | 1.25 M | 4.99 M | 0.13 | 3.28 | 0.449x unmasked |
| base=kernel,fraction=0.5,park=combat | masked | 65,536 | 262,144 | 14.5 M | 57.8 M | 0.11 | 4.53 | 0.673x unmasked |
| base=kernel,fraction=0.5,park=composed | masked | 4,096 | 24,576 | 975 k | 5.85 M | 0.053 | 4.2 | 0.466x unmasked |
| base=kernel,fraction=0.5,park=composed | masked | 65,536 | 393,216 | 10 M | 60.1 M | 0.058 | 6.54 | 0.722x unmasked |
| base=kernel,fraction=0.5,park=modes | masked | 4,096 | 16,384 | 2.5 M | 10 M | 0.09 | 1.64 | 0.127x unmasked |
| base=kernel,fraction=0.5,park=modes | masked | 65,536 | 262,144 | 21.1 M | 84.3 M | 0.1 | 3.11 | 0.613x unmasked |
| base=kernel,fraction=0.5,park=orient | masked | 4,096 | 4,096 | 3.39 M | 3.39 M | 0.15 | 1.21 | 0.752x unmasked |
| base=kernel,fraction=0.5,park=orient | masked | 65,536 | 65,536 | 40.3 M | 40.3 M | 0.18 | 1.63 | 0.8x unmasked |
| base=kernel,fraction=1.0,park=collide | masked | 4,096 | 16,384 | 8.53 M | 34.1 M | 0.093 | 0.48 | 0.867x unmasked |
| base=kernel,fraction=1.0,park=collide | masked | 65,536 | 262,144 | 34.4 M | 138 M | 0.076 | 1.9 | 0.986x unmasked |
| base=kernel,fraction=1.0,park=combat | masked | 4,096 | 16,384 | 1.26 M | 5.03 M | 0.068 | 3.26 | 0.452x unmasked |
| base=kernel,fraction=1.0,park=combat | masked | 65,536 | 262,144 | 14.7 M | 58.9 M | 0.11 | 4.45 | 0.685x unmasked |
| base=kernel,fraction=1.0,park=composed | masked | 4,096 | 24,576 | 983 k | 5.9 M | 0.047 | 4.17 | 0.47x unmasked |
| base=kernel,fraction=1.0,park=composed | masked | 65,536 | 393,216 | 10 M | 60 M | 0.062 | 6.55 | 0.721x unmasked |
| base=kernel,fraction=1.0,park=modes | masked | 4,096 | 16,384 | 2.55 M | 10.2 M | 0.047 | 1.61 | 0.129x unmasked |
| base=kernel,fraction=1.0,park=modes | masked | 65,536 | 262,144 | 20.1 M | 80.5 M | 0.16 | 3.26 | 0.585x unmasked |
| base=kernel,fraction=1.0,park=orient | masked | 4,096 | 4,096 | 3.48 M | 3.48 M | 0.14 | 1.18 | 0.773x unmasked |
| base=kernel,fraction=1.0,park=orient | masked | 65,536 | 65,536 | 41.1 M | 41.1 M | 0.17 | 1.59 | 0.817x unmasked |
| base=kernel,fraction=off,park=collide | kernel | 4,096 | 16,384 | 9.84 M | 39.3 M | 0.13 | 0.416 |  |
| base=kernel,fraction=off,park=collide | kernel | 65,536 | 262,144 | 34.9 M | 140 M | 0.047 | 1.88 |  |
| base=kernel,fraction=off,park=combat | kernel | 4,096 | 16,384 | 2.78 M | 11.1 M | 0.042 | 1.47 |  |
| base=kernel,fraction=off,park=combat | kernel | 65,536 | 262,144 | 21.5 M | 85.9 M | 0.12 | 3.05 |  |
| base=kernel,fraction=off,park=composed | kernel | 4,096 | 24,576 | 2.09 M | 12.6 M | 0.19 | 1.96 |  |
| base=kernel,fraction=off,park=composed | kernel | 65,536 | 393,216 | 13.9 M | 83.2 M | 0.041 | 4.73 |  |
| base=kernel,fraction=off,park=modes | kernel | 4,096 | 16,384 | 19.8 M | 79.1 M | 0.057 | 0.207 |  |
| base=kernel,fraction=off,park=modes | kernel | 65,536 | 262,144 | 34.4 M | 138 M | 0.049 | 1.91 |  |
| base=kernel,fraction=off,park=orient | kernel | 4,096 | 4,096 | 4.51 M | 4.51 M | 0.063 | 0.909 |  |
| base=kernel,fraction=off,park=orient | kernel | 65,536 | 65,536 | 50.4 M | 50.4 M | 0.069 | 1.3 |  |

### Flight modes: follow, orbit, warp (`modes`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| do=0,mode=modes,park=modes | kernel | 256 | 1,024 | 894 k | 3.57 M | 0.1 | 0.286 |  |
| do=0,mode=modes,park=modes | kernel | 2,048 | 8,192 | 5.8 M | 23.2 M | 0.047 | 0.353 |  |
| do=0,mode=modes,park=modes | kernel | 16,384 | 65,536 | 19.8 M | 79.2 M | 0.13 | 0.827 |  |
| do=0,mode=modes,park=modes | kernel | 65,536 | 262,144 | 26.8 M | 107 M | 0.077 | 2.44 |  |
| do=0,mode=modes_graph,park=modes | graph | 256 | 1,024 | 1.01 M | 4.05 M | 0.07 | 0.253 |  |
| do=0,mode=modes_graph,park=modes | graph | 2,048 | 8,192 | 6.28 M | 25.1 M | 0.11 | 0.326 |  |
| do=0,mode=modes_graph,park=modes | graph | 16,384 | 65,536 | 20 M | 80 M | 0.068 | 0.82 |  |
| do=0,mode=modes_graph,park=modes | graph | 65,536 | 262,144 | 27.1 M | 109 M | 0.043 | 2.42 |  |
| do=0,mode=none,park=vec | torch_gpu | 1,024 | 1,024 | 1.25 M | 1.25 M | 0.079 | 0.819 |  |
| do=0,mode=none,park=vec | torch_gpu | 8,192 | 8,192 | 9.09 M | 9.09 M | 0.078 | 0.901 |  |
| do=0,mode=none,park=vec | torch_gpu | 65,536 | 65,536 | 52.3 M | 52.3 M | 0.045 | 1.25 |  |
| do=0,mode=none,park=vec | torch_gpu | 262,144 | 262,144 | 114 M | 114 M | 0.076 | 2.29 |  |
| do=1,mode=modes_do,park=modes | kernel | 256 | 1,024 | 553 k | 2.21 M | 0.036 | 0.463 |  |
| do=1,mode=modes_do,park=modes | kernel | 2,048 | 8,192 | 3.9 M | 15.6 M | 0.047 | 0.525 |  |
| do=1,mode=modes_do,park=modes | kernel | 16,384 | 65,536 | 13.3 M | 53.1 M | 0.038 | 1.23 |  |
| do=1,mode=modes_do,park=modes | kernel | 65,536 | 262,144 | 18.8 M | 75.2 M | 0.055 | 3.49 |  |
| do=1,mode=modes_do_graph,park=modes | graph | 256 | 1,024 | 567 k | 2.27 M | 0.03 | 0.452 |  |
| do=1,mode=modes_do_graph,park=modes | graph | 2,048 | 8,192 | 3.93 M | 15.7 M | 0.045 | 0.521 |  |
| do=1,mode=modes_do_graph,park=modes | graph | 16,384 | 65,536 | 13.3 M | 53.2 M | 0.032 | 1.23 |  |
| do=1,mode=modes_do_graph,park=modes | graph | 65,536 | 262,144 | 18.4 M | 73.6 M | 0.095 | 3.56 |  |
| do=1,mode=turn2,park=orient | kernel | 1,024 | 1,024 | 1.01 M | 1.01 M | 0.056 | 1.02 |  |
| do=1,mode=turn2,park=orient | kernel | 8,192 | 8,192 | 7.77 M | 7.77 M | 0.13 | 1.05 |  |
| do=1,mode=turn2,park=orient | kernel | 65,536 | 65,536 | 45.9 M | 45.9 M | 0.023 | 1.43 |  |
| do=1,mode=turn2,park=orient | kernel | 262,144 | 262,144 | 90.8 M | 90.8 M | 0.1 | 2.89 |  |
| do=1,mode=turn2_graph,park=orient | graph | 1,024 | 1,024 | 1.98 M | 1.98 M | 0.024 | 0.517 |  |
| do=1,mode=turn2_graph,park=orient | graph | 8,192 | 8,192 | 14.5 M | 14.5 M | 0.023 | 0.563 |  |
| do=1,mode=turn2_graph,park=orient | graph | 65,536 | 65,536 | 63.6 M | 63.6 M | 0.06 | 1.03 |  |
| do=1,mode=turn2_graph,park=orient | graph | 262,144 | 262,144 | 95.2 M | 95.2 M | 0.061 | 2.75 |  |

In this table, park=modes runs worlds of four ships (a leader, its follower, its orbiter and a ship warping), and park=orient and park=vec run every ship as its own world.

### Turning: dynamical orientation (`orient`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mode=cuda | kernel | 1,024 | 1,024 | 975 k | 975 k | 0.054 | 1.05 |  |
| mode=cuda | kernel | 8,192 | 8,192 | 7.36 M | 7.36 M | 0.076 | 1.11 |  |
| mode=cuda | kernel | 65,536 | 65,536 | 44.5 M | 44.5 M | 0.1 | 1.47 |  |
| mode=cuda | kernel | 262,144 | 262,144 | 92.6 M | 92.6 M | 0.028 | 2.83 |  |
| mode=cuda_graph | graph | 1,024 | 1,024 | 1.97 M | 1.97 M | 0.041 | 0.519 |  |
| mode=cuda_graph | graph | 8,192 | 8,192 | 14.2 M | 14.2 M | 0.02 | 0.576 |  |
| mode=cuda_graph | graph | 65,536 | 65,536 | 62.7 M | 62.7 M | 0.029 | 1.05 |  |
| mode=cuda_graph | graph | 262,144 | 262,144 | 94.1 M | 94.1 M | 0.059 | 2.79 |  |
| mode=dynamical | torch_cpu | 1,024 | 1,024 | 12.2 k | 12.2 k | 0.0078 | 83.9 |  |
| mode=dynamical | torch_gpu | 1,024 | 1,024 | 4.35 k | 4.35 k | 0.073 | 235 |  |
| mode=dynamical | torch_gpu | 8,192 | 8,192 | 34.5 k | 34.5 k | 0.057 | 237 |  |
| mode=dynamical | torch_gpu | 65,536 | 65,536 | 275 k | 275 k | 0.066 | 239 |  |
| mode=dynamical | torch_gpu | 262,144 | 262,144 | 1.06 M | 1.06 M | 0.047 | 247 |  |
| mode=none | torch_gpu | 1,024 | 1,024 | 1.39 M | 1.39 M | 0.084 | 0.738 |  |
| mode=none | torch_gpu | 8,192 | 8,192 | 9.99 M | 9.99 M | 0.094 | 0.82 |  |
| mode=none | torch_gpu | 65,536 | 65,536 | 56.6 M | 56.6 M | 0.15 | 1.16 |  |
| mode=none | torch_gpu | 262,144 | 262,144 | 120 M | 120 M | 0.07 | 2.18 |  |
| mode=ypr | torch_gpu | 1,024 | 1,024 | 322 k | 322 k | 0.072 | 3.18 |  |
| mode=ypr | torch_gpu | 8,192 | 8,192 | 1.15 M | 1.15 M | 0.11 | 7.15 |  |
| mode=ypr | torch_gpu | 65,536 | 65,536 | 2.29 M | 2.29 M | 0.033 | 28.6 |  |
| mode=ypr | torch_gpu | 262,144 | 262,144 | 2.29 M | 2.29 M | 0.0058 | 114 |  |

### The server layer over three parks (`server`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| layer=direct,mode=direct,park=collide,server=torch | kernel | 256 | 1,024 | 252 k | 1.01 M | 0.14 | 1.01 |  |
| layer=direct,mode=direct,park=collide,server=torch | kernel | 2,048 | 8,192 | 2.08 M | 8.32 M | 0.076 | 0.985 |  |
| layer=direct,mode=direct,park=collide,server=torch | kernel | 16,384 | 65,536 | 16.5 M | 66.1 M | 0.22 | 0.991 |  |
| layer=direct,mode=direct,park=collide,server=torch | kernel | 65,536 | 262,144 | 73 M | 292 M | 0.14 | 0.898 |  |
| layer=direct,mode=direct,park=modes,server=torch | kernel | 256 | 1,024 | 306 k | 1.23 M | 0.049 | 0.836 |  |
| layer=direct,mode=direct,park=modes,server=torch | kernel | 2,048 | 8,192 | 2.26 M | 9.03 M | 0.12 | 0.907 |  |
| layer=direct,mode=direct,park=modes,server=torch | kernel | 16,384 | 65,536 | 18.9 M | 75.4 M | 0.22 | 0.869 |  |
| layer=direct,mode=direct,park=modes,server=torch | kernel | 65,536 | 262,144 | 82.3 M | 329 M | 0.16 | 0.796 |  |
| layer=direct,mode=direct,park=orient,server=torch | kernel | 256 | 1,024 | 184 k | 737 k | 0.11 | 1.39 |  |
| layer=direct,mode=direct,park=orient,server=torch | kernel | 2,048 | 8,192 | 1.37 M | 5.49 M | 0.1 | 1.49 |  |
| layer=direct,mode=direct,park=orient,server=torch | kernel | 16,384 | 65,536 | 11 M | 44.1 M | 0.052 | 1.49 |  |
| layer=direct,mode=direct,park=orient,server=torch | kernel | 65,536 | 262,144 | 47.5 M | 190 M | 0.32 | 1.38 |  |
| layer=park,mode=park,park=collide,server=torch | kernel | 256 | 1,024 | 828 k | 3.31 M | 0.093 | 0.309 |  |
| layer=park,mode=park,park=collide,server=torch | kernel | 2,048 | 8,192 | 6.18 M | 24.7 M | 0.094 | 0.331 |  |
| layer=park,mode=park,park=collide,server=torch | kernel | 16,384 | 65,536 | 49 M | 196 M | 0.28 | 0.335 |  |
| layer=park,mode=park,park=collide,server=torch | kernel | 65,536 | 262,144 | 202 M | 806 M | 0.17 | 0.325 |  |
| layer=park,mode=park,park=modes,server=torch | kernel | 256 | 1,024 | 5.58 M | 22.3 M | 0.16 | 0.0459 |  |
| layer=park,mode=park,park=modes,server=torch | kernel | 2,048 | 8,192 | 47.7 M | 191 M | 0.23 | 0.043 |  |
| layer=park,mode=park,park=modes,server=torch | kernel | 16,384 | 65,536 | 312 M | 1.25 G | 0.13 | 0.0526 |  |
| layer=park,mode=park,park=modes,server=torch | kernel | 65,536 | 262,144 | 452 M | 1.81 G | 0.012 | 0.145 |  |
| layer=park,mode=park,park=orient,server=torch | kernel | 256 | 1,024 | 298 k | 1.19 M | 0.076 | 0.858 |  |
| layer=park,mode=park,park=orient,server=torch | kernel | 2,048 | 8,192 | 2.46 M | 9.86 M | 0.27 | 0.831 |  |
| layer=park,mode=park,park=orient,server=torch | kernel | 16,384 | 65,536 | 19.2 M | 76.8 M | 0.23 | 0.853 |  |
| layer=park,mode=park,park=orient,server=torch | kernel | 65,536 | 262,144 | 91.5 M | 366 M | 0.067 | 0.716 |  |
| layer=server,mode=server,park=collide,server=torch | torch_gpu | 256 | 1,024 | 24.4 k | 97.8 k | 0.15 | 10.5 |  |
| layer=server,mode=server,park=collide,server=torch | torch_gpu | 2,048 | 8,192 | 193 k | 773 k | 0.045 | 10.6 |  |
| layer=server,mode=server,park=collide,server=torch | torch_gpu | 16,384 | 65,536 | 1.55 M | 6.2 M | 0.13 | 10.6 |  |
| layer=server,mode=server,park=collide,server=torch | torch_gpu | 65,536 | 262,144 | 6.59 M | 26.4 M | 0.067 | 9.94 |  |
| layer=server,mode=server,park=modes,server=torch | torch_gpu | 256 | 1,024 | 23.7 k | 94.8 k | 0.18 | 10.8 |  |
| layer=server,mode=server,park=modes,server=torch | torch_gpu | 2,048 | 8,192 | 191 k | 765 k | 0.051 | 10.7 |  |
| layer=server,mode=server,park=modes,server=torch | torch_gpu | 16,384 | 65,536 | 1.49 M | 5.97 M | 0.034 | 11 |  |
| layer=server,mode=server,park=modes,server=torch | torch_gpu | 65,536 | 262,144 | 6.79 M | 27.2 M | 0.092 | 9.65 |  |
| layer=server,mode=server,park=orient,server=torch | torch_gpu | 256 | 1,024 | 23.4 k | 93.7 k | 0.1 | 10.9 |  |
| layer=server,mode=server,park=orient,server=torch | torch_gpu | 2,048 | 8,192 | 190 k | 761 k | 0.071 | 10.8 |  |
| layer=server,mode=server,park=orient,server=torch | torch_gpu | 16,384 | 65,536 | 1.47 M | 5.87 M | 0.028 | 11.2 |  |
| layer=server,mode=server,park=orient,server=torch | torch_gpu | 65,536 | 262,144 | 6.33 M | 25.3 M | 0.11 | 10.4 |  |
| layer=server,mode=server_graph,park=collide,server=torch | graph | 256 | 1,024 | 205 k | 819 k | 0.0083 | 1.25 |  |
| layer=server,mode=server_graph,park=collide,server=torch | graph | 2,048 | 8,192 | 1.49 M | 5.98 M | 0.002 | 1.37 |  |
| layer=server,mode=server_graph,park=collide,server=torch | graph | 16,384 | 65,536 | 8.33 M | 33.3 M | 0.003 | 1.97 |  |
| layer=server,mode=server_graph,park=collide,server=torch | graph | 65,536 | 262,144 | 19.9 M | 79.4 M | 0.0014 | 3.3 |  |
| layer=server,mode=server_graph,park=modes,server=torch | graph | 256 | 1,024 | 190 k | 759 k | 0.0054 | 1.35 |  |
| layer=server,mode=server_graph,park=modes,server=torch | graph | 2,048 | 8,192 | 1.4 M | 5.59 M | 0.0027 | 1.46 |  |
| layer=server,mode=server_graph,park=modes,server=torch | graph | 16,384 | 65,536 | 7.79 M | 31.2 M | 0.0039 | 2.1 |  |
| layer=server,mode=server_graph,park=modes,server=torch | graph | 65,536 | 262,144 | 18.9 M | 75.5 M | 0.0024 | 3.47 |  |
| layer=server,mode=server_graph,park=orient,server=torch | graph | 256 | 1,024 | 183 k | 732 k | 0.009 | 1.4 |  |
| layer=server,mode=server_graph,park=orient,server=torch | graph | 2,048 | 8,192 | 1.35 M | 5.38 M | 0.0039 | 1.52 |  |
| layer=server,mode=server_graph,park=orient,server=torch | graph | 16,384 | 65,536 | 7.54 M | 30.2 M | 0.0021 | 2.17 |  |
| layer=server,mode=server_graph,park=orient,server=torch | graph | 65,536 | 262,144 | 18.2 M | 72.8 M | 0.0028 | 3.6 |  |
| layer=server,mode=server_k,park=modes,server=cuda | kernel | 256 | 1,024 | 168 k | 672 k | 0.087 | 1.52 |  |
| layer=server,mode=server_k,park=modes,server=cuda | kernel | 2,048 | 8,192 | 1.41 M | 5.62 M | 0.063 | 1.46 |  |
| layer=server,mode=server_k,park=modes,server=cuda | kernel | 16,384 | 65,536 | 11.2 M | 44.7 M | 0.079 | 1.47 |  |
| layer=server,mode=server_k,park=modes,server=cuda | kernel | 65,536 | 262,144 | 49.7 M | 199 M | 0.12 | 1.32 |  |
| layer=server,mode=server_k_graph,park=modes,server=cuda | graph | 256 | 1,024 | 2.27 M | 9.1 M | 0.016 | 0.113 |  |
| layer=server,mode=server_k_graph,park=modes,server=cuda | graph | 2,048 | 8,192 | 16.9 M | 67.7 M | 0.0045 | 0.121 |  |
| layer=server,mode=server_k_graph,park=modes,server=cuda | graph | 16,384 | 65,536 | 82.8 M | 331 M | 0.0037 | 0.198 |  |
| layer=server,mode=server_k_graph,park=modes,server=cuda | graph | 65,536 | 262,144 | 179 M | 716 M | 0.0096 | 0.366 |  |

### Bubbles, proximity sensors and the partition (`space`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mode=layer_eager | kernel | 256 | 1,024 | 360 k | 1.44 M | 0.21 | 0.711 |  |
| mode=layer_eager | kernel | 2,048 | 8,192 | 2.98 M | 11.9 M | 0.19 | 0.688 |  |
| mode=layer_eager | kernel | 16,384 | 65,536 | 21.8 M | 87.1 M | 0.18 | 0.753 |  |
| mode=layer_eager | kernel | 65,536 | 262,144 | 69.5 M | 278 M | 0.055 | 0.942 |  |
| mode=layer_graph | graph | 256 | 1,024 | 1.51 M | 6.04 M | 0.024 | 0.169 |  |
| mode=layer_graph | graph | 2,048 | 8,192 | 11.9 M | 47.6 M | 0.024 | 0.172 |  |
| mode=layer_graph | graph | 16,384 | 65,536 | 66.2 M | 265 M | 0.044 | 0.247 |  |
| mode=layer_graph | graph | 65,536 | 262,144 | 80.4 M | 322 M | 0.075 | 0.815 |  |
| mode=park | kernel | 256 | 1,024 | 5.75 M | 23 M | 0.14 | 0.0445 |  |
| mode=park | kernel | 2,048 | 8,192 | 45.4 M | 182 M | 0.19 | 0.0451 |  |
| mode=park | kernel | 16,384 | 65,536 | 325 M | 1.3 G | 0.089 | 0.0505 |  |
| mode=park | kernel | 65,536 | 262,144 | 458 M | 1.83 G | 0.0054 | 0.143 |  |
| mode=partition_graph | graph | 256 | 1,024 | 2.74 M | 10.9 M | 0.0068 | 0.0935 |  |
| mode=partition_graph | graph | 2,048 | 8,192 | 21 M | 84.1 M | 0.0046 | 0.0974 |  |
| mode=partition_graph | graph | 16,384 | 65,536 | 106 M | 424 M | 0.013 | 0.155 |  |
| mode=partition_graph | graph | 65,536 | 262,144 | 174 M | 696 M | 0.036 | 0.377 |  |
| mode=server_torch | torch_gpu | 256 | 1,024 | 14.3 k | 57.2 k | 0.1 | 17.9 |  |
| mode=server_torch | torch_gpu | 2,048 | 8,192 | 112 k | 446 k | 0.086 | 18.4 |  |
| mode=server_torch | torch_gpu | 16,384 | 65,536 | 852 k | 3.41 M | 0.095 | 19.2 |  |

### Proximity maps of any size (`space_primes`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| N=13,R=2000000.0 | graph | 256 | 3,328 | 279 k | 3.63 M | 0.08 | 0.918 |  |
| N=13,R=2000000.0 | graph | 2,048 | 26,624 | 1.9 M | 24.7 M | 0.0079 | 1.08 |  |
| N=13,R=2000000.0 | graph | 8,192 | 106,496 | 6.65 M | 86.5 M | 0.00024 | 1.23 |  |
| N=4,R=5000.0 | graph | 256 | 1,024 | 1.95 M | 7.81 M | 0.063 | 0.131 |  |
| N=4,R=5000.0 | graph | 2,048 | 8,192 | 14.9 M | 59.7 M | 0.062 | 0.137 |  |
| N=4,R=5000.0 | graph | 8,192 | 32,768 | 52.4 M | 209 M | 0.082 | 0.156 |  |
| N=64,R=2000000.0 | graph | 256 | 16,384 | 6.44 k | 412 k | 0.0007 | 39.8 |  |
| N=64,R=2000000.0 | graph | 2,048 | 131,072 | 51.2 k | 3.28 M | 0.0018 | 40 |  |

### The server layer over the partition (`space_server`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ships=4 | torch_gpu | 256 | 1,024 | 12.5 k | 49.9 k | 0.11 | 20.5 |  |
| ships=4 | torch_gpu | 2,048 | 8,192 | 99.7 k | 399 k | 0.053 | 20.6 |  |
| ships=4 | torch_gpu | 16,384 | 65,536 | 779 k | 3.11 M | 0.089 | 21 |  |
| ships=4 | kernel | 256 | 1,024 | 99.4 k | 398 k | 0.028 | 2.57 |  |
| ships=4 | kernel | 2,048 | 8,192 | 801 k | 3.21 M | 0.026 | 2.56 |  |
| ships=4 | kernel | 16,384 | 65,536 | 6.38 M | 25.5 M | 0.092 | 2.57 |  |
| ships=4 | kernel | 65,536 | 262,144 | 25.2 M | 101 M | 0.18 | 2.6 |  |
| ships=4 | graph | 256 | 1,024 | 917 k | 3.67 M | 0.047 | 0.279 |  |
| ships=4 | graph | 2,048 | 8,192 | 6.73 M | 26.9 M | 0.04 | 0.304 |  |
| ships=4 | graph | 16,384 | 65,536 | 37 M | 148 M | 0.035 | 0.443 |  |
| ships=4 | graph | 65,536 | 262,144 | 68.7 M | 275 M | 0.016 | 0.954 |  |

### Collisions with static collidables (`statics`)

| Configuration | Path | Worlds | Ships | World-ticks/s | Ship-ticks/s | Spread | ms/batch tick | Against |
|---|---|---:|---:|---:|---:|---:|---:|---|
| contact_slots=8,ships=4,spheres=4,statics=0 | torch_gpu | 1,024 | 4,096 | 26 k | 104 k | 0.037 | 39.4 |  |
| contact_slots=8,ships=4,spheres=4,statics=0 | torch_gpu | 8,192 | 32,768 | 203 k | 813 k | 0.19 | 40.3 |  |
| contact_slots=8,ships=4,spheres=4,statics=0 | kernel | 1,024 | 4,096 | 2.76 M | 11 M | 0.087 | 0.371 |  |
| contact_slots=8,ships=4,spheres=4,statics=0 | kernel | 8,192 | 32,768 | 15.8 M | 63 M | 0.023 | 0.52 |  |
| contact_slots=8,ships=4,spheres=4,statics=0 | kernel | 65,536 | 262,144 | 35.2 M | 141 M | 0.058 | 1.86 |  |
| contact_slots=8,ships=4,spheres=4,statics=8 | reference | 16 | 64 | 8.38 k | 33.5 k | 0.014 | 1.91 |  |
| contact_slots=8,ships=4,spheres=4,statics=8 | torch_cpu | 256 | 1,024 | 1.42 k | 5.67 k | 0.0097 | 181 | 0.169x reference |
| contact_slots=8,ships=4,spheres=4,statics=8 | torch_gpu | 1,024 | 4,096 | 14.3 k | 57.4 k | 0.15 | 71.4 | 1.71x reference |
| contact_slots=8,ships=4,spheres=4,statics=8 | torch_gpu | 8,192 | 32,768 | 116 k | 463 k | 0.11 | 70.8 | 13.8x reference |
| contact_slots=8,ships=4,spheres=4,statics=8 | kernel | 1,024 | 4,096 | 1.89 M | 7.55 M | 0.051 | 0.542 | 225x reference |
| contact_slots=8,ships=4,spheres=4,statics=8 | kernel | 8,192 | 32,768 | 8.42 M | 33.7 M | 0.007 | 0.973 | 1,000x reference |
| contact_slots=8,ships=4,spheres=4,statics=8 | kernel | 65,536 | 262,144 | 13.6 M | 54.3 M | 0.067 | 4.83 | 1,620x reference |


Coverage: 269 of the 269 registered cells have an eligible record; the others did not complete, were not clean or were not eligible.
