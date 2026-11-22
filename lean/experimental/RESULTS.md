# Physical comparison results

The optional library has **10 registered experimental theorems**. [Experimental declarations](claims.json)

## Terminal speed

| Sphere | Diameter (mm) | Re from observation | Predicted (m/s) | Observed (m/s) | Prediction − observed (m/s) | Rounding interval of the printed speed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S3 | 5.0 | 0.27 | 0.069477 | 0.06 | +0.009477 | outside |
| S4 | 6.0 | 0.49 | 0.100047 | 0.09 | +0.010047 | outside |

Anchoring the radius-squared law to S3 gives **0.0864 m/s** for S4. At the published increment of 0.01 m/s this gives 0.09 m/s. The reported S4 speed is 0.09 m/s. Propagating just the half-unit of rounding in the S3 reading gives 0.0792–0.0936 m/s.

Georgia Tech trial 1 covers 0.361666900 m in 27.32 s, giving **0.013238 m/s**. The authors report that liquid temperature was unrecorded. Absolute calibration for this run is held open.

Inputs, measurement conditions and interpretation are in the [experiment register](README.md#falling-spheres).

## Repeat-collision decisions

| Sweep (s) | Hit fraction | Elapsed (µs) | Engine | Elapsed-time control |
| ---: | ---: | ---: | --- | --- |
| 1 | 8e-05 | 80 | ignored | ignored |
| 2 | 8e-05 | 160 | ignored | retained |
| 0.5 | 0.00015 | 75 | retained | ignored |
| 1 | 0.0001 | 100 | ignored | ignored |
| 1 | 0.00011 | 110 | retained | retained |

## MOV-703 branch decisions

| Step (s) | Engine | Fixed exponent cutoff | Scalar position (m) | Continuous drag position (m) |
| ---: | --- | --- | ---: | ---: |
| 0.5 | regular | special | -413.701855 | 0.125000 |
| 2 | special | regular | 0.000000 | 2.000000 |

Both timing comparisons have reproducible witnesses in the [physical discrepancy log](DISCREPANCIES.md).
