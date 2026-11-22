# Physical comparison results

The optional library has **17 registered experimental theorems**. [Experimental declarations](claims.json)

## Ten mechanics experiments

| Experiment | Numerical result |
| --- | --- |
| Warp length units | All nine converted outputs agree to the replay tolerance at 1,000 m and 10¹⁴ m. The short peak is 750.75 m/s. Using a bare 1 in kilometres gives 1.5 km/s. |
| GOTO sweep clearance | The 1 s ship has a 2.5 m bound. Full and tapered turns deviate 1.094729 m and 0.775601 m at mid-tick. The 0.5 s ship has a 0.15625 m bound. |
| Travel in a history tolerance | At 100 µs the opposed 750 and 250 m/s bodies close 0.100 m. Two co-moving bodies close 0.000 m. One binary64 step below, history returns the stored positions. |
| Unique elastic impulse | Three approaching pairs give impulses 8, 24, 12 kg·m/s. Each leaves with restitution 1. |
| Capped impact energy | Lab ΔK +703.125 J, velocities shifted by +100 m/s -46.875 J, and internal ΔK -23.4375 J. |
| Tracking across frames | The translated, boosted and rotated dyadic control agrees exactly. Cycling the near-radial input from xyz to yzx changes the reported angular speed from NaN to +0 rad/s. |
| Drag in a moving medium | An 8 m/s boost leaves 4.852245 m/s after one second in the fixed medium. The moving-medium force control retains 8 m/s. |
| Sustainable circular speed | At radius 10 m the steady ceiling is 9.395649 m/s. The controller requests 10 m/s, which would require 11.180340 m/s² on a circle with a 10 m/s² budget. |
| STOP damping per second | After one second the vertical speeds are 5.271715, 4.581958 and 3.461381 m/s for 1, 2 and 4 ticks. The one-tick fit is γ = 0.140229 s⁻¹. |
| Sphere resistance | Stokes predicts Cᴅ Re = 24. Sheffield's printed products are S3 30.30 (25.2375–35.3675) and S4 28.45 (25.5825–31.3225). Parentheses give the endpoints from printed rounding. |

The nine engine comparisons use scalar CPU calls. The sphere resistance comparison uses published laboratory data. The two tracking frames and the moving-medium integrator also replay in Lean. With a supplied clone, those calls and the three STOP schedules run against the corresponding compiled Destiny methods. [Assumptions and dispositions](README.md#ten-mechanics-experiments) · [Numerical controls](mechanics_checks.py)

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
