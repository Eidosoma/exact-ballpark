# Real physics experiments

These experiments ask where a physical model describes an Exact Ballpark rule. They share the engine’s Lean package, toolchain, dependencies and build cache. The optional `RealPhysics` library imports individual engine modules. The main proof gate checks the direction of those imports. [Lake targets](../lakefile.toml) · [Engine claim register](../CLAIMS.md)

| Question | Finding | Evidence |
| --- | --- | --- |
| Does ordinary movement describe a Newtonian particle with linear drag? | **Promoted.** The regular source position agrees with the particle’s trajectory with the correct full-tick cache. Mass times acceleration equals the sum of the prescribed force and drag. | [Main Newtonian claim](../CLAIMS.md#claim-182) |
| What happens when the prescribed force exactly balances drag? | **Promoted.** In real arithmetic both source branches return uniform motion for any supplied cache value. | [Main equilibrium claim](../CLAIMS.md#claim-183) |
| Can the drag law predict a falling sphere’s terminal speed? | **Mixed.** Using the smaller brass sphere to predict the larger one gives agreement at the published precision. Both independent absolute predictions fall outside their printed rounding intervals. A second lab’s absolute comparison is open. | [Measured results](RESULTS.md#terminal-speed) · [Stokes model](RealPhysics/FallingSphere.lean) |
| Does the repeat-collision guard mean 100 µs from the start of a sweep? | **Would require an engine change.** The current rule suppresses one hit and retains another relative to that policy. | [COL-715](DISCREPANCIES.md#col-715-repeat-collisions) |
| Does the near-zero-drag guard apply a fixed cutoff to the dimensionless exponent? | **Would require an engine change.** Its effective exponent cutoff grows with the square of the step duration. | [MOV-703](DISCREPANCIES.md#mov-703-drag-cutoff) |

Promotion applies to an exact statement about a named engine rule, with its branch, arithmetic and input assumptions. Its declaration moves into the main library and enters the engine register and axiom audit. Experiments import that declaration. Laboratory comparisons keep their data, measurement assumptions and outcomes here. A physical policy that changes an engine decision goes in the [physical discrepancy log](DISCREPANCIES.md).

## Falling spheres

Take downward as positive, use the bead’s inertial mass for the movement input (agility one), and hold gravity and buoyancy constant. For a rigid sphere of radius `R` in a still Newtonian liquid, the unbounded, creeping-flow model gives drag coefficient `k = 6πηR`. Substituting bead mass and buoyant weight into the engine’s terminal-velocity equation gives `2 R² g (ρsolid − ρfluid) / (9η)`. Lean proves the substitution, velocity convergence from rest with a valid full-tick cache, uniform settling at equilibrium, and the radius-squared scaling. It also relates the executable rational formula to the real formula. [Experimental theorems](RealPhysics/FallingSphere.lean) · [MIT’s Stokes drag derivation, slide 12](https://ocw.mit.edu/courses/6-777j-design-and-fabrication-of-microelectromechanical-devices-spring-2007/ecbd3da50020e79e0cbb85a43a9604c1_07lecture15.pdf#page=12)

**Sheffield.** Carranza Chavez reports two brass spheres settling in pure glycerine. Tables 5.1, 5.2 and 5.5 supply material density, diameters, and tabulated liquid density and viscosity at the measured average temperature of 20 °C. Figure 5.8 and Table 5.8 supply the observed speeds. Both absolute predictions use those independent inputs. The cross-size comparison uses S3’s measured speed to predict S4. The thesis’s own theoretical values use the Haider–Levenspiel correlation in equation 2.10. Our predictions use Stokes drag. [Original measurements and methods](https://etheses.whiterose.ac.uk/id/eprint/15558/1/Thesis_FC_120109432.pdf) · [Transcribed inputs and source digests](falling_spheres.json)

The speeds are printed to 0.01 m/s. The comparison tests agreement at that rounding precision. The repeatability estimate in Table 5.10 concerns S1 in water. A confidence interval for S3 or S4 would require uncertainty estimates for those runs. Their measured Reynolds numbers are below one. Finite inertia, tank walls and the run temperature can affect absolute speeds. The CPU checks at 0.01 s and 1 s compare the engine with the analytic trajectory. The laboratory comparisons concern terminal speed. [Calculated results](RESULTS.md#terminal-speed)

**Georgia Tech.** The original timing worksheet lets us recover the terminal speed of one trial. The authors report Reynolds numbers down to about 0.02, along with an unrecorded liquid temperature and strong temperature dependence of glycerin viscosity. An absolute test needs an independently known viscosity at the run temperature. [Original report](https://ace.gatech.edu/experiments2/6514/flowaround/fall02/ma6514/ma6514.html) · [Timing worksheet](https://ace.gatech.edu/experiments2/6514/flowaround/fall02/ma6514/math64142.xls)

## Reproduce

Run these commands from `lean/` with the existing pinned toolchain and Python 3.13 or later.

```bash
lake build RealPhysics
python3 experimental/check.py
```

The checker audits experimental theorem dependencies, reads the local measurement inputs, compares the exact rational predictions with Lean, and checks the scalar CPU replays and generated results page. With a local Destiny clone, `python3 experimental/check.py --destiny-repo /path/to/destiny` also builds the original movement, sphere sweep and repeat-guard methods directly from the pinned git objects. Use `--write` to regenerate the measurements page after reviewing a change. [Runner](check.py) · [Generated results](RESULTS.md)

The remaining candidates are warp unit conversions, clearance from ship parameters, collision restitution, reference-frame symmetries and sustainable orbit speed.
