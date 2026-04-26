# Uncertainty Quantification in a 1D Thermal Model for Metal AM

This project investigates the propagation of parametric uncertainty in a one-dimensional transient thermal model for **Metal Additive Manufacturing (AM)**, specifically focusing on **Ti-6Al-4V**. By utilizing **Polynomial Chaos Expansion (PCE)** and **Latin Hypercube Sampling (LHS)**, the study quantifies how variability in laser parameters and material properties influences the thermal field.

---

## Project Overview

The core of this project is a reduced-order model of the **Laser Powder Bed Fusion (LPBF)** process. It simulates a Gaussian heat source moving across a 1D domain, governed by the transient heat equation with Robin boundary conditions.

### Key Features
* **Physics-Based Modeling:** Implements an explicit finite difference scheme to solve for temperature distribution over time.
* **Uncertainty Quantification:** Models four uncertain inputs ($\\alpha, Q, A, h$) using independent uniform distributions within a $\\pm 10\%$ range of nominal values.
* **Surrogate Modeling:** Constructs a non-intrusive **Legendre-basis PCE** to replace expensive simulations with an efficient polynomial surrogate.
* **Sensitivity Analysis:** Calculates **First-order Sobol Indices** to identify which parameters most heavily drive variance in the output.

---

## Mathematical Framework

### The Governing Equation
The model solves the 1D heat equation:

$$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2} + \\frac{A Q}{\\rho c_p A_c} s(x,t)$$

Where $s(x,t)$ is a moving **Gaussian source**:

$$s(x,t) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(x - vt)^2}{2\\sigma^2}\\right)$$

### Quantities of Interest (QoIs)
1.  **$T_{max}$**: The peak temperature reached during the scan.
2.  **$t_{melt}$**: The total duration the material remains above the melting point ($1933.0$ K), calculated using a **smooth logistic indicator function** to ensure continuity for the PCE.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `thermal_solver.py` | The core engine; contains the explicit solver and physical constants for Ti-6Al-4V. |
| `nominals.py` | Centralized storage for reference physical parameters. |
| `generate_lhs.py` | Generates 1000 input samples using Latin Hypercube Sampling. |
| `sim.py` | Campaign runner that executes the deterministic solver for all LHS samples. |
| `pce_results.py` | Builds the PCE surrogate, calculates statistical moments, and outputs Sobol indices. |
| `analyze_sim_results.py` | Provides baseline statistics and melting counts for the raw simulation data. |
| `animate_thermal.py` | Generates a visual animation of the laser scan and saves static thermal profiles. |
| `test_thermal_solver.py` | Verification suite for energy conservation and numerical stability (CFL condition). |

---

## How to Run

1.  **Verify the Solver:** Run the test suite to ensure energy conservation and stability.
    ```bash
    python test_thermal_solver.py
    ```
2.  **Generate Samples:** Create the stochastic input space.
    ```bash
    python generate_lhs.py
    ```
3.  **Execute Simulations:** Run the deterministic model for the generated samples.
    ```bash
    python sim.py
    ```
4.  **Perform UQ Analysis:** Generate the PCE surrogate and sensitivity report.
    ```bash
    python pce_results.py
    ```
5.  **Visualize:** View the thermal profile animation.
    ```bash
    python animate_thermal.py
    ```

---

## Expected Results
Based on the analysis, **Laser Power ($Q$)** and **Absorptivity ($A$)** are expected to dominate the variance for both $T_{max}$ and $t_{melt}$, typically accounting for nearly 50% of the variance each. **Thermal Diffusivity ($\\alpha$)** and **Convection ($h$)** generally show negligible first-order sensitivity in this 1D configuration.
