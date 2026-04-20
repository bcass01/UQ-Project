# Project: Uncertainty Quantification in Metal Additive Manufacturing

## Project Overview
This project investigates uncertainty propagation in a 1D transient thermal model for Ti-6Al-4V metal additive manufacturing (AM). The study focuses on thermal response under a localized, moving heat source, governed by the following PDE: 
$$\frac{\partial T}{\partial t}=\alpha\frac{\partial^{2}T}{\partial x^{2}}+AQ~s(x,t)-h(T-T_{\infty})$$ 

Four primary stochastic quantities are modeled as independent uniform distributions ($\pm10\%$ of nominal values): Thermal Diffusivity ($\alpha$), Heat Input ($Q$), Absorptivity ($A$), and Convection Coefficient ($h$).

---

## File Descriptions

### 1. `thermal_solver.py`
* **Purpose**: Implements the deterministic 1D thermal solver.
* **Method**: Uses a custom finite-difference implementation in Python to solve the transient heat equation.
* **Features**: Balances computational efficiency with numerical stability required for repeated evaluations during the UQ campaign.

### 2. `test_thermal_solver.py`
* **Purpose**: A verification suite for the deterministic solver.
* **Functionality**: Tests the solver against stationary laser heat and ensures energy conservation and CFL stability.
* **Significance**: This step is critical to identifying potential pitfalls before moving to the stochastic analysis.

### 3. `lhs_generator.py`
* **Purpose**: Generates the stochastic input sets.
* **Method**: Implements Latin Hypercube Sampling (LHS) to create the baseline samples for $\alpha, Q, A, \text{ and } h$.
* **Scale**: Generates a baseline (1,000 or 10,000 samples) to provide a ground truth for surrogate validation.

### 4. `sim.py`
* **Purpose**: The campaign runner/orchestrator.
* **Logic**: Iterates through the LHS inputs and executes the deterministic solver for each parameter set to propagate uncertainty.
* **Data Logging**: Stores the Quantities of Interest (QoIs)—Peak Temperature ($T_{max}$) and time-above-melting-threshold—into CSV format.

### 5. `analyze_sim_results.py`
* **Purpose**: Statistical post-processing of the sampling results.
* **Output**: Calculates the mean and variance of the QoIs from the simulation data.
* **Role**: Provides the "gold standard" statistics used to validate the accuracy of the PCE surrogate.

### 6. `pce_surrogate.py`
* **Purpose**: Constructs the Polynomial Chaos Expansion (PCE) surrogate.
* **Basis**: Employs a non-intrusive PCE using a Legendre basis, which is the mathematically appropriate choice for uniform input distributions.
* **Optimization**: Provides a computationally efficient meta-model for sensitivity analysis.

### 7. `pce_results.py`
* **Purpose**: Performs the final Global Sensitivity Analysis.
* **Metrics**: Calculates first-order Sobol Indices from the PCE coefficients.
* **Conclusion**: Ranks parameters to identify which physical factors—specifically $Q$ or $A$—most significantly drive the risk of thermal defects.
