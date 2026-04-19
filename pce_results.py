import numpy as np
import pandas as pd
from pce_surrogate import build_pce # Assuming the previous function is in this file

def generate_report_metrics():
    # 1. Build the PCE and get coefficients
    indices, c = build_pce(input_file="lhs_inputs_1k.csv", output_file="uq_results_1k.csv", degree=2)
    
    # 2. Extract PCE-derived Mean
    # In a Legendre-PCE, the first coefficient (index 0,0,0,0) is the mean
    pce_mean = c[0]
    
    # 3. Calculate PCE-derived Variance
    # Variance = sum of squares of coefficients (excluding the mean), 
    # scaled by the norm of the Legendre polynomials (1/(2k+1) per dimension)
    pce_var = 0
    sobol_contributions = {"alpha": 0, "Q": 0, "A": 0, "h": 0}
    param_map = {0: "alpha", 1: "Q", 2: "A", 3: "h"}

    for i, multi_index in enumerate(indices):
        if i == 0: continue # Skip the mean
        
        # Calculate the basis norm squared <Psi^2>
        # For Legendre on [-1,1], <L_k^2> = 1/(2k+1)
        norm_sq = np.prod([1.0 / (2 * k + 1) for k in multi_index])
        term_contribution = (c[i]**2) * norm_sq
        pce_var += term_contribution
        
        # Track first-order contributions for Sobol Indices
        # If only one dimension in the multi-index is non-zero, it's a first-order effect
        if sum(1 for k in multi_index if k > 0) == 1:
            active_dim = next(dim for dim, k in enumerate(multi_index) if k > 0)
            sobol_contributions[param_map[active_dim]] += term_contribution

    # 4. Final Outputs for Bernie's Report
    print("--- PCE VALIDATION RESULTS ---")
    print(f"PCE Mean (a_0): {pce_mean:.4f} K")
    print(f"PCE Total Variance: {pce_var:.4f}")
    print("\n--- GLOBAL SENSITIVITY ANALYSIS (SOBOL INDICES) ---")
    for param, val in sobol_contributions.items():
        s_i = val / pce_var
        print(f"First-order Sobol Index for {param}: {s_i:.4f}")

if __name__ == "__main__":
    generate_report_metrics()