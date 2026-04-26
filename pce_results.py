import numpy as np      #type:ignore
import pandas as pd     #type:ignore
from numpy.polynomial.legendre import legval    #type:ignore
from generate_lhs import NOMINALS

def build_pce(input_file="csv/lhs_inputs_90W.csv", output_file="csv/uq_results_90W.csv", degree=3, target='T_max'):

    if target != "T_max" and target != "t_melt":
        raise ValueError("Invalid target variable provided")

    # 1. Load your data
    inputs = pd.read_csv(input_file)
    outputs = pd.read_csv(output_file)
    
    # 2. Map inputs to [-1, 1]
    # (Using the +/- 10% bounds from your proposal)
    xi = pd.DataFrame()
    for col in ['alpha', 'Q', 'A', 'h']:
        low = NOMINALS[col] * 0.9
        high = NOMINALS[col] * 1.1
        xi[col] = 2 * (inputs[col] - low) / (high - low) - 1    
    # 3. Create Multi-Index (Total Degree p=2, d=4)
    # For a 2nd order PCE with 4 variables, there are 15 terms
    from itertools import product
    indices = [i for i in product(range(degree + 1), repeat=4) if sum(i) <= degree]
    
    # 4. Build Design Matrix (Psi)
    # Each row is a sample, each column is a basis function product
    N_samples = len(inputs)  # Use samples to 'train' the surrogate
    Psi = np.ones((N_samples, len(indices)))
    
    for i in range(N_samples):
        for j, index in enumerate(indices):
            # Compute product of 1D Legendre polynomials: L_n1(xi1) * L_n2(xi2) * ...
            term_val = 1.0
            for dim in range(4):
                # legval coefficients: [0,0,1] represents L2(x)
                coeffs = [0] * (index[dim] + 1)
                coeffs[index[dim]] = 1
                term_val *= legval(xi.iloc[i, dim], coeffs)
            Psi[i, j] = term_val

    # 5. Solve for Coefficients (c) using Least Squares
    # y = Psi * c  => c = (Psi^T * Psi)^-1 * Psi^T * y
    y = outputs[target].values[:N_samples]
    c, residuals, rank, s = np.linalg.lstsq(Psi, y, rcond=None)
    
    return indices, c

# The coefficients c_0 is the PCE-predicted mean. 
# The sum of squares of c_j (for j > 0) relates to the variance.

def generate_report_metrics(t):
    # 1. Build the PCE and get coefficients
    indices, c = build_pce(target=t)
    if t == "T_max":
        unit = "K"
    elif t == "t_melt":
        unit = "s"
        
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
    print("--- PCE VALIDATION RESULTS:", t, "--- ")
    print(f"PCE-predicted mean: {pce_mean:.4f}", unit)
    print(f"PCE-predicted variance: {pce_var:.4f}")
    print("\n--- GLOBAL SENSITIVITY ANALYSIS (SOBOL INDICES) ---")
    for param, val in sobol_contributions.items():
        s_i = val / pce_var
        print(f"First-order Sobol Index for {param}: {s_i:.4f}")
    print()
if __name__ == "__main__":
    generate_report_metrics('T_max')
    generate_report_metrics('t_melt')