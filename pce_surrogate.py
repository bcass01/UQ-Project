import pandas as pd
import numpy as np
from numpy.polynomial.legendre import legval

def build_pce(input_file="lhs_inputs_1k.csv", output_file="uq_results_1k.csv", degree=2):
    # 1. Load your data
    inputs = pd.read_csv(input_file)
    outputs = pd.read_csv(output_file)
    
    # 2. Map inputs to [-1, 1]
    # (Using the +/- 10% bounds from your proposal)
    xi = pd.DataFrame()
    for col in ['alpha', 'Q', 'A', 'h']:
        low, high = inputs[col].min(), inputs[col].max()
        xi[col] = 2 * (inputs[col] - low) / (high - low) - 1
    
    # 3. Create Multi-Index (Total Degree p=2, d=4)
    # For a 2nd order PCE with 4 variables, there are 15 terms
    from itertools import product
    indices = [i for i in product(range(degree + 1), repeat=4) if sum(i) <= degree]
    
    # 4. Build Design Matrix (Psi)
    # Each row is a sample, each column is a basis function product
    N_samples = 1000  # Use a subset of 1k samples to 'train' the surrogate
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
    y = outputs['T_max'].values[:N_samples]
    c, residuals, rank, s = np.linalg.lstsq(Psi, y, rcond=None)
    
    return indices, c

# The coefficients c_0 is the PCE-predicted mean. 
# The sum of squares of c_j (for j > 0) relates to the variance.