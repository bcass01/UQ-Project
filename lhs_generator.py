import pandas as pd
import numpy as np
from scipy.stats import qmc

# Nominal values from project proposal
NOMINALS = {
    "alpha": 2.9e-6,
    "Q": 200.0,
    "A": 0.35,
    "h": 20.0
}

def generate_samples(n=10000, filename="lhs_inputs.csv"):
    sampler = qmc.LatinHypercube(d=len(NOMINALS))
    sample = sampler.random(n=n)
    
    # Define bounds
    l_bounds = [v * 0.9 for v in NOMINALS.values()]
    u_bounds = [v * 1.1 for v in NOMINALS.values()]
    
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
    
    df = pd.DataFrame(scaled_samples, columns=NOMINALS.keys())
    df.to_csv(filename, index_label="sample_id")
    print(f"Successfully generated {n} samples in {filename}")

if __name__ == "__main__":
    generate_samples()