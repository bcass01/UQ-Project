import pandas as pd #type:ignore
from thermal_solver import ThermalParams, solve_thermal_explicit

def run_campaign(input_file="lhs_inputs.csv", output_file="uq_results.csv"):
    inputs = pd.read_csv(input_file)
    results = []
    n = len(inputs)

    print(f"Starting Deterministic Solver: {n} simulations for Ti-6Al-4V model...")

    for index, row in inputs.iterrows():
        # Map CSV row to the solver parameters
        params = ThermalParams(
            alpha=row['alpha'],
            Q=row['Q'],
            A=row['A'],
            h=row['h']
        )
        
        # Execute deterministic solver
        sim_data = solve_thermal_explicit(params)
        
        # Log the Quantities of Interest (QoIs)
        results.append({
            "sample_id": row['sample_id'],
            "T_max": sim_data["T_max"],
            "t_melt": sim_data["t_melt"]
        })

        if index % 1000 == 0:
            print(f"Progress: {index}/{n} simulations complete.")

    # Save results to a separate file
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Simulations complete. Results saved to {output_file}")

if __name__ == "__main__":
    run_campaign()