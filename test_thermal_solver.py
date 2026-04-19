import numpy as np
from thermal_solver import ThermalParams, solve_thermal_explicit

def run_verification():
    print("--- CEE 628: DETERMINISTIC SOLVER VERIFICATION ---")
    
    # TEST 1: The Stationary Burn
    # If the laser stays in one spot (v=0), it MUST melt the Ti-6Al-4V.
    print("\n[Test 1] Stationary Laser Test...")
    p_burn = ThermalParams(v=0.0, t_end=0.5) 
    # Use a small cross-section area (e.g., pi * sigma^2) to concentrate power
    area_cross = np.pi * p_burn.sigma**2
    
    res = solve_thermal_explicit(p_burn) # Ensure your solver uses area_cross!
    
    if res['T_max'] > 1933.0: # T_melt for Ti-6Al-4V [cite: 125, 131]
        print(f"  PASS: Peak Temp {res['T_max']:.2f} K (Melting achieved)")
    else:
        print(f"  FAIL: Peak Temp {res['T_max']:.2f} K (No melting)")

    # TEST 2: Conservation of Energy (Adiabatic)
    # With no cooling (h=0), Total Heat In should match Internal Energy Gain.
    print("\n[Test 2] Energy Conservation (h=0)...")
    p_energy = ThermalParams(h=0.0, t_end=1.0)
    res_e = solve_thermal_explicit(p_energy)
    
    total_q_in = p_energy.A * p_energy.Q * p_energy.t_end
    energy_stored = np.sum(res_e['t_history'][-1] - p_energy.T_inf) * (p_energy.rho_cp * (p_energy.L/p_energy.N) * area_cross)
    
    error = abs(total_q_in - energy_stored) / total_q_in
    if error < 0.05:
        print(f"  PASS: Energy Error {error:.2%} (Conservation holds)")
    else:
        print(f"  FAIL: Energy Error {error:.2%} (Check your source scaling)")

    # TEST 3: Stability Check
    print("\n[Test 3] Stability Check...")
    dx = p_burn.L / p_burn.N
    cfl = (2 * p_burn.alpha * p_burn.dt) / (dx**2)
    if cfl <= 1.0:
        print(f"  PASS: CFL number is {cfl:.3f} (Stable region)")
    else:
        print(f"  FAIL: CFL number is {cfl:.3f} (Reduce dt or N)")

if __name__ == "__main__":
    run_verification()