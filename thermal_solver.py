import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ThermalParams:
    """Physical and numerical parameters for Ti-6Al-4V 1D AM model."""
    # --- Stochastic inputs (±10% nominal) ---
    alpha: float = 2.9e-6      # Thermal diffusivity [m²/s]
    Q:     float = 200.0       # Laser power [W]
    A:     float = 0.35        # Absorptivity [-]
    h:     float = 20.0        # Convection coefficient [W/(m²·K)] 

    # --- Fixed physical parameters ---
    T_inf:    float = 298.15   # Ambient temperature [K]
    k:        float = 7.2      # Thermal conductivity [W/(m·K)]
    rho_cp:   float = 2.45e6   # rho * cp [J/(m³·K)]
    sigma:    float = 5e-4     # Gaussian beam half-width [m]
    v:        float = 0.01     # Laser scan speed [m/s]
    T_melt:   float = 1933.0   # Melting point [K]

    # --- Domain & numerical parameters ---
    L:     float = 0.05        # Rod length [m]
    N:     int   = 200         # Number of spatial nodes
    t_end: float = 5.0         # Simulation end time [s]
    dt:    float = 0.005       # Time step [s] (Check stability!)

def gaussian_source(x: np.ndarray, x_c: float, sigma: float) -> np.ndarray:
    """Calculates normalized Gaussian beam profile."""
    return np.exp(-0.5 * ((x - x_c) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

def solve_thermal_explicit(params: ThermalParams) -> dict:
    """Explicit Forward Euler solver for the 1D AM thermal model."""
    p = params
    x, dx = np.linspace(0, p.L, p.N, retstep=True)
    t_arr = np.arange(0, p.t_end + p.dt, p.dt)
    
    # Stability Check: CFL Condition
    cfl_limit = dx**2 / (2 * p.alpha)
    if p.dt > cfl_limit:
        print(f"Warning: dt ({p.dt}) exceeds stability limit ({cfl_limit:.6f})")

    T = np.full(p.N, p.T_inf)
    T_max_global = p.T_inf
    t_melt_total = 0.0
    
    # Pre-calculate constants
    diff_coeff = p.alpha * p.dt / dx**2
    conv_coeff = (p.h / p.rho_cp) * p.dt
    src_const = (p.A * p.Q / p.rho_cp) * p.dt
    robin_fac = p.h * dx / p.k  # For BCs

    T_history = []
    save_interval = max(1, len(t_arr) // 100)

    for n, t in enumerate(t_arr):
        T_new = np.copy(T)
        
        # Track QoIs 
        current_max = np.max(T)
        T_max_global = max(T_max_global, current_max)
        if current_max >= p.T_melt:
            t_melt_total += p.dt

        if n % save_interval == 0:
            T_history.append(T.copy())

        # 1. Source Term (Laser moving)
        x_c = p.v * t
        s_vec = gaussian_source(x, x_c, p.sigma)

        # 2. Interior Nodes (Forward Euler)
        # T_i_new = T_i + r*(T_i+1 - 2*T_i + T_i-1) + dt*Source - dt*Conv
        T_new[1:-1] = (T[1:-1] + 
                       diff_coeff * (T[2:] - 2*T[1:-1] + T[:-2]) + 
                       src_const * s_vec[1:-1] - 
                       conv_coeff * (T[1:-1] - p.T_inf))

        # 3. Boundary Conditions (Robin using ghost nodes)
        # At x=0: T_new[0] update using central diff for flux
        T_new[0] = (T[0] + 
                    2 * diff_coeff * (T[1] - T[0] - robin_fac * (T[0] - p.T_inf)) + 
                    src_const * s_vec[0] - 
                    conv_coeff * (T[0] - p.T_inf))
        
        # At x=L
        T_new[-1] = (T[-1] + 
                     2 * diff_coeff * (T[-2] - T[-1] - robin_fac * (T[-1] - p.T_inf)) + 
                     src_const * s_vec[-1] - 
                     conv_coeff * (T[-1] - p.T_inf))

        T = T_new

    return {
        "x": x,
        "t_history": np.array(T_history),
        "T_max": T_max_global,
        "t_melt": t_melt_total
    }

if __name__ == "__main__":
    params = ThermalParams()
    results = solve_thermal_explicit(params)
    print(f"Simulation Complete for Bernie Cassidy")
    print(f"Max Temperature: {results['T_max']:.2f} K")
    print(f"Time above Melt: {results['t_melt']:.4f} s")