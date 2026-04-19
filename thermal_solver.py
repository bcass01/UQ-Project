import numpy as np
from dataclasses import dataclass

@dataclass
class ThermalParams:
    """Physical and numerical parameters for Ti-6Al-4V 1D AM model."""
    # --- Stochastic inputs (±10% nominal) --- [cite: 131]
    alpha: float = 2.9e-6      # Thermal diffusivity [m²/s] [cite: 132]
    Q:     float = 200.0       # Laser power [W] [cite: 133]
    A:     float = 0.35        # Absorptivity [-] [cite: 134]
    h:     float = 20.0        # Convection coefficient [W/(m²·K)] [cite: 135]

    # --- Fixed physical parameters ---
    T_inf:    float = 298.15   # Ambient [K]
    k:        float = 7.2      # Thermal conductivity [W/(m·K)]
    rho_cp:   float = 2.45e6   # rho * cp [J/(m³·K)]
    sigma:    float = 5e-4     # Gaussian beam half-width [m]
    v:        float = 0.01     # Laser scan speed [m/s]
    T_melt:   float = 1933.0   # Melting point [K]

    # --- Domain & Numerical ---
    L:     float = 0.05        # Rod length [m]
    N:     int   = 200         # Nodes
    t_end: float = 1.0         # Time [s]
    dt:    float = 0.005       # Time step [s]

    # NEW: Cross-sectional area to concentrate the laser power
    # Using pi * sigma^2 as a representative area for the heat-affected zone
    @property
    def area_c(self):
        return np.pi * self.sigma**2

def gaussian_source(x: np.ndarray, x_c: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - x_c) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

def solve_thermal_explicit(params: ThermalParams) -> dict:
    p = params
    x, dx = np.linspace(0, p.L, p.N, retstep=True)
    t_arr = np.arange(0, p.t_end + p.dt, p.dt)
    
    T = np.full(p.N, p.T_inf)
    T_max_global = p.T_inf
    t_melt_total = 0.0
    
    # Corrected Scaling: Divide Power by (rho_cp * Area_c) to get K/s
    diff_coeff = p.alpha * p.dt / dx**2
    conv_coeff = (p.h / p.rho_cp) * p.dt  
    src_scale  = (p.A * p.Q / (p.rho_cp * p.area_c)) * p.dt 

    T_history = []
    for n, t in enumerate(t_arr):
        T_new = np.copy(T)
        T_max_global = max(T_max_global, np.max(T))
        
        if np.max(T) >= p.T_melt:
            t_melt_total += p.dt

        x_c = p.v * t
        s_vec = gaussian_source(x, x_c, p.sigma)

        # Interior update
        T_new[1:-1] = (T[1:-1] + 
                       diff_coeff * (T[2:] - 2*T[1:-1] + T[:-2]) + 
                       src_scale * s_vec[1:-1] - 
                       conv_coeff * (T[1:-1] - p.T_inf))

        # Boundary Conditions (Robin)
        robin_fac = p.h * dx / p.k
        T_new[0] = T[0] + 2*diff_coeff*(T[1] - T[0] - robin_fac*(T[0]-p.T_inf)) + src_scale*s_vec[0]
        T_new[-1] = T[-1] + 2*diff_coeff*(T[-2] - T[-1] - robin_fac*(T[-1]-p.T_inf)) + src_scale*s_vec[-1]

        T = T_new
        if n % 10 == 0: T_history.append(T.copy())

    return {"T_max": T_max_global, "t_melt": t_melt_total, "t_history": np.array(T_history)}