import numpy as np                  #type:ignore
from dataclasses import dataclass
import nominals

@dataclass
class ThermalParams:
    """Physical and numerical parameters for Ti-6Al-4V 1D AM model."""
    # --- Stochastic inputs (±10% nominal) --- 
    alpha: float = nominals.alpha     # Thermal diffusivity [m²/s] 
    Q:     float = nominals.Q        # Laser power [W] 
    A:     float = nominals.A        # Absorptivity [-] 
    h:     float = nominals.H        # Convection coefficient [W/(m²·K)]

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
    t_end: float = 10.0         # Time [s]
    dt:    float = 0.005       # Time step [s]

    # NEW: Cross-sectional area to concentrate the laser power
    # Using pi * sigma^2 as a representative area for the heat-affected zone
    @property
    def area_c(self):
        return np.pi * self.sigma**2

def gaussian_source(x: np.ndarray, x_c: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - x_c) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

def smooth_melt_indicator(T, T_melt, beta=0.02):
    """
    Smooth approximation of (T >= T_melt).
    beta controls sharpness (smaller = sharper).
    """
    return 1.0 / (1.0 + np.exp(-(T - T_melt) / beta))

def solve_thermal_explicit(params: ThermalParams) -> dict:
    p = params
    x, dx = np.linspace(0, p.L, p.N, retstep=True)
    t_arr = np.arange(0, p.t_end + p.dt, p.dt)
    
    T = np.full(p.N, p.T_inf)
    T_max_global = p.T_inf
    t_melt_total = 0.0
    
    # Corrected Scaling: Divide Power by (rho_cp * Area_c) to get K/s
    diff_coeff = p.alpha * p.dt / dx**2

    T_history = []
    for n, t in enumerate(t_arr):
        T_new = np.copy(T)
        T_max_global = max(T_max_global, np.max(T))
        
        melt_fraction = np.max(smooth_melt_indicator(T, p.T_melt, beta=10.0))
        t_melt_total += melt_fraction * p.dt
        x_c = p.v * t

        # Toggle power: If the beam center has left the rod, turn it off
        # current_Q = p.Q if x_c < p.L else 0.0

        # Fade beam power as it approaches end of rod

        dist_to_end = p.L - x_c
        if dist_to_end < 0.001 and dist_to_end > 0:
            current_Q = p.Q * (dist_to_end / 0.001)
        elif x_c > p.L:
            current_Q = 0.0
        else:
            current_Q = p.Q

        s_vec = gaussian_source(x, x_c, p.sigma)
        src_scale  = (p.A * current_Q / (p.rho_cp * p.area_c)) * p.dt

        # Interior update
        T_new[1:-1] = (
            T[1:-1]
            + diff_coeff * (T[2:] - 2*T[1:-1] + T[:-2])
            + src_scale * s_vec[1:-1]
        )        
        # Boundary Conditions (Robin)
        robin_fac = p.h * dx / p.k
        # Left boundary
        T_new[0] = (
            T[0]
            + 2 * diff_coeff * (T[1] - T[0] - robin_fac * (T[0] - p.T_inf))
            + src_scale * s_vec[0]
        )
        # Right boundary
        T_new[-1] = (
            T[-1]
            + 2 * diff_coeff * (T[-2] - T[-1] - robin_fac * (T[-1] - p.T_inf))
            + src_scale * s_vec[-1]
        )
        T = T_new
        if n % 10 == 0: T_history.append(T.copy())

    return {"T_max": T_max_global,
            "t_melt": t_melt_total,
            "t_history": np.array(T_history),
            "x": x}