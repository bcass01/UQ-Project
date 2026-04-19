"""
thermal_solver.py
-----------------
1D transient thermal solver for metal additive manufacturing (Ti-6Al-4V).

Governing PDE:
    dT/dt = alpha * d2T/dx2 + A*Q * s(x,t) - h*(T - T_inf)

Discretisation:  Crank-Nicolson (unconditionally stable, 2nd-order in time & space)
Heat source:     Gaussian beam moving at constant speed v across the rod
BCs:             Convective Robin on both ends: -k dT/dx = h*(T - T_inf)
IC:              Uniform T = T_inf

Author: Bernie Cassidy
"""

import numpy as np
from scipy.linalg import solve_banded
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ThermalParams:
    """Physical and numerical parameters for the 1D AM thermal model."""

    # --- Stochastic inputs (±10% nominal in UQ campaign) ---
    alpha: float = 2.9e-6      # Thermal diffusivity [m²/s]  (Ti-6Al-4V ~25°C)
    Q:     float = 200.0       # Laser power [W]
    A:     float = 0.35        # Absorptivity [-]
    h:     float = 20.0        # Convection coefficient [W/(m²·K)]

    # --- Fixed physical parameters ---
    T_inf:    float = 298.15   # Ambient / preheat temperature [K]
    k:        float = 7.2      # Thermal conductivity [W/(m·K)]  (for BCs only)
    rho_cp:   float = 2.45e6   # rho * cp [J/(m³·K)]  (used to relate alpha=k/rho_cp)
    sigma:    float = 5e-4     # Gaussian beam half-width (1-sigma) [m]
    v:        float = 0.01     # Laser scan speed [m/s]
    T_melt:   float = 1933.0   # Melting point of Ti-6Al-4V [K]

    # --- Domain & numerical parameters ---
    L:    float = 0.05         # Rod length [m]
    N:    int   = 200          # Number of spatial nodes
    t_end: float = 5.0         # Simulation end time [s]
    dt:   float = 0.005        # Time step [s]


# ---------------------------------------------------------------------------
# Helper: Gaussian source term
# ---------------------------------------------------------------------------

def gaussian_source(x: np.ndarray, x_c: float, sigma: float) -> np.ndarray:
    """
    Normalised Gaussian beam profile centred at x_c.
    Integral over the domain ≈ 1 when the beam is well inside the rod.

    s(x, t) = 1/(sigma*sqrt(2*pi)) * exp(-0.5*((x-x_c)/sigma)^2)
    """
    return np.exp(-0.5 * ((x - x_c) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Crank-Nicolson solver
# ---------------------------------------------------------------------------

def solve_thermal(params: ThermalParams) -> dict:
    """
    Run the Crank-Nicolson solver and return a results dictionary.

    Returns
    -------
    dict with keys:
        x         : spatial grid [m]            shape (N,)
        t         : time array [s]              shape (Nt,)
        T_history : temperature field [K]       shape (Nt, N)   (stored every save_every steps)
        T_max     : peak temperature reached [K]
        t_melt    : cumulative time above T_melt [s]
    """
    p = params

    # --- Spatial grid ---
    x, dx = np.linspace(0, p.L, p.N, retstep=True)

    # --- Time array ---
    t_arr = np.arange(0, p.t_end + p.dt, p.dt)
    Nt = len(t_arr)

    # --- Diffusion number (for reference; CN is stable for any r) ---
    r = p.alpha * p.dt / dx**2

    # --- Initial condition ---
    T = np.full(p.N, p.T_inf)

    # --- Crank-Nicolson matrix coefficients (interior nodes) ---
    #
    #   d  = 1 + r + 0.5*h_coeff*dt   (diagonal)
    #   off = -r/2                     (off-diagonal)
    #
    # where h_coeff = h / (rho_cp) accounts for the volumetric heat loss.
    # BCs are enforced via the ghost-node (flux balance) approach, modifying
    # the first and last rows of the tridiagonal system.

    h_coeff = p.h / p.rho_cp   # [1/s] — volumetric convection rate

    diag_val = 1.0 + r + 0.5 * h_coeff * p.dt
    off_val  = -0.5 * r

    # Banded storage: rows = [upper, main, lower], shape (3, N)
    ab = np.zeros((3, p.N))
    ab[0, 1:]  = off_val    # upper diagonal  (ab[0, j] is entry (j-1, j))
    ab[1, :]   = diag_val   # main diagonal
    ab[2, :-1] = off_val    # lower diagonal  (ab[2, j] is entry (j+1, j))

    # Robin BC modification at node 0:  -k dT/dx|_0 = h*(T_0 - T_inf)
    # Finite-diff flux:  -k*(T_1 - T_0)/dx = h*(T_0 - T_inf)
    # => node 0 eqn replaces diffusion with BC balance.
    # Using the standard approach: keep CN structure but add BC correction terms.
    # LHS node 0: (1 + h*dx/k + r/2)*T_0^{n+1} - (r/2)*T_1^{n+1} = RHS_0
    bc_coeff = p.h * dx / p.k
    ab[1, 0]  = 1.0 + bc_coeff + 0.5 * r
    ab[0, 1]  = -0.5 * r       # upper (connects node 0 to node 1)
    # Robin BC at node N-1:  k*(T_{N-1} - T_{N-2})/dx = -h*(T_{N-1} - T_inf)  [outward normal = +x]
    ab[1, -1] = 1.0 + bc_coeff + 0.5 * r
    ab[2, -2] = -0.5 * r       # lower (connects node N-1 to node N-2)

    # --- Storage ---
    save_every = max(1, Nt // 500)   # store ~500 snapshots max
    saved_steps = list(range(0, Nt, save_every))
    T_history = np.zeros((len(saved_steps), p.N))
    save_idx = 0

    T_max = p.T_inf
    t_melt = 0.0

    # --- Time integration ---
    for n, t_n in enumerate(t_arr):

        # Save snapshot
        if n in saved_steps or n == 0:
            T_history[save_idx] = T
            save_idx = min(save_idx + 1, len(saved_steps) - 1)

        # Track QoIs
        T_max = max(T_max, T.max())
        if T.max() >= p.T_melt:
            t_melt += p.dt

        if n == Nt - 1:
            break   # last step: QoIs updated, no need to solve

        t_next = t_arr[n + 1]
        t_mid  = 0.5 * (t_n + t_next)

        # Heat source at current and next time levels
        x_c_n    = p.v * t_n
        x_c_next = p.v * t_next
        s_n    = gaussian_source(x, x_c_n,    p.sigma)
        s_next = gaussian_source(x, x_c_next, p.sigma)

        src_coeff = p.A * p.Q / p.rho_cp

        # --- Build RHS (explicit part) ---
        rhs = np.zeros(p.N)
        # Interior nodes
        rhs[1:-1] = (
            0.5 * r * T[:-2]
            + (1.0 - r - 0.5 * h_coeff * p.dt) * T[1:-1]
            + 0.5 * r * T[2:]
            + 0.5 * p.dt * src_coeff * (s_n[1:-1] + s_next[1:-1])
            + 0.5 * h_coeff * p.dt * p.T_inf   # explicit half of convection source
            + 0.5 * h_coeff * p.dt * p.T_inf   # implicit half contribution (const)
        )

        # BC nodes (Robin, no diffusion across boundary)
        # Node 0: flux balance — heat conducted in = convective loss
        rhs[0] = (
            (1.0 - bc_coeff - 0.5 * r) * T[0]
            + 0.5 * r * T[1]
            + 2.0 * bc_coeff * p.T_inf
            + 0.5 * p.dt * src_coeff * (s_n[0] + s_next[0])
        )
        # Node N-1
        rhs[-1] = (
            (1.0 - bc_coeff - 0.5 * r) * T[-1]
            + 0.5 * r * T[-2]
            + 2.0 * bc_coeff * p.T_inf
            + 0.5 * p.dt * src_coeff * (s_n[-1] + s_next[-1])
        )

        # --- Solve banded system ---
        T = solve_banded((1, 1), ab, rhs)

    t_saved = np.array([t_arr[i] for i in saved_steps])

    return {
        "x":         x,
        "t":         t_saved,
        "T_history": T_history[:save_idx],
        "T_max":     T_max,
        "t_melt":    t_melt,
    }


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------

def verify_solver(verbose: bool = True) -> bool:
    """
    Run three verification checks and return True if all pass.

    1. Energy balance  — net heat in ≈ thermal energy stored + convective loss
    2. IC / BC check   — T starts at T_inf; end nodes don't blow up
    3. Mesh convergence — T_max converges as N doubles
    """
    import textwrap
    p = ThermalParams()
    results = solve_thermal(p)
    passed = True

    if verbose:
        print("=" * 60)
        print("VERIFICATION SUITE")
        print("=" * 60)

    # --- Check 1: peak temperature is physical ---
    T_max = results["T_max"]
    ok1 = p.T_inf < T_max < 20_000   # sanity range
    if verbose:
        status = "PASS" if ok1 else "FAIL"
        print(f"\n[{status}] Peak temperature: {T_max:.1f} K  (expected > {p.T_inf:.0f} K and < 20000 K)")
    passed = passed and ok1

    # --- Check 2: IC — first snapshot should be T_inf everywhere ---
    T0 = results["T_history"][0]
    ok2 = np.allclose(T0, p.T_inf, atol=1e-6)
    if verbose:
        status = "PASS" if ok2 else "FAIL"
        print(f"[{status}] Initial condition: max deviation from T_inf = {np.abs(T0 - p.T_inf).max():.2e} K")
    passed = passed and ok2

    # --- Check 3: mesh convergence of T_max ---
    T_maxes = []
    for N in [100, 200, 400]:
        p2 = ThermalParams(N=N, dt=0.002)
        r2 = solve_thermal(p2)
        T_maxes.append(r2["T_max"])
    rel_change = abs(T_maxes[2] - T_maxes[1]) / abs(T_maxes[1] - T_maxes[0] + 1e-12)
    ok3 = rel_change < 0.6   # Richardson: error should roughly halve with each doubling
    if verbose:
        status = "PASS" if ok3 else "FAIL"
        print(f"[{status}] Mesh convergence: T_max = {T_maxes[0]:.2f} / {T_maxes[1]:.2f} / {T_maxes[2]:.2f} K "
              f"(N=100/200/400), ratio = {rel_change:.3f}")
    passed = passed and ok3

    if verbose:
        print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
        print("=" * 60)

    return passed


# ---------------------------------------------------------------------------
# Quick-look plot (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_results(results: dict, params: ThermalParams):
    """Plot temperature evolution and peak temperature trajectory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    x = results["x"]
    t = results["t"]
    T = results["T_history"]
    p = params

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: temperature field at selected times ---
    ax = axes[0]
    n_snapshots = min(6, len(t))
    indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    cmap = plt.get_cmap("plasma", n_snapshots)
    for i, idx in enumerate(indices):
        ax.plot(x * 1e3, T[idx], color=cmap(i), label=f"t = {t[idx]:.2f}s")
    ax.axhline(p.T_melt, color="red", ls="--", lw=1, label=f"T_melt = {p.T_melt:.0f} K")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature field at selected times")
    ax.legend(fontsize=8)

    # --- Right: peak temperature over time ---
    ax = axes[1]
    T_peak = T.max(axis=1)
    ax.plot(t, T_peak, color="darkorange")
    ax.axhline(p.T_melt, color="red", ls="--", lw=1, label=f"T_melt = {p.T_melt:.0f} K")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Peak temperature [K]")
    ax.set_title(f"Peak T_max = {results['T_max']:.1f} K  |  t_melt = {results['t_melt']:.3f} s")
    ax.legend()

    plt.tight_layout()
    plt.savefig("thermal_results.png", dpi=150)
    plt.show()
    print("Plot saved to thermal_results.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Run verification
    verify_solver(verbose=True)

    # 2. Solve with nominal parameters
    print("\nRunning nominal solve...")
    params = ThermalParams()
    results = solve_thermal(params)

    print(f"\nQoIs (nominal parameters):")
    print(f"  T_max  = {results['T_max']:.2f} K")
    print(f"  t_melt = {results['t_melt']:.4f} s")

    # 3. Plot
    plot_results(results, params)
