import numpy as np                                              # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
from matplotlib.animation import FuncAnimation                  # type: ignore
from thermal_solver import ThermalParams, solve_thermal_explicit

# 1. Run the simulation
params = ThermalParams(t_end=10.0) # Shorter time for a quick animation
results = solve_thermal_explicit(params)

x = results["x"]
t_hist = results["t_history"]
t_melt = params.T_melt  # 1933.0 K

# 2. Setup the figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x * 1000, t_hist[0], color='blue', lw=2, label='Temperature')
ax.axhline(t_melt, color='red', linestyle='--', label=f'Melt Point ({t_melt}K)')

ax.set_xlim(0, params.L * 1000) # Convert m to mm
ax.set_ylim(params.T_inf - 50, 5000)
ax.set_xlabel('Position (mm)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Laser Scanning Simulation (Ti-6Al-4V)')
ax.legend()
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# 3. Define the update function
def update(frame):
    line.set_ydata(t_hist[frame])
    # Assuming snapshots are saved every 10 steps in your solver
    time_text.set_text(f'Time: {frame * params.dt * 10:.3f} s') 
    return line, time_text

# 4. Create animation
# Interval is the delay between frames in milliseconds
ani = FuncAnimation(fig, update, frames=len(t_hist), interval=50, blit=True)

plt.show()

# To save as a video file (requires ffmpeg):
#ani.save('thermal_scan.mp4', writer='ffmpeg', fps=20)