import matplotlib.pyplot as plt
import numpy as np

# Example data: reference and measured colors in Lab space (L*, a*, b*)
# Only a* and b* are used for the plot
reference_colors = np.array([
    [20, 30],
    [-10, 50],
    [0, 0],
    [40, -20],
    [-30, -40]
])

measured_colors = np.array([
    [25, 35],
    [-5, 45],
    [5, -5],
    [35, -15],
    [-35, -35]
])

# Create a*b* plot
plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

# Plot reference colors (squares)
plt.scatter(reference_colors[:, 0], reference_colors[:, 1], color='blue', label='Reference', marker='s')

# Plot measured colors (circles)
plt.scatter(measured_colors[:, 0], measured_colors[:, 1], color='red', label='Measured', marker='o')

# Draw arrows from reference to measured
for ref, meas in zip(reference_colors, measured_colors):
    plt.arrow(ref[0], ref[1],
              meas[0] - ref[0], meas[1] - ref[1],
              head_width=2, head_length=3, fc='gray', ec='gray', alpha=0.6)

# Styling
plt.title("a*b* Color Error Plot")
plt.xlabel("a* (green to red)")
plt.ylabel("b* (blue to yellow)")
plt.legend()
plt.grid(True)
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
