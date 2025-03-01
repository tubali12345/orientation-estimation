import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from dataset import AudioOrientation

# Define colors for each quadrant
colors = {0: "red", 1: "blue", 2: "green", 3: "yellow"}

# Create a mesh grid for plotting
theta = np.linspace(0, 2 * np.pi, 360)
r = np.linspace(0, 1, 100)
T, R = np.meshgrid(theta, r)
X, Y = R * np.cos(T), R * np.sin(T)

# Compute orientation in degrees and apply labeling
angles = np.degrees(T)
labels = np.vectorize(AudioOrientation._orientation_to_label)(angles)

# Plot the circle
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
c = ax.pcolormesh(T, R, labels, cmap=mcolors.ListedColormap([colors[i] for i in range(4)]))

ax.set_yticklabels([])  # Remove radial labels
ax.set_xticklabels([])  # Remove angular labels
# Annotate transition angles
for angle in [45, 135, 225, 315]:
    ax.text(np.radians(angle), 1.1, f"{angle}°", ha="center", va="center", fontsize=12, color="black")
ax.set_title("Color-coded Orientation Circle")
legend_patches = [mpatches.Patch(color=colors[i], label=f"Class {i}") for i in range(4)]
ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.2, 1.2))
plt.savefig("orientation_classes.png")
plt.show()
