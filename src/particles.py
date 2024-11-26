import numpy as np
import matplotlib.pyplot as plt

# Sample 2D data uniformly at random
np.random.seed(0)
particles = 1 - 2 * np.random.rand(2, 2000)

# Remove points outside the unit circle and inside a smaller circle
mask = np.linalg.norm(particles, axis=0) <= 1
mask &= np.linalg.norm(particles, axis=0) >= 0.2
particles = particles[:, mask]

# Create a scatter plot
plt.scatter(*particles, s=5, color="#6abab1")
plt.gca().set_aspect("equal")

# Remove axes
plt.axis("off")
# Transparent background
plt.gca().set_facecolor("none")
# Save the plot
plt.savefig("particles.svg", bbox_inches="tight", pad_inches=0)
