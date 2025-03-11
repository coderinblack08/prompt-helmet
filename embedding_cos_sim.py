import numpy as np
import matplotlib.pyplot as plt

# Create figure and axis
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Two example vectors
v1 = np.array([3, 4])
v2 = np.array([4, 2])

# Calculate cosine similarity
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(cos_sim)

# Plot vectors with thicker lines
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', 
          width=0.008, label='Vector 1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', 
          width=0.008, label='Vector 2')

# Draw arc to show angle (starting from bottom vector)
radius = 1.0
theta = np.linspace(np.arctan2(v2[1], v2[0]), np.arctan2(v1[1], v1[0]), 100)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
ax.plot(x, y, 'gray', linestyle='--', alpha=0.5)

# Add text for cosine similarity
ax.text(1.0, 0.5, f'cos(θ) = {cos_sim:.2f}', fontsize=12)

# Add LaTeX equation and calculation in top right with larger font
equation = r"$\cos(\theta) = \frac{\mathbf{v_1} \cdot \mathbf{v_2}}{|\mathbf{v_1}| |\mathbf{v_2}|}$"
calculation = r"$= \frac{20}{\sqrt{25}\sqrt{20}}$"
result = r"$= 0.89$"

ax.text(3.2, 5.5, equation, fontsize=20)
ax.text(3.2, 5.1, calculation, fontsize=20)
ax.text(3.2, 4.7, result, fontsize=20)

# Add labels at vector heads
ax.text(3.1, 4.1, "\"My cat purrs when I pet it.\"", color='blue', fontsize=12)
ax.text(4.1, 2.1, "\"My dog wags its tail when I pet it.\"", color='red', fontsize=12)

# Draw gray arrow between vector heads and label it
ax.annotate("", xy=(v2[0], v2[1]), xytext=(v1[0], v1[1]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# Add "Cat vs. Dog" label to the middle of the gray arrow
midpoint_x = (v1[0] + v2[0]) / 2
midpoint_y = (v1[1] + v2[1]) / 2
ax.text(midpoint_x - 0.5, midpoint_y + 0.3, "Cat vs. Dog", color='gray', fontsize=12)

# Set equal aspect ratio and limits
ax.set_aspect('equal')
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)

# Add grid and labels
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Add title
plt.title('Vector Cosine Similarity Visualization')

plt.show()