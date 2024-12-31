import numpy as np
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from gtmp.splines import LayerAkima1DInterpolator
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

# Enable LaTeX-style fonts
matplotlib.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text rendering
    "font.family": "sans-serif",        # Use a serif font
    "font.sans-serif": ["Helvetica"],   # Set specific sans-serif font (e.g., Helvetica)
    "axes.labelsize": 9,          # Set axis label font size
    "font.size": 9,               # Set default font size
    "legend.fontsize": 9,         # Set legend font size
    "xtick.labelsize": 6,         # Set x-tick font size
    "ytick.labelsize": 6          # Set y-tick font size
})

# Define a custom color palette with calm and relaxed colors
relaxed_palette = ['#1D3557',
                   '#F4A261',  # Warm peach
                   '#E9C46A',  # Light mustard
                   '#2A9D8F',  # Soft teal
                   ]  # Deep green-blue

# Set the palette
sns.set_palette(relaxed_palette)
sns.set_context("talk")

length = 5
x = np.linspace(1, length, length)
q_s, q_g = jnp.zeros(2), jnp.zeros(2)

# 3 layers of dream points
p = np.linspace(-1, 1, 4)
X, Y = np.meshgrid(p, p)
points = np.stack([X.flatten(), Y.flatten()], axis=-1)
q_l = jnp.stack([points, points, points], axis=0)

akima = LayerAkima1DInterpolator(x, q_s, q_l, q_g)
path_ids = jnp.array(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14]
    ]
)
num_points = 20
points_s_1, points_layers, points_final_g = akima.get_spline_grid_interpolation(num_points=num_points)

N = points_s_1.shape[0]
# 3d plot
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="3d")
for i in range(3):
    x_i = np.ones_like(q_l[i, :, 0]) * (i + 2)
    ax.plot(x_i, q_l[i, :, 0], q_l[i, :, 1], "o", c='blue', ms=5, label="Waypoints")
ax.plot(x[0], q_s[0], q_s[1], "ro", markersize=5, label="Start")
ax.plot(x[-1], q_g[0], q_g[1], "go", markersize=5, label="Goal")

x_s = np.linspace(1, 2, num=num_points + 1)[:-1]

for i in range(N):
    ax.plot(x_s, points_s_1[i, :, 0], points_s_1[i, :, 1], c='royalblue', alpha=0.5)

for t in range(0, points_layers.shape[0]):
    x_l = np.linspace(t + 2, t + 3, num=num_points + 1)[:-1]
    for i in range(N):
        for j in range(N):
            ax.plot(x_l, points_layers[t, i, j, :, 0], points_layers[t, i, j, :, 1], c='royalblue', alpha=0.5)

x_g = np.linspace(length - 1, length, num=num_points + 1)[:-1]
for i in range(N):
    ax.plot(x_g, points_final_g[i, :, 0], points_final_g[i, :, 1], c='royalblue', alpha=0.5)

ax.set_aspect('equal')
ax.set_axis_off()
ax.grid(False)
fig.tight_layout(pad=0)
plt.show()
