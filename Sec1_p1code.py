import numpy as np
import matplotlib.pyplot as plt

# random seed so results stay the same each run
np.random.seed(42)

# ----------------------------------------
# simulation settings
# ----------------------------------------

num_steps = 100
num_paths = 100

# probability of moving upward
p_up = 0.5

# normalized time axis
t_vals = np.linspace(0, 1, num_steps + 1)

# storing all random walk paths
walk_data = np.zeros((num_paths, num_steps + 1))

# ----------------------------------------
# generate scaled random walks
# ----------------------------------------

for sim_idx in range(num_paths):

    # generate +1 / -1 moves
    rand_moves = 2 * (np.random.rand(num_steps) < p_up) - 1

    # cumulative movement
    walk_pos = np.cumsum(rand_moves)

    # adding starting point manually
    walk_pos = np.concatenate(([0], walk_pos))

    # scaling for Brownian motion approximation
    scaled_walk = walk_pos / np.sqrt(num_steps)

    walk_data[sim_idx] = scaled_walk

    # could vectorize this later but honestly this is easier to read

# ----------------------------------------
# mean path
# ----------------------------------------

mean_path = np.mean(walk_data, axis=0)

# ----------------------------------------
# covariance estimate around t = 0.5
# ----------------------------------------

middle_idx = num_steps // 2

cov_estimates = []

for lag in range(num_steps + 1):

    future_idx = middle_idx + lag

    # avoiding index overflow manually
    if future_idx > num_steps:
        future_idx = num_steps

    cov_val = np.mean(
        walk_data[:, middle_idx] * walk_data[:, future_idx]
    )

    cov_estimates.append(cov_val)

# ----------------------------------------
# plotting
# ----------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# ====================================================
# 1) sample trajectories
# ====================================================

sample_colors = plt.cm.tab10(np.linspace(0, 0.9, 5))

for k in range(5):

    axes[0].plot(
        t_vals,
        walk_data[k],
        color=sample_colors[k],
        lw=1.2,
        label=f'Path {k + 1}'
    )

# average path
axes[0].plot(
    t_vals,
    mean_path,
    'k--',
    lw=2,
    label='Average'
)

axes[0].set_title("Sample Paths of Scaled Random Walk")
axes[0].set_xlabel("t")
axes[0].set_ylabel("W_N(t)")
axes[0].legend(fontsize=7)

# ====================================================
# 2) estimated mean
# ====================================================

axes[1].plot(
    t_vals,
    mean_path,
    color='steelblue',
    lw=2
)

axes[1].fill_between(
    t_vals,
    mean_path,
    0,
    alpha=0.2,
    color='steelblue'
)

axes[1].axhline(
    0,
    color='black',
    lw=0.8,
    linestyle='--',
    alpha=0.5
)

axes[1].set_title("Estimated Mean Function")
axes[1].set_xlabel("t")
axes[1].set_ylabel("Mean")

# ====================================================
# 3) covariance behavior
# ====================================================

lag_axis = np.arange(num_steps + 1) / num_steps

axes[2].plot(
    lag_axis,
    cov_estimates,
    color='darkorange',
    lw=2
)

axes[2].axhline(
    0,
    color='black',
    lw=0.8,
    linestyle='--',
    alpha=0.5
)

axes[2].set_title("Empirical Covariance Near t = 0.5")
axes[2].set_xlabel("normalized lag")
axes[2].set_ylabel("Covariance")

# ----------------------------------------
# final figure formatting
# ----------------------------------------

plt.suptitle(
    "Scaled Symmetric Random Walk",
    fontsize=13,
    fontweight='bold'
)

# had overlap once so leaving this in
plt.tight_layout()

# save figure
plt.savefig(
    "P1_Random_Walk.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("Finished plotting. Saved as P1_Random_Walk.png")