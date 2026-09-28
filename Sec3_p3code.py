import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
np.random.seed(99)

segment_len = 256

# Number of spectra to average together
avg_counts = [10, 20, 50]

# Frequency bins (one-sided FFT frequencies)
freq_axis = np.fft.rfftfreq(segment_len)

# Variance of Uniform(0,1) = 1/12
actual_psd = np.full(len(freq_axis), 1 / 12)

# ---------------------------------------------------------
# Plot setup
# ---------------------------------------------------------
fig, plot_axes = plt.subplots(1, 3, figsize=(16, 4))

plot_colors = ['steelblue', 'darkorange', 'seagreen']

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
for idx, (ax, avg_num, clr) in enumerate(zip(plot_axes, avg_counts, plot_colors)):

    # storing all individual periodograms first
    all_periodograms = np.zeros((avg_num, len(freq_axis)))

    for k in range(avg_num):

        # Generate random uniform sequence
        samples = np.random.uniform(0, 1, segment_len)

        # Removing mean helps suppress the huge DC spike
        centered_signal = samples - samples.mean()

        # FFT calculation
        fft_vals = np.fft.rfft(centered_signal)

        # Standard periodogram estimate
        power_spec = (np.abs(fft_vals) ** 2) / segment_len

        all_periodograms[k] = power_spec

        # old debug line
        # print(power_spec[:5])

    # Averaging multiple periodograms smooths the estimate
    averaged_psd = np.mean(all_periodograms, axis=0)

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    ax.semilogy(
        freq_axis,
        averaged_psd,
        color=clr,
        linewidth=1.8,
        label=f'Averaged PSD (M={avg_num})'
    )

    ax.semilogy(
        freq_axis,
        actual_psd,
        '--',
        color='red',
        linewidth=1.5,
        label='True PSD = 1/12'
    )

    ax.set_title(f"Smoothed Periodogram | M = {avg_num}")

    ax.set_xlabel("Frequency")

    ax.set_ylabel("PSD (log scale)")

    ax.grid(True, which='both', alpha=0.3)

    ax.legend()

# ---------------------------------------------------------
# Final figure adjustments
# ---------------------------------------------------------
plt.suptitle(
    "Spectral Density Estimation using Averaged Periodograms",
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()

save_file = "P3_Periodogram.png"

# Saving before displaying because matplotlib sometimes acts weird
plt.savefig(save_file, dpi=150, bbox_inches='tight')

plt.show()

print("Finished.")
print(f"Saved figure as: {save_file}")