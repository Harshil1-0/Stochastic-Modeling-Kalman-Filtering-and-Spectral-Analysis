import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────
np.random.seed(7)
a       = 0.8     # AR(1) coefficient
Q       = 0.36    # system noise variance  → std dev = 0.6
R       = 1.0     # observation noise variance → std dev = 1.0
n_steps = 100

# ── Generate true AR(1) signal ────────────────────────────────────────────
W       = np.random.normal(0, np.sqrt(Q), n_steps)          # W_n ~ N(0, Q)
N_noise = np.random.normal(0, np.sqrt(R), n_steps + 1)      # N_n ~ N(0, R)

Z = np.zeros(n_steps + 1)
for n in range(1, n_steps + 1):
    Z[n] = a * Z[n - 1] + W[n - 1]

X = Z + N_noise    # observed (noisy) signal

# ── Kalman filter ─────────────────────────────────────────────────────────
Z_est = np.zeros(n_steps + 1)   # state estimates
P     = np.zeros(n_steps + 1)   # error covariance
K_all = np.zeros(n_steps + 1)   # Kalman gains (for plotting)

for n in range(1, n_steps + 1):
    # --- Predict ---
    Z_pred = a * Z_est[n - 1]
    P_pred = a**2 * P[n - 1] + Q
    # --- Update ---
    K         = P_pred / (P_pred + R)
    Z_est[n]  = Z_pred + K * (X[n] - Z_pred)
    P[n]      = (1 - K) * P_pred
    K_all[n]  = K

# ── Plot ──────────────────────────────────────────────────────────────────
t = np.arange(n_steps + 1)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# --- Signal comparison ---
axes[0].plot(t, X,     color='lightgray', lw=0.9, label='Observed X_n (noisy)')
axes[0].plot(t, Z,     color='steelblue', lw=1.8, label='True Signal Z_n')
axes[0].plot(t, Z_est, color='crimson',   lw=2.0, ls='--', label='Kalman Estimate Ẑ_n')
axes[0].set_title('True Signal vs Kalman Filter Estimate')
axes[0].set_xlabel('Time step n')
axes[0].set_ylabel('Amplitude')
axes[0].legend()

# --- Estimation error with uncertainty bounds ---
error = Z - Z_est
axes[1].plot(t, error,           color='darkorange', lw=1.5, label='Estimation error')
axes[1].plot(t,  np.sqrt(P),     color='purple',     lw=1.5, ls=':', label='±1 std dev (√P_n)')
axes[1].plot(t, -np.sqrt(P),     color='purple',     lw=1.5, ls=':')
axes[1].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
axes[1].fill_between(t, error, 0, alpha=0.15, color='darkorange')
axes[1].set_title('Estimation Error and Uncertainty Bounds')
axes[1].set_xlabel('Time step n')
axes[1].set_ylabel('Error')
axes[1].legend()

# --- Kalman gain convergence ---
axes[2].plot(t[1:], K_all[1:], color='teal', lw=2)
axes[2].axhline(K_all[-1], color='k', lw=0.8, ls='--', alpha=0.5,
                label=f'Steady-state K ≈ {K_all[-1]:.3f}')
axes[2].set_title('Kalman Gain Convergence K_n')
axes[2].set_xlabel('Time step n')
axes[2].set_ylabel('Kalman Gain K_n')
axes[2].legend()

plt.suptitle('Kalman Filter — AR(1) Signal  (a=0.8, Q=0.36, R=1.0)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('P2_Kalman_Filter.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done — figure saved as P2_Kalman_Filter.png")