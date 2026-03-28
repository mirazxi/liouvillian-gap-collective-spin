import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def block_liouvillian(k: int, omega: float, delta: float, gamma: float = 1.0) -> np.ndarray:
    qs = np.arange(-k, k + 1, dtype=float)
    n = 2 * k + 1
    L = np.zeros((n, n), dtype=complex)
    for i, q in enumerate(qs):
        L[i, i] = -1j * delta * q - 0.5 * gamma * q * q
        if i < n - 1:
            coeff = -0.5j * omega * math.sqrt(k * (k + 1) - q * (q + 1))
            L[i + 1, i] = coeff
        if i > 0:
            coeff = -0.5j * omega * math.sqrt(k * (k + 1) - q * (q - 1))
            L[i - 1, i] = coeff
    return L

def sector_gap_from_block(k: int, omega: float, delta: float, gamma: float = 1.0) -> float:
    vals = np.linalg.eigvals(block_liouvillian(k, omega, delta, gamma))
    vals = [v for v in vals if abs(v) > 1e-12]
    v = max(vals, key=lambda z: z.real)
    return -float(v.real)

def dipolar_gap_exact_from_cubic(omega: float, delta: float, gamma: float = 1.0) -> float:
    coeffs = [
        1.0,
        gamma,
        delta**2 + omega**2 + gamma**2 / 4.0,
        gamma * omega**2 / 2.0,
    ]
    roots = np.roots(coeffs)
    roots = [r for r in roots if abs(r) > 1e-12]
    winner = max(roots, key=lambda z: z.real)
    return -float(winner.real)

def weak_drive_approx_k1(omega: float, delta: float, gamma: float = 1.0) -> float:
    return 2.0 * gamma * omega**2 / (4.0 * delta**2 + gamma**2)

def strong_drive_asymptote_k1(gamma: float = 1.0) -> float:
    return gamma / 4.0

gamma = 1.0
delta = 0.18
omega_vals = np.linspace(0.0, 2.0, 801)

exact_cubic = np.array([dipolar_gap_exact_from_cubic(om, delta, gamma) for om in omega_vals])
numeric_block = np.array([sector_gap_from_block(1, om, delta, gamma) for om in omega_vals])
strong_asym = np.full_like(omega_vals, strong_drive_asymptote_k1(gamma))

omega_weak = np.linspace(0.0, 0.35, 200)
weak_approx = np.array([weak_drive_approx_k1(om, delta, gamma) for om in omega_weak])

df = pd.DataFrame({
    "Omega_over_gamma": omega_vals,
    "Delta_over_gamma_fixed": np.full_like(omega_vals, delta),
    "dipolar_gap_exact_cubic_over_gamma": exact_cubic,
    "dipolar_gap_numeric_block_over_gamma": numeric_block,
    "strong_drive_asymptote_over_gamma": strong_asym,
    "abs_difference_exact_minus_numeric": np.abs(exact_cubic - numeric_block),
})
df.to_csv(OUTPUT_DIR / "dipolar_gap_comparison_delta_0p18.csv", index=False)

plt.figure(figsize=(7, 5))
plt.plot(omega_vals, exact_cubic, label=r"exact $k=1$ gap (cubic roots)")
plt.plot(omega_vals, numeric_block, label=r"direct diagonalization of $L^{(1)}$")
plt.plot(omega_vals, strong_asym, label=r"strong-drive asymptote")
plt.plot(omega_weak, weak_approx, "--", linewidth=2, label=r"weak-drive asymptote")
plt.xlabel(r"$\Omega/\gamma$")
plt.ylabel(r"$\Delta_L^{(1)}/\gamma$")
plt.title(rf"Dipolar gap at fixed $\Delta/\gamma={delta:.2f}$")
plt.legend()
plt.savefig(OUTPUT_DIR / "dipolar_gap_exact_vs_numeric_delta_0p18.png", dpi=220, bbox_inches="tight")
plt.close()

print("Saved:")
print(OUTPUT_DIR / "dipolar_gap_comparison_delta_0p18.csv")
print(OUTPUT_DIR / "dipolar_gap_exact_vs_numeric_delta_0p18.png")
print("Max |exact-numeric| =", np.max(np.abs(exact_cubic - numeric_block)))
