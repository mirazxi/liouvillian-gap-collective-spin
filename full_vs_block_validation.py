import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collective_spin_operators(N: int):
    J = N / 2.0
    mvals = np.arange(-J, J + 1, 1.0)
    d = len(mvals)

    Jz = np.diag(mvals.astype(complex))
    Jp = np.zeros((d, d), dtype=complex)
    for i, m in enumerate(mvals[:-1]):
        coeff = math.sqrt(J * (J + 1) - m * (m + 1))
        Jp[i + 1, i] = coeff
    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    return Jx, Jz


def full_liouvillian_symmetric(N: int, omega: float, delta: float, gamma: float = 1.0):
    Jx, Jz = collective_spin_operators(N)
    d = Jx.shape[0]
    I = np.eye(d, dtype=complex)
    H = omega * Jx + delta * Jz
    Jz2 = Jz @ Jz

    # vec(A rho B) = (B^T \otimes A) vec(rho)
    L = (
        -1j * (np.kron(I, H) - np.kron(H.T, I))
        + gamma * (
            np.kron(Jz.T, Jz)
            - 0.5 * np.kron(I, Jz2)
            - 0.5 * np.kron(Jz2.T, I)
        )
    )
    return L


def block_liouvillian(k: int, omega: float, delta: float, gamma: float = 1.0):
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


def all_block_eigs(N: int, omega: float, delta: float, gamma: float = 1.0):
    vals = []
    tags = []
    for k in range(0, N + 1):
        eigs = np.linalg.eigvals(block_liouvillian(k, omega, delta, gamma))
        vals.extend(eigs.tolist())
        tags.extend([k] * len(eigs))
    return np.array(vals), np.array(tags)


def nearest_matching_errors(full_vals, block_vals):
    remaining = list(block_vals.copy())
    errs = []
    for z in full_vals:
        distances = [abs(z - w) for w in remaining]
        idx = int(np.argmin(distances))
        errs.append(distances[idx])
        remaining.pop(idx)
    return np.array(errs)


def main():
    gamma = 1.0
    omega = 0.35
    delta = 0.18
    Ns = [4, 6, 8]

    summary_rows = []
    spectra_rows = []

    for N in Ns:
        full_vals = np.linalg.eigvals(full_liouvillian_symmetric(N, omega, delta, gamma))
        block_vals, block_tags = all_block_eigs(N, omega, delta, gamma)

        errs = nearest_matching_errors(full_vals, block_vals)
        summary_rows.append({
            "N": N,
            "dimension_full_liouvillian": (N + 1) ** 2,
            "num_full_eigs": len(full_vals),
            "num_block_eigs": len(block_vals),
            "max_abs_nearest_mismatch": float(np.max(errs)),
            "mean_abs_nearest_mismatch": float(np.mean(errs)),
        })

        for z in full_vals:
            spectra_rows.append({
                "N": N,
                "source": "full",
                "real_part": float(np.real(z)),
                "imag_part": float(np.imag(z)),
                "rank_k": np.nan,
            })
        for z, k in zip(block_vals, block_tags):
            spectra_rows.append({
                "N": N,
                "source": "blocks",
                "real_part": float(np.real(z)),
                "imag_part": float(np.imag(z)),
                "rank_k": int(k),
            })

    summary_df = pd.DataFrame(summary_rows)
    spectra_df = pd.DataFrame(spectra_rows)

    summary_df.to_csv(OUTPUT_DIR / "full_vs_block_validation_summary.csv", index=False)
    spectra_df.to_csv(OUTPUT_DIR / "full_vs_block_validation_spectra.csv", index=False)

    Nplot = 6
    full_plot = spectra_df[(spectra_df["N"] == Nplot) & (spectra_df["source"] == "full")]
    block_plot = spectra_df[(spectra_df["N"] == Nplot) & (spectra_df["source"] == "blocks")]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        block_plot["real_part"],
        block_plot["imag_part"],
        s=26,
        marker="o",
        facecolors="none",
        linewidths=1.1,
        label="block spectra",
    )
    plt.scatter(
        full_plot["real_part"],
        full_plot["imag_part"],
        s=12,
        marker="x",
        label="full symmetric Liouvillian",
    )
    plt.xlabel(r"$\mathrm{Re}\,\lambda/\gamma$")
    plt.ylabel(r"$\mathrm{Im}\,\lambda/\gamma$")
    plt.title(rf"Spectrum validation at $N={Nplot}$, $\Omega/\gamma={omega:.2f}$, $\Delta/\gamma={delta:.2f}$")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "full_vs_block_spectrum_overlay_N6.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()