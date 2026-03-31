import math
from pathlib import Path

import numpy as np
import pandas as pd

GAMMA = 1.0
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunk to run
K_START = 91
K_END = 100

# Parameter box grid
OMEGA_MIN = 0.20
OMEGA_MAX = 0.35
DOMEGA_BOX = 0.01

DELTA_MIN = 0.05
DELTA_MAX = 0.12
DDELTA_BOX = 0.01

# Certification settings
LOWRANK_NSAMPLE = 5
TARGET_DY = 0.10
MIN_NY = 401

# Large-k tail padding:
# outside the scanned strip we use
# sigma_min >= |y| - ||Im(L_k)|| >= outside_pad
SAFETY_MARGIN = 0.05


def block_liouvillian(k: int, omega: float, delta: float, gamma: float = GAMMA) -> np.ndarray:
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


def sector_gap_from_block(k: int, omega: float, delta: float, gamma: float = GAMMA) -> float:
    vals = np.linalg.eigvals(block_liouvillian(k, omega, delta, gamma))
    vals = [v for v in vals if abs(v) > 1e-12]
    winner = max(vals, key=lambda z: z.real)
    return -float(winner.real)


def low_rank_upper_bound_on_box(
    omega_lo: float,
    omega_hi: float,
    delta_lo: float,
    delta_hi: float,
    n_sample: int = LOWRANK_NSAMPLE,
) -> float:
    omegas = np.linspace(omega_lo, omega_hi, n_sample)
    deltas = np.linspace(delta_lo, delta_hi, n_sample)

    vals = []
    for omega in omegas:
        for delta in deltas:
            gap1 = sector_gap_from_block(1, float(omega), float(delta))
            gap2 = sector_gap_from_block(2, float(omega), float(delta))
            vals.append(min(gap1, gap2))

    # Upper bound on the true box minimum of min(gap1, gap2),
    # because the true minimum is <= every sampled value.
    return min(vals)


def imag_op_norm_bound_on_box_for_k(k: int, omega_hi: float, delta_hi: float) -> float:
    L = block_liouvillian(k, omega_hi, delta_hi)
    B = (L - L.conj().T) / (2j)  # Hermitian imaginary-part operator
    evals = np.linalg.eigvalsh(B)
    return float(np.max(np.abs(evals)))


def sigma_min_of_boundary_matrix(k: int, alpha: float, y: float, omega: float, delta: float) -> float:
    n = 2 * k + 1
    A = ((-alpha + 1j * y) * np.eye(n, dtype=complex) - block_liouvillian(k, omega, delta))
    svals = np.linalg.svd(A, compute_uv=False)
    return float(svals[-1])


def choose_ny(y_scan: float, target_dy: float = TARGET_DY, min_ny: int = MIN_NY) -> int:
    ny = max(min_ny, int(math.ceil((2.0 * y_scan) / target_dy)) + 1)
    if ny % 2 == 0:
        ny += 1
    return ny


def certify_k_vs_lowrank_box(
    k: int,
    omega_lo: float,
    omega_hi: float,
    delta_lo: float,
    delta_hi: float,
    n_sample: int = LOWRANK_NSAMPLE,
) -> dict:
    omega_c = 0.5 * (omega_lo + omega_hi)
    delta_c = 0.5 * (delta_lo + delta_hi)

    alpha = low_rank_upper_bound_on_box(
        omega_lo, omega_hi, delta_lo, delta_hi, n_sample=n_sample
    )

    # Bound on ||Im(L_k)|| over the box
    yim_bound = imag_op_norm_bound_on_box_for_k(k, omega_hi, delta_hi)

    # Real-parameter perturbation size:
    # half-width in omega + half-width in delta, times k
    half_omega = 0.5 * (omega_hi - omega_lo)
    half_delta = 0.5 * (delta_hi - delta_lo)
    param_pert = (half_omega + half_delta) * float(k)

    # Outside-strip lower bound
    outside_pad = param_pert + SAFETY_MARGIN
    y_scan = yim_bound + outside_pad

    n_y = choose_ny(y_scan)
    ys = np.linspace(-y_scan, y_scan, n_y)
    dy = ys[1] - ys[0] if len(ys) > 1 else 0.0

    sigmas = np.array(
        [sigma_min_of_boundary_matrix(k, alpha, float(y), omega_c, delta_c) for y in ys],
        dtype=float,
    )
    sigma_center_min = float(sigmas.min())

    # 1-Lipschitz in y on the scanned strip
    sigma_scan_lower = sigma_center_min - 0.5 * abs(dy)

    # Outside the strip:
    # sigma_min(( -alpha + i y )I - L_k) >= |y| - ||Im(L_k)|| >= outside_pad
    sigma_outside_lower = outside_pad

    sigma_global_lower = min(sigma_scan_lower, sigma_outside_lower)
    higher_rank_margin = sigma_global_lower - param_pert

    return {
        "k": k,
        "Omega_lo": omega_lo,
        "Omega_hi": omega_hi,
        "Delta_lo": delta_lo,
        "Delta_hi": delta_hi,
        "alpha_upper_lowrank": alpha,
        "Yim_bound": yim_bound,
        "outside_pad": outside_pad,
        "Yscan": y_scan,
        "sigma_center_min": sigma_center_min,
        "sigma_scan_lower": sigma_scan_lower,
        "sigma_outside_lower": sigma_outside_lower,
        "sigma_global_lower": sigma_global_lower,
        "param_pert": param_pert,
        "higher_rank_margin": higher_rank_margin,
        "passed": bool(higher_rank_margin > 0.0),
    }


def main():
    omega_edges = np.arange(OMEGA_MIN, OMEGA_MAX + 1e-12, DOMEGA_BOX)
    delta_edges = np.arange(DELTA_MIN, DELTA_MAX + 1e-12, DDELTA_BOX)

    rows = []
    ks = list(range(K_START, K_END + 1))
    total = len(ks) * (len(omega_edges) - 1) * (len(delta_edges) - 1)
    count = 0

    for k in ks:
        for i in range(len(omega_edges) - 1):
            for j in range(len(delta_edges) - 1):
                row = certify_k_vs_lowrank_box(
                    k,
                    float(omega_edges[i]),
                    float(omega_edges[i + 1]),
                    float(delta_edges[j]),
                    float(delta_edges[j + 1]),
                    n_sample=LOWRANK_NSAMPLE,
                )
                rows.append(row)

                count += 1
                if count % 50 == 0 or count == total:
                    print(f"Progress: {count}/{total}")

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "higher_rank_largek_adaptive_k91_to_k100.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)

    summary = (
        df.groupby("k")["passed"]
        .agg(total_boxes="count", passed_boxes="sum")
        .reset_index()
    )
    summary["failed_boxes"] = summary["total_boxes"] - summary["passed_boxes"]

    print("\nPer-k box-cert counts:")
    print(summary.to_string(index=False))

    print(f"\nGlobal min higher_rank_margin = {df['higher_rank_margin'].min():.15e}")
    print(f"Global max higher_rank_margin = {df['higher_rank_margin'].max():.15e}")

    cols = [
        "k",
        "Omega_lo",
        "Omega_hi",
        "Delta_lo",
        "Delta_hi",
        "alpha_upper_lowrank",
        "Yim_bound",
        "outside_pad",
        "Yscan",
        "sigma_center_min",
        "sigma_scan_lower",
        "sigma_outside_lower",
        "sigma_global_lower",
        "param_pert",
        "higher_rank_margin",
        "passed",
    ]

    print("\n20 weakest large-k boxes:")
    print(df.nsmallest(20, "higher_rank_margin")[cols].to_string(index=False))

    failed = df[df["passed"] == False].copy()
    if len(failed) > 0:
        print("\nFailed large-k boxes:")
        print(failed[cols[:-1]].to_string(index=False))


if __name__ == "__main__":
    main()
