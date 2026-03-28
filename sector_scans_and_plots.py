import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
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


def sector_gap(k: int, omega: float, delta: float, gamma: float = 1.0) -> float:
    vals = np.linalg.eigvals(block_liouvillian(k, omega, delta, gamma))
    vals = [v for v in vals if abs(v) > 1e-12]
    if not vals:
        return 0.0
    v = max(vals, key=lambda z: z.real)
    return -float(v.real)


def winner_grid(omega_vals, delta_vals, max_k=10, gamma=1.0):
    winners = np.zeros((len(delta_vals), len(omega_vals)), dtype=int)
    gaps = np.zeros((len(delta_vals), len(omega_vals)), dtype=float)
    for i, delta in enumerate(delta_vals):
        for j, omega in enumerate(omega_vals):
            sector_gaps = [sector_gap(k, omega, delta, gamma) for k in range(1, max_k + 1)]
            best_idx = int(np.argmin(sector_gaps))
            winners[i, j] = best_idx + 1
            gaps[i, j] = sector_gaps[best_idx]
    return winners, gaps


def crossings_from_diff(x, y):
    xs = []
    for i in range(len(x) - 1):
        if y[i] == 0:
            xs.append(float(x[i]))
        elif y[i] * y[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            xs.append(float(xc))
    return xs


def save_broad_and_focused_scans():
    omega_broad = np.linspace(0.0, 5.0, 101)
    delta_broad = np.linspace(0.0, 5.0, 101)
    winners_broad, gaps_broad = winner_grid(omega_broad, delta_broad, max_k=10, gamma=1.0)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        winners_broad,
        origin="lower",
        aspect="auto",
        extent=[omega_broad.min(), omega_broad.max(), delta_broad.min(), delta_broad.max()],
    )
    plt.xlabel(r"$\Omega/\gamma$")
    plt.ylabel(r"$\Delta/\gamma$")
    plt.title(r"Winning sector $k_*$ from scan over $k=1,\dots,10$")
    plt.colorbar(label=r"$k_*$")
    plt.savefig(OUTPUT_DIR / "broad_winner_map_k10.png", dpi=220, bbox_inches="tight")
    plt.close()

    broad_rows = []
    for i, delta in enumerate(delta_broad):
        for j, omega in enumerate(omega_broad):
            broad_rows.append({
                "Omega_over_gamma": omega,
                "Delta_over_gamma": delta,
                "winner_k": int(winners_broad[i, j]),
                "global_gap_over_gamma": float(gaps_broad[i, j]),
            })
    pd.DataFrame(broad_rows).to_csv(OUTPUT_DIR / "broad_scan_k10_dataset.csv", index=False)

    omega_focus = np.linspace(0.15, 0.50, 181)
    delta_focus = np.linspace(0.05, 0.35, 161)
    winners_focus, gaps_focus = winner_grid(omega_focus, delta_focus, max_k=4, gamma=1.0)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        winners_focus,
        origin="lower",
        aspect="auto",
        extent=[omega_focus.min(), omega_focus.max(), delta_focus.min(), delta_focus.max()],
    )
    plt.xlabel(r"$\Omega/\gamma$")
    plt.ylabel(r"$\Delta/\gamma$")
    plt.title(r"Focused scan: local region where $k=2$ competes")
    plt.colorbar(label=r"$k_*$")
    plt.savefig(OUTPUT_DIR / "focused_winner_map.png", dpi=220, bbox_inches="tight")
    plt.close()

    focus_rows = []
    for i, delta in enumerate(delta_focus):
        for j, omega in enumerate(omega_focus):
            focus_rows.append({
                "Omega_over_gamma": omega,
                "Delta_over_gamma": delta,
                "winner_k": int(winners_focus[i, j]),
                "global_gap_over_gamma": float(gaps_focus[i, j]),
            })
    pd.DataFrame(focus_rows).to_csv(OUTPUT_DIR / "focused_scan_dataset.csv", index=False)


def save_line_cuts_and_benchmarks():
    delta_values = [0.00, 0.10, 0.18, 0.30, 0.50]
    omega_line = np.linspace(0.0, 1.0, 801)
    max_k = 4

    all_rows = []
    cross_rows = []

    for delta in delta_values:
        gaps = {k: np.array([sector_gap(k, om, delta, 1.0) for om in omega_line]) for k in range(1, max_k + 1)}

        for idx, om in enumerate(omega_line):
            row = {"Omega_over_gamma": om, "Delta_over_gamma": delta}
            for k in range(1, max_k + 1):
                row[f"gap_k{k}_over_gamma"] = gaps[k][idx]
            row["winner_k_among_1_to_4"] = int(min(range(1, max_k + 1), key=lambda kk: gaps[kk][idx]))
            all_rows.append(row)

        diff12 = gaps[1] - gaps[2]
        for c in crossings_from_diff(omega_line, diff12):
            cross_rows.append({
                "Delta_over_gamma": delta,
                "crossing_type": "k1_equals_k2",
                "Omega_over_gamma": c,
            })

        plt.figure(figsize=(7, 5))
        plt.plot(omega_line, gaps[1], label=r"$\Delta_L^{(1)}/\gamma$")
        plt.plot(omega_line, gaps[2], label=r"$\Delta_L^{(2)}/\gamma$")
        plt.plot(omega_line, gaps[3], label=r"$\Delta_L^{(3)}/\gamma$")
        plt.plot(omega_line, gaps[4], label=r"$\Delta_L^{(4)}/\gamma$")
        plt.xlabel(r"$\Omega/\gamma$")
        plt.ylabel(r"sector gap$/\gamma$")
        plt.title(rf"Sector gaps at fixed $\Delta/\gamma={delta:.2f}$")
        plt.legend()
        name = str(delta).replace(".", "p")
        plt.savefig(OUTPUT_DIR / f"line_cuts_delta_{name}.png", dpi=220, bbox_inches="tight")
        plt.close()

    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "multi_cut_gap_dataset.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(OUTPUT_DIR / "k1_k2_crossings_summary.csv", index=False)

    points = [
        ("resonance weak drive", 0.10, 0.00),
        ("resonance strong drive", 1.50, 0.00),
        ("inside k=2 wedge", 0.35, 0.18),
        ("near wedge boundary", 0.33, 0.18),
        ("outside wedge", 0.50, 0.18),
        ("higher-rank reorder", 0.17614656, 0.10116011),
    ]
    bench_rows = []
    for label, omega, delta in points:
        vals = {k: sector_gap(k, omega, delta, 1.0) for k in range(1, 7)}
        winner = min(vals, key=vals.get)
        bench_rows.append({
            "label": label,
            "Omega_over_gamma": omega,
            "Delta_over_gamma": delta,
            "winner_k": winner,
            **{f"gap_k{k}": vals[k] for k in range(1, 7)},
        })
    pd.DataFrame(bench_rows).to_csv(OUTPUT_DIR / "benchmark_gap_table.csv", index=False)


def main():
    save_broad_and_focused_scans()
    save_line_cuts_and_benchmarks()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()