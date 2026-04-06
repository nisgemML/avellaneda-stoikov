"""
sensitivity.py

Grid search over (gamma, kappa, sigma) to show how each parameter
affects the market maker's key outputs:
  - Final PnL
  - Inventory variance  (lower = better inventory control)
  - Sharpe ratio
  - Fill rate

This kind of analysis — "does the theory hold empirically as parameters vary?" —
is exactly what a quant researcher would do before deploying a strategy.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.market import MarketParams
from strategies.avellaneda_stoikov import AvellanedaStoikov
from simulation.engine import Simulator

N_PATHS  = 100
BASE     = dict(sigma=2.0, kappa=1.5, A=140.0, dt=0.005, T=1.0)

# ── helper ──────────────────────────────────────────────────────────────────

def run_grid(param_name: str, values: list[float]) -> dict:
    results = {k: [] for k in ["mean_pnl", "std_pnl", "sharpe", "inv_var", "fill_rate"]}
    for v in values:
        kw = {k: v2 for k, v2 in BASE.items()}
        if param_name != "gamma":
            kw[param_name] = v
        params = MarketParams(**kw)
        pnls, inv_vars, fills = [], [], []

        for seed in range(N_PATHS):
            gamma  = 0.1   # fixed unless we're sweeping gamma
            if param_name == "gamma":
                gamma = v
            strat  = AvellanedaStoikov(params, gamma=gamma)
            result = Simulator(params, strat, "AS", seed=seed).run()
            pnls.append(result.final_pnl)
            inv_vars.append(np.var(result.inventory_series))
            bid_f, ask_f = result.fill_count
            fills.append((bid_f + ask_f) / params.n_steps)

        pnls = np.array(pnls)
        results["mean_pnl"].append(float(np.mean(pnls)))
        results["std_pnl"].append(float(np.std(pnls)))
        results["sharpe"].append(float(np.mean(pnls) / (np.std(pnls) + 1e-9)))
        results["inv_var"].append(float(np.mean(inv_vars)))
        results["fill_rate"].append(float(np.mean(fills)))
    return results


# ── sweep parameters ────────────────────────────────────────────────────────
print("Sweeping gamma  (risk aversion)...")
gamma_vals   = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

# For sigma and kappa, gamma is fixed at 0.1 (BASE)
print("Sweeping sigma  (volatility)...")
sigma_vals   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

print("Sweeping kappa  (book resilience)...")
kappa_vals   = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

gamma_res = run_grid("gamma", gamma_vals)
sigma_res = run_grid("sigma", sigma_vals)
kappa_res = run_grid("kappa", kappa_vals)

print("Done. Plotting...")

# ── Plot ────────────────────────────────────────────────────────────────────
LIGHT = "#f5f5f0"
DARK  = "#1a1a2e"
TEAL  = "#1D9E75"
CORAL = "#D85A30"
AMBER = "#EF9F27"
GRAY  = "#888780"

fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor=LIGHT)
fig.suptitle(
    "Parameter Sensitivity Analysis — Avellaneda-Stoikov Market Maker",
    fontsize=14, fontweight="bold", color=DARK, y=0.98
)

sweeps = [
    ("gamma (risk aversion γ)", gamma_vals, gamma_res, TEAL),
    ("sigma (volatility σ)",    sigma_vals, sigma_res, CORAL),
    ("kappa (book resilience κ)", kappa_vals, kappa_res, AMBER),
]

metrics = [
    ("mean_pnl",   "Mean final PnL",       "Higher = better"),
    ("sharpe",     "Sharpe ratio",          "Higher = better"),
    ("inv_var",    "Inventory variance",    "Lower = better"),
    ("fill_rate",  "Fill rate (fills/step)","Higher = more active"),
]

for row, (param_label, x_vals, res, color) in enumerate(sweeps):
    for col, (metric_key, metric_label, direction) in enumerate(metrics):
        ax = axes[row][col]
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRAY)
        ax.spines[["left", "bottom"]].set_linewidth(0.5)
        ax.tick_params(colors=DARK, labelsize=8)

        y = res[metric_key]
        ax.plot(x_vals, y, color=color, lw=2.0, marker="o", markersize=4)
        ax.fill_between(x_vals, y, alpha=0.12, color=color)

        if row == 0:
            ax.set_title(f"{metric_label}\n({direction})", fontsize=9,
                         fontweight="bold", color=DARK)
        if col == 0:
            ax.set_ylabel(param_label, fontsize=8, color=DARK)
        ax.set_xlabel(param_label.split(" ")[0], fontsize=8, color=GRAY)

        # mark optimal point
        if "Higher" in direction:
            best_idx = int(np.argmax(y))
        else:
            best_idx = int(np.argmin(y))
        ax.axvline(x_vals[best_idx], color=DARK, lw=0.8, linestyle="--", alpha=0.4)
        ax.scatter([x_vals[best_idx]], [y[best_idx]],
                   color=DARK, s=40, zorder=5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("results", exist_ok=True)
plt.savefig("results/sensitivity.png", dpi=150, bbox_inches="tight", facecolor=LIGHT)
print("Saved results/sensitivity.png")
