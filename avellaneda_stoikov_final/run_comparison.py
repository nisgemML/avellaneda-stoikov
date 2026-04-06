"""
run_comparison.py

Runs A-S vs Naive over multiple Monte Carlo paths and produces a
four-panel tearsheet saved to results/tearsheet.png.

Panels:
  1. PnL paths (mean + 1-sigma band) for both strategies
  2. Inventory distribution — A-S should be tighter around zero
  3. Spread over time — A-S narrows as T approaches, naive is flat
  4. Summary statistics table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from core.market import MarketParams
from strategies.avellaneda_stoikov import AvellanedaStoikov
from strategies.naive import NaiveMarketMaker
from simulation.engine import Simulator

# ── Config ─────────────────────────────────────────────────────────────────
N_PATHS   = 200
SEED_BASE = 0
GAMMA     = 0.1
PARAMS    = MarketParams(sigma=2.0, kappa=1.5, A=140.0, dt=0.005, T=1.0)
OUT_PATH  = "results/tearsheet.png"

# ── Run simulations ─────────────────────────────────────────────────────────
print(f"Running {N_PATHS} Monte Carlo paths...")

as_pnls, naive_pnls = [], []
as_invs, naive_invs = [], []
as_spreads = []

for seed in range(N_PATHS):
    as_strat    = AvellanedaStoikov(PARAMS, gamma=GAMMA)
    naive_strat = NaiveMarketMaker(fixed_spread=0.5)

    as_res    = Simulator(PARAMS, as_strat,    "AS",    seed=SEED_BASE + seed).run()
    naive_res = Simulator(PARAMS, naive_strat, "Naive", seed=SEED_BASE + seed).run()

    as_pnls.append(as_res.pnl_series)
    naive_pnls.append(naive_res.pnl_series)
    as_invs.append(as_res.inventory_series)
    naive_invs.append(naive_res.inventory_series)
    as_spreads.append(as_res.spread_series)

as_pnls    = np.array(as_pnls)
naive_pnls = np.array(naive_pnls)
as_invs    = np.array(as_invs)
naive_invs = np.array(naive_invs)
as_spreads = np.array(as_spreads)

t_axis = np.linspace(0, PARAMS.T, PARAMS.n_steps + 1)

# ── Summary stats ───────────────────────────────────────────────────────────
def stats(pnl_matrix, inv_matrix):
    final_pnls = pnl_matrix[:, -1]
    inv_flat   = inv_matrix.flatten()
    return {
        "Mean final PnL":     np.mean(final_pnls),
        "Std final PnL":      np.std(final_pnls),
        "Sharpe (annualised)": np.mean(final_pnls) / (np.std(final_pnls) + 1e-9),
        "% paths profitable": 100 * np.mean(final_pnls > 0),
        "Mean |inventory|":   np.mean(np.abs(inv_flat)),
        "Max |inventory|":    np.max(np.abs(inv_flat)),
    }

as_stats    = stats(as_pnls, as_invs)
naive_stats = stats(naive_pnls, naive_invs)

print("\n  Strategy          |  A-S (optimal)  |  Naive (fixed)")
print("  " + "-" * 58)
for k in as_stats:
    print(f"  {k:<22}|  {as_stats[k]:>12.3f}   |  {naive_stats[k]:>12.3f}")

# ── Plot ────────────────────────────────────────────────────────────────────
DARK   = "#1a1a2e"
LIGHT  = "#f5f5f0"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
GRAY   = "#888780"
YELLOW = "#EF9F27"

fig = plt.figure(figsize=(16, 10), facecolor=LIGHT)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                        left=0.07, right=0.96, top=0.88, bottom=0.08)

ax1 = fig.add_subplot(gs[0, :2])   # PnL paths — wide
ax2 = fig.add_subplot(gs[0, 2])    # Inventory distribution
ax3 = fig.add_subplot(gs[1, :2])   # Spread over time — wide
ax4 = fig.add_subplot(gs[1, 2])    # Stats table

def style_ax(ax, title):
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(colors=DARK, labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=DARK, pad=8)

# Panel 1 — PnL
style_ax(ax1, "PnL paths — Avellaneda-Stoikov vs Naive (200 Monte Carlo paths)")

as_mean    = as_pnls.mean(0)
as_std     = as_pnls.std(0)
naive_mean = naive_pnls.mean(0)
naive_std  = naive_pnls.std(0)

ax1.fill_between(t_axis, as_mean - as_std, as_mean + as_std,
                 color=TEAL, alpha=0.18, label="_nolegend_")
ax1.fill_between(t_axis, naive_mean - naive_std, naive_mean + naive_std,
                 color=CORAL, alpha=0.18, label="_nolegend_")
ax1.plot(t_axis, as_mean,    color=TEAL,  lw=2.0, label="A-S optimal")
ax1.plot(t_axis, naive_mean, color=CORAL, lw=2.0, label="Naive fixed spread", linestyle="--")
ax1.axhline(0, color=GRAY, lw=0.5, linestyle=":")
ax1.set_xlabel("Time", fontsize=9, color=DARK)
ax1.set_ylabel("Mark-to-market PnL", fontsize=9, color=DARK)
ax1.legend(fontsize=9, framealpha=0.7)

# Panel 2 — Inventory distribution
style_ax(ax2, "Inventory distribution")

all_as_inv    = as_invs.flatten()
all_naive_inv = naive_invs.flatten()
bins = np.arange(
    min(all_as_inv.min(), all_naive_inv.min()),
    max(all_as_inv.max(), all_naive_inv.max()) + 2
)
ax2.hist(all_naive_inv, bins=bins, color=CORAL, alpha=0.5, label="Naive", density=True)
ax2.hist(all_as_inv,    bins=bins, color=TEAL,  alpha=0.6, label="A-S",   density=True)
ax2.axvline(0, color=GRAY, lw=0.8, linestyle="--")
ax2.set_xlabel("Inventory (units)", fontsize=9, color=DARK)
ax2.set_ylabel("Density", fontsize=9, color=DARK)
ax2.legend(fontsize=9, framealpha=0.7)

inv_var_as    = np.var(all_as_inv)
inv_var_naive = np.var(all_naive_inv)
ax2.text(0.97, 0.97,
         f"Var(AS) = {inv_var_as:.1f}\nVar(Naive) = {inv_var_naive:.1f}",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=8, color=DARK,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=GRAY))

# Panel 3 — Spread
style_ax(ax3, "Quote spread over time  (A-S narrows as horizon approaches; naive is constant)")

mean_spread = as_spreads.mean(0)
ax3.plot(t_axis, mean_spread, color=TEAL, lw=2.0, label="A-S spread (mean)")
ax3.axhline(0.5, color=CORAL, lw=2.0, linestyle="--", label="Naive spread (fixed = 0.5)")
ax3.fill_between(t_axis, as_spreads.min(0), as_spreads.max(0),
                 color=TEAL, alpha=0.08)
ax3.set_xlabel("Time", fontsize=9, color=DARK)
ax3.set_ylabel("Bid-ask spread", fontsize=9, color=DARK)
ax3.legend(fontsize=9, framealpha=0.7)

# Panel 4 — Stats table
ax4.axis("off")
style_ax(ax4, "Summary statistics")

rows = []
for k in as_stats:
    av = as_stats[k]
    nv = naive_stats[k]
    better = TEAL if av > nv else CORAL
    rows.append((k, f"{av:.3f}", f"{nv:.3f}", better))

y = 0.92
for label, av, nv, color in rows:
    ax4.text(0.02, y, label,   transform=ax4.transAxes, fontsize=8,  color=DARK)
    ax4.text(0.62, y, av,      transform=ax4.transAxes, fontsize=8.5, color=color,  fontweight="bold")
    ax4.text(0.82, y, nv,      transform=ax4.transAxes, fontsize=8.5, color=CORAL)
    y -= 0.13

ax4.text(0.02, 1.01, "Metric",         transform=ax4.transAxes, fontsize=8,  color=GRAY)
ax4.text(0.62, 1.01, "A-S",            transform=ax4.transAxes, fontsize=8,  color=TEAL, fontweight="bold")
ax4.text(0.82, 1.01, "Naive",          transform=ax4.transAxes, fontsize=8,  color=CORAL)
ax4.axhline(0, color=GRAY, lw=0.3)

# Title
fig.text(0.5, 0.95,
         "Avellaneda-Stoikov (2008) Optimal Market Maker vs Naive Fixed-Spread Baseline",
         ha="center", fontsize=14, fontweight="bold", color=DARK)
fig.text(0.5, 0.918,
         r"$r(s,q,t) = s - q\gamma\sigma^2(T-t)$   |   "
         r"$\delta^* = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right)$",
         ha="center", fontsize=10, color=GRAY)

os.makedirs("results", exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=LIGHT)
print(f"\nTearsheet saved to {OUT_PATH}")
