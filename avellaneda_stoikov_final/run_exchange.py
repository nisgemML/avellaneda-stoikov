"""
run_exchange.py

Runs the full multi-agent exchange simulation and produces a
three-panel chart showing:
  1. Mid-price + individual trade executions
  2. PnL: market maker vs noise trader vs informed trader
  3. Market maker inventory and spread over time
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.market import MarketParams
from simulation.exchange import MultiAgentExchange

PARAMS = MarketParams(sigma=2.0, kappa=1.5, A=140.0, dt=0.005, T=1.0)

print("Running multi-agent exchange simulation...")
exchange = MultiAgentExchange(
    params             = PARAMS,
    gamma              = 0.1,
    noise_intensity    = 30.0,
    informed_intensity = 3.0,
    informed_edge      = 0.05,
    seed               = 7,
)
result = exchange.run()

print(f"  Total trades executed : {len(result.trade_prices)}")
print(f"  MM final PnL          : {result.mm_pnl[-1]:.2f}")
print(f"  Noise trader final PnL: {result.noise_pnl[-1]:.2f}")
print(f"  Informed final PnL    : {result.informed_pnl[-1]:.2f}")

# ── Plot ────────────────────────────────────────────────────────────────────
LIGHT = "#f5f5f0"
DARK  = "#1a1a2e"
TEAL  = "#1D9E75"
CORAL = "#D85A30"
AMBER = "#EF9F27"
PURPLE= "#534AB7"
GRAY  = "#888780"

fig, axes = plt.subplots(3, 1, figsize=(14, 11), facecolor=LIGHT,
                          gridspec_kw={"height_ratios": [2, 1.5, 1]})
fig.suptitle(
    "Multi-Agent Exchange: Market Maker vs Noise Traders vs Informed Traders",
    fontsize=13, fontweight="bold", color=DARK, y=0.99
)

def style(ax, title):
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(colors=DARK, labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color=DARK, pad=6)

# Panel 1 — mid-price + trade dots
ax = axes[0]
style(ax, "Mid-price process with trade executions")
ax.plot(result.t_axis, result.mid_prices, color=GRAY, lw=1.0, label="Mid-price", zorder=1)
if result.trade_times:
    ax.scatter(result.trade_times, result.trade_prices,
               s=8, color=TEAL, alpha=0.4, zorder=2, label="Executed trades")
ax.set_ylabel("Price", fontsize=9, color=DARK)
ax.legend(fontsize=9, framealpha=0.7)

# Panel 2 — PnL by agent
ax = axes[1]
style(ax, "Mark-to-market PnL by agent")
ax.plot(result.t_axis, result.mm_pnl,       color=TEAL,   lw=2.0, label="Market maker (A-S)")
ax.plot(result.t_axis, result.noise_pnl,    color=CORAL,  lw=1.5, label="Noise traders",  linestyle="--")
ax.plot(result.t_axis, result.informed_pnl, color=AMBER,  lw=1.5, label="Informed trader", linestyle="-.")
ax.axhline(0, color=GRAY, lw=0.5, linestyle=":")
ax.set_ylabel("PnL", fontsize=9, color=DARK)
ax.legend(fontsize=9, framealpha=0.7)

# Panel 3 — MM inventory + spread
ax3a = axes[2]
style(ax3a, "Market maker: inventory (bars) and quote spread (line)")
ax3b = ax3a.twinx()

ax3a.bar(result.t_axis, result.mm_inventory,
         width=PARAMS.dt, color=TEAL, alpha=0.35, label="Inventory")
ax3b.plot(result.t_axis, result.mm_spreads,
          color=CORAL, lw=1.2, alpha=0.8, label="Spread")

ax3a.set_ylabel("Inventory (units)", fontsize=9, color=TEAL)
ax3b.set_ylabel("Spread", fontsize=9, color=CORAL)
ax3a.set_xlabel("Time", fontsize=9, color=DARK)
ax3a.tick_params(axis="y", colors=TEAL)
ax3b.tick_params(axis="y", colors=CORAL)
ax3b.spines[["top"]].set_visible(False)

lines1, labels1 = ax3a.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3a.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("results", exist_ok=True)
plt.savefig("results/exchange.png", dpi=150, bbox_inches="tight", facecolor=LIGHT)
print("Saved results/exchange.png")
