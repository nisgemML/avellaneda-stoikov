"""
Tests for the A-S market maker.

Two categories:
1. Unit tests  -- closed-form math matches known analytical values
2. Invariant tests -- properties that must hold for ALL valid inputs
   (we grid-search a wide parameter space)

Run with:  python tests/test_strategy.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.market import MarketParams
from strategies.avellaneda_stoikov import AvellanedaStoikov
from strategies.naive import NaiveMarketMaker
from simulation.engine import Simulator

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []

def test(name, condition, detail=""):
    ok = bool(condition)
    _results.append(ok)
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))

def approx_eq(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

params = MarketParams(sigma=2.0, kappa=1.5, A=140.0, dt=0.005, T=1.0)
strat  = AvellanedaStoikov(params, gamma=0.1)

# -- 1. Reservation price --------------------------------------------------
section("Reservation price -- unit tests")

test("q=0  => reservation price == mid",
     approx_eq(strat.reservation_price(100.0, 0, 0.0), 100.0))
test("q>0  => reservation price < mid",
     strat.reservation_price(100.0, 10, 0.0) < 100.0)
test("q<0  => reservation price > mid",
     strat.reservation_price(100.0, -10, 0.0) > 100.0)
test("t=T  => inventory penalty vanishes",
     approx_eq(strat.reservation_price(100.0, 50, params.T), 100.0))

r = strat.reservation_price(100.0, 1, 0.0)
test(f"known value: q=1,t=0 => 99.6 (got {r:.6f})", approx_eq(r, 99.6))

r0 = strat.reservation_price(100.0, 0, 0.0)
r1 = strat.reservation_price(100.0, 1, 0.0)
r2 = strat.reservation_price(100.0, 2, 0.0)
test("linear in inventory", approx_eq(r2 - r1, r1 - r0))

# -- 2. Optimal spread -----------------------------------------------------
section("Optimal spread -- unit tests")

for t_val in [0.0, 0.5, 0.99]:
    test(f"spread > 0 at t={t_val}", strat.optimal_spread(t_val) > 0)

s0, s5, s9 = strat.optimal_spread(0.0), strat.optimal_spread(0.5), strat.optimal_spread(0.99)
test("spread decreases toward horizon", s0 > s5 > s9)

expected_T = (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
got_T = strat.optimal_spread(params.T)
test(f"terminal spread matches closed form (got {got_T:.4f}, expected {expected_T:.4f})",
     approx_eq(got_T, expected_T))

lo = AvellanedaStoikov(params, gamma=0.05)
hi = AvellanedaStoikov(params, gamma=0.30)
test("higher gamma => wider spread", hi.optimal_spread(0.5) > lo.optimal_spread(0.5))

# -- 3. Quote structure ----------------------------------------------------
section("Quote structure -- unit tests")

qt = strat.quotes(100.0, 0, 0.0)
test("bid < ask", qt.bid < qt.ask)
test("spread == ask - bid", approx_eq(qt.ask - qt.bid, qt.spread))
test("symmetric around reservation price",
     approx_eq(qt.reservation_price - qt.bid, qt.ask - qt.reservation_price))

# -- 4. Invariant sweep ----------------------------------------------------
section("Invariant tests -- exhaustive parameter sweep")

bid_lt_ask_fail = monotone_fail = spread_neg_fail = 0
total = 0

for gamma in [0.01, 0.05, 0.1, 0.2, 0.5]:
    s = AvellanedaStoikov(params, gamma=gamma)
    for price in [50.0, 100.0, 200.0]:
        for t in np.linspace(0, 0.99, 20):
            for inv in range(-50, 51, 5):
                total += 1
                q2 = s.quotes(price, inv, t)
                if q2.bid >= q2.ask: bid_lt_ask_fail += 1
                r_q  = s.reservation_price(price, inv,     t)
                r_q1 = s.reservation_price(price, inv + 1, t)
                if r_q1 >= r_q: monotone_fail += 1
                if s.optimal_spread(t) <= 0: spread_neg_fail += 1

test(f"bid < ask across {total} combinations",       bid_lt_ask_fail == 0)
test(f"reservation price monotone in q ({total})",   monotone_fail   == 0)
test(f"spread > 0 for all gamma/time",               spread_neg_fail == 0)

# -- 5. Simulation tests ---------------------------------------------------
section("Simulation tests")

as_strat = AvellanedaStoikov(params, gamma=0.1)
result = Simulator(params, as_strat, "AS", max_inventory=20, seed=42).run()

test("PnL series has no NaN",                   not np.any(np.isnan(result.pnl_series)))
test("inventory never exceeds hard limit",       result.max_inventory <= 20)
bid_f, ask_f = result.fill_count
test(f"bid fills occur (got {bid_f})",           bid_f > 10)
test(f"ask fills occur (got {ask_f})",           ask_f > 10)

as_vars, naive_vars = [], []
for seed in range(30):
    ar = Simulator(params, AvellanedaStoikov(params, gamma=0.1), "AS",    seed=seed).run()
    nr = Simulator(params, NaiveMarketMaker(0.5),                "Naive", seed=seed).run()
    as_vars.append(np.var(ar.inventory_series))
    naive_vars.append(np.var(nr.inventory_series))

test(f"A-S inventory variance < naive (AS={np.mean(as_vars):.1f} vs Naive={np.mean(naive_vars):.1f})",
     np.mean(as_vars) < np.mean(naive_vars))

# -- Summary ---------------------------------------------------------------
print(f"\n{'='*60}")
passed = sum(_results)
total_t = len(_results)
col = "\033[92m" if passed == total_t else "\033[91m"
print(f"  {col}{passed}/{total_t} tests passed\033[0m")
print(f"{'='*60}\n")
if passed < total_t:
    sys.exit(1)
