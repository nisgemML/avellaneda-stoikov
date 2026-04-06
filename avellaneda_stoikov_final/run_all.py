"""
run_all.py  —  single entry point

Runs:
  1. All tests (order book + strategy)
  2. MC comparison: A-S vs Naive  → results/tearsheet.png
  3. Multi-agent exchange sim      → results/exchange.png
  4. Parameter sensitivity sweep  → results/sensitivity.png

Usage:
    python run_all.py
"""

import sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def banner(text):
    print(f"\n{BOLD}{CYAN}{'━'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'━'*60}{RESET}")

def run_step(label, fn):
    print(f"\n  {YELLOW}▶{RESET} {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"{GREEN}done{RESET} {DIM}({elapsed:.1f}s){RESET}")
        return True
    except Exception as e:
        print(f"{RED}FAILED{RESET}\n    {e}")
        return False

os.makedirs("results", exist_ok=True)

# ── 1. Tests ────────────────────────────────────────────────────────────────
banner("Step 1 — Running all tests")

for test_file in ["tests/test_order_book.py", "tests/test_strategy.py"]:
    result = subprocess.run([sys.executable, test_file], capture_output=False)
    if result.returncode != 0:
        print(f"{RED}Tests failed. Fix before proceeding.{RESET}")
        sys.exit(1)

# ── 2. MC Comparison ────────────────────────────────────────────────────────
banner("Step 2 — Monte Carlo comparison (A-S vs Naive, 200 paths)")

def run_mc():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_comparison", "run_comparison.py")
    mod  = importlib.util.load_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)

# Run as subprocess to avoid matplotlib state issues
result = subprocess.run([sys.executable, "run_comparison.py"], capture_output=False)

# ── 3. Exchange simulation ──────────────────────────────────────────────────
banner("Step 3 — Multi-agent exchange simulation")
result = subprocess.run([sys.executable, "run_exchange.py"], capture_output=False)

# ── 4. Sensitivity analysis ─────────────────────────────────────────────────
banner("Step 4 — Parameter sensitivity sweep")
result = subprocess.run([sys.executable, "sensitivity.py"], capture_output=False)

# ── Summary ─────────────────────────────────────────────────────────────────
banner("Done")
print(f"""
  Output files:
    results/tearsheet.png    — A-S vs Naive PnL, inventory, spread
    results/exchange.png     — Multi-agent simulation (noise + informed traders)
    results/sensitivity.png  — Effect of gamma, sigma, kappa on performance

  To push to GitHub:
    git init && git add . && git commit -m "Avellaneda-Stoikov optimal market maker"
""")
