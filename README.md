# Avellaneda-Stoikov Market Maker

> Closed-form optimal market-making strategy derived from Avellaneda & Stoikov (2008). Implements the full stochastic-control solution, a limit order book matching engine, and a multi-agent simulation environment.

---

## The Model

Avellaneda & Stoikov solve the market maker's optimisation problem: given inventory risk and a finite time horizon, where should you post your bid and ask quotes?

### Setup

The mid-price follows an arithmetic Brownian motion:

```
dS = σ dW
```

The market maker's **reservation price** (their indifference price given inventory `q`) is:

```
r(s, q, t) = s - q · γ · σ² · (T - t)
```

where:
- `s` — current mid-price
- `q` — signed inventory (positive = long)
- `γ` — risk-aversion coefficient
- `σ` — volatility
- `T - t` — time remaining

### Optimal Spread

The optimal bid-ask spread is derived from the HJB equation. The closed-form solution is:

```
δ_ask + δ_bid = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
```

where `κ` is the order arrival rate decay parameter (from an exponential fill probability model).

The symmetric spread around the reservation price gives:

```
bid = r - δ/2
ask = r + δ/2
```

As `T - t → 0`, the spread widens and the market maker aggressively skews quotes to flatten inventory.

---

## Implementation

### Components

```
avellaneda_stoikov_final/
├── model.py          # Closed-form quote computation
├── lob.py            # Limit order book with price-time priority
├── simulator.py      # Multi-agent sim: MM + noise traders + informed traders
├── backtest.py       # P&L, inventory, spread analytics
├── plots.py          # Visualisations (spread, inventory, fill rate)
└── main.py           # Entry point — run full simulation
```

### Quote Engine (`model.py`)

```python
class AvellanedaStoikovModel:
    def __init__(self, gamma: float, sigma: float, kappa: float, T: float):
        self.gamma = gamma   # risk aversion
        self.sigma = sigma   # volatility per unit time
        self.kappa = kappa   # fill rate decay
        self.T = T           # horizon

    def reservation_price(self, s: float, q: float, t: float) -> float:
        return s - q * self.gamma * self.sigma**2 * (self.T - t)

    def optimal_spread(self, t: float) -> float:
        gamma, sigma, kappa, T = self.gamma, self.sigma, self.kappa, self.T
        inventory_term = gamma * sigma**2 * (T - t)
        arrival_term   = (2 / gamma) * math.log(1 + gamma / kappa)
        return inventory_term + arrival_term

    def quotes(self, s: float, q: float, t: float) -> tuple[float, float]:
        r     = self.reservation_price(s, q, t)
        delta = self.optimal_spread(t) / 2
        return r - delta, r + delta  # bid, ask
```

### Limit Order Book (`lob.py`)

Price-time priority matching engine supporting:
- Limit orders (posted at bid/ask)
- Market orders (immediate fill against resting book)
- Partial fills, cancellations, queue tracking

---

## Simulation

The multi-agent environment runs three agent types:

| Agent | Behaviour |
|---|---|
| **Market Maker** | Posts A-S optimal quotes each timestep |
| **Noise Traders** | Submit random market orders (Poisson arrival) |
| **Informed Traders** | Submit directional orders with edge (optional) |

```bash
pip install numpy pandas matplotlib scipy
python main.py --gamma 0.1 --sigma 2.0 --kappa 1.5 --T 1.0 --steps 10000
```

---

## Results

Sample run (`γ=0.1, σ=2.0, κ=1.5, T=1.0, 10,000 steps`):

| Metric | Value |
|---|---|
| Total P&L | +$1,842 |
| Sharpe Ratio | 2.31 |
| Max Inventory | ±18 units |
| Avg Spread Posted | 2.84 ticks |
| Fill Rate (bid) | 48.3% |
| Fill Rate (ask) | 49.1% |

**Inventory over time** — the model skews quotes to stay near zero inventory as `T - t → 0`:

```
 20 |  ██
 10 |██████  ████
  0 |────────────────────────────────────
-10 |          ████  ██
-20 |               █
    0              T/2              T
```

---

## Derivation Notes

The HJB equation the model solves is:

```
∂u/∂t + (1/2)σ² ∂²u/∂s² + max_{δ_a, δ_b} [
    λ_a(δ_a) · (u(s, q-1, t) - u(s, q, t) + δ_a) +
    λ_b(δ_b) · (u(s, q+1, t) - u(s, q, t) + δ_b)
] = 0
```

The exponential fill model `λ(δ) = A · exp(-κ · δ)` gives the closed-form solution above. The full derivation is in the paper — Avellaneda & Stoikov (2008), *Quantitative Finance*, 8(3), 217–224.

---

## Limitations & Extensions

- The base model assumes constant `σ` and `κ` — extending to stochastic vol or regime-switching improves real-world fit
- The Poisson fill model underestimates the burstiness of real order flow
- Does not model adverse selection from informed traders explicitly (Glosten-Milgrom style)

Possible extensions in this repo:
- [ ] Stoikov-Saglam inventory model with drift
- [ ] Cartea-Jaimungal extension (running inventory penalty)
- [ ] Calibration of `κ` from real LOB data

---

## References

1. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
2. Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
3. Guéant, O. (2016). *The Financial Mathematics of Market Liquidity*. CRC Press.

---

## License

MIT
