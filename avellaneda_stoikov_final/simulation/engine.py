"""
Simulation engine.

Runs a market-making strategy over a simulated price path and
records the full state history for analysis.

State at each step:
- mid-price (from Brownian motion)
- quotes posted by the strategy
- fills (order arrivals sampled from Poisson model)
- inventory (net position in shares)
- cash (accumulated from fills)
- PnL: cash + inventory * mid  (mark-to-market)

Design decisions
----------------
- We separate "did an order arrive?" (OrderArrivalModel) from
  "did it fill?" (always yes if it arrived — we simplify by
  assuming arrivals ARE fills, consistent with A-S 2008).
- Inventory is bounded by max_inventory to prevent blow-up in
  adversarial price paths. This is a practical addition not in
  the paper — real market makers always have hard limits.
- Trade size is fixed at 1 unit. Extending to variable size
  is straightforward but obscures the core dynamics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from core.market import MarketParams, MidPriceProcess, OrderArrivalModel


@dataclass
class StepRecord:
    t: float
    mid: float
    bid: float
    ask: float
    reservation_price: float
    spread: float
    inventory: int
    cash: float
    pnl: float
    bid_fill: bool
    ask_fill: bool


@dataclass
class SimulationResult:
    records: list[StepRecord]
    params: MarketParams
    strategy_name: str

    @property
    def pnl_series(self) -> np.ndarray:
        return np.array([r.pnl for r in self.records])

    @property
    def inventory_series(self) -> np.ndarray:
        return np.array([r.inventory for r in self.records])

    @property
    def spread_series(self) -> np.ndarray:
        return np.array([r.spread for r in self.records])

    @property
    def final_pnl(self) -> float:
        return self.records[-1].pnl

    @property
    def max_inventory(self) -> int:
        return int(np.max(np.abs(self.inventory_series)))

    @property
    def pnl_sharpe(self) -> float:
        """Sharpe ratio of per-step PnL changes."""
        deltas = np.diff(self.pnl_series)
        if deltas.std() == 0:
            return 0.0
        return (deltas.mean() / deltas.std()) * np.sqrt(len(deltas))

    @property
    def fill_count(self) -> tuple[int, int]:
        """(bid_fills, ask_fills)"""
        bids = sum(1 for r in self.records if r.bid_fill)
        asks = sum(1 for r in self.records if r.ask_fill)
        return bids, asks


class Simulator:
    """
    Runs any strategy that implements .quotes(s, q, t) -> Quote.
    """

    def __init__(
        self,
        params: MarketParams,
        strategy,
        strategy_name: str,
        s0: float = 100.0,
        max_inventory: int = 50,
        seed: int | None = 42,
    ):
        self.params = params
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.max_inventory = max_inventory
        self.rng = np.random.default_rng(seed)

        self._price_process = MidPriceProcess(params, s0, seed=seed)
        self._arrival_model = OrderArrivalModel(params, self.rng)

    def run(self) -> SimulationResult:
        prices = self._price_process.simulate()
        p = self.params

        inventory = 0
        cash = 0.0
        records: list[StepRecord] = []

        for i, s in enumerate(prices):
            t = i * p.dt
            quote = self.strategy.quotes(s, inventory, t)

            # Sample order arrivals — only fill if within inventory limits
            bid_fill = False
            ask_fill = False

            if inventory < self.max_inventory:
                # A market sell hits our bid → we buy → inventory increases
                bid_fill = self._arrival_model.has_arrival(quote.bid_delta)
                if bid_fill:
                    inventory += 1
                    cash -= quote.bid

            if inventory > -self.max_inventory:
                # A market buy hits our ask → we sell → inventory decreases
                ask_fill = self._arrival_model.has_arrival(quote.ask_delta)
                if ask_fill:
                    inventory -= 1
                    cash += quote.ask

            pnl = cash + inventory * s

            records.append(StepRecord(
                t=t,
                mid=s,
                bid=quote.bid,
                ask=quote.ask,
                reservation_price=quote.reservation_price,
                spread=quote.spread,
                inventory=inventory,
                cash=cash,
                pnl=pnl,
                bid_fill=bid_fill,
                ask_fill=ask_fill,
            ))

        return SimulationResult(records, p, self.strategy_name)
