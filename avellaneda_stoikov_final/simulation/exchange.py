"""
Multi-agent exchange simulation.

Three agent types compete on a shared limit order book:

1. NoiseTradersAgent
   Random walk traders with no information. They submit market orders
   at random intervals with random direction. They are the "dumb money"
   that provides the market maker's edge.

2. InformedTraderAgent
   Has a signal about future price direction (drawn from the true price
   process). Trades aggressively when the signal is strong. Their
   presence creates adverse selection risk for the market maker.

3. MarketMakerAgent (wraps A-S strategy)
   Posts two-sided quotes using the A-S optimal formula. Gets filled
   by both noise traders and informed traders — only the former is
   profitable.

Design note: agent interaction
-------------------------------
Each agent sees the same order book snapshot at each time step.
Agents act sequentially (not simultaneously) — a simplification
vs. real concurrent exchange, but sufficient to study the economics.
Order of action: market maker posts first, then other agents react.
This is realistic: market makers must commit to quotes before
knowing who will fill them.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from core.order_book import LimitOrderBook, Order, Side, OrderType, Trade
from core.market import MarketParams, MidPriceProcess
from strategies.avellaneda_stoikov import AvellanedaStoikov


@dataclass
class AgentState:
    name:       str
    inventory:  int   = 0
    cash:       float = 0.0
    order_count: int  = 0

    def pnl(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def mark_fill(self, side: Side, price: float, qty: int):
        if side == Side.BID:   # we bought
            self.inventory += qty
            self.cash      -= price * qty
        else:                  # we sold
            self.inventory -= qty
            self.cash      += price * qty
        self.order_count += 1


@dataclass
class ExchangeResult:
    t_axis:         np.ndarray
    mid_prices:     np.ndarray
    mm_pnl:         np.ndarray
    mm_inventory:   np.ndarray
    noise_pnl:      np.ndarray
    informed_pnl:   np.ndarray
    trade_prices:   list[float]
    trade_times:    list[float]
    mm_spreads:     np.ndarray


class MultiAgentExchange:
    """
    Full exchange simulation with three competing agent types.

    Parameters
    ----------
    params : MarketParams
    gamma : float
        A-S risk aversion for the market maker.
    noise_intensity : float
        Expected noise trader orders per unit time.
    informed_intensity : float
        Expected informed trader orders per unit time.
    informed_edge : float
        How many steps ahead the informed trader can see (as fraction of T).
    seed : int
    """

    def __init__(
        self,
        params:             MarketParams,
        gamma:              float = 0.1,
        noise_intensity:    float = 5.0,
        informed_intensity: float = 1.0,
        informed_edge:      float = 0.05,
        seed:               int   = 42,
    ):
        self.params             = params
        self.gamma              = gamma
        self.noise_intensity    = noise_intensity
        self.informed_intensity = informed_intensity
        self.informed_edge      = informed_edge
        self.rng                = np.random.default_rng(seed)

        self._as_strategy = AvellanedaStoikov(params, gamma=gamma)
        self._order_id    = 0

    def _next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def run(self) -> ExchangeResult:
        params = self.params
        prices = MidPriceProcess(params, seed=int(self.rng.integers(1_000_000))).simulate()

        lob = LimitOrderBook()

        mm      = AgentState("MarketMaker")
        noise   = AgentState("NoiseTrades")
        informed = AgentState("InformedTrader")

        mm_pnls      = np.zeros(params.n_steps + 1)
        mm_invs      = np.zeros(params.n_steps + 1, dtype=int)
        noise_pnls   = np.zeros(params.n_steps + 1)
        informed_pnls= np.zeros(params.n_steps + 1)
        mm_spreads   = np.zeros(params.n_steps + 1)

        trade_prices: list[float] = []
        trade_times:  list[float] = []

        # look-ahead steps for informed trader
        edge_steps = max(1, int(self.informed_edge * params.n_steps))

        mm_bid_id = mm_ask_id = None

        for i in range(params.n_steps + 1):
            s = prices[i]
            t = i * params.dt

            # ── Market maker: post quotes first, cancel after fill window ──
            quote = self._as_strategy.quotes(s, mm.inventory, t)
            mm_spreads[i] = quote.spread

            if mm_bid_id is not None:
                lob.cancel_order(mm_bid_id)
                mm_bid_id = None
            if mm_ask_id is not None:
                lob.cancel_order(mm_ask_id)
                mm_ask_id = None

            bid_price = round(quote.bid, 2)
            ask_price = round(quote.ask, 2)

            if bid_price < ask_price:
                bid_order = Order(self._next_id(), Side.BID, bid_price, 1, i)
                ask_order = Order(self._next_id(), Side.ASK, ask_price, 1, i)
                lob.add_order(bid_order)
                lob.add_order(ask_order)
                mm_bid_id = bid_order.order_id
                mm_ask_id = ask_order.order_id

            # ── Noise traders: random market orders ─────────────────────
            if self.rng.random() < self.noise_intensity * params.dt:
                side = Side.BID if self.rng.random() < 0.5 else Side.ASK
                mo   = Order(self._next_id(), side, 0, 1, i, OrderType.MARKET)
                trades = lob.add_order(mo)
                for tr in trades:
                    trade_prices.append(tr.price)
                    trade_times.append(t)
                    # noise trader always the aggressor
                    if side == Side.BID:
                        noise.mark_fill(Side.BID, tr.price, tr.quantity)
                        mm.mark_fill(Side.ASK,   tr.price, tr.quantity)
                    else:
                        noise.mark_fill(Side.ASK, tr.price, tr.quantity)
                        mm.mark_fill(Side.BID,    tr.price, tr.quantity)

            # ── Informed traders: directional market orders ─────────────
            if self.rng.random() < self.informed_intensity * params.dt:
                future_idx = min(i + edge_steps, params.n_steps)
                future_price = prices[future_idx]
                # trade in direction of expected move
                if future_price > s + 0.05:
                    side = Side.BID
                elif future_price < s - 0.05:
                    side = Side.ASK
                else:
                    side = None

                if side is not None:
                    mo = Order(self._next_id(), side, 0, 1, i, OrderType.MARKET)
                    trades = lob.add_order(mo)
                    for tr in trades:
                        trade_prices.append(tr.price)
                        trade_times.append(t)
                        if side == Side.BID:
                            informed.mark_fill(Side.BID, tr.price, tr.quantity)
                            mm.mark_fill(Side.ASK,       tr.price, tr.quantity)
                        else:
                            informed.mark_fill(Side.ASK, tr.price, tr.quantity)
                            mm.mark_fill(Side.BID,       tr.price, tr.quantity)

            mm_pnls[i]       = mm.pnl(s)
            mm_invs[i]       = mm.inventory
            noise_pnls[i]    = noise.pnl(s)
            informed_pnls[i] = informed.pnl(s)

        t_axis = np.linspace(0, params.T, params.n_steps + 1)
        return ExchangeResult(
            t_axis         = t_axis,
            mid_prices     = prices,
            mm_pnl         = mm_pnls,
            mm_inventory   = mm_invs,
            noise_pnl      = noise_pnls,
            informed_pnl   = informed_pnls,
            trade_prices   = trade_prices,
            trade_times    = trade_times,
            mm_spreads     = mm_spreads,
        )
