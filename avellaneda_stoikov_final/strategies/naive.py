"""
Naive symmetric market maker — the baseline A-S beats.

Posts a fixed spread around the mid-price at all times.
No inventory awareness, no time-to-horizon adjustment.

This is what most "market making tutorials" implement.
We include it to demonstrate concretely what A-S adds.
"""

from __future__ import annotations
from dataclasses import dataclass
from strategies.avellaneda_stoikov import Quote


class NaiveMarketMaker:
    """
    Fixed-spread market maker. Zero sophistication.

    Always quotes: bid = mid - half_spread, ask = mid + half_spread.
    Ignores inventory, volatility regime, and time horizon.
    """

    def __init__(self, fixed_spread: float = 0.5):
        self.fixed_spread = fixed_spread

    def quotes(self, s: float, q: int, t: float) -> Quote:
        half = self.fixed_spread / 2.0
        return Quote(
            bid=s - half,
            ask=s + half,
            reservation_price=s,   # no skew
            spread=self.fixed_spread,
            mid=s,
        )
