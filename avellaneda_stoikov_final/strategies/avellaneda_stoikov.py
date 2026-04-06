"""
Avellaneda-Stoikov (2008) optimal market-making strategy.

The paper solves a stochastic optimal control problem:
    maximise E[u(W_T + q_T * S_T)]
    subject to:
        dS = sigma * dW          (Brownian mid-price)
        arrivals ~ Poisson(A * exp(-kappa * delta))

where u(x) = -exp(-gamma * x) is CARA utility with risk-aversion gamma.

The key results (equations 7 and 13 in the paper):

1. Reservation price:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

   The market maker shades the mid-price DOWN when long (q > 0)
   to reflect the risk of holding inventory.

2. Optimal spread:
        delta* = gamma * sigma^2 * (T - t)
                 + (2/gamma) * ln(1 + gamma/kappa)

   This is the TOTAL spread. Each side gets half.

Design decision: we separate the "indifference price" (reservation price)
from the "spread" so each can be tested independently.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from core.market import MarketParams


@dataclass
class Quote:
    bid: float
    ask: float
    reservation_price: float
    spread: float
    mid: float

    @property
    def bid_delta(self) -> float:
        """Distance of bid from mid."""
        return self.mid - self.bid

    @property
    def ask_delta(self) -> float:
        """Distance of ask from mid."""
        return self.ask - self.mid


class AvellanedaStoikov:
    """
    Closed-form optimal quotes from A-S 2008.

    Parameters
    ----------
    params : MarketParams
        Market dynamics (sigma, kappa, A).
    gamma : float
        Risk-aversion coefficient. Higher gamma -> tighter inventory
        control, wider spreads, more aggressive reservation price shading.
        Typical range: 0.01 - 0.5.

    Design rationale
    ----------------
    The two outputs — reservation price and spread — are independent:
    - Reservation price encodes WHERE the mid of our quotes sits (skewed
      away from inventory).
    - Spread encodes HOW WIDE our quotes are (wider when volatile or
      near end of horizon).

    This separation makes it easy to test each component in isolation
    and to ablate either one (e.g., "what if we use optimal spread but
    no inventory skew?").
    """

    def __init__(self, params: MarketParams, gamma: float = 0.1):
        self.params = params
        self.gamma = gamma
        # Pre-compute the time-independent part of the spread
        self._spread_const = (2.0 / gamma) * np.log(1.0 + gamma / params.kappa)

    def reservation_price(self, s: float, q: int, t: float) -> float:
        """
        Equation (7) from A-S 2008.

        r = s - q * gamma * sigma^2 * (T - t)

        Interpretation: if we're long q units, the price at which we're
        indifferent between buying/selling is BELOW the mid-price.
        The further from expiry, the larger the inventory penalty.
        """
        time_remaining = self.params.T - t
        return s - q * self.gamma * (self.params.sigma ** 2) * time_remaining

    def optimal_spread(self, t: float) -> float:
        """
        Equation (13) from A-S 2008.

        delta* = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

        Two components:
        1. gamma * sigma^2 * (T-t): widens spread when volatile and far from
           horizon (more time for inventory to hurt us).
        2. (2/gamma) * ln(1 + gamma/kappa): constant term — the minimum
           spread that compensates for adverse selection given order-book
           depth (kappa).
        """
        time_remaining = self.params.T - t
        return self.gamma * (self.params.sigma ** 2) * time_remaining + self._spread_const

    def quotes(self, s: float, q: int, t: float) -> Quote:
        """
        Compute bid and ask quotes at time t.

        The bid/ask are symmetric around the RESERVATION PRICE (not mid).
        This is the key insight: inventory skew shifts the entire quote
        pair, it does not widen the spread asymmetrically.
        """
        r = self.reservation_price(s, q, t)
        half_spread = self.optimal_spread(t) / 2.0
        return Quote(
            bid=r - half_spread,
            ask=r + half_spread,
            reservation_price=r,
            spread=self.optimal_spread(t),
            mid=s,
        )
