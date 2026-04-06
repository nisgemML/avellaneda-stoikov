"""
Market dynamics model.

Models the mid-price as a Brownian motion (as in A-S 2008) and
order arrivals as Poisson processes whose intensity decays
exponentially with distance from the mid-price.

Reference: Avellaneda & Stoikov (2008), Section 2.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketParams:
    """
    Parameters governing mid-price dynamics and order flow.

    Attributes
    ----------
    sigma : float
        Volatility of the mid-price (annual, or per unit time).
    kappa : float
        Order-book resilience. Controls how fast order-arrival
        intensity decays with distance from mid: lambda = A * exp(-kappa * delta).
    A : float
        Baseline order arrival rate (orders per unit time at zero spread).
    dt : float
        Simulation time step.
    T : float
        Total horizon (same units as dt).
    """
    sigma: float = 2.0
    kappa: float = 1.5
    A: float = 140.0
    dt: float = 0.005
    T: float = 1.0

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)


class MidPriceProcess:
    """
    Arithmetic Brownian motion for mid-price.

    dS = sigma * dW

    No drift term — A-S assume a symmetric, zero-drift reference price.
    This is appropriate for a market maker who has no directional view.
    """

    def __init__(self, params: MarketParams, s0: float = 100.0, seed: int | None = None):
        self.params = params
        self.s0 = s0
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> np.ndarray:
        """
        Returns array of mid-prices of length n_steps + 1.
        Uses exact simulation: S(t+dt) = S(t) + sigma * sqrt(dt) * Z.
        """
        p = self.params
        shocks = self.rng.normal(0, p.sigma * np.sqrt(p.dt), p.n_steps)
        prices = np.empty(p.n_steps + 1)
        prices[0] = self.s0
        np.cumsum(shocks, out=prices[1:])
        prices[1:] += self.s0
        return prices


class OrderArrivalModel:
    """
    Poisson order arrivals with intensity decaying in quote distance.

    lambda_b(delta_b) = A * exp(-kappa * delta_b)   [buy arrivals at ask]
    lambda_a(delta_a) = A * exp(-kappa * delta_a)   [sell arrivals at bid]

    A large kappa means orders are very sensitive to how far your quotes
    are from the mid — posting wide spreads gets you very few fills.
    """

    def __init__(self, params: MarketParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def arrival_intensity(self, delta: float) -> float:
        """Expected arrivals per unit time at quote distance delta."""
        return self.params.A * np.exp(-self.params.kappa * delta)

    def has_arrival(self, delta: float) -> bool:
        """
        Bernoulli trial for one time step dt.
        P(arrival) = lambda * dt  (valid for small dt).
        """
        prob = self.arrival_intensity(delta) * self.params.dt
        return self.rng.random() < prob
