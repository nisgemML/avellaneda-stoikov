"""
Limit Order Book with price-time priority matching engine.

Design decisions
----------------
- Bids stored as max-heap (price negated), asks as min-heap.
  heapq only gives min-heap; negating price gives us max-heap for bids.
- Each price level is a deque — O(1) append and popleft for FIFO within level.
- Orders are lazily deleted: cancellation sets remaining=0 and the order
  is skipped when encountered during matching. This avoids O(n) heap surgery.
- Trade records are immutable dataclasses — safe to share across threads
  and easy to serialise.

Invariants (verified by tests):
- best_bid() < best_ask() at all times (no crossed book)
- All resting orders have remaining > 0
- trade.price is always a resting order's price (price improvement goes to aggressor)
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Side(Enum):
    BID = auto()
    ASK = auto()


class OrderType(Enum):
    LIMIT  = auto()
    MARKET = auto()


@dataclass
class Order:
    order_id:   int
    side:       Side
    price:      float          # ignored for MARKET orders
    quantity:   int
    timestamp:  int            # logical clock; ties broken FIFO
    order_type: OrderType = OrderType.LIMIT
    remaining:  int = field(init=False)

    def __post_init__(self):
        self.remaining = self.quantity

    def is_filled(self) -> bool:
        return self.remaining == 0


@dataclass(frozen=True)
class Trade:
    trade_id:      int
    buy_order_id:  int
    sell_order_id: int
    price:         float   # resting order's price (aggressor gets price improvement)
    quantity:      int
    timestamp:     int

    @property
    def notional(self) -> float:
        return self.price * self.quantity


class LimitOrderBook:
    """
    Price-time priority limit order book.

    Public interface
    ----------------
    add_order(order)     -> list[Trade]   insert + match
    cancel_order(id)     -> bool          cancel resting order
    best_bid()           -> float | None
    best_ask()           -> float | None
    mid_price()          -> float | None
    spread()             -> float | None
    bid_depth(levels)    -> list[(price, qty)]
    ask_depth(levels)    -> list[(price, qty)]
    snapshot(levels)     -> str           human-readable
    """

    def __init__(self):
        # price_level -> deque[Order]  (FIFO within each price level)
        self._bids: dict[float, deque] = defaultdict(deque)
        self._asks: dict[float, deque] = defaultdict(deque)

        # heaps for O(log n) best-price lookup
        # bids: store (-price,) so heapq gives max price
        # asks: store ( price,) so heapq gives min price
        self._bid_heap: list = []
        self._ask_heap: list = []

        # active price sets — used to skip stale heap entries (lazy deletion)
        self._active_bids: set[float] = set()
        self._active_asks: set[float] = set()

        # order registry: id -> Order  (for cancellation)
        self._orders: dict[int, Order] = {}

        self._trade_counter = 0
        self.trade_log: list[Trade] = []

    # ── Public interface ──────────────────────────────────────────────────

    def add_order(self, order: Order) -> list[Trade]:
        """Insert order and immediately attempt matching. Returns trades."""
        self._orders[order.order_id] = order

        if order.order_type == OrderType.MARKET:
            return self._match_market(order)

        trades = self._match_limit(order)

        if not order.is_filled():
            self._rest_order(order)

        return trades

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a resting order.

        Lazy deletion: we zero out `remaining` and remove from registry.
        The heap/deque entries remain but are skipped on next access.
        Returns True if the order existed and was active.
        """
        order = self._orders.get(order_id)
        if order is None or order.is_filled():
            return False
        side = order.side
        price = order.price
        order.remaining = 0
        del self._orders[order_id]

        # Eagerly remove from active set if level is now empty
        if side == Side.BID:
            level = self._bids.get(price)
            if level is not None and all(o.remaining == 0 for o in level):
                self._active_bids.discard(price)
        else:
            level = self._asks.get(price)
            if level is not None and all(o.remaining == 0 for o in level):
                self._active_asks.discard(price)

        return True

    def best_bid(self) -> Optional[float]:
        self._clean_bid_heap()
        return -self._bid_heap[0][0] if self._bid_heap else None

    def best_ask(self) -> Optional[float]:
        self._clean_ask_heap()
        return self._ask_heap[0][0] if self._ask_heap else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if (bb is not None and ba is not None) else None

    def bid_depth(self, levels: int = 5) -> list[tuple[float, int]]:
        """Top N bid price levels, best (highest) first."""
        return self._depth(self._bids, self._active_bids, levels, descending=True)

    def ask_depth(self, levels: int = 5) -> list[tuple[float, int]]:
        """Top N ask price levels, best (lowest) first."""
        return self._depth(self._asks, self._active_asks, levels, descending=False)

    def snapshot(self, levels: int = 5) -> str:
        asks = list(reversed(self.ask_depth(levels)))
        bids = self.bid_depth(levels)
        mid  = self.mid_price()
        W    = 36
        lines = ["=" * W, f"{'ASKS':>{W//2}}"]
        lines.append("-" * W)
        for price, qty in asks:
            lines.append(f"  {price:>10.4f}  |  {qty:>8} ")
        mid_str = f"{mid:.4f}" if mid else "---"
        lines.append(f"  {'─── mid ' + mid_str + ' ───':>{W-4}}")
        for price, qty in bids:
            lines.append(f"  {price:>10.4f}  |  {qty:>8} ")
        lines += ["-" * W, f"{'BIDS':>{W//2}}", "=" * W]
        return "\n".join(lines)

    # ── Matching logic ────────────────────────────────────────────────────

    def _match_limit(self, order: Order) -> list[Trade]:
        trades = []
        if order.side == Side.BID:
            while order.remaining > 0:
                best = self.best_ask()
                if best is None or best > order.price:
                    break
                trades += self._fill_level(order, self._asks[best], best)
                if not self._asks[best]:
                    self._active_asks.discard(best)
                    del self._asks[best]
        else:
            while order.remaining > 0:
                best = self.best_bid()
                if best is None or best < order.price:
                    break
                trades += self._fill_level(order, self._bids[best], best)
                if not self._bids[best]:
                    self._active_bids.discard(best)
                    del self._bids[best]
        return trades

    def _match_market(self, order: Order) -> list[Trade]:
        """Walk the book until filled or exhausted."""
        trades = []
        if order.side == Side.BID:
            for price in sorted(self._asks.keys()):
                if order.remaining == 0:
                    break
                trades += self._fill_level(order, self._asks[price], price)
                if not self._asks[price]:
                    self._active_asks.discard(price)
                    del self._asks[price]
        else:
            for price in sorted(self._bids.keys(), reverse=True):
                if order.remaining == 0:
                    break
                trades += self._fill_level(order, self._bids[price], price)
                if not self._bids[price]:
                    self._active_bids.discard(price)
                    del self._bids[price]
        return trades

    def _fill_level(self, aggressor: Order, level: deque, price: float) -> list[Trade]:
        """Fill aggressor against all resting orders at one price level."""
        trades = []
        while level and aggressor.remaining > 0:
            resting = level[0]
            if resting.remaining == 0:   # lazily-deleted
                level.popleft()
                continue

            fill_qty = min(aggressor.remaining, resting.remaining)
            aggressor.remaining -= fill_qty
            resting.remaining   -= fill_qty

            self._trade_counter += 1
            if aggressor.side == Side.BID:
                buy_id, sell_id = aggressor.order_id, resting.order_id
            else:
                buy_id, sell_id = resting.order_id, aggressor.order_id

            trade = Trade(
                trade_id      = self._trade_counter,
                buy_order_id  = buy_id,
                sell_order_id = sell_id,
                price         = price,
                quantity      = fill_qty,
                timestamp     = aggressor.timestamp,
            )
            trades.append(trade)
            self.trade_log.append(trade)

            if resting.is_filled():
                level.popleft()
                self._orders.pop(resting.order_id, None)

        return trades

    def _rest_order(self, order: Order):
        if order.side == Side.BID:
            self._bids[order.price].append(order)
            if order.price not in self._active_bids:
                self._active_bids.add(order.price)
                heapq.heappush(self._bid_heap, (-order.price,))
        else:
            self._asks[order.price].append(order)
            if order.price not in self._active_asks:
                self._active_asks.add(order.price)
                heapq.heappush(self._ask_heap, (order.price,))

    # ── Heap maintenance (lazy deletion) ─────────────────────────────────

    def _clean_bid_heap(self):
        while self._bid_heap and (-self._bid_heap[0][0]) not in self._active_bids:
            heapq.heappop(self._bid_heap)

    def _clean_ask_heap(self):
        while self._ask_heap and self._ask_heap[0][0] not in self._active_asks:
            heapq.heappop(self._ask_heap)

    def _depth(self, book, active, levels, descending) -> list[tuple[float, int]]:
        prices = sorted(active, reverse=descending)[:levels]
        result = []
        for p in prices:
            qty = sum(o.remaining for o in book[p] if o.remaining > 0)
            if qty > 0:
                result.append((p, qty))
        return result
