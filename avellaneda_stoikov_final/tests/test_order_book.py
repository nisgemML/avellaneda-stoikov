"""
Tests for the Limit Order Book matching engine.

Invariants verified:
- No crossed book: best_bid < best_ask at all times
- Price-time priority: best price always fills first
- FIFO within level: earlier order fills before later at same price
- Fill quantity conservation: aggressor filled == resting filled
- Market orders walk the book in price order
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.order_book import LimitOrderBook, Order, Side, OrderType

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def test(name, condition, detail=""):
    ok = bool(condition)
    _results.append(ok)
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def make_order(oid, side, price, qty, ts, otype=OrderType.LIMIT):
    return Order(oid, side, price, qty, ts, otype)


# ── helpers ────────────────────────────────────────────────────────────────

def fresh_book():
    return LimitOrderBook()


# ── 1. Basic insertion ─────────────────────────────────────────────────────
section("Basic insertion -- no crossing")

lob = fresh_book()
lob.add_order(make_order(1, Side.BID, 99.0, 10, 0))
lob.add_order(make_order(2, Side.ASK, 101.0, 10, 1))

test("best_bid correct after insert",  lob.best_bid() == 99.0)
test("best_ask correct after insert",  lob.best_ask() == 101.0)
test("mid_price correct",              lob.mid_price() == 100.0)
test("spread correct",                 lob.spread() == 2.0)
test("no trades on non-crossing add",  len(lob.trade_log) == 0)


# ── 2. Simple limit match ──────────────────────────────────────────────────
section("Limit order matching")

lob = fresh_book()
lob.add_order(make_order(1, Side.ASK, 100.0, 5, 0))   # resting ask
trades = lob.add_order(make_order(2, Side.BID, 100.0, 3, 1))  # crossing bid

test("trade generated",             len(trades) == 1)
test("trade price = resting price", trades[0].price == 100.0)
test("trade quantity correct",      trades[0].quantity == 3)
test("resting order partially filled",
     lob.ask_depth()[0][1] == 2)   # 5 - 3 = 2 remaining
test("aggressor fully filled (gone from book)",
     lob.best_bid() is None)        # bid was fully consumed


# ── 3. Price-time priority ─────────────────────────────────────────────────
section("Price-time priority")

lob = fresh_book()
lob.add_order(make_order(1, Side.ASK, 101.0, 5, 0))  # worse ask
lob.add_order(make_order(2, Side.ASK, 100.0, 5, 1))  # better ask
trades = lob.add_order(make_order(3, Side.BID, 102.0, 5, 2))

test("fills at best ask price first (100, not 101)",
     all(t.price == 100.0 for t in trades))
test("correct fill quantity",  sum(t.quantity for t in trades) == 5)
test("worse ask still resting", lob.best_ask() == 101.0)


# ── 4. FIFO within same price level ────────────────────────────────────────
section("FIFO within price level")

lob = fresh_book()
lob.add_order(make_order(10, Side.ASK, 100.0, 3, 0))  # earlier (ts=0)
lob.add_order(make_order(11, Side.ASK, 100.0, 3, 1))  # later   (ts=1)

# Fill exactly 3 -- should consume order 10 entirely
trades = lob.add_order(make_order(12, Side.BID, 100.0, 3, 2))
filled_ids = [t.sell_order_id for t in trades]

test("earlier order filled first",   all(oid == 10 for oid in filled_ids))
test("later order untouched",        lob.ask_depth()[0][1] == 3)


# ── 5. Full book consumption ────────────────────────────────────────────────
section("Full book consumption")

lob = fresh_book()
for i, p in enumerate([101.0, 102.0, 103.0]):
    lob.add_order(make_order(i, Side.ASK, p, 2, i))

# Market buy for 6 -- should consume all three levels
mo = make_order(99, Side.BID, 0, 6, 10, OrderType.MARKET)
trades = lob.add_order(mo)

test("all 6 units filled",      sum(t.quantity for t in trades) == 6)
test("prices in ascending order",
     [t.price for t in trades] == [101.0, 102.0, 103.0])
test("book empty after full consumption", lob.best_ask() is None)


# ── 6. Cancellation ────────────────────────────────────────────────────────
section("Order cancellation")

lob = fresh_book()
lob.add_order(make_order(1, Side.BID, 99.0, 10, 0))
lob.add_order(make_order(2, Side.BID, 98.0, 10, 1))

test("cancel returns True for active order",  lob.cancel_order(1))
test("best_bid updated after cancel",         lob.best_bid() == 98.0)
test("cancel returns False for unknown id",   not lob.cancel_order(999))
test("cancel returns False for already-cancelled", not lob.cancel_order(1))


# ── 7. Partial fills ───────────────────────────────────────────────────────
section("Partial fills and quantity conservation")

lob = fresh_book()
lob.add_order(make_order(1, Side.ASK, 100.0, 10, 0))
trades = lob.add_order(make_order(2, Side.BID, 100.0, 3, 1))

test("partial fill quantity",           sum(t.quantity for t in trades) == 3)
test("resting order has 7 remaining",   lob.ask_depth()[0][1] == 7)


# ── 8. Invariant sweep ─────────────────────────────────────────────────────
section("Invariant sweep -- random order sequence")

rng = np.random.default_rng(42)
crossed_count = 0
N_TRIALS = 500

for trial in range(N_TRIALS):
    lob = fresh_book()
    oid = 0
    for _ in range(rng.integers(10, 40)):
        side  = Side.BID if rng.random() < 0.5 else Side.ASK
        price = round(float(rng.uniform(98, 102)), 1)
        qty   = int(rng.integers(1, 10))
        oid  += 1
        lob.add_order(make_order(oid, side, price, qty, oid))

        bb, ba = lob.best_bid(), lob.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            crossed_count += 1

test(f"no crossed book across {N_TRIALS} random sequences",
     crossed_count == 0, f"{crossed_count} violations")

# trade price conservation: buy_order_id and sell_order_id must differ
lob2 = fresh_book()
oid  = 0
for _ in range(200):
    side  = Side.BID if rng.random() < 0.5 else Side.ASK
    price = round(float(rng.uniform(99, 101)), 1)
    qty   = 1
    oid  += 1
    lob2.add_order(make_order(oid, side, price, qty, oid))

self_trade = any(t.buy_order_id == t.sell_order_id for t in lob2.trade_log)
test("no self-trades in random sequence", not self_trade)


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
passed = sum(_results)
total  = len(_results)
col    = "\033[92m" if passed == total else "\033[91m"
print(f"  {col}{passed}/{total} tests passed\033[0m")
print(f"{'='*60}\n")

import sys
if passed < total:
    sys.exit(1)
