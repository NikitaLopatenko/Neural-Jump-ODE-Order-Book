"""LOBSTER message and order-book files, turned into one row per event.

Column layout, from the LOBSTER output notes:

message: time, type, order id, size, price, direction
order book, repeated per level: ask price, ask size, bid price, bid size

Row k of the book is the book after message k. Prices are dollars times 10,000.
Time is seconds after midnight. Types 1 through 5 are the order events.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PRICE_SCALE = 10_000.0
N_FEATURES = 11
# Columns after the five-way event type. Those five stay as indicators.
ZSCORE_COLUMNS = np.arange(5, N_FEATURES)

FEATURE_NAMES = (
    "new_limit",
    "partial_cancel",
    "full_delete",
    "visible_execution",
    "hidden_execution",
    "direction",
    "log_size",
    "price_vs_mid_in_spreads",
    "spread_over_mid",
    "imbalance",
    "log_touch_size",
)


class ColumnScaler:
    """Mean and variance from the training clock only."""

    def __init__(self, columns: np.ndarray = ZSCORE_COLUMNS):
        self.columns = np.asarray(columns)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> None:
        part = features[:, self.columns]
        self.mean = part.mean(axis=0)
        self.std = np.maximum(part.std(axis=0), 1e-6)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit the scaler on the morning rows before transforming")
        scaled = np.array(features, dtype=np.float32, copy=True)
        scaled[:, self.columns] = (scaled[:, self.columns] - self.mean) / self.std
        return scaled


def find_pair(data_dir: Path) -> tuple[Path, Path]:
    messages = sorted(data_dir.glob("*message*.csv"))
    if not messages:
        raise FileNotFoundError(f"no LOBSTER message csv in {data_dir}")
    for message in messages:
        book = Path(str(message).replace("message", "orderbook"))
        if book.exists():
            return message, book
    raise FileNotFoundError(f"no order-book csv matched to {messages[0].name}")


def load_frame(message_path: Path, book_path: Path) -> dict[str, np.ndarray]:
    message = np.loadtxt(message_path, delimiter=",", usecols=(0, 1, 3, 4, 5))
    book = np.loadtxt(book_path, delimiter=",", usecols=(0, 1, 2, 3))
    if message.ndim == 1:
        message = message.reshape(1, -1)
        book = book.reshape(1, -1)
    if len(message) != len(book):
        raise ValueError(f"row mismatch: {len(message)} messages, {len(book)} book rows")
    times = message[:, 0]
    if np.any(np.diff(times) < -1e-6):
        raise ValueError("message timestamps go backwards")

    kind = np.rint(message[:, 1]).astype(np.int64)
    keep = (kind >= 1) & (kind <= 5)
    ask = book[:, 0] / PRICE_SCALE
    ask_size = book[:, 1]
    bid = book[:, 2] / PRICE_SCALE
    bid_size = book[:, 3]
    keep &= (ask > 0) & (bid > 0) & np.isfinite(ask) & np.isfinite(bid)

    kind = kind[keep]
    size = message[keep, 2]
    price = message[keep, 3] / PRICE_SCALE
    direction = message[keep, 4]
    times = times[keep]
    ask, ask_size, bid, bid_size = ask[keep], ask_size[keep], bid[keep], bid_size[keep]

    spread = ask - bid
    mid = 0.5 * (ask + bid)
    touch = np.maximum(bid_size + ask_size, 0.0)
    micro = (ask * bid_size + bid * ask_size) / np.maximum(touch, 1.0)
    imbalance = (bid_size - ask_size) / np.maximum(touch, 1.0)

    one_hot = np.zeros((len(times), 5), dtype=np.float64)
    one_hot[np.arange(len(times)), kind - 1] = 1.0
    denom = np.maximum(spread, 0.01)
    relative = np.clip((price - mid) / denom, -5.0, 5.0)
    relative[(price <= 0) | ~np.isfinite(relative)] = 0.0
    features = np.column_stack(
        [
            one_hot,
            direction,
            np.log1p(np.maximum(size, 0.0)),
            relative,
            spread / np.maximum(mid, 0.01),
            imbalance,
            np.log1p(touch),
        ]
    )
    dt = np.zeros(len(times), dtype=np.float64)
    if len(times) > 1:
        dt[1:] = np.diff(times)
        dt[dt < 0] = 0.0

    return {
        "features": features,
        "dt": dt,
        "times": times,
        "mid": mid,
        "micro": micro,
        "spread": spread,
        "imbalance": imbalance,
        "n_raw": np.array(len(message)),
        "n_kept": np.array(len(times)),
    }
