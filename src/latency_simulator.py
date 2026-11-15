"""Taker execution after a delay.

The fill is the quote in force once the delay has elapsed. Crossing that quote
pays the full spread plus the taker fee. A zero direction is an abstention and
pays nothing. This is not a maker queue: the LOBSTER row does not say where our
order would stand in line, so a resting fill is not invented.
"""

from __future__ import annotations

import numpy as np

from src.labels import future_index

# One basis point is 0.0001 of the mid. 0.5 bp sits on top of the spread.
FEE_BPS = 0.5


def net_bps(
    direction: np.ndarray,
    mid_entry: np.ndarray,
    mid_exit: np.ndarray,
    spread_entry: np.ndarray,
    fee_bps: float = FEE_BPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Gross mid move, and the same move after the full spread plus the taker fee."""
    mid = np.maximum(mid_entry, 1e-8)
    gross = direction * (mid_exit - mid_entry) / mid * 1e4
    cost = spread_entry / mid * 1e4 + fee_bps
    net = np.where(direction == 0.0, 0.0, gross - cost)
    gross = np.where(direction == 0.0, 0.0, gross)
    return gross, net


def window_mids(
    times: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    start: int,
    stop: int,
    delay: float,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Entry and exit quotes for events in [start, stop), after an optional delay."""
    entry = future_index(times, delay)[start:stop]
    exit_ = future_index(times, delay + horizon)[start:stop]
    alive = (times + delay + horizon) <= (times[-1] + 1e-9)
    return mid[entry], mid[exit_], spread[entry], alive[start:stop]


def information_ratio(times: np.ndarray, net_bps_per_event: np.ndarray, seconds: float = 1.0) -> dict:
    """Mean over std of pnl summed into fixed clock bins. One day is one sample of this ratio."""
    times = np.asarray(times)
    pnl = np.asarray(net_bps_per_event, dtype=np.float64)
    if times.shape[0] != pnl.shape[0] or pnl.shape[0] == 0:
        return {"information_ratio": float("nan"), "n_bins": 0, "mean_bps": float("nan"), "std_bps": float("nan")}
    bins = np.floor(times / seconds).astype(np.int64)
    order = np.argsort(bins, kind="mergesort")
    bins = bins[order]
    pnl = pnl[order]
    cuts = np.flatnonzero(np.diff(bins)) + 1
    summed = np.add.reduceat(pnl, np.concatenate(([0], cuts)))
    mean = float(np.mean(summed))
    std = float(np.std(summed))
    ratio = mean / std if std > 0 else float("nan")
    return {
        "information_ratio": ratio,
        "n_bins": int(summed.shape[0]),
        "mean_bps": mean,
        "std_bps": std,
        "bin_seconds": seconds,
    }


def cumulative_pnl(net_bps_per_event: np.ndarray) -> np.ndarray:
    """Running sum of per-event pnl, in basis points."""
    return np.cumsum(np.asarray(net_bps_per_event, dtype=np.float64))
