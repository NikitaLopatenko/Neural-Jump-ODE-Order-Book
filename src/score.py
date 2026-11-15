"""Class scores. Execution cost lives in latency_simulator."""

from __future__ import annotations

import numpy as np

from src.labels import DOWN, FLAT, UP
from src.latency_simulator import FEE_BPS, cumulative_pnl, information_ratio, net_bps, window_mids

IMBALANCE_THRESHOLD = 0.05

__all__ = [
    "FEE_BPS",
    "cumulative_pnl",
    "information_ratio",
    "net_bps",
    "window_mids",
]


def class_to_direction(pred_class: np.ndarray) -> np.ndarray:
    table = np.array([-1.0, 0.0, 1.0])
    return table[pred_class]


def direction_to_class(direction: np.ndarray) -> np.ndarray:
    return (np.rint(direction).astype(np.int64) + 1)


def imbalance_direction(imbalance: np.ndarray, threshold: float = IMBALANCE_THRESHOLD) -> np.ndarray:
    direction = np.zeros(len(imbalance), dtype=np.float64)
    direction[imbalance > threshold] = 1.0
    direction[imbalance < -threshold] = -1.0
    return direction


def summarize(pred_class: np.ndarray, label: np.ndarray, gross: np.ndarray, net: np.ndarray) -> dict:
    valid = label != -100
    pred = pred_class[valid]
    lab = label[valid]
    gross_v = gross[valid]
    net_v = net[valid]
    trades = pred != FLAT
    return {
        "n": int(valid.sum()),
        "accuracy": _mean(pred == lab),
        "flat_base_rate": _mean(lab == FLAT),
        "up_rate": _mean(lab == UP),
        "down_rate": _mean(lab == DOWN),
        "trade_rate": _mean(trades),
        "mean_net_bps_per_event": _mean(net_v),
        "mean_net_bps_per_trade": _mean(net_v[trades]),
        "mean_gross_bps_per_trade": _mean(gross_v[trades]),
    }


def binary_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUC of a ranking. Ties take the midpoint. NaN when a class is missing."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.shape[0], dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < score.shape[0]:
        stop = start + 1
        while stop < score.shape[0] and sorted_score[stop] == sorted_score[start]:
            stop += 1
        # ranks are 1-based. The tied block shares its average rank.
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _mean(values) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))
