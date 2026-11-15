"""Labels for the sign of the micro-price. The label may look ahead. The features may not."""

from __future__ import annotations

import numpy as np

# Seconds. The brief asks for 50 ms, and the band out to 250 ms.
HORIZONS = (0.05, 0.10, 0.25)

DOWN, FLAT, UP = 0, 1, 2


def future_index(times: np.ndarray, delay: float) -> np.ndarray:
    """Last event at or before each timestamp plus delay.

    If nothing new arrives inside the window, the index stays on the current
    event. The book in force is then the book we already have.
    """
    return np.searchsorted(times, times + delay, side="right") - 1


def covered(times: np.ndarray, delay: float, horizon: float) -> np.ndarray:
    """False when the file ends before the window does."""
    return (times + delay + horizon) <= (times[-1] + 1e-9)


def make_labels(
    times: np.ndarray,
    micro: np.ndarray,
    spread: np.ndarray,
    horizons: tuple[float, ...] = HORIZONS,
    tick: float = 0.01,
) -> np.ndarray:
    """Class of the micro-price move over each horizon.

    Flat unless the move clears one tick or half the spread, whichever is
    larger. A crossed or empty quote is ignore-index -100.
    """
    labels = np.empty((len(times), len(horizons)), dtype=np.int64)
    for column, horizon in enumerate(horizons):
        index = future_index(times, horizon)
        move = micro[index] - micro
        threshold = np.maximum(tick, 0.5 * np.maximum(spread, 0.0))
        label = np.full(len(times), FLAT, dtype=np.int64)
        label[move > threshold] = UP
        label[move < -threshold] = DOWN
        bad = (~np.isfinite(move)) | (spread <= 0) | (~covered(times, 0.0, horizon))
        label[bad] = -100
        labels[:, column] = label
    return labels
