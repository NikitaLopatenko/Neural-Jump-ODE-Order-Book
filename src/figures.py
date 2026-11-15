"""Two figures for the afternoon test: cancel intensity, and taker equity.

The equity curve is the running sum of basis points after the spread and the
taker fee. It is there to show the fees the model does not pay, not a profit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.download import ensure_sample
from src.labels import HORIZONS, make_labels
from src.latency_simulator import cumulative_pnl, net_bps, window_mids
from src.lobster import ColumnScaler, load_frame
from src.model import JumpODE
from src.score import class_to_direction, imbalance_direction
from src.train import NOON, cap_range, predict_outputs, segment_tensors, split_ranges

ROOT = Path(__file__).resolve().parents[1]
DELAY = 0.001
HORIZON = HORIZONS[0]


def main() -> None:
    data_dir = ROOT / "data"
    runs = ROOT / "runs"
    message, book = ensure_sample(data_dir)
    frame = load_frame(message, book)
    labels = make_labels(frame["times"], frame["micro"], frame["spread"])
    train_r, _, test_r = split_ranges(frame["times"])
    scaler = ColumnScaler()
    scaler.fit(frame["features"][train_r[0] : train_r[1]])
    test_s, test_e = cap_range(*test_r, 300_000, tail=False)
    warm = max(0, test_s - 256)
    features, dt, _, warmup = segment_tensors(frame, scaler, labels, warm, test_s, test_e)

    model = JumpODE()
    model.load_state_dict(torch.load(runs / "jump_ode.pt", map_location="cpu", weights_only=True))
    pred, _, cancel = predict_outputs(model, features, dt, warmup, 128)

    times = frame["times"][test_s:test_e]
    raw = frame["features"][test_s:test_e]
    is_cancel = (raw[:, 1] + raw[:, 2]) > 0.5
    _intensity_figure(runs / "hawkes_cancel.png", times, cancel, is_cancel)

    mid_entry, mid_exit, spread_entry, alive = window_mids(
        frame["times"], frame["mid"], frame["spread"], test_s, test_e, DELAY, HORIZON
    )
    _, model_net = net_bps(class_to_direction(pred[:, 0]), mid_entry, mid_exit, spread_entry)
    imbalance = imbalance_direction(frame["imbalance"][test_s:test_e])
    _, base_net = net_bps(imbalance, mid_entry, mid_exit, spread_entry)
    model_net = np.where(alive, model_net, 0.0)
    base_net = np.where(alive, base_net, 0.0)
    _equity_figure(runs / "pnl_drawdown.png", times, model_net, base_net)
    print(f"wrote {runs / 'hawkes_cancel.png'}", flush=True)
    print(f"wrote {runs / 'pnl_drawdown.png'}", flush=True)
    print(
        f"afternoon cumulative taker pnl, 50 ms, delay 1 ms: "
        f"model {float(np.sum(model_net)):.1f} bp, imbalance {float(np.sum(base_net)):.1f} bp",
        flush=True,
    )


def _intensity_figure(path: Path, times: np.ndarray, cancel: np.ndarray, is_cancel: np.ndarray) -> None:
    second = np.floor(times).astype(np.int64)
    # The busiest second of cancels, plus a short margin, so the line is readable.
    counts = np.bincount(second - second[0], weights=is_cancel.astype(np.float64))
    center = int(second[0] + np.argmax(counts))
    keep = (times >= center - 0.4) & (times <= center + 1.2)
    clock = times[keep] - center
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(clock, cancel[keep], color="#1f4e79", linewidth=1.2, label="cancel intensity")
    marks = clock[is_cancel[keep]]
    ax.vlines(marks, 0, np.nanmax(cancel[keep]) * 0.15, color="#b85c38", linewidth=0.6, label="real cancel")
    ax.set_xlabel("seconds from the busiest cancel second")
    ax.set_ylabel("intensity")
    ax.set_title("Predicted cancel intensity and the cancels that actually printed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _equity_figure(path: Path, times: np.ndarray, model_net: np.ndarray, base_net: np.ndarray) -> None:
    clock = (times - NOON) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    axes[0].plot(clock, cumulative_pnl(base_net), color="#b85c38", linewidth=1.0, label="imbalance")
    axes[0].plot(clock, cumulative_pnl(model_net), color="#1f4e79", linewidth=1.0, label="jump-ODE")
    axes[0].axhline(0, color="#888888", linewidth=0.6)
    axes[0].set_ylabel("basis points")
    axes[0].set_title("Same scale: the imbalance rule pays the spread on most ticks")
    axes[0].legend(frameon=False)
    axes[1].plot(clock, cumulative_pnl(model_net), color="#1f4e79", linewidth=1.0)
    axes[1].axhline(0, color="#888888", linewidth=0.6)
    axes[1].set_ylabel("basis points")
    axes[1].set_xlabel("hours after noon")
    axes[1].set_title("Jump-ODE alone. It still loses, by much less, because it rarely trades")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
