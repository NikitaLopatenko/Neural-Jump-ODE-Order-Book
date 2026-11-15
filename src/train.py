"""Train the jump-ODE on one LOBSTER sample day and score it with the spread."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.download import ensure_sample
from src.labels import HORIZONS, make_labels
from src.lobster import ColumnScaler, load_frame
from src.model import JumpODE
from src.score import (
    FEE_BPS,
    binary_auc,
    class_to_direction,
    direction_to_class,
    imbalance_direction,
    information_ratio,
    net_bps,
    summarize,
    window_mids,
)

ROOT = Path(__file__).resolve().parents[1]
NOON = 12 * 3600
HALF_PAST_ONE = int(13.5 * 3600)


def split_ranges(times: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    n = len(times)
    if times[0] < NOON < times[-1] and times[-1] > HALF_PAST_ONE:
        train_end = int(np.searchsorted(times, NOON))
        val_end = int(np.searchsorted(times, HALF_PAST_ONE))
        return (0, train_end), (train_end, val_end), (val_end, n)
    first, second = int(n * 0.6), int(n * 0.8)
    return (0, first), (first, second), (second, n)


def cap_range(start: int, stop: int, max_n: int, tail: bool) -> tuple[int, int]:
    if stop - start > max_n:
        if tail:
            start = stop - max_n
        else:
            stop = start + max_n
    return start, stop


def class_weights(labels: np.ndarray) -> torch.Tensor:
    valid = labels[labels != -100]
    counts = np.bincount(valid, minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_segment(
    model: JumpODE,
    features: torch.Tensor,
    dt: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    warmup: int,
    chunk: int,
) -> float:
    train = optimizer is not None
    model.train(train)
    state = model.initial_state(features)
    if warmup:
        with torch.no_grad():
            _, _, state = model.forward_chunk(state, features[:warmup], dt[:warmup])
            state = state.detach()
    total = 0.0
    count = 0
    cursor = warmup
    next_report = warmup + 20_000
    while cursor < features.shape[0]:
        end = min(features.shape[0], cursor + chunk)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits, rates, state = model.forward_chunk(state, features[cursor:end], dt[cursor:end])
        target = labels[cursor:end].reshape(-1)
        valid = int((target != -100).sum())
        if valid == 0:
            state = state.detach()
            cursor = end
            continue
        loss = nn.functional.cross_entropy(logits.reshape(-1, 3), target, weight=weights, ignore_index=-100)
        loss = loss + 0.05 * _hawkes_nll(rates, features[cursor:end], dt[cursor:end], model.reset_gap)
        if train and torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if torch.isfinite(loss):
            total += float(loss.detach()) * valid
            count += valid
        state = state.detach()
        cursor = end
        if train and cursor >= next_report:
            print(f"    {cursor - warmup} events", flush=True)
            next_report += 20_000
    return total / max(count, 1)


def _hawkes_nll(rates: torch.Tensor, features: torch.Tensor, dt: torch.Tensor, reset_gap: float) -> torch.Tensor:
    """Piecewise-constant Hawkes likelihood of the next event. The next row is a target, not an input."""
    if rates.shape[0] < 2:
        return rates.sum() * 0.0
    gap = dt[1:]
    kind = features[1:, :5].argmax(dim=-1)
    now = rates[:-1]
    chosen = now[torch.arange(kind.shape[0]), kind]
    total = now.sum(dim=-1)
    usable = (gap > 0) & (gap <= reset_gap)
    if int(usable.sum()) == 0:
        return rates.sum() * 0.0
    nll = -torch.log(chosen) + total * gap
    return nll[usable].mean()


@torch.no_grad()
def predict_outputs(
    model: JumpODE,
    features: torch.Tensor,
    dt: torch.Tensor,
    warmup: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class, class probability, and cancellation intensity (partial cancel + full delete)."""
    model.eval()
    state = model.initial_state(features)
    if warmup:
        _, _, state = model.forward_chunk(state, features[:warmup], dt[:warmup])
        state = state.detach()
    classes = []
    probs = []
    cancel = []
    cursor = warmup
    while cursor < features.shape[0]:
        end = min(features.shape[0], cursor + chunk)
        logits, rates, state = model.forward_chunk(state, features[cursor:end], dt[cursor:end])
        classes.append(logits.argmax(dim=-1).cpu())
        probs.append(torch.softmax(logits, dim=-1).cpu())
        cancel.append((rates[:, 1] + rates[:, 2]).cpu())
        state = state.detach()
        cursor = end
    return (
        torch.cat(classes, dim=0).numpy(),
        torch.cat(probs, dim=0).numpy(),
        torch.cat(cancel, dim=0).numpy(),
    )


def score_predictions(
    pred: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    start: int,
    fee_bps: float,
    delay: float,
) -> dict:
    report = {}
    for column, horizon in enumerate(HORIZONS):
        mid_entry, mid_exit, spread_entry, alive = window_mids(
            times, mid, spread, start, start + len(pred), delay, horizon
        )
        direction = class_to_direction(pred[:, column])
        gross, net = net_bps(direction, mid_entry, mid_exit, spread_entry, fee_bps)
        label = labels[:, column].copy()
        label[~alive] = -100
        report[f"{int(horizon * 1000)}ms"] = summarize(pred[:, column], label, gross, net)
    return report


def directional_auc(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Up versus down, among moves that are not flat. Score is P(up) / (P(up)+P(down))."""
    report = {}
    for column, horizon in enumerate(HORIZONS):
        label = labels[:, column]
        moved = (label == 0) | (label == 2)
        up = probs[:, column, 2]
        down = probs[:, column, 0]
        score = up / np.maximum(up + down, 1e-8)
        report[f"{int(horizon * 1000)}ms"] = {
            "auc": binary_auc(score[moved], label[moved] == 2),
            "n_moves": int(moved.sum()),
        }
    return report


def score_imbalance(
    imbalance: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    start: int,
    fee_bps: float,
) -> dict:
    direction = imbalance_direction(imbalance)
    pred = direction_to_class(direction)
    report = {}
    for column, horizon in enumerate(HORIZONS):
        mid_entry, mid_exit, spread_entry, alive = window_mids(
            times, mid, spread, start, start + len(pred), 0.0, horizon
        )
        gross, net = net_bps(direction, mid_entry, mid_exit, spread_entry, fee_bps)
        label = labels[:, column].copy()
        label[~alive] = -100
        report[f"{int(horizon * 1000)}ms"] = summarize(pred, label, gross, net)
    return report


def segment_tensors(frame, scaler, labels, warm: int, start: int, stop: int):
    features = torch.tensor(scaler.transform(frame["features"][warm:stop]))
    dt = torch.tensor(frame["dt"][warm:stop], dtype=torch.float32)
    target = torch.tensor(labels[warm:stop], dtype=torch.long)
    return features, dt, target, start - warm


def fit_model(name: str, predict_flow: bool, tensors, labels_for_weights, epochs: int, chunk: int, lr: float):
    features, dt, target, warmup = tensors
    model = JumpODE(predict_flow=predict_flow)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = class_weights(labels_for_weights)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = run_segment(model, features, dt, target, weights, optimizer, warmup, chunk)
        history.append(loss)
        print(f"  {name} epoch {epoch}: train loss {loss:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    scored = features.shape[0] - warmup
    return model, {
        "train_loss": history,
        "seconds": elapsed,
        "events_scored": int(scored),
        "ms_per_event_train": elapsed / max(scored, 1) / epochs * 1000.0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "predict_flow": predict_flow,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the limit-order-book jump-ODE")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=300_000)
    parser.add_argument("--max-val", type=int, default=300_000)
    parser.add_argument("--max-test", type=int, default=300_000)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fee-bps", type=float, default=FEE_BPS)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    data_dir = ROOT / "data"
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)

    message, book = ensure_sample(data_dir)
    print(f"loading {message.name}", flush=True)
    frame = load_frame(message, book)
    labels = make_labels(frame["times"], frame["micro"], frame["spread"])
    train_r, val_r, test_r = split_ranges(frame["times"])
    scaler = ColumnScaler()
    scaler.fit(frame["features"][train_r[0] : train_r[1]])

    train_s, train_e = cap_range(*train_r, args.max_train, tail=True)
    val_s, val_e = cap_range(*val_r, args.max_val, tail=False)
    test_s, test_e = cap_range(*test_r, args.max_test, tail=False)
    train_w = max(train_r[0], train_s - args.warmup)
    val_w = max(0, val_s - args.warmup)
    test_w = max(0, test_s - args.warmup)

    print(
        f"kept {int(frame['n_kept'])} of {int(frame['n_raw'])} events"
        f"  mid {np.median(frame['mid']):.2f}"
        f"  spread {np.median(frame['spread']):.4f}",
        flush=True,
    )
    print(
        f"train {train_s}:{train_e}  val {val_s}:{val_e}  test {test_s}:{test_e}",
        flush=True,
    )

    train_tensors = segment_tensors(frame, scaler, labels, train_w, train_s, train_e)
    model, train_info = fit_model(
        "jump-ode",
        True,
        train_tensors,
        labels[train_s:train_e],
        args.epochs,
        args.chunk,
        args.lr,
    )
    torch.save(model.state_dict(), runs / "jump_ode.pt")

    ablation = None
    if not args.no_ablation:
        ablation_model, ablation = fit_model(
            "jump-only",
            False,
            train_tensors,
            labels[train_s:train_e],
            args.epochs,
            args.chunk,
            args.lr,
        )
        torch.save(ablation_model.state_dict(), runs / "jump_only.pt")
    else:
        ablation_model = None

    def pack(split_name, warm, start, stop, fitted, fitted_info):
        features, dt, _, warmup = segment_tensors(frame, scaler, labels, warm, start, stop)
        started = time.perf_counter()
        pred, probs, cancel = predict_outputs(fitted, features, dt, warmup, args.chunk)
        eval_seconds = time.perf_counter() - started
        label_slice = labels[start:stop]
        auc = directional_auc(probs, label_slice)
        # Taker: cross the spread and pay the fee. Entry is delayed by 1 µs and by 1 ms.
        delayed = {}
        ratios = {}
        for delay, name in ((1e-6, "1us"), (1e-3, "1ms"), (5e-3, "5ms")):
            delayed[name] = score_predictions(
                pred, label_slice, frame["times"], frame["mid"], frame["spread"], start, args.fee_bps, delay
            )
            ratios[name] = {}
            for column, horizon in enumerate(HORIZONS):
                mid_entry, mid_exit, spread_entry, alive = window_mids(
                    frame["times"], frame["mid"], frame["spread"], start, stop, delay, horizon
                )
                _, net = net_bps(
                    class_to_direction(pred[:, column]), mid_entry, mid_exit, spread_entry, args.fee_bps
                )
                net = np.where(alive, net, 0.0)
                ratios[name][f"{int(horizon * 1000)}ms"] = information_ratio(frame["times"][start:stop], net)
        next_type = features[warmup + 1 :, :5].argmax(dim=1).numpy() if features.shape[0] > warmup + 1 else np.array([])
        next_is_cancel = np.isin(next_type, (1, 2)) if next_type.size else np.array([])
        cancel_now = cancel[:-1] if cancel.size else cancel
        part = {
            "model": score_predictions(
                pred, label_slice, frame["times"], frame["mid"], frame["spread"], start, args.fee_bps, 0.0
            ),
            "model_entry_delayed_5ms": delayed["5ms"],
            "imbalance": score_imbalance(
                frame["imbalance"][start:stop],
                label_slice,
                frame["times"],
                frame["mid"],
                frame["spread"],
                start,
                args.fee_bps,
            ),
            "directional_auc": auc,
            "taker_information_ratio": ratios,
            "hawkes_cancel_intensity_mean": float(np.mean(cancel)) if cancel.size else float("nan"),
            "hawkes_cancel_intensity_when_next_is_cancel": float(np.mean(cancel_now[next_is_cancel]))
            if next_is_cancel.size and int(next_is_cancel.sum())
            else float("nan"),
            "hawkes_cancel_intensity_when_next_is_not": float(np.mean(cancel_now[~next_is_cancel]))
            if next_is_cancel.size and int((~next_is_cancel).sum())
            else float("nan"),
            "eval_ms_per_event": eval_seconds / max(len(pred), 1) * 1000.0,
            "n_events": int(len(pred)),
            "train": fitted_info,
        }
        _print_split(split_name, part)
        return part

    report = {
        "data": "LOBSTER AMZN 2012-06-21 level 10 sample",
        "caveat": "One free sample day. Not a trading result.",
        "fee_bps": args.fee_bps,
        "cost": "Taker: full quoted spread plus the fee, charged only when the model trades. Entry can be delayed by 1 microsecond, 1 ms, or 5 ms. Score is the mid-price move. Information ratio is mean/std of that pnl summed into 1-second bins.",
        "horizons_ms": [int(h * 1000) for h in HORIZONS],
        "split_seconds_after_midnight": {
            "train": [float(frame["times"][train_s]), float(frame["times"][train_e - 1])],
            "test": [float(frame["times"][test_s]), float(frame["times"][test_e - 1])],
        },
        "median_mid": float(np.median(frame["mid"])),
        "median_spread": float(np.median(frame["spread"])),
        "test_median_spread": float(np.median(frame["spread"][test_s:test_e])),
        "jump_ode": {
            "val": pack("val jump-ode", val_w, val_s, val_e, model, train_info),
            "test": pack("test jump-ode", test_w, test_s, test_e, model, train_info),
        },
    }
    if ablation_model is not None:
        report["jump_only"] = {
            "val": pack("val jump-only", val_w, val_s, val_e, ablation_model, ablation),
            "test": pack("test jump-only", test_w, test_s, test_e, ablation_model, ablation),
        }
    path = runs / "metrics.json"
    path.write_text(json.dumps(_json_ready(report), indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _print_split(name: str, part: dict) -> None:
    print(f"{name}  eval {part['eval_ms_per_event']:.2f} ms/event", flush=True)
    header = f"  {'horizon':<8} {'acc':>7} {'flat':>7} {'trade':>7} {'net/evt':>8} {'net/trd':>8} {'imb net':>8} {'auc':>7} {'IR 1ms':>8}"
    print(header, flush=True)
    for horizon, row in part["model"].items():
        base = part["imbalance"][horizon]
        auc = part["directional_auc"][horizon]["auc"]
        ratio = part["taker_information_ratio"]["1ms"][horizon]["information_ratio"]
        print(
            f"  {horizon:<8} {row['accuracy']:7.3f} {row['flat_base_rate']:7.3f}"
            f" {row['trade_rate']:7.3f} {row['mean_net_bps_per_event']:8.3f}"
            f" {row['mean_net_bps_per_trade']:8.3f} {base['mean_net_bps_per_event']:8.3f}"
            f" {auc:7.3f} {ratio:8.3f}",
            flush=True,
        )
    print(
        f"  cancel intensity {part['hawkes_cancel_intensity_mean']:.3f}"
        f"  next-cancel {part['hawkes_cancel_intensity_when_next_is_cancel']:.3f}"
        f"  other {part['hawkes_cancel_intensity_when_next_is_not']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
