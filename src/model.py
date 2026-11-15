"""Neural jump-ODE.

Between events the latent state follows

    dh/dt = tanh(f_theta(h)) / tau

with tau = 50 ms, integrated by a fixed number of Euler steps. At an event the
state jumps by a bounded network of the pre-jump state and that event's
features. The forecast then flows for a fixed horizon with no further events.
A later message is not an input to an earlier forecast.
"""

from __future__ import annotations

import torch
from torch import nn

from src.labels import HORIZONS
from src.lobster import N_FEATURES


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


class JumpODE(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = 32,
        horizons: tuple[float, ...] = HORIZONS,
        tau: float = 0.05,
        reset_gap: float = 1.0,
        interval_steps: int = 1,
        horizon_steps: int = 4,
        predict_flow: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.horizons = tuple(horizons)
        self.tau = tau
        self.reset_gap = reset_gap
        self.interval_steps = interval_steps
        self.horizon_steps = horizon_steps
        self.predict_flow = predict_flow

        self.drift = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.jump = nn.Sequential(
            nn.Linear(hidden + n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 3)
        # Five LOBSTER event types. softplus makes a Hawkes intensity.
        self.intensity = nn.Linear(hidden, 5)
        # A zero vector field at the start. Gradients still pass through tanh'(0) = 1.
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)
        nn.init.zeros_(self.jump[-1].weight)
        nn.init.zeros_(self.jump[-1].bias)
        nn.init.zeros_(self.intensity.weight)
        nn.init.zeros_(self.intensity.bias)

    def initial_state(self, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros(self.hidden)

    def flow(self, state: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
        step = dt / float(steps)
        scale = step / self.tau
        for _ in range(steps):
            state = state + torch.tanh(self.drift(state)) * scale
        return state

    def advance(self, state: torch.Tensor, dt) -> torch.Tensor:
        gap = _as_float(dt)
        if gap > self.reset_gap:
            # A halt-sized hole is not a one-second integral of the last event.
            return state.new_zeros(state.shape)
        if gap <= 0.0:
            return state
        return self.flow(state, gap, self.interval_steps)

    def rates(self, state: torch.Tensor) -> torch.Tensor:
        """Hawkes intensity of the five event types, read from the state after this event."""
        return torch.nn.functional.softplus(self.intensity(state)) + 1e-4

    def forward_chunk(
        self,
        state: torch.Tensor,
        features: torch.Tensor,
        dts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {features.shape[-1]}")
        logits = []
        intensities = []
        for i in range(features.shape[0]):
            state = self.advance(state, dts[i])
            # Jump uses this event only. The forecast below does not read features[i+1:].
            delta = torch.tanh(self.jump(torch.cat([state, features[i]], dim=0)))
            state = state + delta
            intensities.append(self.rates(state))
            horizon_logits = []
            for horizon in self.horizons:
                if self.predict_flow:
                    future = self.flow(state, horizon, self.horizon_steps)
                else:
                    future = state
                horizon_logits.append(self.head(future))
            logits.append(torch.stack(horizon_logits, dim=0))
        return torch.stack(logits, dim=0), torch.stack(intensities, dim=0), state
