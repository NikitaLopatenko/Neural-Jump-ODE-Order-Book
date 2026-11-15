import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import JumpODE


class CausalTests(unittest.TestCase):
    def test_future_messages_do_not_change_the_forecast(self):
        torch.manual_seed(0)
        model = JumpODE(n_features=11, hidden=16)
        with torch.no_grad():
            model.jump[-1].weight.normal_(0, 0.3)
            model.drift[-1].weight.normal_(0, 0.3)
            model.head.weight.normal_(0, 0.3)
        features = torch.randn(20, 11)
        dt = torch.rand(20) * 0.04
        dt[0] = 0
        original, original_rate = _logits(model, features, dt)

        changed_event = features.clone()
        changed_event[10:] += 3
        after_event, after_rate = _logits(model, changed_event, dt)
        self.assertTrue(torch.allclose(original[:10], after_event[:10], atol=1e-6))
        self.assertTrue(torch.allclose(original_rate[:10], after_rate[:10], atol=1e-6))
        self.assertFalse(torch.allclose(original[10:], after_event[10:], atol=1e-5))

        changed_gap = dt.clone()
        changed_gap[10:] += 1.5
        after_gap, after_gap_rate = _logits(model, features, changed_gap)
        self.assertTrue(torch.allclose(original[:10], after_gap[:10], atol=1e-6))
        self.assertTrue(torch.allclose(original_rate[:10], after_gap_rate[:10], atol=1e-6))

    def test_a_long_gap_drops_the_state(self):
        model = JumpODE(hidden=8)
        state = torch.ones(8)
        reset = model.advance(state, 5.0)
        self.assertTrue(torch.equal(reset, torch.zeros(8)))
        kept = model.advance(state, 0.01)
        self.assertFalse(torch.equal(kept, torch.zeros(8)))

    def test_flow_does_not_overwrite_its_input(self):
        model = JumpODE(hidden=8)
        state = torch.randn(8)
        snapshot = state.clone()
        model.flow(state, 0.25, 4)
        self.assertTrue(torch.equal(state, snapshot))

    def test_one_step_reaches_the_jump_and_the_drift(self):
        torch.manual_seed(0)
        model = JumpODE(n_features=11, hidden=8)
        features = torch.randn(16, 11)
        dt = torch.full((16,), 0.02)
        target = torch.randint(0, 3, (16, len(model.horizons)))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        jump_before = model.jump[-1].weight.detach().clone()
        drift_before = model.drift[-1].weight.detach().clone()
        logits, _, _ = model.forward_chunk(model.initial_state(features), features, dt)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))
        loss.backward()
        optimizer.step()
        self.assertGreater(float((model.jump[-1].weight - jump_before).detach().abs().sum()), 0.0)
        self.assertGreater(float((model.drift[-1].weight - drift_before).detach().abs().sum()), 0.0)


def _logits(model, features, dt):
    model.eval()
    with torch.no_grad():
        logits, intensity, _ = model.forward_chunk(model.initial_state(features), features, dt)
    return logits, intensity


if __name__ == "__main__":
    unittest.main()
