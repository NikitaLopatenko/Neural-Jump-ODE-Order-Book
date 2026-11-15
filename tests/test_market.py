import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labels import FLAT, UP, make_labels
from src.lobster import N_FEATURES, ColumnScaler, load_frame
from src.score import binary_auc, class_to_direction, information_ratio, net_bps


class LabelTests(unittest.TestCase):
    def test_flat_when_the_window_has_no_new_print(self):
        times = np.array([0.0, 0.01, 0.20])
        micro = np.array([10.0, 10.05, 9.0])
        spread = np.array([0.02, 0.02, 0.02])
        labels = make_labels(times, micro, spread, horizons=(0.05,))
        self.assertEqual(int(labels[0, 0]), UP)
        self.assertEqual(int(labels[1, 0]), FLAT)
        # 0.20 + 0.05 runs off the end of the file.
        self.assertEqual(int(labels[2, 0]), -100)

    def test_later_book_row_does_not_change_earlier_features(self):
        first = _write_pair(self, last_ask="2239500")
        second = _write_pair(self, last_ask="2245000")
        early = load_frame(*first)["features"]
        shifted = load_frame(*second)["features"]
        np.testing.assert_allclose(early[0], shifted[0])
        self.assertFalse(np.allclose(early[1], shifted[1]))
        self.assertEqual(early.shape[1], N_FEATURES)

    def test_scaler_leaves_the_event_indicators_alone(self):
        features = np.zeros((4, N_FEATURES))
        features[:, 0] = 1.0
        features[:, 5] = np.array([-1.0, 1.0, 1.0, -1.0])
        scaler = ColumnScaler()
        scaler.fit(features)
        scaled = scaler.transform(features)
        np.testing.assert_allclose(scaled[:, 0], 1.0)

    def test_real_sample_prefix(self):
        root = Path(__file__).resolve().parents[1] / "data"
        message = root / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
        book = root / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
        if not message.exists():
            self.skipTest("sample not downloaded")
        with tempfile.TemporaryDirectory() as tmp:
            short_message = Path(tmp) / "message.csv"
            short_book = Path(tmp) / "orderbook.csv"
            short_message.write_text("".join(message.read_text().splitlines(True)[:30]))
            short_book.write_text("".join(book.read_text().splitlines(True)[:30]))
            frame = load_frame(short_message, short_book)
        self.assertGreater(frame["times"][0], 34000)
        self.assertLess(frame["times"][0], 35000)
        self.assertTrue(np.all(np.diff(frame["times"]) >= -1e-9))
        self.assertGreater(np.median(frame["mid"]), 100)
        self.assertLess(np.median(frame["mid"]), 400)


class FeeTests(unittest.TestCase):
    def test_a_one_tick_win_does_not_pay_a_wide_spread(self):
        direction = np.array([1.0])
        gross, net = net_bps(direction, np.array([220.0]), np.array([220.01]), np.array([0.05]), fee_bps=0.5)
        self.assertGreater(float(gross[0]), 0.0)
        self.assertLess(float(net[0]), 0.0)

    def test_auc_ranks_a_perfect_score_above_a_tied_one(self):
        perfect = binary_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True]))
        self.assertAlmostEqual(perfect, 1.0)
        tied = binary_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([False, False, True, True]))
        self.assertAlmostEqual(tied, 0.5)

    def test_information_ratio_uses_clock_bins(self):
        times = np.array([0.1, 0.2, 1.2, 1.4])
        pnl = np.array([1.0, 1.0, -1.0, -1.0])
        report = information_ratio(times, pnl, seconds=1.0)
        self.assertEqual(report["n_bins"], 2)
        self.assertAlmostEqual(report["information_ratio"], 0.0)

    def test_abstention_is_free(self):
        _, net = net_bps(
            class_to_direction(np.array([1])),
            np.array([220.0]),
            np.array([219.0]),
            np.array([0.05]),
        )
        self.assertEqual(float(net[0]), 0.0)


def _write_pair(case: unittest.TestCase, last_ask: str):
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    root = Path(tmp.name)
    message = root / "m_message.csv"
    book = root / "m_orderbook.csv"
    message.write_text(
        "\n".join(
            [
                "34200.000,1,10,100,2238100,1",
                "34200.010,4,10,40,2238200,-1",
                "34200.400,3,11,50,2237000,1",
            ]
        )
        + "\n"
    )
    book.write_text(
        "\n".join(
            [
                "2239500,100,2231800,100,2240000,50,2230000,40",
                f"{last_ask},100,2238100,21,2240000,50,2230000,40",
                "2239500,80,2238100,21,2240000,50,2230000,40",
            ]
        )
        + "\n"
    )
    return message, book


if __name__ == "__main__":
    unittest.main()
