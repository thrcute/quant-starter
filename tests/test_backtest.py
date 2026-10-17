"""Offline accounting regressions; no exchange client or trading loop is used."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import Config
from strategy import compute_signals


def candles(opens, closes=None, freq="h"):
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": 1.0,
        },
        index=pd.date_range("2025-01-01", periods=len(opens), freq=freq, tz="UTC", name="datetime"),
    )


def cash_and_units_equity(opens, close_signals, cost):
    """Independent execution ledger: buy with cash, hold units, sell for cash."""
    cash, units = 1.0, 0.0
    values = []
    for i, price in enumerate(opens):
        target = close_signals[i - 1] if i else 0
        if target and units == 0:
            units = (cash - cash * cost) / price
            cash = 0.0
        elif not target and units > 0:
            proceeds = units * price
            cash = proceeds - proceeds * cost
            units = 0.0
        values.append(cash + units * price)
    return values


class BacktestAccountingTests(unittest.TestCase):
    def setUp(self):
        # Prevent accidental network use if backtest imports/calls change later.
        network = patch("socket.socket", side_effect=AssertionError("tests must remain offline"))
        network.start()
        self.addCleanup(network.stop)
        self.cfg = Config(
            fee_rate=0.001, slippage_pct=0.0005,
            fast_ma=2, slow_ma=3, rsi_len=2, rsi_buy=55, rsi_sell=45,
        )

    def run_signals(self, opens, signals, cfg=None, freq="h"):
        df = candles(opens, freq=freq)
        sig = SimpleNamespace(long=pd.Series(signals, index=df.index, dtype=float))
        with patch("backtest.compute_signals", return_value=sig):
            result = run_backtest(df, cfg or self.cfg)
        return df, result

    def assert_ledger(self, result, opens, signals, cfg=None):
        cfg = cfg or self.cfg
        expected = cash_and_units_equity(opens, signals, cfg.fee_rate + cfg.slippage_pct)
        np.testing.assert_allclose(result.equity, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(result.stats["total_return_pct"], (expected[-1] - 1) * 100)

    def test_reported_entry_gap_regression(self):
        opens = [100, 100, 110, 121, 121]
        signals = [0, 1, 1, 0, 0]
        df, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        np.testing.assert_allclose(result.equity, [1, 1, 0.9985, 1.09835, 1.096702475])
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade.entry_time, df.index[2])
        self.assertEqual(trade.exit_time, df.index[4])
        self.assertEqual(trade.entry_px, 110)
        self.assertEqual(trade.exit_px, 121)
        self.assertAlmostEqual(trade.pnl_pct, 0.096702475)
        self.assertAlmostEqual(result.equity.iloc[-1], 1 + trade.pnl_pct)

    def test_exit_gap_is_earned_but_post_exit_move_is_not(self):
        for exit_px in (80, 125):
            with self.subTest(exit_px=exit_px):
                opens, signals = [100, 100, exit_px, 500], [1, 0, 0, 0]
                _, result = self.run_signals(opens, signals)
                self.assert_ledger(result, opens, signals)
                self.assertAlmostEqual(result.equity.iloc[2], exit_px / 100 * 0.9985**2)
                self.assertEqual(result.equity.iloc[2], result.equity.iloc[3])

    def test_zero_cost_round_trip_is_exit_over_entry(self):
        cfg = Config(fee_rate=0, slippage_pct=0)
        opens, signals = [100, 100, 110, 121, 121], [0, 1, 1, 0, 0]
        _, result = self.run_signals(opens, signals, cfg)
        self.assert_ledger(result, opens, signals, cfg)
        self.assertAlmostEqual(result.equity.iloc[-1], 121 / 110)
        self.assertAlmostEqual(result.trades.iloc[0].pnl_pct, 121 / 110 - 1)

    def test_fee_and_slippage_are_each_charged_once_per_side(self):
        for fee, slip in ((0, 0), (0.02, 0), (0, 0.03), (0.02, 0.03)):
            with self.subTest(fee=fee, slippage=slip):
                cfg = Config(fee_rate=fee, slippage_pct=slip)
                opens, signals = [100] * 5, [1, 1, 1, 0, 0]
                _, result = self.run_signals(opens, signals, cfg)
                self.assert_ledger(result, opens, signals, cfg)
                np.testing.assert_allclose(result.equity, [1, 1-fee-slip, 1-fee-slip, 1-fee-slip, (1-fee-slip)**2])
                self.assertAlmostEqual(result.trades.iloc[0].pnl_pct, (1-fee-slip)**2 - 1)

    def test_multiple_closed_trades_compound_to_equity(self):
        opens = [100, 110, 121, 90, 99, 80, 200]
        signals = [1, 0, 1, 0, 1, 0, 0]
        df, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        self.assertEqual(len(result.trades), 3)
        factor = 1.0
        for trade in result.trades.itertuples():
            self.assertAlmostEqual(1 + trade.pnl_pct, trade.exit_px / trade.entry_px * 0.9985**2)
            factor *= 1 + trade.pnl_pct
            self.assertAlmostEqual(result.equity.loc[trade.exit_time], factor)
        self.assertAlmostEqual(result.equity.loc[df.index[-1]], factor)

    def test_no_position_has_no_returns_or_costs(self):
        opens, signals = [100, 200, 50, 150], [0, 0, 0, 0]
        _, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        np.testing.assert_array_equal(result.equity, np.ones(4))
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.stats["trades"], 0)

    def test_open_position_is_marked_at_last_open_with_only_entry_cost(self):
        # The final close's exit signal cannot execute without a following open.
        opens, signals = [100, 100, 110, 121], [0, 1, 1, 0]
        _, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        self.assertTrue(result.trades.empty)
        self.assertAlmostEqual(result.equity.iloc[-1], 0.9985 * 121 / 110)

    def test_closed_trades_plus_open_position_reconcile(self):
        opens, signals = [100, 100, 110, 120, 150, 180], [1, 0, 0, 1, 1, 0]
        _, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        self.assertEqual(len(result.trades), 1)
        closed_factor = (1 + result.trades.pnl_pct).prod()
        open_factor = 0.9985 * 180 / 150
        self.assertAlmostEqual(result.equity.iloc[-1], closed_factor * open_factor)

    def test_entry_at_last_open_charges_cost_without_future_return(self):
        opens, signals = [100, 100, 200], [0, 1, 0]
        _, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        self.assertAlmostEqual(result.equity.iloc[-1], 0.9985)
        self.assertTrue(result.trades.empty)

    def test_signal_at_last_close_cannot_trade(self):
        opens, signals = [100, 110, 200], [0, 0, 1]
        _, result = self.run_signals(opens, signals)
        self.assert_ledger(result, opens, signals)
        np.testing.assert_array_equal(result.equity, np.ones(3))
        self.assertTrue(result.trades.empty)

    def test_real_strategy_signals_execute_at_next_open(self):
        closes = [100, 99, 101, 103, 102, 98, 97, 100, 103, 99, 95, 96]
        opens = [100, 100, 99, 110, 112, 102, 88, 97, 105, 120, 99, 90]
        df = candles(opens, closes)
        signals = compute_signals(df, self.cfg).long
        # At index 2 the two moving averages are equal, so entry waits until 3.
        self.assertEqual(signals.tolist(), [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        result = run_backtest(df, self.cfg)
        self.assert_ledger(result, opens, signals.tolist())
        self.assertEqual(result.trades.entry_time.tolist(), [df.index[4], df.index[8]])
        self.assertEqual(result.trades.exit_time.tolist(), [df.index[6], df.index[10]])
        self.assertAlmostEqual(result.equity.iloc[-1], (1 + result.trades.pnl_pct).prod())

    def test_daily_sharpe_uses_compounded_equity_changes(self):
        # Two opens per day; include offsetting +/- intraday price changes.
        opens = [100, 100, 120, 100, 100, 110, 99, 100]
        signals = [1] * 8
        _, result = self.run_signals(opens, signals, freq="12h")
        self.assert_ledger(result, opens, signals)
        daily_returns = pd.Series([-0.0015, 0, 0.1, 100 / 110 - 1])
        expected = daily_returns.mean() / (daily_returns.std() + 1e-9) * np.sqrt(252)
        self.assertAlmostEqual(result.stats["sharpe"], expected)


if __name__ == "__main__":
    unittest.main()
