"""Regression checks for the daily clock and settlement across missing quote dates."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtests.engine import Config, Curve, Market, advance, open_trade, run


def market(date, previous=None):
    date = pd.Timestamp(date)
    times = [0, 11]
    curve = Curve.from_discounts(times, np.exp(-.035 * np.array(times)))
    treasury = Curve.from_discounts(times, np.exp(-.045 * np.array(times)))
    n = (date - previous.date).days if previous else 0
    step = 1 + .04 / 360
    return Market(date, date, date, treasury, curve, .04, .04, .04,
                  "sofr_scenario", step**n, "daily", (step,) * n)


class DailyClock(unittest.TestCase):
    config = Config.read("configs/sofr_scenario.json")

    def test_signal_precedes_execution(self):
        dates = []
        for day in ["2022-01-06", "2022-01-07", "2022-01-10"]:
            dates.append(market(day, dates[-1] if dates else None))
        result = run(dates, replace(self.config, tenors=(2,)), funded=True)
        trade, = result["trades"].itertuples()
        self.assertEqual(trade.signal_date, dates[0].date)
        self.assertEqual(trade.entry_date, dates[1].date)
        self.assertEqual(trade.age, 3)
        self.assertEqual(trade.swap_notional, 1_000_000)
        self.assertEqual(trade.bond_notional, 1_000_000)
        self.assertLess(result["portfolio"].reconciliation_error.abs().max(), 1e-7)

    def test_anniversary_and_maturity_between_quotes(self):
        start = market("2021-01-29")
        trade = open_trade(1, 2, 1, 0, start, start, 1_000_000, self.config)
        self.assertAlmostEqual(trade.prices(start)["bond_dirty"], 1)
        self.assertAlmostEqual(trade.prices(start)["swap_unit"], 0)
        annual = market("2022-01-31", start)
        flows = advance(trade, annual, self.config)
        step = 1 + .04 / 360
        self.assertAlmostEqual(flows["swap_fixed_cash"], 1_000_000 * trade.swap_coupon * 365/360)
        self.assertAlmostEqual(flows["swap_float_cash"], -1_000_000 * (step**365 - 1), places=6)
        self.assertAlmostEqual(trade.overnight_accum, step**2)
        final = market("2023-01-30", annual)
        flows = advance(trade, final, self.config)
        self.assertEqual(trade.remaining, 0)
        self.assertEqual(trade.age, 730)
        self.assertEqual(flows["treasury_principal_cash"], -1_000_000)
        self.assertEqual(flows["funding_principal_cash"], 1_000_000)
        self.assertAlmostEqual(flows["swap_float_cash"], -1_000_000 * (step**365 - 1), places=6)
        self.assertAlmostEqual(flows["financing_gross_cash"], 1_000_000 * (step**363 - 1), places=6)
