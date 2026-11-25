"""Calendar-dated valuation and cash flows for the daily SOFR scenario."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .engine import Trade, empty_flows


@lru_cache(maxsize=8192)
def calendar_schedule(start, months):
    """Unadjusted semiannual anniversaries, always anchored to the original day."""
    return np.array([(start + pd.DateOffset(months=m)).toordinal()
                     for m in range(0, months + 1, 6)], dtype=int)


def discount_tables(market):
    """All cohorts on one date share the same two curves and parallel 1 bp shifts."""
    if not hasattr(market, "_daily_tables"):
        last = round(market.swap.times[-1] * 365)
        years = np.arange(last + 1) / 365
        market._daily_tables = (market.treasury.df(years), market.swap.df(years),
                                np.sinh(.0001 * years))
    return market._daily_tables


def par_spread(market, schedule):
    today = market.date.toordinal()
    first = schedule.searchsorted(today, side="right")
    future = schedule[first:] - today
    if not len(future):
        raise ValueError("No remaining maturity for a daily signal")
    bond, swap, _ = discount_tables(market)
    # Front stub on a par instrument starting on the signal date.
    stub = future[0] / (schedule[first] - schedule[first - 1])
    def par(discounts):
        dfs = discounts[future]
        return (1 - dfs[-1]) / (.5 * (stub * dfs[0] + dfs[1:].sum()))
    return par(swap) - par(bond) - market.short_spread


class DailyTrade(Trade):
    @property
    def remaining(self):
        return max(self.schedule[-1] - self.today, 0)

    def remaining_signal(self, market):
        return par_spread(market, self.schedule)

    def valuation(self, market):
        if getattr(self, "_valuation_date", None) == market.date:
            return self._valuation
        today = market.date.toordinal()
        if self.remaining == 0:
            prices = dict(bond_dirty=0., bond_accrued=0., bond_clean=0.,
                          swap_unit=0., swap_fixed_pv=0., swap_float_pv=0.)
            result = (prices, 0., (0., 0.))
        else:
            bond, swap, shift = discount_tables(market)
            first = self.schedule.searchsorted(today, side="right")
            days = self.schedule[first:] - today
            start, end = self.schedule[first - 1:first + 1]
            bond_dfs = bond[days]
            bond_pv = self.bond_coupon * .5 * bond_dfs.sum() + bond_dfs[-1]
            accrued = self.bond_coupon * .5 * (today - start) / (end - start)
            annual = self.schedule[::2]
            first_annual = annual.searchsorted(today, side="right")
            swap_days = annual[first_annual:] - today
            discounted_accruals = np.diff(annual[first_annual - 1:]) / 360 * swap[swap_days]
            annuity = discounted_accruals.sum()
            fixed = self.swap_coupon * annuity
            floating = self.overnight_accum - swap[days[-1]]
            prices = dict(bond_dirty=bond_pv, bond_accrued=accrued, bond_clean=bond_pv-accrued,
                          swap_unit=fixed-floating, swap_fixed_pv=fixed, swap_float_pv=floating)
            bond_dv = self.bond_coupon * .5 * (bond_dfs @ shift[days]) + bond_dfs[-1]*shift[days[-1]]
            swap_dv = self.swap_coupon*(discounted_accruals @ shift[swap_days]) + swap[days[-1]]*shift[days[-1]]
            result = (prices, annuity, (swap_dv, bond_dv))
        self._valuation_date, self._valuation = market.date, result
        return result

    def prices(self, market):
        return self.valuation(market)[0]

    def dv01(self, market):
        return self.valuation(market)[2]

    def execution_cost(self, market, config):
        if not self.remaining:
            return 0., 0.
        annuity = self.valuation(market)[1]
        return (self.bond_notional / 100 * config.treasury_full_spread / 2,
                self.swap_notional * config.swap_full_spread_bp / 20000 * annuity)

    def advance_daily(self, market, config):
        flows = empty_flows()
        previous = self.today
        self.today = market.date.toordinal()
        end = min(self.today, self.schedule[-1])
        self.age = end - self.schedule[0]  # Calendar days held through contractual maturity.
        d, ns, nb = self.direction, self.swap_notional, self.bond_notional
        cursor = previous
        first = self.schedule.searchsorted(previous, side="right")
        stop = self.schedule.searchsorted(end, side="right")
        for i in range(first, stop):
            due = self.schedule[i]
            flows["treasury_coupon_cash"] -= d * nb * self.bond_coupon * .5
            if i % 2 == 0:
                self.overnight_accum *= np.prod(market.daily_factors[cursor-previous:due-previous])
                accrual = (due - self.schedule[i-2]) / 360
                flows["swap_fixed_cash"] += d * ns * self.swap_coupon * accrual
                flows["swap_float_cash"] -= d * ns * (self.overnight_accum - 1)
                self.overnight_accum = 1.
                cursor = due
        self.overnight_accum *= np.prod(market.daily_factors[cursor-previous:end-previous])
        accrual = (end - previous) / 360
        factor = np.prod(market.daily_factors[:end-previous])
        basis = market.repo - market.floating
        financing = d * nb * (factor - 1 + basis * accrual)
        drag = nb * (config.repo_full_spread_bp / 2 + config.financing_stress_bp) / 10000 * accrual
        flows.update(financing_gross_cash=financing, financing_net_cash=financing-drag)
        if not self.remaining:
            flows.update(treasury_principal_cash=-d*nb, funding_principal_cash=d*nb)
        coupon_cash = flows["treasury_coupon_cash"] + flows["swap_fixed_cash"] + flows["swap_float_cash"]
        self.cash_gross += coupon_cash + financing
        self.cash_net += coupon_cash + financing - drag
        self.mark(market, config)
        return flows


def new_daily_trade(id, tenor, direction, index, market, signal_market, notional):
    schedule = calendar_schedule(market.date, tenor * 12)
    bond, swap, _ = discount_tables(market)
    days = schedule - schedule[0]
    bond_coupon = (1 - bond[days[-1]]) / (.5 * bond[days[1:]].sum())
    annual = days[::2]
    swap_coupon = (1 - swap[days[-1]]) / ((np.diff(annual) / 360) @ swap[annual[1:]])
    trade = DailyTrade(id, tenor, direction, index, market.date, signal_market.date,
                       signal_market.signal(tenor * 12), notional, notional, swap_coupon,
                       bond_coupon, market.floating, market.repo, market.mode)
    trade.schedule, trade.today = schedule, schedule[0]
    return trade
