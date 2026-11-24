"""Monthly swap-spread engine. Assumptions and usage: ../README.md."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# Data loading

def bloomberg_csv(path):
    rows = list(csv.reader(Path(path).open(encoding="utf-8-sig", newline="")))
    header = next(i for i, row in enumerate(rows) if row and row[0] == "Dates")
    instruments = rows[header - 2]
    columns, current = [], ""
    for i, field in enumerate(rows[header][1:], 1):
        if instruments[i].strip():
            current = instruments[i].strip()
        columns.append(f"{current}|{field}")
    raw = pd.DataFrame(rows[header + 1:], columns=["date", *columns])
    dates = pd.to_datetime(raw.pop("date"), format="%m/%d/%Y", errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path}: duplicate or unordered dates")
    raw.index = pd.DatetimeIndex(dates, name="date")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    return raw, numeric



# Curves

def month_end(date, months=0):
    return (pd.Timestamp(date).to_period("M") + months).to_timestamp("M")


def act360(start, end):
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 360.0


def payment_months(remaining, frequency):
    """Front stub, then regular periods, all measured from valuation date."""
    if remaining <= 0:
        return np.array([], dtype=int)
    return np.arange((remaining - 1) % frequency + 1, remaining + 1, frequency)


@dataclass
class Curve:
    times: np.ndarray
    discounts: np.ndarray
    forwards: np.ndarray

    @classmethod
    def from_discounts(cls, times, discounts):
        times, discounts = np.asarray(times, float), np.asarray(discounts, float)
        if (len(times) < 2 or len(times) != len(discounts) or times[0] != 0 or
                discounts[0] != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
                not np.isfinite(discounts).all() or np.any(discounts <= 0)):
            raise ValueError("Invalid discount nodes")
        averages = -np.diff(np.log(discounts)) / np.diff(times)
        forwards = [averages[0]]
        for average in averages:
            forwards.append(2 * average - forwards[-1])
        return cls(times, discounts, np.asarray(forwards))

    @classmethod
    def bootstrap(cls, par_rates, anchor_discount, date, frequency=6, daycount="bond"):
        knots = np.array(sorted(par_rates), float)
        rates = np.array([par_rates[x] for x in knots], float)
        step = frequency / 12
        if (len(knots) < 2 or knots[0] > step or not np.isfinite(rates).all() or
                not 0 < anchor_discount < 2):
            raise ValueError("Missing short-end par inputs or invalid anchor")
        spline = CubicSpline(knots, rates, bc_type="natural", extrapolate=False)
        grid = np.arange(step, knots[-1] + 1e-8, step)
        dfs, annuity = [], 0.0
        for t in grid:
            m = int(round(t * 12))
            a = step if daycount == "bond" else act360(month_end(date, m-frequency), month_end(date, m))
            c = float(spline(t))
            df = (1 - c * annuity) / (1 + c * a)
            if not np.isfinite(df) or df <= 0:
                raise ValueError("Par bootstrap produced nonpositive discount factor")
            dfs.append(df)
            annuity += a * df
        return cls.from_discounts([0, .25, *grid], [1, anchor_discount, *dfs])

    def df(self, t):
        t = np.asarray(t, float)
        if np.any(t < -1e-10) or np.any(t > self.times[-1] + 1e-10):
            raise ValueError("Curve extrapolation is forbidden")
        t = np.clip(t, 0, self.times[-1])
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times)-2)
        u = t - self.times[idx]
        slope = (self.forwards[idx+1]-self.forwards[idx])/(self.times[idx+1]-self.times[idx])
        return self.discounts[idx] * np.exp(-self.forwards[idx]*u - slope*u*u/2)

    def bumped(self, shift):
        return Curve.from_discounts(self.times, self.discounts * np.exp(-shift*self.times))

    def annuity(self, remaining, frequency=6, date=None, daycount="bond"):
        months = payment_months(remaining, frequency)
        previous = np.r_[0, months[:-1]]
        accrual = (months-previous)/12 if daycount == "bond" else np.array([
            act360(month_end(date, int(a)), month_end(date, int(b))) for a, b in zip(previous, months)])
        return float(np.sum(accrual * self.df(months/12)))

    def par(self, remaining, frequency=6, date=None, daycount="bond"):
        if remaining <= 0:
            raise ValueError("No par quote at or after maturity")
        return float((1-self.df(remaining/12)) / self.annuity(remaining, frequency, date, daycount))



# Market inputs

@dataclass
class Market:
    date: pd.Timestamp
    observed_at: pd.Timestamp
    available_at: pd.Timestamp
    treasury: Curve
    swap: Curve
    floating: float
    repo: float
    cash_rate: float
    mode: str = "libor"
    overnight_factor: float = 1.0  # Realized factor since previous market observation.

    @property
    def short_spread(self):
        return self.floating - self.repo

    def signal(self, remaining):
        return self.swap.par(remaining) - self.treasury.par(remaining) - self.short_spread


def validate_markets(markets):
    if len(markets) < 2:
        raise ValueError("At least two monthly observations required")
    for i, m in enumerate(markets):
        if m.date != month_end(m.date) or m.available_at > m.date or m.observed_at > m.available_at:
            raise ValueError("Invalid month-end clock or future/unavailable observation")
        if i and m.date != month_end(markets[i-1].date, 1):
            raise ValueError("Missing monthly observations; do not forward fill curves")
        if m.mode != markets[0].mode or not np.isfinite([m.floating, m.repo, m.cash_rate, m.overnight_factor]).all():
            raise ValueError("Mixed rate regimes or invalid rates")
        if m.overnight_factor <= 0:
            raise ValueError("Invalid realized overnight factor")


def load_baseline(path):
    """JSON contract, deliberately not compatible with raw SOFR Bloomberg exports."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("regime") != "LIBOR_3M_GC_REPO_3M" or payload.get("units") != "decimal":
        raise ValueError("Baseline needs decimal LIBOR 3M / GC repo 3M inputs; SOFR substitution rejected")
    required = {.5, 1., 2., 3., 5., 7., 10.}
    markets = []
    for r in payload["observations"]:
        date = pd.Timestamp(r["date"])
        treasury, swap = ({float(k): float(v) for k, v in r[name].items()}
                          for name in ("treasury_par", "swap_par"))
        if not required.issubset(treasury) or not required.issubset(swap):
            raise ValueError("Baseline missing Appendix A short/long par curve nodes")
        alpha = act360(date, month_end(date, 3))
        tdf = 1/(1+r["treasury_3m_act360"]*alpha)
        sdf = 1/(1+r["libor_3m_act360"]*alpha)
        markets.append(Market(date, pd.Timestamp(r["observed_at"]), pd.Timestamp(r["available_at"]),
                              Curve.bootstrap(treasury, tdf, date), Curve.bootstrap(swap, sdf, date),
                              r["libor_3m_act360"], r["gc_repo_3m_act360"], r["cash_rate_act360"]))
    validate_markets(markets)
    return markets


def sofr_scenario(root, funding_spread_bp=0.0):
    """Hypothetical par bonds and SOFR-plus-spread funding; never called baseline."""
    root = Path(root) / "CSV's"
    dfs = {p.name: bloomberg_csv(p)[1] for p in root.glob("*.csv")}
    treasury = pd.concat([dfs["USGG.csv"], dfs["USGG_ext.xlsx.csv"]], axis=1)
    swaps = pd.concat([dfs["USOSFR.csv"], dfs["USOSFR_ext.csv"]], axis=1)
    # A dated overnight fixing is conservatively available only on the next exported business row.
    rates = dfs["ONF_MR.csv"]["SOFRRATE Index|PX_LAST"].shift(1).dropna()/100
    common = treasury.index.intersection(swaps.index).intersection(rates.index)
    final_month = common.max().to_period("M")
    dates = pd.Series(common, index=common).groupby(common.to_period("M")).last()
    markets = []
    for period, observed in dates.items():
        if period >= final_month:
            continue  # Incomplete final month is never promoted to a full monthly observation.
        date = period.to_timestamp("M")
        t = {float(k): float(treasury.loc[observed, f"USGG{k}YR Index|PX_LAST"])/100
             for k in (2, 3, 5, 7, 10)}
        s = {float(k): float(swaps.loc[observed, f"USOSFR{k} Curncy|PX_LAST"])/100
             for k in (2, 3, 5, 7, 10)}
        if not np.isfinite([*t.values(), *s.values()]).all():
            raise ValueError(f"Missing scenario curve on {observed}; no forward filling")
        overnight = float(rates.loc[observed])
        # Explicit unsupported short-end assumptions, recorded in scenario metadata.
        t.update({.5: t[2.], 1.: t[2.]})
        tdf = (1+t[2.]/2)**(-.5)
        days = (month_end(date, 12)-date).days
        s[1.] = ((1+overnight/360)**days - 1)/(days/360)
        sdf = (1+overnight/360)**(-(month_end(date, 3)-date).days)
        factor = 1.0
        if markets:
            daily = pd.date_range(markets[-1].date, date-pd.Timedelta(days=1), freq="D")
            known = rates.reindex(rates.index.union(daily)).sort_index().ffill().reindex(daily)
            if known.isna().any():
                raise ValueError("Missing realized SOFR fixings")
            factor = float(np.prod(1+known.to_numpy()/360))
        markets.append(Market(date, pd.Timestamp(observed), pd.Timestamp(observed),
                              Curve.bootstrap(t, tdf, date),
                              Curve.bootstrap(s, sdf, date, frequency=12, daycount="act360"),
                              overnight, overnight+funding_spread_bp/10000, overnight,
                              "sofr_scenario", factor))
    validate_markets(markets)
    return markets



# Configuration and signals

@dataclass(frozen=True)
class Config:
    name: str = "paper_baseline"
    source: str = "baseline"
    market_file: str = "data/baseline_monthly.json"
    tenors: tuple = (2, 3, 5, 10)
    notional: float = 100.0
    threshold_bp: float = 10.0
    hedge: str = "equal_notional"
    lag_months: int = 0
    index_denominator: str = "active"
    treasury_full_spread: float = 1/32
    swap_full_spread_bp: float = 1.0
    repo_full_spread_bp: float = 10.0
    funding_spread_bp: float = 0.0
    financing_stress_bp: float = 0.0
    force_close_final: bool = False
    initial_cash: float = 1000.0
    treasury_haircut: float = .02
    swap_initial_margin: float = .02
    max_gross_leverage: float = 10.0
    max_repo_borrowing: float = 5000.0
    max_unsecured_borrowing: float = 0.0
    unsecured_spread_bp: float = 200.0
    risk_sizing: bool = False
    risk_lookback: int = 24
    risk_min_history: int = 12
    risk_target_notional_vol: float = .01
    max_size_multiplier: float = 2.0

    def __post_init__(self):
        if self.source not in ("baseline", "sofr_scenario"):
            raise ValueError("Unknown data regime")
        if self.hedge not in ("equal_notional", "dv01") or self.index_denominator not in ("active", "retain_cash"):
            raise ValueError("Unknown hedge or cohort denominator")
        if self.lag_months not in (0, 1) or not self.tenors or any(t not in (2, 3, 5, 10) for t in self.tenors):
            raise ValueError("Supported tenors 2/3/5/10; signal lag 0 or 1 month")
        positive = [self.notional, self.initial_cash, self.max_gross_leverage, self.max_size_multiplier,
                    self.risk_target_notional_vol]
        nonnegative = [self.threshold_bp, self.treasury_full_spread, self.swap_full_spread_bp,
                       self.repo_full_spread_bp, self.financing_stress_bp, self.treasury_haircut,
                       self.swap_initial_margin, self.max_repo_borrowing, self.max_unsecured_borrowing,
                       self.unsecured_spread_bp]
        if any(not np.isfinite(x) or x <= 0 for x in positive) or any(not np.isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError("Invalid size/cost/capital setting")
        if not np.isfinite(self.funding_spread_bp) or not 2 <= self.risk_min_history <= self.risk_lookback:
            raise ValueError("Invalid funding or risk history setting")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)


def entry_direction(signal, threshold_bp=10):
    threshold = threshold_bp/10000
    # Tolerance prevents decimal representation from trading exact boundaries.
    if signal > threshold + 1e-12:
        return 1
    if signal < -threshold - 1e-12:
        return -1
    return 0


def converged(direction, remaining_signal):
    return direction*remaining_signal <= 1e-12


def prior_size(history, config):
    if not config.risk_sizing or len(history) < config.risk_min_history:
        return 1.0
    vol = np.std(history[-config.risk_lookback:], ddof=1)*np.sqrt(12)
    if vol < 1e-12:
        return config.max_size_multiplier
    return min(config.max_size_multiplier, config.risk_target_notional_vol/vol)



# Instrument valuation

@dataclass
class Trade:
    id: int
    tenor: int
    direction: int
    entry_index: int
    entry_date: object
    signal_date: object
    entry_signal: float
    swap_notional: float
    bond_notional: float
    swap_coupon: float
    bond_coupon: float
    float_reset: float
    repo_reset: float
    mode: str
    age: int = 0
    overnight_accum: float = 1.0
    cash_gross: float = 0.0
    cash_net: float = 0.0
    value_gross: float = 0.0
    value_net: float = 0.0
    total_gross: float = 0.0
    total_net: float = 0.0
    closed: bool = False
    exit_reason: str = ""
    exit_date: object = None

    @property
    def remaining(self):
        return self.tenor*12-self.age

    @property
    def fixed_frequency(self):
        return 12 if self.mode == "sofr_scenario" else 6

    def future_fixed(self, frequency):
        return np.arange((self.age//frequency+1)*frequency, self.tenor*12+1, frequency)

    def fixed_accruals(self, schedule, frequency, daycount):
        if daycount == "bond":
            return np.full(len(schedule), frequency/12)
        return np.array([act360(month_end(self.entry_date, int(m-frequency)),
                               month_end(self.entry_date, int(m))) for m in schedule])

    def fixed_annuity(self, curve):
        schedule = self.future_fixed(self.fixed_frequency)
        accrual = self.fixed_accruals(schedule, self.fixed_frequency,
                                     "act360" if self.mode == "sofr_scenario" else "bond")
        return float(np.sum(accrual*curve.df((schedule-self.age)/12)))

    def prices(self, market, swap_curve=None, treasury_curve=None):
        if self.remaining == 0:
            return dict(bond_dirty=0., bond_accrued=0., bond_clean=0., swap_unit=0., swap_fixed_pv=0., swap_float_pv=0.)
        sc = market.swap if swap_curve is None else swap_curve
        tc = market.treasury if treasury_curve is None else treasury_curve
        schedule = self.future_fixed(6)
        bond = float(self.bond_coupon*.5*np.sum(tc.df((schedule-self.age)/12)) + tc.df(self.remaining/12))
        last = self.age//6*6
        start, end = month_end(self.entry_date, last), month_end(self.entry_date, last+6)
        accrued = self.bond_coupon*.5*(market.date-start).days/(end-start).days
        fixed = self.swap_coupon*self.fixed_annuity(sc)
        if self.mode == "sofr_scenario":
            floating = self.overnight_accum-float(sc.df(self.remaining/12))
        else:
            last = self.age//3*3
            next_month = min(last+3, self.tenor*12)
            a = act360(month_end(self.entry_date, last), month_end(self.entry_date, next_month))
            dfnext = float(sc.df((next_month-self.age)/12))
            floating = (1+self.float_reset*a)*dfnext-float(sc.df(self.remaining/12))
        return dict(bond_dirty=bond, bond_accrued=accrued, bond_clean=bond-accrued,
                    swap_unit=fixed-floating, swap_fixed_pv=fixed, swap_float_pv=floating)

    def dv01(self, market):
        key = (id(market), self.age, self.overnight_accum, self.float_reset)
        if getattr(self, "_dv01_key", None) == key:
            return self._dv01_value
        up = self.prices(market, market.swap.bumped(.0001), market.treasury.bumped(.0001))
        down = self.prices(market, market.swap.bumped(-.0001), market.treasury.bumped(-.0001))
        self._dv01_key = key
        self._dv01_value = ((down["swap_unit"]-up["swap_unit"])/2,
                            (down["bond_dirty"]-up["bond_dirty"])/2)
        return self._dv01_value

    def funding_accrual(self, market, config):
        if self.mode == "sofr_scenario" or self.remaining == 0:
            return 0., 0.
        start = month_end(self.entry_date, self.age//3*3)
        a = act360(start, market.date)
        gross = self.direction*self.bond_notional*self.repo_reset*a
        drag = self.bond_notional*(config.repo_full_spread_bp/2+config.financing_stress_bp)/10000*a
        return gross, gross-drag

    def execution_cost(self, market, config):
        if self.remaining == 0:
            return 0., 0.
        return (self.bond_notional/100*config.treasury_full_spread/2,
                self.swap_notional*config.swap_full_spread_bp/20000*self.fixed_annuity(market.swap))

    def mark(self, market, config):
        prices = self.prices(market)
        accrued_g, accrued_n = self.funding_accrual(market, config)
        base = 0. if self.remaining == 0 else self.direction*(
            self.swap_notional*prices["swap_unit"] - self.bond_notional*prices["bond_dirty"] + self.bond_notional)
        self.value_gross, self.value_net = base+accrued_g, base+accrued_n
        self.total_gross = self.cash_gross+self.value_gross
        self.total_net = self.cash_net+self.value_net
        return prices, accrued_g, accrued_n


def open_trade(id, tenor, direction, index, market, signal_market, notional, config):
    frequency = 12 if market.mode == "sofr_scenario" else 6
    dc = "act360" if frequency == 12 else "bond"
    trade = Trade(id, tenor, direction, index, market.date, signal_market.date,
                  signal_market.signal(tenor*12), notional, notional,
                  market.swap.par(tenor*12, frequency, market.date, dc), market.treasury.par(tenor*12),
                  market.floating, market.repo, market.mode)
    if config.hedge == "dv01":
        swap_dv, bond_dv = trade.dv01(market)
        if swap_dv <= 0 or bond_dv <= 0:
            raise ValueError("Nonpositive instrument DV01 cannot define hedge")
        trade.bond_notional = trade.swap_notional*swap_dv/bond_dv
    tb, sw = trade.execution_cost(market, config)
    trade.cash_net = -tb-sw
    trade.mark(market, config)
    return trade



# Trade accounting and simulation

def empty_flows():
    return dict(swap_fixed_cash=0., swap_float_cash=0., treasury_coupon_cash=0.,
                financing_gross_cash=0., financing_net_cash=0., treasury_principal_cash=0.,
                funding_principal_cash=0., swap_unwind_cash=0., treasury_trade_cash=0.,
                treasury_execution_cost=0., swap_execution_cost=0.)


def close_trade(trade, market, config, reason, flows):
    p, fg, fn = trade.mark(market, config)
    d = trade.direction
    if trade.remaining:
        flows["swap_unwind_cash"] += d*trade.swap_notional*p["swap_unit"]
        flows["treasury_trade_cash"] -= d*trade.bond_notional*p["bond_dirty"]
        flows["funding_principal_cash"] += d*trade.bond_notional
        flows["financing_gross_cash"] += fg
        flows["financing_net_cash"] += fn
        tb, sw = trade.execution_cost(market, config)
        flows["treasury_execution_cost"] += tb
        flows["swap_execution_cost"] += sw
        trade.cash_gross += trade.value_gross
        trade.cash_net += trade.value_net-tb-sw
    trade.value_gross = trade.value_net = 0.
    trade.total_gross, trade.total_net = trade.cash_gross, trade.cash_net
    trade.closed, trade.exit_reason, trade.exit_date = True, reason, market.date


def advance(trade, market, config):
    flows = empty_flows()
    trade.age += 1
    d, ns, nb, age = trade.direction, trade.swap_notional, trade.bond_notional, trade.age
    if age % 6 == 0:
        flows["treasury_coupon_cash"] = -d*nb*trade.bond_coupon*.5
    freq = trade.fixed_frequency
    if age % freq == 0:
        a = act360(month_end(trade.entry_date, age-freq), market.date) if freq == 12 else .5
        flows["swap_fixed_cash"] = d*ns*trade.swap_coupon*a
    if trade.mode == "sofr_scenario":
        trade.overnight_accum *= market.overnight_factor
        if age % 12 == 0:
            flows["swap_float_cash"] = -d*ns*(trade.overnight_accum-1)
            trade.overnight_accum = 1.
        a = act360(month_end(market.date, -1), market.date)
        basis = market.repo-market.floating
        flows["financing_gross_cash"] = d*nb*(market.overnight_factor-1+basis*a)
        drag = nb*(config.repo_full_spread_bp/2+config.financing_stress_bp)/10000*a
        flows["financing_net_cash"] = flows["financing_gross_cash"]-drag
    elif age % 3 == 0:
        a = act360(month_end(trade.entry_date, age-3), market.date)
        flows["swap_float_cash"] = -d*ns*trade.float_reset*a
        flows["financing_gross_cash"] = d*nb*trade.repo_reset*a
        drag = nb*(config.repo_full_spread_bp/2+config.financing_stress_bp)/10000*a
        flows["financing_net_cash"] = flows["financing_gross_cash"]-drag
        trade.float_reset, trade.repo_reset = market.floating, market.repo
    if trade.remaining == 0:
        flows["treasury_principal_cash"] = -d*nb
        flows["funding_principal_cash"] = d*nb
    common = sum(flows[k] for k in ("swap_fixed_cash", "swap_float_cash", "treasury_coupon_cash",
                                    "treasury_principal_cash", "funding_principal_cash"))
    trade.cash_gross += common+flows["financing_gross_cash"]
    trade.cash_net += common+flows["financing_net_cash"]
    trade.mark(market, config)
    return flows


def record(trade, market, config, event, flows, before, reason="", exit_signal=np.nan):
    old_g, old_n, old_cg, old_cn, old_vg, old_vn = before
    p = trade.prices(market)
    sdv, bdv = (0., 0.) if trade.closed else trade.dv01(market)
    d = trade.direction
    accrued_g, accrued_n = (0., 0.) if trade.closed else trade.funding_accrual(market, config)
    row = dict(date=market.date, trade_id=trade.id, tenor=trade.tenor, event=event,
               reason=reason, direction=d, signal_date=trade.signal_date,
               entry_signal_bp=trade.entry_signal*10000, exit_signal_bp=exit_signal*10000,
               age_months=trade.age, remaining_months=trade.remaining,
               swap_notional=trade.swap_notional, treasury_face=trade.bond_notional,
               swap_coupon=trade.swap_coupon, treasury_coupon=trade.bond_coupon,
               floating_reset=trade.float_reset, repo_reset=trade.repo_reset,
               overnight_accum=trade.overnight_accum,
               **p, **flows, funding_accrued_gross=accrued_g, funding_accrued_net=accrued_n,
               restricted_repo_asset=0. if trade.closed else max(d*trade.bond_notional, 0),
               repo_debt=0. if trade.closed else max(-d*trade.bond_notional, 0),
               swap_dv01=d*trade.swap_notional*sdv,
               treasury_dv01=-d*trade.bond_notional*bdv,
               mark_gross=trade.value_gross, mark_net=trade.value_net,
               mark_change_gross=trade.value_gross-old_vg, mark_change_net=trade.value_net-old_vn,
               cash_change_gross=trade.cash_gross-old_cg, cash_change_net=trade.cash_net-old_cn,
               cumulative_gross_pnl=trade.total_gross, cumulative_net_pnl=trade.total_net,
               gross_pnl=trade.total_gross-old_g, net_pnl=trade.total_net-old_n,
               open_after=not trade.closed)
    row["total_cost_drag"] = row["gross_pnl"]-row["net_pnl"]
    row["reconciliation_error"] = row["net_pnl"]-row["cash_change_net"]-row["mark_change_net"]
    return row


def state(trade):
    return (trade.total_gross, trade.total_net, trade.cash_gross, trade.cash_net, trade.value_gross, trade.value_net)


def exposure(trades, market, config):
    out = dict(gross_notional=0., repo_debt=0., restricted_repo_asset=0., collateral_required=0.,
               swap_dv01=0., treasury_dv01=0., marks_net=0., treasury_market_value=0.)
    for t in trades:
        if t.closed:
            continue
        p = t.prices(market)
        sdv, bdv = t.dv01(market)
        out["gross_notional"] += t.swap_notional+t.bond_notional
        out["repo_debt"] += max(-t.direction*t.bond_notional, 0)
        out["restricted_repo_asset"] += max(t.direction*t.bond_notional, 0)
        out["collateral_required"] += (config.treasury_haircut*t.bond_notional*p["bond_dirty"] +
                                       config.swap_initial_margin*t.swap_notional +
                                       max(-t.direction*t.swap_notional*p["swap_unit"], 0) +
                                       max(t.direction*t.bond_notional*(p["bond_dirty"]-1), 0))
        out["swap_dv01"] += t.direction*t.swap_notional*sdv
        out["treasury_dv01"] -= t.direction*t.bond_notional*bdv
        out["marks_net"] += t.value_net
        out["treasury_market_value"] -= t.direction*t.bond_notional*p["bond_dirty"]
    out["net_dv01"] = out["swap_dv01"]+out["treasury_dv01"]
    return out


def fits(cash, exp, config):
    equity = cash+exp["marks_net"]
    return (equity > 0 and cash-exp["collateral_required"] >= -config.max_unsecured_borrowing-1e-8 and
            exp["gross_notional"] <= config.max_gross_leverage*equity+1e-8 and
            exp["repo_debt"] <= config.max_repo_borrowing+1e-8)


def run(markets, config, funded=False):
    validate_markets(markets)
    if (config.source == "baseline") != (markets[0].mode == "libor"):
        raise ValueError("Configuration and instrument regime disagree")
    trades, ledger, decisions, portfolio, indexes = [], [], [], [], []
    cash, previous_equity = config.initial_cash, config.initial_cash
    history, pending_first, ever_cohorts = [], {t: [0., 0.] for t in config.tenors}, {t: 0 for t in config.tenors}
    for i, m in enumerate(markets):
        month_rows = []
        active_before = {tenor: sum(not t.closed and t.tenor == tenor for t in trades) for tenor in config.tenors}
        cash_interest = 0.
        if funded and i:
            rate = markets[i-1].cash_rate + (config.unsecured_spread_bp/10000 if cash < 0 else 0)
            cash_interest = cash*rate*act360(markets[i-1].date, m.date)
            cash += cash_interest
        for t in [t for t in trades if not t.closed]:
            before = state(t)
            flows = advance(t, m, config)
            sm = markets[max(0, i-config.lag_months)]
            remaining_at_decision = t.remaining+config.lag_months
            signal = sm.signal(remaining_at_decision) if t.remaining > 0 else np.nan
            reason = "maturity" if t.remaining == 0 else "convergence" if converged(t.direction, signal) else ""
            if config.force_close_final and i == len(markets)-1 and not reason:
                reason = "sample_end_liquidation"
            if reason:
                close_trade(t, m, config, reason, flows)
            row = record(t, m, config, "exit" if reason else "mark", flows, before, reason, signal)
            month_rows.append(row)
            cash += row["cash_change_net"]
        if funded:
            for t in [t for t in trades if not t.closed]:
                if fits(cash, exposure(trades, m, config), config):
                    break
                before, flows = state(t), empty_flows()
                close_trade(t, m, config, "capital_constraint", flows)
                row = record(t, m, config, "exit", flows, before, "capital_constraint")
                month_rows.append(row)
                cash += row["cash_change_net"]
        can_start = i < len(markets)-1
        size = prior_size(history, config)
        for tenor in config.tenors:
            if not can_start:
                continue
            ever_cohorts[tenor] += 1
            sm = markets[max(0, i-config.lag_months)]
            signal = sm.signal(tenor*12)
            direction = entry_direction(signal, config.threshold_bp) if i >= config.lag_months else 0
            decision = dict(date=m.date, signal_date=sm.date, tenor=tenor, signal_bp=signal*10000,
                            direction=direction, size_multiplier=size,
                            reason="inside_band" if i >= config.lag_months else "signal_warmup")
            if direction:
                candidate = open_trade(len(trades)+1, tenor, direction, i, m, sm,
                                       config.notional*size, config)
                exp = exposure([*trades, candidate], m, config)
                if funded and not fits(cash+candidate.cash_net, exp, config):
                    decision["reason"] = "entry_rejected_capital"
                else:
                    trades.append(candidate)
                    reason = "spread_above_upper_band" if direction == 1 else "spread_below_lower_band"
                    decision["reason"] = reason
                    flows = empty_flows()
                    tb, sw = candidate.execution_cost(m, config)
                    flows.update(treasury_execution_cost=tb, swap_execution_cost=sw,
                                 treasury_trade_cash=-direction*candidate.bond_notional,
                                 funding_principal_cash=direction*candidate.bond_notional)
                    row = record(candidate, m, config, "entry", flows, (0.,)*6, reason)
                    month_rows.append(row)
                    cash += row["cash_change_net"]
            decisions.append(decision)
        exp = exposure(trades, m, config)
        equity = cash+exp["marks_net"]
        pnl_net = sum(r["net_pnl"] for r in month_rows)
        pnl_gross = sum(r["gross_pnl"] for r in month_rows)
        portfolio.append(dict(date=m.date, cash=cash, cash_interest=cash_interest, equity=equity,
                              capital_reserved=exp["collateral_required"],
                              free_cash=cash-exp["collateral_required"],
                              gross_pnl=pnl_gross, net_pnl=pnl_net,
                              excess_return=pnl_net/previous_equity if previous_equity > 0 else np.nan,
                              total_return=(equity-previous_equity)/previous_equity if previous_equity > 0 else np.nan,
                              reconciliation_error=equity-previous_equity-pnl_net-cash_interest,
                              active_trades=sum(not t.closed for t in trades),
                              constraints_satisfied=fits(cash, exp, config), **exp))
        previous_equity = equity
        for tenor in config.tenors:
            g = sum(r["gross_pnl"] for r in month_rows if r["tenor"] == tenor)
            n = sum(r["net_pnl"] for r in month_rows if r["tenor"] == tenor)
            if i == 0:
                pending_first[tenor] = [g, n]
                continue
            if i == 1:
                g += pending_first[tenor][0]
                n += pending_first[tenor][1]
            active_denom = active_before[tenor]+int(can_start)
            retained_denom = ever_cohorts[tenor]
            denom = active_denom if config.index_denominator == "active" else retained_denom
            indexes.append(dict(date=m.date, tenor=tenor, active_denominator=active_denom,
                                retained_denominator=retained_denom, denominator=denom,
                                sum_gross_pnl=g, sum_net_pnl=n,
                                index_gross_pnl=g/max(denom, 1), index_net_pnl=n/max(denom, 1),
                                retain_cash_net_pnl=n/max(retained_denom, 1)))
        ledger.extend(month_rows)
        # Completed unit returns only; current date sizing was already decided above.
        prior_notional = sum(t.swap_notional for t in trades if not t.closed)
        history.append(pnl_net/max(prior_notional, config.notional))
    summary = pd.DataFrame([asdict(t) for t in trades], columns=[f.name for f in fields(Trade)])
    ledger_frame=pd.DataFrame(ledger) if ledger else pd.DataFrame(columns=[
        "date","trade_id","tenor","event","reason","gross_pnl","net_pnl",
        "cash_change_net","mark_change_net","reconciliation_error"])
    return dict(trades=summary, ledger=ledger_frame, decisions=pd.DataFrame(decisions),
                portfolio=pd.DataFrame(portfolio), index=pd.DataFrame(indexes))

