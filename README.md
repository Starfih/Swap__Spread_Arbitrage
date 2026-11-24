# Swap-spread backtest

A small monthly backtest with one funded portfolio and four output files.

```text
backtest.py              Run the backtest
backtests/
  engine.py              Load data, build curves, value and account for trades
  report.py              Statistics, trade annotations and chart
configs/
  sofr_scenario.json     Main configuration for the supplied data
  sofr_financing_stress.json   Optional additional 50 bp funding stress
  baseline.json          Original LIBOR rules; required data is unavailable
outputs/
  summary.md             Read this first: drawdown, Sharpe, P&L and assumptions
  trades.csv             One row per trade: dates, direction, sizes, reasons, P&L
  equity.csv             Monthly equity, P&L, drawdown and exposures
  chart.png              Equity, drawdown and annotated entry/exit dates
SSA_Data/                Original supplied data, preserved
*.pdf                    Original supplied references, preserved
requirements.txt         Python dependencies
```

## Run

From this folder, using the existing Python environment:

```powershell
./.venv/Scripts/python backtest.py
```

For a fresh environment: `python -m venv .venv`, then
`./.venv/Scripts/python -m pip install -r requirements.txt`.
The hidden `.venv` folder is the local Python environment.

To change rules, edit `configs/sofr_scenario.json` and rerun. Parameters include
`tenors`, `notional`, `threshold_bp`, `initial_cash`, funding and cost assumptions.
An alternative run can use:

```powershell
./.venv/Scripts/python backtest.py --config configs/sofr_financing_stress.json --output outputs/stress
```

The default command replaces the same four files. Other output directories are
created only when explicitly requested. There is no package installation step.

## How to read the output

**Finalized P&L** means net P&L from closed trades. Open positions are shown
separately with their life-to-date marked P&L. Total trading P&L combines both.
Cash interest is separate. Ending equity equals initial capital plus net trading
P&L plus cash interest. Set `force_close_final` to `true` to liquidate all remaining
positions at the sample end and charge exit costs; default `false` preserves the
previous run's behavior.

Sharpe uses monthly trading excess returns, annualized by sqrt(12). Maximum
drawdown uses portfolio equity, including cash interest. Neither measure uses
full-sample volatility normalization. Annualized total return includes cash
interest. Win rate and profit factor use closed trades only; undefined figures
are `n/a`. Monetary amounts use the configured dollars, not scaled percentages.

The chart marks each executed entry with a triangle and exit with a cross on
its maturity's modeled spread. Multiple cohorts can trade on the same date.
Use `trades.csv` for exact IDs, dates and reasons. It also records modeled bond
clean prices per $100 face and unit-receiver swap values times notional before
costs. Costs are charged separately. Blank exit dates identify open trades.
Dates are contractual calendar month ends, not actual exchange fill timestamps.

## Assumptions and limits

1. **Hypothetical SOFR/par-bond scenario.** Supplied data covers 2020-2026 and
   contains SOFR swaps and generic Treasury yields, not historical LIBOR/term-repo
   data or identified tradable bond returns. This cannot replicate Table 1.
   October 2026 is incomplete and excluded; current results end September 2026.
2. **Rules.** Trade 2/3/5/10-year maturities monthly. If swap minus Treasury par
   spread exceeds floating minus funding spread by more than +10 bp, receive
   swap fixed and short Treasury. Below -10 bp, reverse. Equality does not enter.
   Default swap and Treasury notionals are equal. Overlapping cohorts retain
   their original coupon/maturity; exits test convergence at remaining maturity.
3. **Timing.** The SOFR scenario uses the previous month's signal and current
   modeled execution prices. Overnight fixings become usable on the next
   exported weekday; known rates carry across weekends/holidays. Daily SOFR
   compounds into annual ACT/360 floating coupons. No publication vintages,
   actual settlement calendar, payment lag or observation shift is available.
4. **Missing curves.** USGG benchmark yields are assumed to be hypothetical par
   rates. Treasury 6M/1Y rates are flat at 2Y; its 3M discount factor is
   `(1+y_2Y/2)^(-0.5)`. A missing 1Y OIS quote and 3M anchor are inferred from
   flat overnight SOFR. Natural cubic par interpolation, coupon bootstrapping
   and discount-preserving linear forwards are used. The starting forward equals
   the first interval average. This boundary convention and missing short-end
   quotes produce material forward oscillations (about -4.4% to 10.3% in this
   sample). Intermediate marks and convergence are sensitive to these assumptions.
5. **Instrument values.** Hypothetical Treasury bonds pay semiannual coupons;
   SOFR swaps pay annual ACT/360 fixed/compounded floating coupons. Bonds retain
   their original schedules; dirty marks include accrued interest. Swaps use a
   single discount/projection curve. For signal comparison, OIS curves are
   converted to semiannual par rates. No actual CUSIP prices or total returns
   are inferred from benchmark changes.
6. **Funding.** Scenario funding is rolling overnight SOFR plus the configured
   basis, settled monthly on fixed face principal. It is not observed 3M GC repo.
   Full repo width 10 bp adds 5 bp adverse annual financing drag; stress adds
   further adverse drag. Capital cash earns the previous month's SOFR. In the
   LIBOR mode, known LIBOR/repo resets lock quarterly; early repo termination
   is assumed possible at accrued carrying value without a break fee.
7. **Costs.** Full Treasury width is $1/32 per $100 face; charge half each side.
   Full swap width is 1 bp; charge half times notional times remaining fixed-leg
   annuity as a cash fee on entry/unwind while retaining the mid coupon. This is
   an execution-PV equivalent, with different cash timing from an off-mid coupon.
   Maturity redemption has no exit spread. No actual quoted bid/ask is substituted.
8. **Capital.** Default $1,000 capital, $100 swap notional per admitted cohort,
   2% Treasury haircut, 2% swap initial margin, 10x gross notional/equity limit,
   $5,000 repo borrowing cap and no unsecured borrowing. Restricted short-sale
   proceeds cannot fund other trades. Cash reserves cover adverse swap/bond marks.
   Reject entries that do not fit; liquidate oldest trades after monthly breaches.
   No intraday margin, specialness, counterparty default or liquidity simulation.
9. **Original baseline.** The retained LIBOR configuration requires verified
   decimal 3M LIBOR, 3M GC repo, 3M Treasury ACT/360 rate and semiannual Treasury/
   swap par nodes at 0.5/1/2/3/5/7/10Y. The loader enforces its regime and monthly
   availability. The original reference is DLY, supplied `769 (2).pdf`, Section 1
   pp. 772-775, Table 1 p. 776, Appendix A pp. 802-803. Spline boundaries,
   financing closeout and cohort details remain replication assumptions. The
   user's chosen paper-index convention removes closed cohorts after the exit
   month and includes each new cash cohort once; the compact output presents the
   funded portfolio rather than the paper return index. Jha ch. 9 pp. 225-229
   supports optional DV01 matching; Jermann pp. 1-3 and 18 motivates funding
   stress. These extensions do not replace equal-notional baseline rules.

The consolidated engine passed the existing 30 checks before the old test and
research folders were removed. The compact result was checked against the
previous funded account; cash flows, P&L and equity reconcile. The runner also
checks reconciliation each time it writes results.
