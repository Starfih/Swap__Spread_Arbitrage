# sofr_par_bond_scenario

HYPOTHETICAL SOFR / PAR-BOND SCENARIO
Period: 2020-01-31 to 2026-09-30 (80 monthly intervals).
All money amounts are dollars. Unscaled funded portfolio; no retrospective volatility scaling.

| Statistic | Result |
|---|---:|
| Initial capital | 1,000.000 |
| Ending equity | 1,293.103 |
| Gross trading P&L, closed + marked open trades | 109.698 |
| Costs and adverse financing drag | 22.583 |
| Net trading P&L, closed + marked open trades | 87.115 |
| Finalized P&L: closed trades only, net | 45.141 |
| Open trades: life-to-date marked P&L, net | 41.974 |
| Cash interest, separate from trading P&L | 205.988 |
| Total return including cash interest (%) | 29.310 |
| Annualized total return including cash interest (%) | 3.933 |
| Maximum NAV drawdown (%) | -2.927 |
| Sharpe: monthly trading excess return, annualized | 0.599 |
| Annualized trading excess-return volatility (%) | 1.902 |
| Trades entered / closed / still open | 131 / 85 / 46 |
| Closed-trade win rate (%) | 91.765 |
| Closed-trade profit factor | 40.768 |

## Trade locations

`chart.png`: triangles mark executed entries and crosses mark exits for each maturity.
`trades.csv`: exact trade IDs, signal/entry/exit dates, position, size, reasons,
modeled clean prices per $100 face, swap values, holding months and gross/net P&L.
Execution uses modeled prices plus separately charged spread costs. Rates are percentages.
Matured Treasury redemption is $100; its exit clean-price field is blank after payment.
Open trades have no exit date and retain marked P&L. Set `force_close_final` to true to
liquidate them at the last observation and charge exit costs.

## Assumptions used for this run

- Sizing: `equal_notional`; base swap notional $100; prior-risk sizing False.
  Tenors [2, 3, 5, 10]; entry threshold ±10 bp.
- Signal lag: 1 month(s). Convergence uses each trade's remaining maturity.
- Monthly overlapping trades; each has its own original coupon and maturity.
- Treasury full bid-ask 0.03125 dollars per $100; swap 1 bp;
  repo 10 bp. Half widths charged per side; repo drag accrues over time.
- Funding basis 0 bp over SOFR; extra adverse funding stress 0 bp.
- Treasury haircut 2%; swap margin 2%;
  gross notional/equity limit 10x; repo borrowing cap $5000.
- Missing short-end curves, hypothetical par bonds and funding assumptions materially affect results.
  The supplied data cannot establish Table 1 replication or actual-security returns. See README.md.
