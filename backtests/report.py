"""Four readable outputs from one funded portfolio."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results(result, markets, config, output):
    output.mkdir(parents=True, exist_ok=True)
    account, trades, ledger = result["portfolio"].copy(), result["trades"].copy(), result["ledger"]
    if account.reconciliation_error.abs().max() > 1e-8:
        raise ValueError("Portfolio P&L does not reconcile")
    if not ledger.empty and (ledger.reconciliation_error.abs().max() > 1e-8 or
                             abs(ledger.net_pnl.sum()-trades.total_net.sum()) > 1e-8):
        raise ValueError("Trade P&L does not reconcile")
    account["drawdown_pct"] = 100*(account.equity/account.equity.cummax().clip(lower=config.initial_cash)-1)
    account["cumulative_net_trading_pnl"] = account.net_pnl.cumsum()
    account[["date", "equity", "cash", "gross_pnl", "net_pnl", "cash_interest",
             "cumulative_net_trading_pnl", "drawdown_pct", "active_trades", "gross_notional",
             "repo_debt", "free_cash"]].to_csv(output/"equity.csv", index=False, float_format="%.8f")

    columns = ["trade_id", "tenor_years", "position", "signal_date", "entry_date", "exit_date",
               "status", "entry_reason", "exit_reason", "entry_signal_bp", "exit_signal_bp",
               "swap_notional", "treasury_face", "swap_fixed_rate_pct", "treasury_coupon_pct",
               "entry_bond_clean", "exit_bond_clean", "entry_swap_value", "exit_swap_value",
               "months_held", "gross_pnl", "costs", "net_pnl"]
    annotations = []
    for t in trades.itertuples():
        events = ledger[ledger.trade_id == t.id]
        entry = events[events.event == "entry"].iloc[0]
        exits = events[events.event == "exit"]
        last = exits.iloc[-1] if not exits.empty else None
        annotations.append(dict(trade_id=t.id, tenor_years=t.tenor,
            position="Receive swap fixed / short Treasury" if t.direction == 1 else "Pay swap fixed / long Treasury",
            signal_date=t.signal_date, entry_date=t.entry_date, exit_date=t.exit_date,
            status="CLOSED" if t.closed else "OPEN: marked at sample end",
            entry_reason=entry.reason, exit_reason=t.exit_reason if t.closed else "Still open",
            entry_signal_bp=t.entry_signal*10000, exit_signal_bp=last.exit_signal_bp if last is not None else np.nan,
            swap_notional=t.swap_notional, treasury_face=t.bond_notional,
            swap_fixed_rate_pct=t.swap_coupon*100, treasury_coupon_pct=t.bond_coupon*100,
            entry_bond_clean=entry.bond_clean*100,
            exit_bond_clean=last.bond_clean*100 if last is not None and t.age < t.tenor*12 else np.nan,
            entry_swap_value=entry.swap_unit*t.swap_notional,
            exit_swap_value=last.swap_unit*t.swap_notional if last is not None else np.nan,
            months_held=t.age, gross_pnl=t.total_gross, costs=t.total_gross-t.total_net, net_pnl=t.total_net))
    pd.DataFrame(annotations, columns=columns).to_csv(output/"trades.csv", index=False, float_format="%.8f")

    # Fold initial execution costs into the first complete return interval.
    previous = account.equity.shift().to_numpy()[1:].copy()
    previous[0] = config.initial_cash
    changes = np.diff(account.equity.to_numpy())
    changes[0] += account.equity.iloc[0]-config.initial_cash
    returns = changes/previous
    excess = (changes-account.cash_interest.to_numpy()[1:])/previous
    vol = np.std(excess, ddof=1)*np.sqrt(12) if len(excess)>1 else np.nan
    sharpe = np.mean(excess)*12/vol if vol > 1e-12 else np.nan
    closed = trades[trades.closed] if not trades.empty else trades
    open_trades = trades[~trades.closed] if not trades.empty else trades
    realized = closed.total_net.sum()
    gains, losses = closed.total_net.clip(lower=0).sum(), -closed.total_net.clip(upper=0).sum()
    years = (account.date.iloc[-1]-account.date.iloc[0]).days/365.25
    cagr = 100*((account.equity.iloc[-1]/config.initial_cash)**(1/years)-1) if years>0 and account.equity.iloc[-1]>0 else np.nan
    win = 100*(closed.total_net>0).mean() if len(closed) else np.nan
    def fmt(value):
        return f"{value:,.3f}" if np.isfinite(value) else "n/a"
    summary = f"""# {config.name}

{'HYPOTHETICAL SOFR / PAR-BOND SCENARIO' if config.source=='sofr_scenario' else 'LIBOR baseline rules; replication unverified'}
Period: {account.date.iloc[0].date()} to {account.date.iloc[-1].date()} ({len(account)-1} monthly intervals).
All money amounts are dollars. Unscaled funded portfolio; no retrospective volatility scaling.

| Statistic | Result |
|---|---:|
| Initial capital | {fmt(config.initial_cash)} |
| Ending equity | {fmt(account.equity.iloc[-1])} |
| Gross trading P&L, closed + marked open trades | {fmt(account.gross_pnl.sum())} |
| Costs and adverse financing drag | {fmt(account.gross_pnl.sum()-account.net_pnl.sum())} |
| Net trading P&L, closed + marked open trades | {fmt(account.net_pnl.sum())} |
| Finalized P&L: closed trades only, net | {fmt(realized)} |
| Open trades: life-to-date marked P&L, net | {fmt(open_trades.total_net.sum())} |
| Cash interest, separate from trading P&L | {fmt(account.cash_interest.sum())} |
| Total return including cash interest (%) | {fmt(100*(account.equity.iloc[-1]/config.initial_cash-1))} |
| Annualized total return including cash interest (%) | {fmt(cagr)} |
| Maximum NAV drawdown (%) | {fmt(account.drawdown_pct.min())} |
| Sharpe: monthly trading excess return, annualized | {fmt(sharpe)} |
| Annualized trading excess-return volatility (%) | {fmt(vol*100)} |
| Trades entered / closed / still open | {len(trades)} / {len(closed)} / {len(open_trades)} |
| Closed-trade win rate (%) | {fmt(win)} |
| Closed-trade profit factor | {fmt(gains/losses if losses>0 else np.nan)} |

## Trade locations

`chart.png`: triangles mark executed entries and crosses mark exits for each maturity.
`trades.csv`: exact trade IDs, signal/entry/exit dates, position, size, reasons,
modeled clean prices per $100 face, swap values, holding months and gross/net P&L.
Execution uses modeled prices plus separately charged spread costs. Rates are percentages.
Matured Treasury redemption is $100; its exit clean-price field is blank after payment.
Open trades have no exit date and retain marked P&L. Set `force_close_final` to true to
liquidate them at the last observation and charge exit costs.

## Assumptions used for this run

- Sizing: `{config.hedge}`; base swap notional ${config.notional:g}; prior-risk sizing {config.risk_sizing}.
  Tenors {list(config.tenors)}; entry threshold ±{config.threshold_bp:g} bp.
- Signal lag: {config.lag_months} month(s). Convergence uses each trade's remaining maturity.
- Monthly overlapping trades; each has its own original coupon and maturity.
- Treasury full bid-ask {config.treasury_full_spread:g} dollars per $100; swap {config.swap_full_spread_bp:g} bp;
  repo {config.repo_full_spread_bp:g} bp. Half widths charged per side; repo drag accrues over time.
- Funding basis {config.funding_spread_bp:g} bp over SOFR; extra adverse funding stress {config.financing_stress_bp:g} bp.
- Treasury haircut {config.treasury_haircut*100:g}%; swap margin {config.swap_initial_margin*100:g}%;
  gross notional/equity limit {config.max_gross_leverage:g}x; repo borrowing cap ${config.max_repo_borrowing:g}.
- Missing short-end curves, hypothetical par bonds and funding assumptions materially affect results.
  The supplied data cannot establish Table 1 replication or actual-security returns. See README.md.
"""
    (output/"summary.md").write_text(summary, encoding="utf-8")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    axes[0].plot(account.date, account.equity, label="Equity including cash interest")
    axes[0].axhline(config.initial_cash, color="gray", linewidth=.7)
    axes[0].set(ylabel="Dollars", title="Funded account")
    axes[0].legend(fontsize=8)
    axes[1].fill_between(account.date, account.drawdown_pct, 0, alpha=.4)
    axes[1].set(ylabel="Drawdown (%)")
    colors = plt.get_cmap("tab10")
    for i, tenor in enumerate(config.tenors):
        dates = [m.date for m in markets]
        spread = [(m.swap.par(tenor*12)-m.treasury.par(tenor*12))*10000 for m in markets]
        axes[2].plot(dates, spread, color=colors(i), linewidth=1, label=f"{tenor}Y modeled spread")
        if not ledger.empty:
            for kind, marker in (("entry", "^"), ("exit", "x")):
                events = ledger[(ledger.tenor==tenor)&(ledger.event==kind)]
                points = pd.Series(spread, index=pd.DatetimeIndex(dates)).reindex(events.date).to_numpy()
                axes[2].scatter(events.date, points, marker=marker, color=colors(i), s=24, zorder=3)
    axes[2].set(ylabel="Basis points", title="Executed trades: triangle = entry, cross = exit (IDs and dates in trades.csv)")
    axes[2].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(config.name.replace("_", " "))
    fig.savefig(output/"chart.png", dpi=150)
    plt.close(fig)
