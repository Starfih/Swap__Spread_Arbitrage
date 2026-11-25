"""Run from anywhere: python backtest.py --config configs/sofr_scenario.json."""
import argparse
from pathlib import Path
import sys
from backtests.engine import Config, load_baseline, sofr_scenario, run
from backtests.report import save_results

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Funded swap-spread backtest")
    parser.add_argument("--config", default="configs/sofr_scenario.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    try:
        config = Config.read(ROOT / args.config)
        if config.source == "baseline":
            path = ROOT / config.market_file
            if not path.exists():
                raise ValueError("LIBOR baseline data is missing. Use the SOFR scenario config; see README.md assumptions.")
            markets = load_baseline(path)
        else:
            markets = sofr_scenario(ROOT / "SSA_Data", config.funding_spread_bp, config.frequency)
        result = run(markets, config, funded=True)
        save_results(result, markets, config, ROOT / args.output)
        print(f"Saved summary.md, trades.csv, equity.csv and chart.png to {ROOT / args.output}")
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Cannot run: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
