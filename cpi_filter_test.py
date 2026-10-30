"""
CPI risk filter test for the Adaptive Rotation strategy.

Idea: US CPI is released at 8:30 AM New York time, before the market opens,
so the big CPI move happens between the previous close and the CPI-day close.
This script keeps the strategy's weekly weights unchanged, but cuts exposure
to a fraction (default 50%) at the close BEFORE each CPI release and restores
it at the CPI-day close. It then compares the result with the unfiltered
strategy, computed the same way.

Run AFTER a normal backtest, so the weights file exists:
    python cpi_filter_test.py --start 2022-01-01 --end 2022-12-31
    python cpi_filter_test.py --start 2025-01-01 --end 2026-10-27 --scale 0.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# US CPI release dates, from the BLS archive:
# https://www.bls.gov/bls/news-release/cpi.htm
# (October 2025 CPI was never published, so there is no November 2025 release.)
CPI_DATES = pd.to_datetime([
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11",
    "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13",
    "2022-11-10", "2022-12-13",
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
    "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
    "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24",
    "2025-12-18",
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14",
])

META_COLS = ["date", "cash", "regime"]


def load_prices(data_dir: Path, symbols):
    prices = {}
    for sym in symbols:
        path = data_dir / f"{sym}_daily.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            prices[sym] = df.set_index("date")["close"]
        else:
            print(f"  Warning: no price file for {sym}")
    return pd.DataFrame(prices).sort_index()


def daily_weights(weights_df: pd.DataFrame, trading_days: pd.DatetimeIndex):
    """Hold each weekly weight vector from its decision date until the next one."""
    w = weights_df.set_index("date").drop(columns=["cash", "regime"], errors="ignore")
    w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # The weight chosen at a decision-date close earns the NEXT day's return.
    w = w.reindex(trading_days, method="ffill").shift(1)
    return w.fillna(0.0)


def metrics(daily_ret: pd.Series):
    equity = (1 + daily_ret).cumprod()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1]
    ann_ret = total ** (1 / years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0  # same definition as deploy.sh (no risk-free rate)
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "Cumulative Return": f"{total:.2f}x",
        "Annualized Return": f"{ann_ret:.2%}",
        "Annualized Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
    }, equity


def main():
    p = argparse.ArgumentParser(description="Test a CPI-day exposure cut on Adaptive Rotation weights")
    p.add_argument("--start", required=True, help="Backtest start date used in deploy.sh, e.g. 2022-01-01")
    p.add_argument("--end", required=True, help="Backtest end date used in deploy.sh, e.g. 2022-12-31")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Exposure kept on CPI days: 0.5 = half, 0.0 = all cash (default 0.5)")
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="Trading cost per side in basis points on the weight traded (default 5)")
    p.add_argument("--weights-dir", default="src/strategies/output/weights/adaptive_rotation")
    p.add_argument("--data-dir", default="data/fmp_daily")
    args = p.parse_args()

    weights_file = Path(args.weights_dir) / f"ars_portfolio_weights_{args.start}_to_{args.end}.csv"
    if not weights_file.exists():
        raise SystemExit(f"Weights file not found: {weights_file}\nRun deploy.sh for these dates first.")

    weights_df = pd.read_csv(weights_file, parse_dates=["date"])
    symbols = [c for c in weights_df.columns if c not in META_COLS]
    prices = load_prices(Path(args.data_dir), symbols)

    start, end = weights_df["date"].min(), weights_df["date"].max()
    prices = prices.loc[start:end]
    asset_ret = prices.pct_change().fillna(0.0)
    w = daily_weights(weights_df, asset_ret.index)[asset_ret.columns.intersection(symbols)]
    asset_ret = asset_ret[w.columns]

    base_ret = (w * asset_ret).sum(axis=1)

    # CPI filter: on CPI day the position is cut at the previous close.
    cpi_days = asset_ret.index.intersection(CPI_DATES)
    is_cpi = asset_ret.index.isin(cpi_days)
    exposure = np.where(is_cpi, args.scale, 1.0)
    filt_ret = base_ret * exposure

    # Cost: sell (1 - scale) of the invested weight before CPI, buy it back after.
    invested = w.sum(axis=1)
    traded = invested * (1 - args.scale) * 2
    cost = np.where(is_cpi, traded * args.cost_bps / 10_000, 0.0)
    filt_ret = filt_ret - cost

    base_m, base_eq = metrics(base_ret)
    filt_m, filt_eq = metrics(pd.Series(filt_ret, index=base_ret.index))

    print("\n" + "=" * 62)
    print(f"CPI Filter Test  {args.start} to {args.end}")
    print(f"CPI days in period: {len(cpi_days)}   Exposure on CPI days: {args.scale:.0%}   "
          f"Cost: {args.cost_bps:.0f} bps/side")
    print("=" * 62)
    print(f"{'Metric':<25s}{'No filter':>15s}{'CPI filter':>15s}")
    print("-" * 55)
    for k in base_m:
        print(f"{k:<25s}{base_m[k]:>15s}{filt_m[k]:>15s}")

    # Evidence check: are CPI days actually more volatile for this portfolio?
    cpi_abs = base_ret[is_cpi].abs().mean()
    other_abs = base_ret[~is_cpi].abs().mean()
    print("\nAverage absolute daily move (no filter):")
    print(f"  CPI days:   {cpi_abs:.2%}")
    print(f"  Other days: {other_abs:.2%}")
    print(f"  Ratio:      {cpi_abs / other_abs:.2f}x" if other_abs > 0 else "")
    print(f"  Total return earned on CPI days: {(1 + base_ret[is_cpi]).prod() - 1:.2%}")

    out = Path(args.weights_dir) / f"cpi_filter_{args.start}_to_{args.end}.csv"
    pd.DataFrame({"no_filter": base_eq, "cpi_filter": filt_eq}).to_csv(out)
    print(f"\nDaily equity curves saved to: {out}")
    print("Note: daily returns use the weekly weights held fixed each day, so the 'No filter' "
          "column can differ slightly from deploy.sh's weekly numbers.")


if __name__ == "__main__":
    main()
