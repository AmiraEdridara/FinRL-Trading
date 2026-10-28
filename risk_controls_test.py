"""
Apply the Adaptive Rotation strategy's DAILY risk controls to its returns.

In the original backtest, fast risk-off cuts and stop-loss exits are only
printed. This script reads the events saved by the patched backtest and
applies them, then compares:

  - Weekly only: the original result (daily controls ignored)
  - Stops only:  stop-loss exits sell the position until the next weekly rebalance
  - Fast only:   fast risk-off cuts invested exposure to (1 - cash_floor),
                 50% by default, until it expires
  - Both:        stops and fast cuts together

Run AFTER a backtest with the patched runner:
    python risk_controls_test.py --start 2022-01-01 --end 2022-12-31
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["date", "cash", "regime"]


def load_prices(data_dir: Path, symbols):
    prices = {}
    for sym in symbols:
        path = data_dir / f"{sym}_daily.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            prices[sym] = df.set_index("date")["close"]
    return pd.DataFrame(prices).sort_index()


def metrics(daily_ret: pd.Series, rf: float):
    equity = (1 + daily_ret).cumprod()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "Cumulative Return": f"{equity.iloc[-1]:.2f}x",
        "Annualized Return": f"{ann_ret:.2%}",
        "Annualized Volatility": f"{ann_vol:.2%}",
        f"Sharpe Ratio (rf {rf:.1%})": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
    }, equity


def main():
    p = argparse.ArgumentParser(description="Apply daily risk controls to Adaptive Rotation returns")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--rf", type=float, default=0.0,
                   help="Annual risk-free rate for Sharpe, e.g. 0.045 (default 0, same as deploy.sh)")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Cost per side in basis points (default 5)")
    p.add_argument("--weights-dir", default="src/strategies/output/weights/adaptive_rotation")
    p.add_argument("--data-dir", default="data/fmp_daily")
    args = p.parse_args()

    wdir = Path(args.weights_dir)
    weights_file = wdir / f"ars_portfolio_weights_{args.start}_to_{args.end}.csv"
    events_file = wdir / f"risk_events_{args.start}_to_{args.end}.csv"
    for f in (weights_file, events_file):
        if not f.exists():
            raise SystemExit(f"Missing {f}\nRe-run deploy.sh for these dates with the patched runner.")

    weekly = pd.read_csv(weights_file, parse_dates=["date"])
    events = pd.read_csv(events_file, parse_dates=["date"])
    symbols = [c for c in weekly.columns if c not in META_COLS]

    prices = load_prices(Path(args.data_dir), symbols)
    prices = prices.loc[weekly["date"].min():weekly["date"].max()]
    asset_ret = prices.pct_change().fillna(0.0)
    days = asset_ret.index
    symbols = [s for s in symbols if s in asset_ret.columns]
    asset_ret = asset_ret[symbols]

    weekly_w = weekly.set_index("date")[symbols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Target weights at each day's CLOSE (they earn the next day's return).
    base_close = weekly_w.reindex(days, method="ffill").fillna(0.0)
    weekly_dates = set(weekly_w.index)
    stops_by_day = {d: str(s).split(";") for d, s in
                    events.loc[events["type"] == "stop_loss"].set_index("date")["symbols"].items()}
    fast_by_day = events.loc[events["type"] == "fast_track"].drop_duplicates("date").set_index("date")

    def apply_controls(use_stops, use_fast):
        close = base_close.copy()
        stopped, cut_until, cut_invested = set(), None, 1.0
        n_stop = n_fast = 0
        for d in days:
            if d in weekly_dates:
                stopped = set()                      # weekly rebalance resets holdings
            if use_stops and d in stops_by_day:
                stopped.update(stops_by_day[d])
                n_stop += 1
            if use_fast and d in fast_by_day.index:
                row = fast_by_day.loc[d]
                cut_until = pd.Timestamp(row["until"])
                floor = float(row["cash_floor"]) if pd.notna(row["cash_floor"]) else 0.5
                cut_invested = 1.0 - floor
                n_fast += 1
            w = close.loc[d].copy()
            for s in stopped:
                if s in w.index:
                    w[s] = 0.0                       # sold, proceeds stay in cash
            if cut_until is not None and d <= cut_until:
                invested = w.sum()
                if invested > cut_invested:
                    w *= cut_invested / invested     # scale down, rest to cash
            close.loc[d] = w
        held = close.shift(1).fillna(0.0)            # weight held during day t
        ret = (held * asset_ret).sum(axis=1)
        # Charge costs only on the extra trades the controls create.
        extra = (close.diff().abs().sum(axis=1)
                 - base_close.diff().abs().sum(axis=1)).clip(lower=0).shift(1).fillna(0.0)
        ret = ret - extra * args.cost_bps / 10_000
        return ret, held, n_stop, n_fast

    variants = {
        "Weekly only": (False, False),
        "Stops only": (True, False),
        "Fast only": (False, True),
        "Both": (True, True),
    }
    results, curves, invested = {}, {}, {}
    for name, (st, fa) in variants.items():
        ret, held, n_stop, n_fast = apply_controls(st, fa)
        results[name], curves[name] = metrics(ret, args.rf)
        invested[name] = held.sum(axis=1).mean()
        if name == "Both":
            counts = (n_stop, n_fast)

    print("\n" + "=" * 87)
    print(f"Risk Controls Test  {args.start} to {args.end}")
    print(f"Stop-loss days: {counts[0]}   Fast risk-off cuts: {counts[1]}   Cost: {args.cost_bps:.0f} bps/side")
    print("=" * 87)
    print(f"{'Metric':<27s}" + "".join(f"{n:>15s}" for n in variants))
    print("-" * 87)
    for k in results["Weekly only"]:
        print(f"{k:<27s}" + "".join(f"{results[n][k]:>15s}" for n in variants))
    print(f"{'Average invested':<27s}" + "".join(f"{invested[n]:>15.1%}" for n in variants))

    out = wdir / f"risk_controls_{args.start}_to_{args.end}.csv"
    pd.DataFrame(curves).to_csv(out)
    print(f"Daily equity curves saved to: {out}")
    print("Note: all columns use daily returns, so 'Weekly only' can differ slightly "
          "from deploy.sh's weekly numbers.")


if __name__ == "__main__":
    main()
