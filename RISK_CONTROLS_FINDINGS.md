# Do the daily risk controls help? (Adaptive Rotation)

## The finding

In the original backtest, the daily risk controls (fast risk-off cuts and stop-loss
exits) are only printed. The performance numbers come from the weekly weights alone,
so the controls never affect returns.

I saved every daily risk event during the backtest, applied them to the returns, and
compared four versions across three very different markets.

## Method

- **Weekly only:** the original strategy (daily controls ignored)
- **Stops only:** a stop-loss exit sells that asset until the next weekly rebalance
- **Fast only:** a fast risk-off trigger cuts invested exposure to 50% for 10 days
- **Both:** stops and fast cuts together

Daily returns, 5 bps cost per side on the extra trades, cash earns 0%.
Sharpe uses a period-appropriate risk-free rate (2% for 2022, 5% for 2023 to 2024,
4% for 2025 to 2026).

## Results: Sharpe ratio

| Period | Market | Weekly only | Stops only | Fast only | Both |
|---|---|---|---|---|---|
| 2022 | Bear, choppy | 0.35 | **0.37** | 0.22 | 0.24 |
| 2023 to 2024 | Strong bull | **1.55** | 1.42 | 1.29 | 1.15 |
| 2025 to 2026 | Sharp crashes | 1.26 | **1.41** | 1.22 | 1.27 |
| **Average** | | 1.05 | **1.07** | 0.91 | 0.89 |

## Results: maximum drawdown

| Period | Weekly only | Stops only | Fast only | Both |
|---|---|---|---|---|
| 2022 | -16.52% | -16.44% | -14.24% | -13.52% |
| 2023 to 2024 | -17.45% | -15.97% | -18.38% | -17.14% |
| 2025 to 2026 | -13.30% | -10.24% | -10.20% | -9.01% |

## Confirmation in the real backtest (`--stops-only`)

The stops-only mode was then built into the backtest itself (weekly statistics,
no risk-free rate, as in the original report):

| Period | Version | Return | Sharpe | Max drawdown |
|---|---|---|---|---|
| 2022 | Original | 8.15% | 0.45 | -15.23% |
| 2022 | Stops only | 7.99% | 0.45 | -15.15% |
| 2023 to 2024 | Original | 41.18% | 1.73 | -15.93% |
| 2023 to 2024 | Stops only | 38.43% | 1.62 | -13.78% |
| 2025 to 2026 | Original | 26.77% | 1.64 | -10.00% |
| 2025 to 2026 | Stops only | 27.44% | 1.71 | -9.67% |
| **Average** | Original | | **1.27** | |
| **Average** | Stops only | | **1.26** | |

## Conclusions

1. **Stop-losses reduce drawdown at roughly the same Sharpe.** Their worst fall was
   smaller in every period, while the average Sharpe was essentially unchanged. They
   cost return in the 2023 to 2024 bull market and added it in the 2025 to 2026 crashes.
2. **Fast risk-off cuts hurt.** They lowered the Sharpe ratio in all three periods.
   They sell half the portfolio after a sharp drop and then miss the rebound.
3. **Recommendation:** for a smoother ride, run with stop-losses applied and fast cuts off:
   `./deploy.sh --strategy adaptive_rotation --mode backtest --stops-only`
   Never enable the fast cuts as currently designed.

## Limitations

- Only three test periods, so the evidence is suggestive rather than conclusive.
- Cash earns 0%, which slightly penalises versions that hold more cash (mostly fast cuts).
- Asset groups and settings were chosen by the original authors with hindsight, so all
  versions may look better in backtests than they would live.
- The next step is out-of-sample paper trading from October 2026.

## Scripts

- `risk_controls_test.py`: applies the saved risk events and prints the four versions
- `cpi_filter_test.py`: tests cutting exposure around US CPI releases
