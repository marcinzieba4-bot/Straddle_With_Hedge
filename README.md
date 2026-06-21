# Straddle With Hedge — ETHUSD / BTCUSD backtest

Monthly ATM short straddle (call + put), hedged with 1 unit of the
underlying future whose side (long/short) is driven by a daily Renko
signal.

## Rules

- Each calendar month, sell ATM call + ATM put. Strike = spot rounded to
  the nearest strike step. Premium per side = a fixed % of spot, plus/minus
  the intrinsic moneyness created by rounding spot to that strike.
- On the initiation day, open 1 unit of future: **long** if the Renko signal
  is up, **short** if down. This is the starting hedge.
- During the month, a Renko flip alone is ignored. The hedge only flips
  when BOTH hold: price crosses the strike (neutral point) AND Renko
  agrees with the new direction.
- At expiry, options are cash-settled against intrinsic value and any open
  hedge leg is closed at the settlement price.
- Renko signal ported from the "Universal Renko Bars by SiddWolf" Pine
  indicator (ATR Based method, multiplier 0.15, tick_trend=2,
  tick_reversal=4, open_offset=0). ATR(14) is approximated from
  close-to-close ranges with Wilder's RMA smoothing (true OHLC ATR isn't
  available from the daily close-only price feed used here).

## Data

Daily closes pulled from Yahoo Finance's chart API, covering 2023-01-01 to
2026-06-21 (~42 months):

- `data/eth_usd_daily.csv` — ETHUSD
- `data/btc_usd_daily.csv` — BTCUSD

## Running

```
pip install matplotlib
python3 backtest.py --data data/eth_usd_daily.csv --premium-rate 0.068 --strike-step 50 --asset ETHUSD --out-csv results_eth.csv --out-png equity_curve_eth.png
python3 backtest.py --data data/btc_usd_daily.csv --premium-rate 0.05 --strike-step 500 --asset BTCUSD --out-csv results_btc.csv --out-png equity_curve_btc.png
```

Outputs a `--out-csv` (per-month P&L breakdown) and `--out-png`
(cumulative P&L chart), and prints a summary table to stdout.

## Result snapshots (42 months, 1 unit notional)

**ETHUSD** (6.8% premium per side, $50 strike step)
- 23/42 winning months
- Total P&L: ~$2,943 (~4.44% of spot per month on average)

**BTCUSD** (5% premium per side, $500 strike step)
- 31/42 winning months
- Total P&L: ~$87,888 (~9.88% of spot per month on average)

This is a simplified, fixed-premium model (not real options market prices)
intended to illustrate the hedge-switching mechanics, not as a production
P&L estimate.
