# Straddle With Hedge — ETHUSD backtest

Monthly ATM short straddle (call + put) on ETHUSD, hedged with 1 unit of
ETH future whose side (long/short) is driven by a daily Renko signal.

## Rules

- Each calendar month, sell ATM call + ATM put. Strike = spot rounded to the
  nearest $50. Premium per side = 6.8% of spot (fixed), plus/minus the
  intrinsic moneyness created by rounding spot to that strike.
- On the initiation day, open 1 unit of future: **long** if the Renko signal
  is up, **short** if down. This is the starting hedge.
- During the month, the hedge flips whenever the Renko signal flips
  (long ↔ short), regardless of where price sits relative to the strike.
- At expiry, options are cash-settled against intrinsic value and any open
  hedge leg is closed at the settlement price.
- The Renko signal uses an ATR(14)-based brick size, recomputed from
  close-to-close ranges (true OHLC ATR isn't available from the daily
  close-only price feed used here).

## Data

`data/eth_usd_daily.csv` — ETHUSD daily closes pulled from Yahoo Finance's
chart API, covering 2023-01-01 to 2026-06-21 (~42 months).

## Running

```
pip install matplotlib
python3 backtest.py
```

Outputs `results.csv` (per-month P&L breakdown) and `equity_curve.png`
(cumulative P&L chart), and prints a summary table to stdout.

## Result snapshot (42 months, 1 unit ETH notional)

- 24/42 winning months
- Total P&L: ~$2,435 (~3.83% of spot per month on average)

This is a simplified, fixed-premium model (not real options market prices)
intended to illustrate the hedge-switching mechanics, not as a production
P&L estimate.
