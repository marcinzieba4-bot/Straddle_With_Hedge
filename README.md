# Straddle With Hedge — ETHUSD backtest

Monthly ATM short straddle (call + put) on ETHUSD, hedged with 1 unit of
ETH future whose side (long/short) is driven by a daily Renko signal.

## Rules

- Each calendar month, sell ATM call + ATM put. Strike = spot rounded to the
  nearest $50. Premium per side = 6.8% of spot (fixed), plus/minus the
  intrinsic moneyness created by rounding spot to that strike.
- On the initiation day, open 1 unit of future: **long** if the Renko signal
  is up, **short** if down. This is the starting hedge.
- During the month, a Renko flip alone is ignored.
- The hedge only switches sides when **both** hold:
  - long → short: price < strike (neutral point) **and** Renko is down
  - short → long: price > strike (neutral point) **and** Renko is up
- At expiry, options are cash-settled against intrinsic value and any open
  hedge leg is closed at the settlement price.
- The Renko signal uses an ATR(14)-based brick size, recomputed from
  close-to-close ranges (true OHLC ATR isn't available from the daily
  close-only price feed used here).

## Data

`data/eth_usd_daily.csv` — ETHUSD daily closes pulled from CoinGecko's
public market_chart API. The free tier caps historical daily data at 365
days, so the backtest covers ~12 months (2025-06-22 to 2026-06-21).

## Running

```
pip install matplotlib
python3 backtest.py
```

Outputs `results.csv` (per-month P&L breakdown) and `equity_curve.png`
(cumulative P&L chart), and prints a summary table to stdout.

## Result snapshot (12 months, 1 unit ETH notional)

- 8/12 winning months
- Total P&L: ~$579 (~1.84% of spot per month on average)

This is a simplified, fixed-premium model (not real options market prices)
intended to illustrate the hedge-switching mechanics, not as a production
P&L estimate.
