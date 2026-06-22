"""
Combine all per-ticker real-period straddle-with-hedge backtests
(results_*_real.csv) into one equal-weighted monthly P&L index.

Why % return, equal-weighted, instead of summing raw dollar P&L:
each per-ticker backtest is run on 1 share notional, and spot prices
across tickers span ~$30 to >$1000. Summing raw period_pnl across
tickers would just reflect which stock happens to be expensive, not a
meaningful combined return. Converting each ticker's period_pnl to a
% of spot0, then equal-weighting (1/N) across tickers that have a
period expiring in a given calendar month, gives every ticker the
same notional weight regardless of share price.

Steps:
- Load every results_*_real.csv, derive ticker from the filename.
- pct_return = period_pnl / spot0 for each period.
- Bucket periods by the calendar month of their expiry_date (where
  the option P&L crystallizes). If a ticker has >1 period expiring in
  the same month, sum its pct returns for that month (sequential
  periods compound additively over the month).
- For each month, average the per-ticker monthly pct returns across
  all tickers with data that month (equal weight, simple mean).
- Compound monthly returns into a cumulative index, base 100.
"""
import csv
import glob
import os
import re
import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ticker_monthly_returns(path):
    """Return {year_month: summed pct_return} for one ticker's results CSV."""
    monthly = defaultdict(float)
    with open(path) as f:
        for row in csv.DictReader(f):
            spot0 = float(row["spot0"])
            period_pnl = float(row["period_pnl"])
            expiry = datetime.date.fromisoformat(row["expiry_date"])
            ym = (expiry.year, expiry.month)
            monthly[ym] += period_pnl / spot0
    return monthly


def main():
    paths = sorted(glob.glob("results_*_real.csv"))
    ticker_monthly = {}
    for path in paths:
        m = re.match(r"results_(.+)_real\.csv", os.path.basename(path))
        ticker = m.group(1).upper()
        ticker_monthly[ticker] = load_ticker_monthly_returns(path)

    all_months = sorted({ym for monthly in ticker_monthly.values() for ym in monthly})

    rows = []
    index_value = 100.0
    for ym in all_months:
        rets = [monthly[ym] for monthly in ticker_monthly.values() if ym in monthly]
        n = len(rets)
        ew_return = sum(rets) / n
        index_value *= (1.0 + ew_return)
        rows.append({
            "month": f"{ym[0]:04d}-{ym[1]:02d}",
            "n_tickers": n,
            "ew_return_pct": ew_return * 100.0,
            "index_value": index_value,
        })

    with open("results_combined_real_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month", "n_tickers", "ew_return_pct", "index_value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'Month':9} {'N':>3} {'Return%':>8} {'Index':>10}")
    for r in rows:
        print(f"{r['month']:9} {r['n_tickers']:3d} {r['ew_return_pct']:8.2f} {r['index_value']:10.2f}")

    n_months = len(rows)
    wins = sum(1 for r in rows if r["ew_return_pct"] > 0)
    total_return = (rows[-1]["index_value"] / 100.0 - 1.0) * 100.0
    print()
    print(f"Tickers combined        : {len(ticker_monthly)}")
    print(f"Months in index         : {n_months}")
    print(f"Winning months          : {wins} ({wins/n_months*100:.0f}%)")
    print(f"Total index return      : {total_return:.1f}%")
    print(f"Final index value       : {rows[-1]['index_value']:.2f} (base 100)")

    xs = [datetime.date(int(r["month"][:4]), int(r["month"][5:7]), 1) for r in rows]
    ys = [r["index_value"] for r in rows]
    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, marker="o", markersize=3)
    plt.axhline(100, color="grey", linewidth=0.8)
    plt.title(f"Equal-weighted straddle-with-hedge P&L index ({len(ticker_monthly)} tickers, real option premiums)")
    plt.xlabel("Month")
    plt.ylabel("Index value (base 100)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve_combined_real_index.png", dpi=130)


if __name__ == "__main__":
    main()
