"""
NVDA straddle-with-hedge backtest using REAL recovered option premium
history (data/nvda_options_raw/NVDA_history_summary.csv) instead of a
fixed assumed premium rate.

For each real option observation:
- period = [observation_date, expiry] exactly as it traded (~25-28 DTE),
  not a calendar month.
- strike = the real strike reported by the data collector (already the
  ATM choice made at entry), so the ATM/moneyness top-up is the actual
  gap between that real strike and the real entry spot, not a rounded
  approximation.
- premium per leg = the real per-leg premium % (entry_premium / spot at
  observation) recovered from the put-leg dataset, applied symmetrically
  to both call and put legs (no real call-leg data survived; the
  recovered optionsDataCall set is a duplicate-of-put bug, see README).

Hedge-switch mechanics (Renko-gated, flips only when price crosses the
strike AND Renko agrees) are unchanged from backtest.py.
"""
import csv
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import (
    load_prices, compute_atr_proxy, compute_renko_signal, ATR_LEN,
)


def load_option_periods(path):
    periods = []
    with open(path) as f:
        for row in csv.DictReader(f):
            periods.append({
                "obs_date": datetime.date.fromisoformat(row["observation_date"]),
                "expiry": datetime.date.fromisoformat(row["expiry"]),
                "spot0": float(row["stock_price"]),
                "strike": float(row["strike"]),
                "entry_premium": float(row["entry_premium"]),
            })
    periods.sort(key=lambda p: p["obs_date"])
    return periods


def nearest_index(dates, target):
    """Index of the trading day on/after target (first date >= target)."""
    for i, d in enumerate(dates):
        if d >= target:
            return i
    return None


def run_real_backtest(dates, closes, signal, periods):
    results = []
    equity = 0.0

    for p in periods:
        i0 = nearest_index(dates, p["obs_date"])
        i1 = nearest_index(dates, p["expiry"])
        if i0 is None or i1 is None or i1 <= i0:
            continue
        idxs = [i for i in range(i0, i1 + 1) if signal[i] is not None]
        if len(idxs) < 2:
            continue

        i0 = idxs[0]
        spot0 = closes[i0]                      # actual price series spot on entry day
        strike = p["strike"]                     # real strike chosen at entry (true ATM pick)
        premium_pct = p["entry_premium"] / p["spot0"]  # real per-leg premium rate

        premium_call = premium_pct * spot0 + max(0.0, spot0 - strike)
        premium_put = premium_pct * spot0 + max(0.0, strike - spot0)
        premium_income = premium_call + premium_put

        hedge_dir = signal[i0]
        hedge_entry = spot0
        hedge_pnl = 0.0
        switches = []

        for i in idxs[1:]:
            price = closes[i]
            sig = signal[i]
            if hedge_dir == 1 and price < strike and sig == -1:
                hedge_pnl += hedge_dir * (price - hedge_entry)
                hedge_dir = -1
                hedge_entry = price
                switches.append((dates[i], "long->short", price))
            elif hedge_dir == -1 and price > strike and sig == 1:
                hedge_pnl += hedge_dir * (price - hedge_entry)
                hedge_dir = 1
                hedge_entry = price
                switches.append((dates[i], "short->long", price))

        settle = closes[idxs[-1]]
        hedge_pnl += hedge_dir * (settle - hedge_entry)

        call_payoff = -max(0.0, settle - strike)
        put_payoff = -max(0.0, strike - settle)
        option_pnl = premium_income + call_payoff + put_payoff

        period_pnl = option_pnl + hedge_pnl
        equity += period_pnl

        results.append({
            "period": f"{dates[i0]}_{dates[idxs[-1]]}",
            "init_date": dates[i0],
            "expiry_date": dates[idxs[-1]],
            "spot0": spot0,
            "strike": strike,
            "premium_pct": premium_pct * 100,
            "settle": settle,
            "premium_income": premium_income,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "n_switches": len(switches),
            "period_pnl": period_pnl,
            "equity": equity,
        })

    return results


def print_report(results):
    print(f"{'Period':24} {'Spot0':>8} {'Strike':>7} {'Prem%':>6} {'Settle':>8} "
          f"{'Premium':>8} {'OptPnL':>8} {'HedgePnL':>9} {'Sw':>3} "
          f"{'PeriodPnL':>10} {'Equity':>10}")
    for r in results:
        print(f"{r['period']:24} {r['spot0']:8.1f} {r['strike']:7.0f} {r['premium_pct']:6.2f} "
              f"{r['settle']:8.1f} {r['premium_income']:8.1f} {r['option_pnl']:8.1f} "
              f"{r['hedge_pnl']:9.1f} {r['n_switches']:3d} {r['period_pnl']:10.1f} {r['equity']:10.1f}")

    n = len(results)
    if n == 0:
        print("No periods simulated.")
        return
    wins = sum(1 for r in results if r["period_pnl"] > 0)
    total_pnl = results[-1]["equity"]
    avg = total_pnl / n
    avg_pct = sum(r["period_pnl"] / r["spot0"] for r in results) / n * 100
    print()
    print(f"Periods simulated     : {n}")
    print(f"Winning periods        : {wins} ({wins/n*100:.0f}%)")
    print(f"Total P&L ($)           : {total_pnl:.1f}")
    print(f"Avg period P&L ($)      : {avg:.1f}")
    print(f"Avg period P&L (% spot) : {avg_pct:.2f}%")
    print(f"Avg real premium rate   : {sum(r['premium_pct'] for r in results)/n:.2f}% per leg")


def plot_equity(results, path, asset="NVDA"):
    xs = [r["expiry_date"] for r in results]
    ys = [r["equity"] for r in results]
    plt.figure(figsize=(11, 5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title(f"Equity curve: ATM short straddle + Renko-hedged future ({asset}, real option premiums)")
    plt.xlabel("Period expiry")
    plt.ylabel("Cumulative P&L ($, 1 unit notional)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nvda_daily.csv")
    parser.add_argument("--options-history", default="data/nvda_options_raw/NVDA_history_summary.csv")
    parser.add_argument("--asset", default="NVDA")
    parser.add_argument("--out-csv", default="results_nvda_real.csv")
    parser.add_argument("--out-png", default="equity_curve_nvda_real.png")
    args = parser.parse_args()

    dates, closes = load_prices(args.data)
    atr = compute_atr_proxy(closes, ATR_LEN)
    signal = compute_renko_signal(dates, closes, atr)
    periods = load_option_periods(args.options_history)

    results = run_real_backtest(dates, closes, signal, periods)
    print_report(results)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "init_date", "expiry_date", "spot0", "strike", "premium_pct",
                    "settle", "premium_income", "option_pnl", "hedge_pnl", "n_switches",
                    "period_pnl", "equity"])
        for r in results:
            w.writerow([r["period"], r["init_date"], r["expiry_date"], r["spot0"], r["strike"],
                        r["premium_pct"], r["settle"], r["premium_income"], r["option_pnl"],
                        r["hedge_pnl"], r["n_switches"], r["period_pnl"], r["equity"]])

    plot_equity(results, args.out_png, args.asset)


if __name__ == "__main__":
    main()
