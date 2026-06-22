"""
Stress-tested variant of backtest_nvda_real.py (real recovered option
premium history), adding two frictions to test robustness of the
straddle-with-hedge results:

- Transaction cost: 0.05% of the underlying price, charged on every
  hedge switch (each Renko-gated flip closes the old future position
  and opens the new one).
- Premium haircut: the real per-leg premium rate is reduced by 0.1
  percentage points (absolute), modelling worse fills / wider spreads
  than the recorded last-trade premium.

Everything else (hedge-switch mechanics, ATM strike-gap top-up, cash
settlement) is identical to backtest_nvda_real.py.
"""
import csv
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import (
    load_prices, compute_atr_proxy, compute_renko_signal, ATR_LEN,
)
from backtest_nvda_real import load_option_periods, nearest_index

TXN_COST_PCT = 0.0005      # 0.05% of price, per hedge switch
PREMIUM_HAIRCUT_PCT = 0.001  # 0.1 percentage points off the real premium rate


def run_stress_backtest(dates, closes, signal, periods):
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
        spot0 = closes[i0]
        strike = p["strike"]
        premium_pct = max(0.0, p["entry_premium"] / p["spot0"] - PREMIUM_HAIRCUT_PCT)

        premium_call = premium_pct * spot0 + max(0.0, spot0 - strike)
        premium_put = premium_pct * spot0 + max(0.0, strike - spot0)
        premium_income = premium_call + premium_put

        hedge_dir = signal[i0]
        hedge_entry = spot0
        hedge_pnl = 0.0
        txn_cost = 0.0
        switches = []

        for i in idxs[1:]:
            price = closes[i]
            sig = signal[i]
            if hedge_dir == 1 and price < strike and sig == -1:
                hedge_pnl += hedge_dir * (price - hedge_entry)
                txn_cost += TXN_COST_PCT * price
                hedge_dir = -1
                hedge_entry = price
                switches.append((dates[i], "long->short", price))
            elif hedge_dir == -1 and price > strike and sig == 1:
                hedge_pnl += hedge_dir * (price - hedge_entry)
                txn_cost += TXN_COST_PCT * price
                hedge_dir = 1
                hedge_entry = price
                switches.append((dates[i], "short->long", price))

        settle = closes[idxs[-1]]
        hedge_pnl += hedge_dir * (settle - hedge_entry)
        hedge_pnl -= txn_cost

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
            "txn_cost": txn_cost,
            "n_switches": len(switches),
            "period_pnl": period_pnl,
            "equity": equity,
        })

    return results


def print_report(results):
    print(f"{'Period':24} {'Spot0':>8} {'Strike':>7} {'Prem%':>6} {'Settle':>8} "
          f"{'Premium':>8} {'OptPnL':>8} {'HedgePnL':>9} {'TxnCost':>7} {'Sw':>3} "
          f"{'PeriodPnL':>10} {'Equity':>10}")
    for r in results:
        print(f"{r['period']:24} {r['spot0']:8.1f} {r['strike']:7.0f} {r['premium_pct']:6.2f} "
              f"{r['settle']:8.1f} {r['premium_income']:8.1f} {r['option_pnl']:8.1f} "
              f"{r['hedge_pnl']:9.1f} {r['txn_cost']:7.2f} {r['n_switches']:3d} "
              f"{r['period_pnl']:10.1f} {r['equity']:10.1f}")

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
    print(f"Avg real premium rate   : {sum(r['premium_pct'] for r in results)/n:.2f}% per leg (after haircut)")
    print(f"Total txn cost ($)      : {sum(r['txn_cost'] for r in results):.1f}")


def plot_equity(results, path, asset="NVDA"):
    xs = [r["expiry_date"] for r in results]
    ys = [r["equity"] for r in results]
    plt.figure(figsize=(11, 5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title(f"Equity curve: ATM short straddle + Renko-hedged future ({asset}, stress: -0.1% premium, 0.05% txn cost)")
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
    parser.add_argument("--out-csv", default="stress_BT/results_nvda_real_stress.csv")
    parser.add_argument("--out-png", default="stress_BT/equity_curve_nvda_real_stress.png")
    args = parser.parse_args()

    dates, closes = load_prices(args.data)
    atr = compute_atr_proxy(closes, ATR_LEN)
    signal = compute_renko_signal(dates, closes, atr)
    periods = load_option_periods(args.options_history)

    results = run_stress_backtest(dates, closes, signal, periods)
    print_report(results)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "init_date", "expiry_date", "spot0", "strike", "premium_pct",
                    "settle", "premium_income", "option_pnl", "hedge_pnl", "txn_cost",
                    "n_switches", "period_pnl", "equity"])
        for r in results:
            w.writerow([r["period"], r["init_date"], r["expiry_date"], r["spot0"], r["strike"],
                        r["premium_pct"], r["settle"], r["premium_income"], r["option_pnl"],
                        r["hedge_pnl"], r["txn_cost"], r["n_switches"], r["period_pnl"], r["equity"]])

    plot_equity(results, args.out_png, args.asset)


if __name__ == "__main__":
    main()
