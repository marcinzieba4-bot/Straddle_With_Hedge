"""
Straddle-with-hedge backtest for the US 10Y Treasury, since there's no
real recovered option-chain data for rates (unlike the single-name
equities). Two things have to be modelled instead of observed:

1. A tradeable "price" series, since the underlying we have is a
   yield (^TNX), not a price. Daily yield changes are converted to
   daily price returns using a fixed modified duration of 8 (per the
   "yield x 8" instruction): price_return = -DURATION * d(yield)/100.
   This price proxy (starting at 100) is what the Renko hedge signal
   and the hedge P&L are computed on, same as every other backtest in
   this repo.

2. A monthly ATM straddle premium, since there's no recovered options
   feed to read it from. It's derived from the MOVE index (ICE BofA
   MOVE, annualized bp yield volatility for ~1m options) at the start
   of each month:
     price_vol_annual = DURATION * (MOVE / 10000)
     premium_pct_per_leg = 0.4 * price_vol_annual * sqrt(1/12)
   (0.4 ~= the standard ATM-straddle-per-leg approximation,
   0.5 * 0.7979 * sigma * sqrt(T), so total straddle ~= 0.8 * S *
   sigma * sqrt(T)). This makes the premium move with the MOVE index
   month to month, as requested, rather than being a fixed rate.

Hedge-switch mechanics (Renko-gated, flips only when price crosses the
strike AND Renko agrees) and the calendar-month structure are
unchanged from backtest.py.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import (
    compute_atr_proxy, compute_renko_signal, nearest_strike,
    monthly_groups, ATR_LEN, STRIKE_STEP,
)

DURATION = 8.0


def load_yield_and_move(yield_path, move_path):
    def load(path):
        d, v = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                d.append(datetime.date.fromisoformat(row["date"]))
                v.append(float(row["close"]))
        return d, v

    y_dates, yields = load(yield_path)
    m_dates, moves = load(move_path)
    move_by_date = dict(zip(m_dates, moves))

    dates, yields_aligned, moves_aligned = [], [], []
    for d, y in zip(y_dates, yields):
        if d in move_by_date:
            dates.append(d)
            yields_aligned.append(y)
            moves_aligned.append(move_by_date[d])
    return dates, yields_aligned, moves_aligned


def build_price_proxy(yields, duration=DURATION):
    """Synthetic price index from yield changes: price_return = -duration * d(yield)/100."""
    prices = [100.0]
    for i in range(1, len(yields)):
        dy_decimal = (yields[i] - yields[i - 1]) / 100.0
        price_return = -duration * dy_decimal
        prices.append(prices[-1] * (1.0 + price_return))
    return prices


def premium_pct_from_move(move_value, duration=DURATION, dte_years=1 / 12):
    price_vol_annual = duration * (move_value / 10000.0)
    return 0.4 * price_vol_annual * math.sqrt(dte_years)


def run_backtest(dates, prices, moves, signal, strike_step=STRIKE_STEP):
    months = monthly_groups(dates)
    results = []
    equity = 0.0

    for idxs in months:
        idxs = [i for i in idxs if signal[i] is not None]
        if len(idxs) < 2:
            continue

        i0 = idxs[0]
        spot0 = prices[i0]
        strike = nearest_strike(spot0, strike_step)
        premium_rate = premium_pct_from_move(moves[i0])

        premium_call = premium_rate * spot0 + max(0.0, spot0 - strike)
        premium_put = premium_rate * spot0 + max(0.0, strike - spot0)
        premium_income = premium_call + premium_put

        hedge_dir = signal[i0]
        hedge_entry = spot0
        hedge_pnl = 0.0
        switches = []

        for i in idxs[1:]:
            price = prices[i]
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

        settle = prices[idxs[-1]]
        hedge_pnl += hedge_dir * (settle - hedge_entry)

        call_payoff = -max(0.0, settle - strike)
        put_payoff = -max(0.0, strike - settle)
        option_pnl = premium_income + call_payoff + put_payoff

        month_pnl = option_pnl + hedge_pnl
        equity += month_pnl

        results.append({
            "month": dates[i0].strftime("%Y-%m"),
            "init_date": dates[i0],
            "expiry_date": dates[idxs[-1]],
            "spot0": spot0,
            "strike": strike,
            "move0": moves[i0],
            "premium_pct": premium_rate * 100,
            "settle": settle,
            "premium_income": premium_income,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "n_switches": len(switches),
            "month_pnl": month_pnl,
            "equity": equity,
        })

    return results


def print_report(results):
    print(f"{'Month':8} {'Spot0':>8} {'Strike':>7} {'MOVE':>6} {'Prem%':>6} {'Settle':>8} "
          f"{'Premium':>8} {'OptPnL':>8} {'HedgePnL':>9} {'Sw':>3} "
          f"{'MonthPnL':>9} {'Equity':>9}")
    for r in results:
        print(f"{r['month']:8} {r['spot0']:8.2f} {r['strike']:7.1f} {r['move0']:6.1f} {r['premium_pct']:6.2f} "
              f"{r['settle']:8.2f} {r['premium_income']:8.3f} {r['option_pnl']:8.3f} "
              f"{r['hedge_pnl']:9.3f} {r['n_switches']:3d} {r['month_pnl']:9.3f} {r['equity']:9.3f}")

    n = len(results)
    if n == 0:
        print("No months simulated.")
        return
    wins = sum(1 for r in results if r["month_pnl"] > 0)
    total_pnl = results[-1]["equity"]
    avg = total_pnl / n
    avg_pct = sum(r["month_pnl"] / r["spot0"] for r in results) / n * 100
    print()
    print(f"Months simulated       : {n}")
    print(f"Winning months         : {wins} ({wins/n*100:.0f}%)")
    print(f"Total P&L (price-index pts) : {total_pnl:.2f}")
    print(f"Avg monthly P&L         : {avg:.3f}")
    print(f"Avg monthly P&L (% spot): {avg_pct:.2f}%")
    print(f"Avg premium rate        : {sum(r['premium_pct'] for r in results)/n:.2f}% per leg")
    print(f"Avg MOVE level          : {sum(r['move0'] for r in results)/n:.1f}")


def plot_equity(results, path):
    xs = [r["expiry_date"] for r in results]
    ys = [r["equity"] for r in results]
    plt.figure(figsize=(11, 5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title("Equity curve: ATM short straddle + Renko-hedged future (US10Y price proxy, MOVE-scaled premium)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative P&L (price-index points, 1 unit notional)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yield-data", default="data/us10y_yield_daily.csv")
    parser.add_argument("--move-data", default="data/move_daily.csv")
    parser.add_argument("--strike-step", type=float, default=1.0)
    parser.add_argument("--out-csv", default="results_us10y.csv")
    parser.add_argument("--out-png", default="equity_curve_us10y.png")
    args = parser.parse_args()

    dates, yields, moves = load_yield_and_move(args.yield_data, args.move_data)
    prices = build_price_proxy(yields)
    atr = compute_atr_proxy(prices, ATR_LEN)
    signal = compute_renko_signal(dates, prices, atr)

    results = run_backtest(dates, prices, moves, signal, args.strike_step)
    print_report(results)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "init_date", "expiry_date", "spot0", "strike", "move0",
                    "premium_pct", "settle", "premium_income", "option_pnl", "hedge_pnl",
                    "n_switches", "month_pnl", "equity"])
        for r in results:
            w.writerow([r["month"], r["init_date"], r["expiry_date"], r["spot0"], r["strike"],
                        r["move0"], r["premium_pct"], r["settle"], r["premium_income"],
                        r["option_pnl"], r["hedge_pnl"], r["n_switches"], r["month_pnl"], r["equity"]])

    plot_equity(results, args.out_png)


if __name__ == "__main__":
    main()
