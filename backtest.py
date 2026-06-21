"""
Backtest: monthly ATM short straddle on ETHUSD, hedged with 1 unit of future
whose side (long/short) is driven by a daily Renko signal.

Rules
-----
- Each calendar month: sell ATM call + ATM put (strike = nearest $50 to spot
  on initiation day). Premium per side = 6.8% of spot, adjusted by the
  intrinsic moneyness created by rounding to the nearest strike (i.e. you are
  not selling exactly at spot, so the fixed premium is topped up/down by the
  strike/spot gap).
- Renko signal (ATR-based brick size, computed from close-to-close ranges)
  is evaluated daily. On the initiation day, open 1 unit of future: long if
  Renko is up, short if Renko is down.
- During the month, the hedge flips whenever the Renko signal flips
  (long <-> short), regardless of where price sits relative to the strike.
- At month end, options are cash-settled against intrinsic value, and any
  open hedge leg is closed at the settlement price.
"""
import datetime
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREMIUM_RATE = 0.068      # 6.8% per side, fixed
STRIKE_STEP = 50          # ETH strikes quoted in $50 increments (Deribit-like)
ATR_LEN = 14
TAIL_PUT_OTM = 0.15       # tail-hedge put struck 15% below spot
TAIL_PUT_PREMIUM_RATE = 0.04  # fixed cost, 4% of spot


def load_prices(path):
    dates, closes = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            dates.append(datetime.date.fromisoformat(row["date"]))
            closes.append(float(row["close"]))
    return dates, closes


def compute_atr_proxy(closes, length=ATR_LEN):
    """True-range proxy from close-only data: |close[t]-close[t-1]|."""
    tr = [None] + [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    atr = [None] * len(closes)
    for i in range(length, len(closes)):
        window = tr[i - length + 1:i + 1]
        atr[i] = sum(window) / length
    return atr


def compute_renko_signal(dates, closes, atr):
    """
    Daily Renko direction using an ATR-based brick size that is re-evaluated
    each time a new brick forms. Returns a list of +1/-1/None (None during
    ATR warm-up).
    """
    signal = [None] * len(closes)
    direction = None
    ref = None
    start = next(i for i, a in enumerate(atr) if a is not None)
    ref = closes[start]
    for i in range(start, len(closes)):
        brick = atr[i]
        if brick is None or brick <= 0:
            signal[i] = direction
            continue
        if direction is None:
            if closes[i] - ref >= brick:
                direction = 1
                ref = ref + brick * math.floor((closes[i] - ref) / brick)
            elif ref - closes[i] >= brick:
                direction = -1
                ref = ref - brick * math.floor((ref - closes[i]) / brick)
        elif direction == 1:
            if closes[i] - ref >= brick:
                ref = ref + brick * math.floor((closes[i] - ref) / brick)
            elif ref - closes[i] >= brick:
                direction = -1
                ref = ref - brick * math.floor((ref - closes[i]) / brick)
        else:
            if ref - closes[i] >= brick:
                ref = ref - brick * math.floor((ref - closes[i]) / brick)
            elif closes[i] - ref >= brick:
                direction = 1
                ref = ref + brick * math.floor((closes[i] - ref) / brick)
        signal[i] = direction
    return signal


def nearest_strike(spot, step=STRIKE_STEP):
    return round(spot / step) * step


def monthly_groups(dates):
    """Group day indices by (year, month)."""
    groups = {}
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        groups.setdefault(key, []).append(i)
    return [groups[k] for k in sorted(groups)]


def run_backtest(dates, closes, signal):
    months = monthly_groups(dates)
    results = []
    equity = 0.0

    for idxs in months:
        idxs = [i for i in idxs if signal[i] is not None]
        if len(idxs) < 2:
            continue  # skip incomplete / warm-up month

        i0 = idxs[0]
        spot0 = closes[i0]
        strike = nearest_strike(spot0)

        premium_call = PREMIUM_RATE * spot0 + max(0.0, spot0 - strike)
        premium_put = PREMIUM_RATE * spot0 + max(0.0, strike - spot0)
        premium_income = premium_call + premium_put

        tail_put_strike = nearest_strike(spot0 * (1 - TAIL_PUT_OTM))
        tail_put_cost = TAIL_PUT_PREMIUM_RATE * spot0

        hedge_dir = signal[i0]            # +1 long, -1 short
        hedge_entry = spot0
        hedge_pnl = 0.0
        switches = []

        for i in idxs[1:]:
            price = closes[i]
            sig = signal[i]
            if sig != hedge_dir:
                hedge_pnl += hedge_dir * (price - hedge_entry)
                label = "long->short" if hedge_dir == 1 else "short->long"
                hedge_dir = sig
                hedge_entry = price
                switches.append((dates[i], label, price))

        settle = closes[idxs[-1]]
        hedge_pnl += hedge_dir * (settle - hedge_entry)

        call_payoff = -max(0.0, settle - strike)
        put_payoff = -max(0.0, strike - settle)
        option_pnl = premium_income + call_payoff + put_payoff

        tail_put_payoff = max(0.0, tail_put_strike - settle)
        tail_put_pnl = tail_put_payoff - tail_put_cost

        month_pnl = option_pnl + hedge_pnl + tail_put_pnl
        equity += month_pnl

        results.append({
            "month": dates[i0].strftime("%Y-%m"),
            "init_date": dates[i0],
            "expiry_date": dates[idxs[-1]],
            "spot0": spot0,
            "strike": strike,
            "settle": settle,
            "premium_income": premium_income,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "n_switches": len(switches),
            "switches": switches,
            "tail_put_strike": tail_put_strike,
            "tail_put_pnl": tail_put_pnl,
            "month_pnl": month_pnl,
            "equity": equity,
        })

    return results


def print_report(results):
    print(f"{'Month':8} {'Spot0':>9} {'Strike':>7} {'Settle':>9} "
          f"{'Premium':>9} {'OptPnL':>9} {'HedgePnL':>9} {'Switch':>6} "
          f"{'TailPut':>9} {'MonthPnL':>10} {'Equity':>10}")
    for r in results:
        print(f"{r['month']:8} {r['spot0']:9.1f} {r['strike']:7.0f} {r['settle']:9.1f} "
              f"{r['premium_income']:9.1f} {r['option_pnl']:9.1f} {r['hedge_pnl']:9.1f} "
              f"{r['n_switches']:6d} {r['tail_put_pnl']:9.1f} {r['month_pnl']:10.1f} {r['equity']:10.1f}")

    total_pnl = results[-1]["equity"] if results else 0.0
    n = len(results)
    wins = sum(1 for r in results if r["month_pnl"] > 0)
    print()
    print(f"Months simulated : {n}")
    print(f"Winning months   : {wins} ({wins/n*100:.0f}%)")
    print(f"Total P&L ($)    : {total_pnl:.1f}")
    if results:
        avg = total_pnl / n
        print(f"Avg monthly P&L  : {avg:.1f}")
        print(f"Avg monthly P&L %: {avg / results[0]['spot0'] * 100:.2f}% of initial spot")


def plot_equity(results, path):
    xs = [r["expiry_date"] for r in results]
    ys = [r["equity"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title("Equity curve: ATM short straddle + Renko hedge + tail put (ETHUSD)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative P&L ($, 1 unit notional)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    dates, closes = load_prices("data/eth_usd_daily.csv")
    atr = compute_atr_proxy(closes, ATR_LEN)
    signal = compute_renko_signal(dates, closes, atr)
    results = run_backtest(dates, closes, signal)
    print_report(results)

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "init_date", "expiry_date", "spot0", "strike", "settle",
                    "premium_income", "option_pnl", "hedge_pnl", "n_switches",
                    "tail_put_strike", "tail_put_pnl", "month_pnl", "equity"])
        for r in results:
            w.writerow([r["month"], r["init_date"], r["expiry_date"], r["spot0"],
                        r["strike"], r["settle"], r["premium_income"], r["option_pnl"],
                        r["hedge_pnl"], r["n_switches"], r["tail_put_strike"],
                        r["tail_put_pnl"], r["month_pnl"], r["equity"]])

    plot_equity(results, "equity_curve.png")


if __name__ == "__main__":
    main()
