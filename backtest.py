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
- Renko signal ported from the "Universal Renko Bars by SiddWolf" Pine
  indicator (ATR Based method, multiplier 0.15, tick_trend=2,
  tick_reversal=4, open_offset=0), evaluated once per daily close. On the
  initiation day, open 1 unit of future: long if Renko is up, short if down.
- During the month, a Renko flip alone is ignored. The hedge only flips when
  BOTH hold: price crosses the strike (neutral point) AND Renko agrees with
  the new direction (long->short needs price < strike and Renko down;
  short->long needs price > strike and Renko up).
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
RENKO_ATR_MULT = 0.15     # "ATR 14 Multiplier" default from the Pine indicator
RENKO_TICK_TREND = 2      # ticks needed to continue the trend
RENKO_TICK_REVERSAL = 4   # ticks needed to reverse the trend
RENKO_OPEN_OFFSET = 0


def load_prices(path):
    dates, closes = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            dates.append(datetime.date.fromisoformat(row["date"]))
            closes.append(float(row["close"]))
    return dates, closes


def compute_atr_proxy(closes, length=ATR_LEN):
    """
    True-range proxy from close-only data: |close[t]-close[t-1]|, smoothed
    with Wilder's RMA (matching Pine's ta.atr) since real OHLC isn't
    available from the daily close-only price feed used here.
    """
    tr = [None] + [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    atr = [None] * len(closes)
    seed_idx = length  # first index with `length` TR values (tr[1..length])
    if seed_idx >= len(closes):
        return atr
    atr[seed_idx] = sum(tr[1:seed_idx + 1]) / length
    for i in range(seed_idx + 1, len(closes)):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def get_tick_size(price_ref, atr_value, atr_multiplier=RENKO_ATR_MULT):
    """Port of getEffectiveTickSize() for the 'ATR Based' method."""
    if atr_value is None:
        return price_ref * 0.001
    return atr_value * atr_multiplier


def compute_renko_signal(dates, closes, atr, tick_trend=RENKO_TICK_TREND,
                          tick_reversal=RENKO_TICK_REVERSAL, open_offset=RENKO_OPEN_OFFSET):
    """
    Port of the "Universal Renko Bars" Pine indicator (ATR Based method),
    evaluated once per daily close (equivalent to one bar in Pine). A new
    brick confirms when close crosses bar_max (continuation/reversal up) or
    bar_min (continuation/reversal down); the brick closes exactly at that
    level (not at the raw price), same as the Pine script. Continuation
    needs `tick_trend` ticks, reversal needs `tick_reversal` ticks (ticks are
    asymmetric, as in the original script's defaults of 2 / 4).

    Returns a list of +1/-1/None (None until the first brick confirms).
    """
    n = len(closes)
    signal = [None] * n
    start = next((i for i, a in enumerate(atr) if a is not None), None)
    if start is None:
        return signal

    tick0 = get_tick_size(closes[start], atr[start])
    bar_open = closes[start]
    bar_direction = 0
    bar_max = bar_open + tick_trend * tick0
    bar_min = bar_open - tick_trend * tick0

    for i in range(start + 1, n):
        price = closes[i]
        max_exceeded = price > bar_max
        min_exceeded = price < bar_min

        if max_exceeded or min_exceeded:
            bar_direction = 1 if max_exceeded else -1
            this_close = bar_max if max_exceeded else bar_min
            current_tick = get_tick_size(this_close, atr[i])
            bar_open = this_close - open_offset * current_tick * bar_direction
            if bar_direction > 0:
                bar_max = this_close + tick_trend * current_tick
                bar_min = this_close - tick_reversal * current_tick
            else:
                bar_max = this_close + tick_reversal * current_tick
                bar_min = this_close - tick_trend * current_tick

        signal[i] = bar_direction if bar_direction != 0 else None

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


def run_backtest(dates, closes, signal, premium_rate=PREMIUM_RATE, strike_step=STRIKE_STEP):
    months = monthly_groups(dates)
    results = []
    equity = 0.0

    for idxs in months:
        idxs = [i for i in idxs if signal[i] is not None]
        if len(idxs) < 2:
            continue  # skip incomplete / warm-up month

        i0 = idxs[0]
        spot0 = closes[i0]
        strike = nearest_strike(spot0, strike_step)

        premium_call = premium_rate * spot0 + max(0.0, spot0 - strike)
        premium_put = premium_rate * spot0 + max(0.0, strike - spot0)
        premium_income = premium_call + premium_put

        hedge_dir = signal[i0]            # +1 long, -1 short
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

        month_pnl = option_pnl + hedge_pnl
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
            "month_pnl": month_pnl,
            "equity": equity,
        })

    return results


def print_report(results):
    print(f"{'Month':8} {'Spot0':>9} {'Strike':>7} {'Settle':>9} "
          f"{'Premium':>9} {'OptPnL':>9} {'HedgePnL':>9} {'Switch':>6} "
          f"{'MonthPnL':>10} {'Equity':>10}")
    for r in results:
        print(f"{r['month']:8} {r['spot0']:9.1f} {r['strike']:7.0f} {r['settle']:9.1f} "
              f"{r['premium_income']:9.1f} {r['option_pnl']:9.1f} {r['hedge_pnl']:9.1f} "
              f"{r['n_switches']:6d} {r['month_pnl']:10.1f} {r['equity']:10.1f}")

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


def plot_equity(results, path, asset="ETHUSD"):
    xs = [r["expiry_date"] for r in results]
    ys = [r["equity"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title(f"Equity curve: ATM short straddle + Renko-hedged future ({asset})")
    plt.xlabel("Month")
    plt.ylabel("Cumulative P&L ($, 1 unit notional)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eth_usd_daily.csv")
    parser.add_argument("--premium-rate", type=float, default=PREMIUM_RATE)
    parser.add_argument("--strike-step", type=float, default=STRIKE_STEP)
    parser.add_argument("--asset", default="ETHUSD")
    parser.add_argument("--out-csv", default="results.csv")
    parser.add_argument("--out-png", default="equity_curve.png")
    args = parser.parse_args()

    dates, closes = load_prices(args.data)
    atr = compute_atr_proxy(closes, ATR_LEN)
    signal = compute_renko_signal(dates, closes, atr)
    results = run_backtest(dates, closes, signal, args.premium_rate, args.strike_step)
    print_report(results)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "init_date", "expiry_date", "spot0", "strike", "settle",
                    "premium_income", "option_pnl", "hedge_pnl", "n_switches",
                    "month_pnl", "equity"])
        for r in results:
            w.writerow([r["month"], r["init_date"], r["expiry_date"], r["spot0"],
                        r["strike"], r["settle"], r["premium_income"], r["option_pnl"],
                        r["hedge_pnl"], r["n_switches"], r["month_pnl"], r["equity"]])

    plot_equity(results, args.out_png, args.asset)


if __name__ == "__main__":
    main()
