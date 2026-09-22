"""Short monthly call + fast long/flat momentum on SPY, 2007-2026.

Structure: sell one monthly call every month (ATM or 30-delta, priced at
90% of the Black-Scholes mark at VIX, settled month-end at intrinsic);
hold 1 unit of SPY long whenever a fast momentum signal is up, flat
otherwise (signal read at the close, position changed at the NEXT open,
1bp per fill; the position persists across month boundaries).

What it is: momentum long -> covered call (~short put); momentum flat ->
naked short call, which earns in the falling markets where momentum sits
out. Directional premium selling steered by trend.

Momentum signals compared (the user asked to check Renko vs moving
averages — this is a comparison grid, not a tuned pick):
    renko  - the repo's Universal Renko signal, cc ATR, long iff up
    p>20d  - close above its 20-day MA
    p>50d  - close above its 50-day MA
    10x50  - 10-day MA above 50-day MA

References: covered call always (BXM-like), naked short call always,
buy & hold. Same caveats as everything in this repo: VIX > sellable ATM
IV, month-end settlement hides intramonth marks, no collateral yield.
Naked short calls additionally carry unlimited-loss margin treatment
that this model does not price.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest import nearest_strike
from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, bs_price, delta_strike, compute_atr,
                         compute_renko_signal, SHORT_MARK_FRAC, SPY_COSTS)
from backtest_putwrite_trend import month_frames
from regime_analysis import sigma_at
from risk_report_v3 import stats as full_stats

DIR = "data/ext2007"
START = "2007-01"
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")


def ma(closes, n):
    out = [None] * len(closes)
    s = 0.0
    for i, c in enumerate(closes):
        s += c
        if i >= n:
            s -= closes[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def make_signals(dates, highs, lows, closes):
    atr = compute_atr(closes, source="cc")
    renko = compute_renko_signal(closes, atr)
    m10, m20, m50 = ma(closes, 10), ma(closes, 20), ma(closes, 50)
    sig = {"renko": [1 if s == 1 else 0 if s is not None else None
                     for s in renko],
           "p>20d": [None if m is None else int(c > m)
                     for c, m in zip(closes, m20)],
           "p>50d": [None if m is None else int(c > m)
                     for c, m in zip(closes, m50)],
           "10x50": [None if (a is None or b is None) else int(a > b)
                     for a, b in zip(m10, m50)]}
    return sig


def run(dates, opens, closes, dvol, frames, signal, call_delta,
        hedge_on=True, always_long=False):
    """Monthly returns (% of that month's init close)."""
    costs = SPY_COSTS
    out = {}
    pos = None
    for label, i0, il in frames:
        if label < START:
            # still evolve the position so it is correct at START
            for i in range(max(i0, 1), il + 1):
                s = signal[i - 1]
                if s is not None:
                    pos = 1 if (always_long or s) else 0
            continue
        spot0 = closes[i0]
        sigma = sigma_at(dvol, dates[i0])
        if sigma is None:
            continue
        sigma /= 100.0
        days = max((dates[il] - dates[i0]).days, 1)
        T = days / 365.0
        if call_delta is None:
            K = nearest_strike(spot0, 1)
        else:
            K = delta_strike(spot0, sigma, T, call_delta, "c", 1)
        prem = SHORT_MARK_FRAC * bs_price(spot0, K, sigma, T, "c")
        pnl = prem - (costs["opt_fee_bps"] / 1e4) * spot0

        if hedge_on:
            if pos is None:
                s = signal[i0 - 1] if i0 > 0 else None
                pos = 1 if (always_long or (s == 1)) else 0
            for i in range(i0 + 1, il + 1):
                s = signal[i - 1]
                want = 1 if (always_long or (s == 1 and s is not None)) \
                    else 0
                if want != pos:
                    pnl += pos * (opens[i] - closes[i - 1])
                    pnl += want * (closes[i] - opens[i])
                    pnl -= (costs["fut_fee_bps"] / 1e4) * opens[i]
                    pos = want
                else:
                    pnl += pos * (closes[i] - closes[i - 1])
        pnl -= max(0.0, closes[il] - K)
        out[label] = pnl / spot0 * 100.0
    return out


if __name__ == "__main__":
    dates, opens, highs, lows, closes = load_ohlc(f"{DIR}/spy_ohlc.csv")
    dvol = load_dvol(f"{DIR}/vix_daily.csv")
    frames = month_frames(dates, opens, closes)
    signals = make_signals(dates, highs, lows, closes)

    variants = {}
    for sname, sig in signals.items():
        for dlt, dname in [(None, "ATM"), (0.30, "30d")]:
            variants[f"{sname}/{dname}"] = run(dates, opens, closes, dvol,
                                               frames, sig, dlt)
    variants["covered-call ATM"] = run(dates, opens, closes, dvol, frames,
                                       signals["p>50d"], None,
                                       always_long=True)
    variants["naked-call ATM"] = run(dates, opens, closes, dvol, frames,
                                     signals["p>50d"], None, hedge_on=False)
    # buy & hold, monthly
    bh = {}
    for label, i0, il in frames:
        if label >= START:
            bh[label] = (closes[il] - closes[i0]) / closes[i0] * 100
    variants["buy&hold"] = bh

    months = sorted(set.intersection(*(set(v) for v in variants.values())))
    cols = ["win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "CVaR95", "skew", "total_pp"]
    print(f"window {months[0]} -> {months[-1]} ({len(months)} months)\n")
    print(f"{'variant':18}" + "".join(f"{c:>9}" for c in cols))
    rows_csv = []
    for name, v in variants.items():
        s = full_stats([v[m] for m in months])
        print(f"{name:18}" + "".join(f"{s[c]:9.2f}" for c in cols))
        rows_csv.append([name] + [round(s[c], 3) for c in cols])

    with open("results_callmom.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
        w.writerows(rows_csv)

    # chart: best-in-class curves vs references
    dts = [datetime.date(int(m[:4]), int(m[5:]), 1) for m in months]

    def cumsum(v):
        out, r = [], 0.0
        for m in months:
            r += v[m]
            out.append(r)
        return out

    def dd(cum):
        peak, out = -1e18, []
        for x in cum:
            peak = max(peak, x)
            out.append(x - peak)
        return out

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.7],
                                          "hspace": 0.11})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
    show = [("buy&hold", "#c3c2b7", 1.2), ("covered-call ATM", "#86b6ef", 1.3),
            ("renko/30d", "#eb6834", 1.5), ("p>50d/30d", "#1baf7a", 1.5),
            ("10x50/ATM", "#e87ba4", 1.2), ("10x50/30d", INK, 2.4)]
    ax = axes[0]
    for lbl, color, lw in show:
        cum = cumsum(variants[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Short monthly ATM call + long/flat momentum, SPY "
                 "2007 → 2026", color=INK, fontsize=11.5, loc="left",
                 pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    for lbl, color, lw in [("covered-call ATM", "#86b6ef", 1.3),
                           ("renko/30d", "#eb6834", 1.4),
                           ("10x50/30d", INK, 2.0)]:
        ax.plot(dts, dd(cumsum(variants[lbl])), color=color, linewidth=lw,
                label=lbl)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig("report_callmom.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote results_callmom.csv, report_callmom.png")
