"""VRP-switched call+momentum on SPY, 2007-2026.

At each month's initiation (walk-forward, expanding daily-VRP terciles,
>=126 prior obs — the signal validated OOS earlier in this repo):

  VRP rich (top tercile)  -> COMBO: sell the monthly 30-delta call and
                             run the 10x50 momentum long/flat hedge.
  otherwise ("vol cheap") -> TREND only: momentum long/flat, no call,
                             with an optional STOP-LOSS.

Stop-loss (trend leg only): trailing exit at the next open when the
close falls k x ATR14 below the highest close since entry (or a fixed
trailing % of that peak). After a stop-out, re-entry requires a FRESH
10x50 up-cross (the signal must turn down and cross up again) — no
extra re-entry parameter. Widths {2xATR, 3xATR, 5%} are reported as a
grid next to no-stop, for the trend leg standalone and inside the
switch, so the effect of the stop is visible rather than optimized.

References: combo always (best variant from backtest_callmom.py) and
trend always. Usual caveats: VIX-priced calls (overstated - see the
x0.90/x0.85 sensitivity in backtest_callmom.py), month-end settlement,
no collateral yield.
"""
import csv
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, bs_price, delta_strike, compute_atr,
                         SHORT_MARK_FRAC, SPY_COSTS)
from backtest_putwrite_trend import month_frames
from backtest_callmom import make_signals
from regime_analysis import sigma_at, rv_ann
from risk_report_v3 import stats as full_stats

DIR = "data/ext2007"
START = "2007-01"
MIN_DAILY_OBS = 126
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")


def run(dates, opens, closes, atr, dvol, frames, sig, rich_at,
        mode_rule, stop=None):
    """mode_rule: 'switch' | 'combo' | 'trend'. stop: None | ('atr',k) |
    ('pct',p). Returns {month: ret%}, mode log."""
    costs = SPY_COSTS
    out, modes = {}, {}
    pos, peak, stopped = 0, None, False
    for label, i0, il in frames:
        sell_call = (mode_rule == "combo" or
                     (mode_rule == "switch" and rich_at.get(label, False)))
        trend_mode = not sell_call or mode_rule == "trend"
        if mode_rule == "trend":
            sell_call = False
        use_stop = stop is not None and not sell_call
        pnl, spot0 = 0.0, closes[i0]
        K = prem = None
        if label >= START and sell_call:
            sigma = sigma_at(dvol, dates[i0])
            if sigma is not None:
                sigma /= 100.0
                T = max((dates[il] - dates[i0]).days, 1) / 365.0
                K = delta_strike(spot0, sigma, T, 0.30, "c", 1)
                prem = SHORT_MARK_FRAC * bs_price(spot0, K, sigma, T, "c")
                pnl += prem - (costs["opt_fee_bps"] / 1e4) * spot0
        for i in range(max(i0, 1) + 1, il + 1):
            s = sig[i - 1]
            if s == 0:
                stopped = False          # signal down: stop resets
            want = 1 if (s == 1 and not (use_stop and stopped)) else 0
            if use_stop and pos == 1 and peak is not None and atr[i - 1]:
                trigger = (peak - stop[1] * atr[i - 1] if stop[0] == "atr"
                           else peak * (1 - stop[1]))
                if closes[i - 1] < trigger:
                    want, stopped = 0, True
            if want != pos:
                pnl += pos * (opens[i] - closes[i - 1])
                pnl += want * (closes[i] - opens[i])
                pnl -= (costs["fut_fee_bps"] / 1e4) * opens[i]
                pos = want
                peak = closes[i] if pos == 1 else None
            else:
                pnl += pos * (closes[i] - closes[i - 1])
            if pos == 1:
                peak = max(peak, closes[i]) if peak else closes[i]
        if label >= START and K is not None:
            pnl -= max(0.0, closes[il] - K)
        if label >= START:
            out[label] = pnl / spot0 * 100.0
            modes[label] = "COMBO" if sell_call else "TREND"
    return out, modes


if __name__ == "__main__":
    dates, opens, highs, lows, closes = load_ohlc(f"{DIR}/spy_ohlc.csv")
    dvol = load_dvol(f"{DIR}/vix_daily.csv")
    frames = month_frames(dates, opens, closes)
    sig = make_signals(dates, highs, lows, closes)["10x50"]
    atr = compute_atr(closes, highs, lows, source="ohlc")

    daily_vrp = []
    for i, d in enumerate(dates):
        iv = sigma_at(dvol, d)
        rv = rv_ann(closes, i)
        if iv is not None and rv is not None:
            daily_vrp.append((d, iv - rv))
    rich_at = {}
    for label, i0, il in frames:
        hist = [v for (d, v) in daily_vrp if d < dates[i0]]
        cur = next((v for (d, v) in reversed(daily_vrp)
                    if d <= dates[i0]), None)
        if len(hist) >= MIN_DAILY_OBS and cur is not None:
            rich_at[label] = cur > sorted(hist)[2 * len(hist) // 3]

    STOPS = [None, ("atr", 2), ("atr", 3), ("pct", 0.05)]

    def sname(st):
        return ("nostop" if st is None else
                f"{st[1]}xATR" if st[0] == "atr" else f"{st[1]:.0%}")

    variants = {}
    variants["COMBO always"], _ = run(dates, opens, closes, atr, dvol,
                                      frames, sig, rich_at, "combo")
    for st in STOPS:
        variants[f"TREND {sname(st)}"], _ = run(
            dates, opens, closes, atr, dvol, frames, sig, rich_at,
            "trend", st)
    switch_modes = None
    for st in STOPS:
        variants[f"SWITCH {sname(st)}"], m = run(
            dates, opens, closes, atr, dvol, frames, sig, rich_at,
            "switch", st)
        switch_modes = m

    months = sorted(set.intersection(*(set(v) for v in variants.values())))
    n_combo = sum(1 for m in months if switch_modes[m] == "COMBO")
    print(f"window {months[0]} -> {months[-1]} ({len(months)} months); "
          f"switch in COMBO {n_combo} months "
          f"({100*n_combo/len(months):.0f}%), TREND rest\n")

    cols = ["win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "CVaR95", "skew", "total_pp"]
    print(f"{'variant':16}" + "".join(f"{c:>9}" for c in cols))
    rows_csv = []
    for name, v in variants.items():
        s = full_stats([v[m] for m in months])
        print(f"{name:16}" + "".join(f"{s[c]:9.2f}" for c in cols))
        rows_csv.append([name] + [round(s[c], 3) for c in cols])

    # per-mode attribution for the headline switch
    for st_lbl in ("nostop", "3xATR"):
        v = variants[f"SWITCH {st_lbl}"]
        for mode in ("COMBO", "TREND"):
            sub = [v[m] for m in months if switch_modes[m] == mode]
            print(f"SWITCH {st_lbl} {mode} months: n={len(sub)} "
                  f"avg {sum(sub)/len(sub):+.2f}%/mo "
                  f"win {100*sum(1 for x in sub if x>0)/len(sub):.0f}%")

    with open("results_vrp_switch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
        w.writerows(rows_csv)

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

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.6, 0.6],
                                          "hspace": 0.11})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax = axes[0]
    for lbl, color, lw in [("COMBO always", "#86b6ef", 1.3),
                           ("TREND nostop", "#eb6834", 1.3),
                           ("SWITCH nostop", "#1baf7a", 1.6),
                           ("SWITCH 3xATR", INK, 2.4)]:
        cum = cumsum(variants[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("VRP-switched call+momentum, SPY 2007 → 2026 — rich VRP "
                 "sells the call, cheap VRP runs trend only", color=INK,
                 fontsize=11.5, loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    for lbl, color, lw in [("COMBO always", "#86b6ef", 1.3),
                           ("SWITCH nostop", "#1baf7a", 1.4),
                           ("SWITCH 3xATR", INK, 2.0)]:
        ax.plot(dts, dd(cumsum(variants[lbl])), color=color, linewidth=lw,
                label=lbl)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[2]
    ax.bar(dts, [1 if switch_modes[m] == "COMBO" else 0 for m in months],
           width=25, color="#2a78d6", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("COMBO", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig("report_vrp_switch.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote results_vrp_switch.csv, report_vrp_switch.png")
