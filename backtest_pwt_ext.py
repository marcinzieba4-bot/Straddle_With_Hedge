"""Extended-window (2020 -> present) run of put-write / trend / combo vs
the hedged straddle, on data/ext (see fetch_ext.py). This is the decisive
test the 2023+ window could not provide: it contains the COVID crash
(Feb-Mar 2020) and the 2022 bear market, so the short-vol sleeves finally
pay their bill and the trend sleeve finally gets something to do.

Same sleeves and costs as backtest_putwrite_trend.py; straddle reference
uses each asset's tearsheet config. Sub-period stats are printed for
2020, 2021, 2022 and 2023+. Caveat inherited from the monthly model:
settlement is month-end European with monthly strike resets, so
intramonth paths (Apr-2025 -21% V-shape, Mar-2020 mid-month lows) are
invisible — real short options would have been margin-called on marks
the model never sees. Long-window numbers here are still OPTIMISTIC for
every short-vol sleeve.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_v2 import load_dvol
from backtest_v3 import load_ohlc, SPY_COSTS, GOLD_COSTS, ETF_COSTS
from backtest_putwrite_trend import (month_frames, put_write, trend,
                                     straddle_same_window, stats, fmt)

ASSETS = [
    ("SPY", "data/ext/spy_ohlc.csv", "data/ext/vix_daily.csv", 1, SPY_COSTS,
     "cc", "flip"),
    ("XLK", "data/ext/xlk_ohlc.csv", "data/ext/xlk_vol.csv", 1, ETF_COSTS,
     "cc", "flip"),
    ("XLV", "data/ext/xlv_ohlc.csv", "data/ext/xlv_vol.csv", 1, ETF_COSTS,
     "cc", "cross"),
    ("XLY", "data/ext/xly_ohlc.csv", "data/ext/xly_vol.csv", 1, ETF_COSTS,
     "ohlc", "itm"),
    ("GOLD", "data/ext/gold_ohlc.csv", "data/ext/gvz_daily.csv", 5,
     GOLD_COSTS, "cc", "flip"),
]
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")

SUBPERIODS = [("2020 (COVID)", "2020-01", "2020-12"),
              ("2021", "2021-01", "2021-12"),
              ("2022 (bear)", "2022-01", "2022-12"),
              ("2023+ (bull)", "2023-01", "2099-12")]


if __name__ == "__main__":
    per_asset, straddle = {}, {}
    for name, ohlc_csv, vol_csv, step, costs, atr_src, rule in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(vol_csv)
        frames = month_frames(dates, opens, closes)
        tr, expo = trend(frames, dates, opens, closes, costs)
        per_asset[name] = {
            "PW-30d": put_write(frames, dates, closes, dvol, step, costs, 0.30),
            "TREND": tr, "expo": expo,
        }
        straddle[name] = straddle_same_window(
            dates, opens, highs, lows, closes, dvol, step, costs,
            atr_src, rule)

    months = None
    for name in per_asset:
        keys = (set(per_asset[name]["TREND"]) & set(per_asset[name]["PW-30d"])
                & set(straddle[name]))
        months = keys if months is None else months & keys
    months = sorted(months)
    print(f"window: {months[0]} -> {months[-1]} ({len(months)} months)\n")

    ew = {}
    for lbl in ("PW-30d", "TREND", "COMBO", "STRADDLE"):
        cols = []
        for name in per_asset:
            if lbl == "COMBO":
                s = [(per_asset[name]["PW-30d"][m] +
                      per_asset[name]["TREND"][m]) / 2 for m in months]
            elif lbl == "STRADDLE":
                s = [straddle[name][m] for m in months]
            else:
                s = [per_asset[name][lbl][m] for m in months]
            cols.append(s)
        ew[lbl] = [sum(c) / len(c) for c in zip(*cols)]

    rows_csv = []
    print("=== EQUAL-WEIGHT, full window ===")
    for lbl in ("PW-30d", "TREND", "COMBO", "STRADDLE"):
        s = stats(ew[lbl])
        print("  " + fmt(lbl, s))
        rows_csv.append(["full", lbl] + [round(s[k], 3) for k in
                        ("n", "win", "avg", "sd", "sharpe", "mdd",
                         "worst", "total")])
    for pname, m0, m1 in SUBPERIODS:
        idx = [i for i, m in enumerate(months) if m0 <= m <= m1]
        if len(idx) < 3:
            continue
        print(f"\n=== EQUAL-WEIGHT, {pname} ({len(idx)} months) ===")
        for lbl in ("PW-30d", "TREND", "COMBO", "STRADDLE"):
            sub = [ew[lbl][i] for i in idx]
            s = stats(sub)
            print("  " + fmt(lbl, s))
            rows_csv.append([pname, lbl] + [round(s[k], 3) for k in
                            ("n", "win", "avg", "sd", "sharpe", "mdd",
                             "worst", "total")])

    print("\ntrend exposure (long % of months): " + ", ".join(
        f"{n} {100*sum(per_asset[n]['expo'][m] for m in months)/len(months):.0f}%"
        for n in per_asset))

    with open("results_pwt_ext.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "sleeve", "n", "win%", "avg%/mo", "sd",
                    "sharpe", "maxDD_pp", "worst", "total_pp"])
        w.writerows(rows_csv)

    dts = [datetime.date(int(m[:4]), int(m[5:]), 1) for m in months]

    def cumsum(r):
        out, run = [], 0.0
        for x in r:
            run += x
            out.append(run)
        return out

    def dd(cum):
        peak, out = -1e18, []
        for x in cum:
            peak = max(peak, x)
            out.append(x - peak)
        return out

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.8), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.7],
                                          "hspace": 0.12})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        for m0, m1 in [("2020-02", "2020-04"), ("2022-01", "2022-10")]:
            ax.axvspan(datetime.date(int(m0[:4]), int(m0[5:]), 1),
                       datetime.date(int(m1[:4]), int(m1[5:]), 28),
                       color="#e34948", alpha=0.06, linewidth=0)
    ax = axes[0]
    for lbl, color, lw in [("PW-30d", "#2a78d6", 1.4),
                           ("TREND", "#eb6834", 1.4),
                           ("STRADDLE", "#898781", 1.6),
                           ("COMBO", INK, 2.6)]:
        cum = cumsum(ew[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8.5, fontweight="bold",
                    va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Equal-weight cumulative return, 2020 → 2026 — shaded: "
                 "COVID crash, 2022 bear", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    for lbl, color, lw in [("PW-30d", "#2a78d6", 1.2),
                           ("STRADDLE", "#898781", 1.6),
                           ("COMBO", INK, 2.2)]:
        ax.plot(dts, dd(cumsum(ew[lbl])), color=color, linewidth=lw,
                label=lbl)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.savefig("report_pwt_ext.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("wrote results_pwt_ext.csv, report_pwt_ext.png")
