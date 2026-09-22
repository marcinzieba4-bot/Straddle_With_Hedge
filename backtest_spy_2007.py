"""SPY 2007 -> 2026: the twenty-year test, VIX-based premiums.

Runs the five strategy variants on SPY only (the one asset with a REAL
vol index over the whole period; sector indices/proxies would be pure
fiction pre-2011, so no EW basket here):

  STRADDLE   - tearsheet config (cc/flip), always on
  GATED-WF   - same, but toxic months (VRP <= expanding daily tercile
               AND close < 200dma) to cash. IMPORTANT: the rule FORM was
               discovered on 2020-2026 data, so 2007-2019 - including
               the GFC - is genuinely out-of-sample for the gate.
  PW-30d     - monthly cash-secured 30-delta put
  TREND      - 12m momentum, long/flat, next-open fills
  COMBO      - (PW-30d + TREND)/2

Data: data/ext2007/ (SPY OHLC + VIX from 2004-01, fetched on first run,
warmup for 200dma / 12m momentum / expanding VRP), trading from 2007-01.

Read the numbers critically:
- premium = VIX x Black-Scholes at 90% of mark; VIX sits above sellable
  ATM IV (worst exactly in panics), so short-vol income is overstated;
- month-end European settlement: intramonth marks (Oct-2008, Aug-2011,
  Feb-2018, Mar-2020, Apr-2025) are invisible - real margin calls and
  forced covers are not modeled, and 2008/2009 fills were far worse than
  the 1bp/leg assumed here;
- no collateral yield (would ADD material T-bill carry 2007-08, 2023+);
- single asset: no diversification, but 237 months gives ~3x the
  statistical power of every earlier window in this repo.
"""
import csv
import datetime
import json
import math
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_v3 import load_ohlc, SPY_COSTS
from backtest_v2 import load_dvol
from backtest_putwrite_trend import (month_frames, put_write, trend,
                                     straddle_same_window)
from regime_analysis import sigma_at, rv_ann
from risk_report_v3 import stats as full_stats

P1, P2 = 1072915200, 1790812800   # 2004-01-01 .. 2026-10-01
DIR = "data/ext2007"
MIN_DAILY_OBS = 126
START = "2007-01"
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")


def fetch():
    os.makedirs(DIR, exist_ok=True)
    for sym, path, ohlc in [("SPY", f"{DIR}/spy_ohlc.csv", True),
                            ("%5EVIX", f"{DIR}/vix_daily.csv", False)]:
        if os.path.exists(path):
            continue
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?"
               f"period1={P1}&period2={P2}&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        res = d["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        rows, seen = [], set()
        for i, t in enumerate(res["timestamp"]):
            day = datetime.datetime.utcfromtimestamp(t).date()
            if day in seen:
                continue
            if ohlc:
                v = (q["open"][i], q["high"][i], q["low"][i], q["close"][i])
                if None in v:
                    continue
                rows.append(f"{day},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f}")
            else:
                if q["close"][i] is None:
                    continue
                rows.append(f"{day},{q['close'][i]:.4f}")
            seen.add(day)
        hdr = "date,open,high,low,close" if ohlc else "date,dvol_close"
        with open(path, "w") as f:
            f.write(hdr + "\n" + "\n".join(rows) + "\n")
        print(path, len(rows), "rows")


if __name__ == "__main__":
    fetch()
    dates, opens, highs, lows, closes = load_ohlc(f"{DIR}/spy_ohlc.csv")
    dvol = load_dvol(f"{DIR}/vix_daily.csv")
    frames = month_frames(dates, opens, closes)
    costs = SPY_COSTS

    st = straddle_same_window(dates, opens, highs, lows, closes, dvol, 1,
                              costs, "cc", "flip")
    pw = put_write(frames, dates, closes, dvol, 1, costs, 0.30)
    tr, expo = trend(frames, dates, opens, closes, costs)

    daily_vrp = []
    vrp_by_day = {}
    for i, d in enumerate(dates):
        iv = sigma_at(dvol, d)
        rv = rv_ann(closes, i)
        if iv is None or rv is None:
            continue
        vrp_by_day[d] = iv - rv
        daily_vrp.append((d, iv - rv))

    gated, toxic = {}, {}
    for label, i0, il in frames:
        if label not in st or i0 < 200 or dates[i0] not in vrp_by_day:
            continue
        hist = [v for (d, v) in daily_vrp if d < dates[i0]]
        if len(hist) < MIN_DAILY_OBS:
            continue
        edge = sorted(hist)[len(hist) // 3]
        below_ma = closes[i0] < sum(closes[i0 - 199:i0 + 1]) / 200
        tox = vrp_by_day[dates[i0]] <= edge and below_ma
        toxic[label] = tox
        gated[label] = 0.0 if tox else st[label]

    months = sorted(m for m in (set(st) & set(pw) & set(tr) & set(gated))
                    if m >= START)
    series = {
        "STRADDLE": [st[m] for m in months],
        "GATED-WF": [gated[m] for m in months],
        "PW-30d": [pw[m] for m in months],
        "TREND": [tr[m] for m in months],
    }
    series["COMBO"] = [(p + t) / 2 for p, t in
                       zip(series["PW-30d"], series["TREND"])]
    VARIANTS = ("STRADDLE", "GATED-WF", "PW-30d", "TREND", "COMBO")

    n_tox = sum(toxic[m] for m in months)
    print(f"\nwindow {months[0]} -> {months[-1]} ({len(months)} months); "
          f"gate fired {n_tox}x "
          f"({', '.join(sorted({m[:4] for m in months if toxic[m]}))})")

    cols = ["n", "win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "VaR95", "CVaR95", "skew", "total_pp"]

    def table(title, sel):
        idx = [i for i, m in enumerate(months) if sel(m)]
        if len(idx) < 6:
            return []
        print(f"\n=== {title} ({len(idx)} months) ===")
        print(f"{'':10}" + "".join(f"{c:>9}" for c in cols))
        out = []
        for v in VARIANTS:
            s = full_stats([series[v][i] for i in idx])
            print(f"{v:10}" + "".join(
                f"{s[c]:9.0f}" if c == "n" else f"{s[c]:9.2f}" for c in cols))
            out.append([title, v] + [round(s[c], 3) for c in cols])
        return out

    rows_csv = []
    rows_csv += table("FULL 2007-2026", lambda m: True)
    rows_csv += table("2007-2019 (gate rule OOS)", lambda m: m < "2020-01")
    rows_csv += table("2007-2009 (GFC)", lambda m: m < "2010-01")
    rows_csv += table("2010-2019", lambda m: "2010-01" <= m < "2020-01")
    rows_csv += table("2020-2026", lambda m: m >= "2020-01")

    with open("results_spy_2007.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "variant"] + cols)
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

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.8, 0.7],
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
    for lbl, color, lw in [("STRADDLE", "#898781", 1.4),
                           ("PW-30d", "#86b6ef", 1.2),
                           ("TREND", "#eb6834", 1.2),
                           ("COMBO", "#1baf7a", 1.6),
                           ("GATED-WF", INK, 2.4)]:
        cum = cumsum(series[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("SPY 2007 → 2026, VIX-priced premiums — gate rule is "
                 "out-of-sample before 2020", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    for lbl, color, lw in [("STRADDLE", "#898781", 1.4),
                           ("COMBO", "#1baf7a", 1.4),
                           ("GATED-WF", INK, 2.0)]:
        ax.plot(dts, dd(cumsum(series[lbl])), color=color, linewidth=lw,
                label=lbl)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[2]
    ax.bar(dts, [1 if toxic[m] else 0 for m in months], width=25,
           color="#e34948", alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("gate", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig("report_spy_2007.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote results_spy_2007.csv, report_spy_2007.png")
