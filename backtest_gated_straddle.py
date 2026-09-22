"""Regime-gated straddle: in-sample vs walk-forward, plus a bad-regime
replacement leg. Window 2020-01 -> 2026-09 on data/ext (five assets, EW).

Gate ("toxic" month, per asset, decided at initiation with no lookahead
in the walk-forward version):
    VRP = vol index - trailing 21d realized vol, at init
    toxic  <=>  VRP below its tercile edge  AND  close < 200dma
  - IS  gate: tercile edge from the FULL sample (the in-sample diagnostic
    from regime_analysis.py, kept for comparison).
  - WF  gate: tercile edge = 33rd percentile of the asset's own DAILY VRP
    history up to the init day (expanding window, >=126 prior daily obs).

Variants (equal-weight across assets; non-traded sleeves hold cash at 0%):
  STRADDLE        - tearsheet configs, always on (reference)
  GATED-IS        - straddle, toxic months to cash (in-sample edges)
  GATED-WF        - same, expanding edges (honest version)
  GATED-WF+PUT    - same, but toxic months BUY the ATM put instead of
                    sitting in cash: pay 110% of the Black-Scholes mark
                    (LONG_MARK_FRAC) + fees, collect intrinsic at expiry.
                    Chosen ex-ante on economics (cheap IV + downtrend =
                    underpriced gamma), not screened from a menu.

Prints full stats (Sharpe, Sortino, max DD, Calmar, VaR/CVaR, skew),
gate diagnostics (fire rate, WF-vs-IS agreement), writes
results_gated.csv and report_gated.png.
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
from backtest_v3 import (load_ohlc, bs_price, LONG_MARK_FRAC,
                         SPY_COSTS, GOLD_COSTS, ETF_COSTS)
from backtest_putwrite_trend import month_frames, straddle_same_window
from regime_analysis import sigma_at, rv_ann
from risk_report_v3 import stats as full_stats

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
MIN_DAILY_OBS = 126
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")


def long_atm_put(dates, closes, i0, il, dvol, step, costs):
    """Buy 1 ATM put at init (110% of mark), settle at intrinsic."""
    spot0, settle = closes[i0], closes[il]
    days = max((dates[il] - dates[i0]).days, 1)
    sigma = sigma_at(dvol, dates[i0])
    if sigma is None:
        return None
    K = nearest_strike(spot0, step)
    cost = LONG_MARK_FRAC * bs_price(spot0, K, sigma / 100.0, days / 365.0,
                                     "p")
    fees = (costs["opt_fee_bps"] / 1e4) * spot0
    pnl = max(0.0, K - settle) - cost - fees
    return pnl / spot0 * 100.0


if __name__ == "__main__":
    per_asset = {}
    for name, ohlc_csv, vol_csv, step, costs, atr_src, rule in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(vol_csv)
        frames = month_frames(dates, opens, closes)
        st = straddle_same_window(dates, opens, highs, lows, closes, dvol,
                                  step, costs, atr_src, rule)

        # daily VRP series for expanding tercile edges
        daily_vrp = []
        vrp_by_day = {}
        for i, d in enumerate(dates):
            iv = sigma_at(dvol, d)
            rv = rv_ann(closes, i)
            if iv is None or rv is None:
                continue
            vrp_by_day[d] = iv - rv
            daily_vrp.append((d, iv - rv))

        recs = {}
        month_vrps = []
        for label, i0, il in frames:
            if label not in st or i0 < 200:
                continue
            d0 = dates[i0]
            if d0 not in vrp_by_day:
                continue
            vrp = vrp_by_day[d0]
            below_ma = closes[i0] < sum(closes[i0 - 199:i0 + 1]) / 200
            hist = [v for (d, v) in daily_vrp if d < d0]
            wf_edge = (sorted(hist)[len(hist) // 3]
                       if len(hist) >= MIN_DAILY_OBS else None)
            put = long_atm_put(dates, closes, i0, il, dvol, step, costs)
            recs[label] = {
                "straddle": st[label], "put": put, "vrp": vrp,
                "below_ma": below_ma,
                "wf_toxic": (wf_edge is not None and vrp <= wf_edge
                             and below_ma),
            }
            month_vrps.append((label, vrp))
        # in-sample edge from the traded months (as in regime_analysis)
        is_edge = sorted(v for _, v in month_vrps)[len(month_vrps) // 3]
        for label in recs:
            recs[label]["is_toxic"] = (recs[label]["vrp"] <= is_edge
                                       and recs[label]["below_ma"])
        per_asset[name] = recs

    months = sorted(set.intersection(*(set(r) for r in per_asset.values())))
    print(f"window {months[0]} -> {months[-1]} ({len(months)} months)")

    n_obs = len(months) * len(per_asset)
    n_is = sum(per_asset[a][m]["is_toxic"] for a in per_asset for m in months)
    n_wf = sum(per_asset[a][m]["wf_toxic"] for a in per_asset for m in months)
    agree = sum(per_asset[a][m]["wf_toxic"] == per_asset[a][m]["is_toxic"]
                for a in per_asset for m in months)
    both = sum(per_asset[a][m]["wf_toxic"] and per_asset[a][m]["is_toxic"]
               for a in per_asset for m in months)
    print(f"gate fires: IS {n_is}/{n_obs}, WF {n_wf}/{n_obs}; "
          f"agreement {100*agree/n_obs:.0f}%, overlap {both}/{max(n_is,1)} "
          f"of IS-toxic months also WF-toxic\n")

    def ew(kind):
        out = []
        for m in months:
            vals = []
            for a in per_asset:
                r = per_asset[a][m]
                if kind == "STRADDLE":
                    vals.append(r["straddle"])
                elif kind == "GATED-IS":
                    vals.append(0.0 if r["is_toxic"] else r["straddle"])
                elif kind == "GATED-WF":
                    vals.append(0.0 if r["wf_toxic"] else r["straddle"])
                elif kind == "GATED-WF+PUT":
                    if r["wf_toxic"]:
                        vals.append(r["put"] if r["put"] is not None else 0.0)
                    else:
                        vals.append(r["straddle"])
            out.append(sum(vals) / len(vals))
        return out

    VARIANTS = ("STRADDLE", "GATED-IS", "GATED-WF", "GATED-WF+PUT")
    series = {v: ew(v) for v in VARIANTS}

    cols = ["n", "win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "best", "VaR95", "CVaR95", "skew", "total_pp"]
    print(f"{'':14}" + "".join(f"{c:>9}" for c in cols))
    rows_csv = []
    for v in VARIANTS:
        s = full_stats(series[v])
        print(f"{v:14}" + "".join(
            f"{s[c]:9.0f}" if c == "n" else f"{s[c]:9.2f}" for c in cols))
        rows_csv.append([v] + [round(s[c], 3) for c in cols])

    # WF-toxic months only: cash vs long put (does the replacement earn?)
    tox_put, tox_str = [], []
    for a in per_asset:
        for m in months:
            r = per_asset[a][m]
            if r["wf_toxic"] and r["put"] is not None:
                tox_put.append(r["put"])
                tox_str.append(r["straddle"])
    if tox_put:
        print(f"\nWF-toxic asset-months (n={len(tox_put)}): "
              f"straddle avg {sum(tox_str)/len(tox_str):+.2f}%/mo, "
              f"long ATM put avg {sum(tox_put)/len(tox_put):+.2f}%/mo, "
              f"put win rate {100*sum(1 for x in tox_put if x>0)/len(tox_put):.0f}%")

    with open("results_gated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
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

    wf_frac = [sum(per_asset[a][m]["wf_toxic"] for a in per_asset) / 5
               for m in months]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.7, 0.8],
                                          "hspace": 0.12})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax = axes[0]
    for lbl, color, lw in [("STRADDLE", "#898781", 1.5),
                           ("GATED-IS", "#86b6ef", 1.2),
                           ("GATED-WF", "#2a78d6", 2.0),
                           ("GATED-WF+PUT", INK, 2.6)]:
        cum = cumsum(series[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Regime-gated straddle, equal-weight, 2020 → 2026 — "
                 "walk-forward gate uses expanding VRP terciles",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    for lbl, color, lw in [("STRADDLE", "#898781", 1.5),
                           ("GATED-WF", "#2a78d6", 1.6),
                           ("GATED-WF+PUT", INK, 2.2)]:
        ax.plot(dts, dd(cumsum(series[lbl])), color=color, linewidth=lw,
                label=lbl)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[2]
    ax.bar(dts, wf_frac, width=22, color="#e34948", alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel("WF gate\n(frac. assets)", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.savefig("report_gated.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("wrote results_gated.csv, report_gated.png")
