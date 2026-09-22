"""Put-writing + trend overlay — the clean two-premia version of what the
hedged straddle accidentally mixes (see the attribution/placebo audit:
most straddle P&L was trend beta; always-long + short straddle = 2x
put-write and beat the Renko version on SPY/XLY).

Sleeves, per asset (SPY, XLK, XLV, XLY, GOLD), 1-unit notional:

- PW-ATM / PW-30d: each calendar month sell one cash-secured put (ATM
  strike, or the 30-delta strike) at 90% of the Black-Scholes mark using
  the asset's vol index at initiation (same premium model, mark haircut
  and option fees as backtest_v3 — including its known optimistic bias:
  index vol > sellable ATM IV, and collateral yield is ignored, which for
  a real cash-secured put would ADD ~4%/yr of T-bill carry).
- TREND: canonical 12-month time-series momentum, long/flat: hold 1 unit
  long for month m if the prior month-end close is above the close 12
  months before, else flat. Entry/exit at the month's first open,
  futures/ETF fill fee on each position change.
- COMBO: (PW-30d + TREND) / 2.

The 12-month lookback makes 2024-02 the first tradable month, so the
hedged-straddle tearsheet configs are re-run on the SAME window for the
head-to-head. Outputs results_pwt.csv, report_pwt.png.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest import nearest_strike, monthly_groups
from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, compute_atr, compute_renko_signal,
                         run_month, bs_price, delta_strike,
                         SHORT_MARK_FRAC, SPY_COSTS, GOLD_COSTS, ETF_COSTS)

ASSETS = [
    ("SPY", "data/spy_ohlc.csv", "data/vix_daily.csv", 1, SPY_COSTS,
     "cc", "flip", "#2a78d6"),
    ("XLK", "data/xlk_ohlc.csv", "data/xlk_ivproxy.csv", 1, ETF_COSTS,
     "cc", "flip", "#eb6834"),
    ("XLV", "data/xlv_ohlc.csv", "data/xlv_ivproxy.csv", 1, ETF_COSTS,
     "cc", "cross", "#1baf7a"),
    ("XLY", "data/xly_ohlc.csv", "data/xly_ivproxy.csv", 1, ETF_COSTS,
     "ohlc", "itm", "#eda100"),
    ("GOLD", "data/gold_ohlc.csv", "data/gvz_daily.csv", 5, GOLD_COSTS,
     "cc", "flip", "#e87ba4"),
]
MOM_LOOKBACK = 12   # months
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")
POS, NEG = "#2a78d6", "#e34948"


def sigma_at(dvol, day):
    for back in range(6):
        v = dvol.get((day - datetime.timedelta(days=back)).isoformat())
        if v is not None:
            return v / 100.0
    return None


def month_frames(dates, opens, closes):
    """[(label, i0, ilast)] per calendar month."""
    out = []
    for idxs in monthly_groups(dates):
        if len(idxs) < 2:
            continue
        out.append((dates[idxs[0]].strftime("%Y-%m"), idxs[0], idxs[-1]))
    return out


def put_write(frames, dates, closes, dvol, step, costs, target_delta=None):
    """Monthly short put; ATM if target_delta is None. ret % of spot0."""
    out = {}
    for label, i0, il in frames:
        spot0, settle = closes[i0], closes[il]
        days = max((dates[il] - dates[i0]).days, 1)
        T = days / 365.0
        sigma = sigma_at(dvol, dates[i0])
        if sigma is None:
            continue
        if target_delta:
            K = delta_strike(spot0, sigma, T, target_delta, "p", step)
        else:
            K = nearest_strike(spot0, step)
        premium = SHORT_MARK_FRAC * bs_price(spot0, K, sigma, T, "p")
        fees = (costs["opt_fee_bps"] / 1e4) * spot0
        pnl = premium - max(0.0, K - settle) - fees
        out[label] = pnl / spot0 * 100.0
    return out


def trend(frames, dates, opens, closes, costs):
    """12m momentum long/flat; signal at prior month end, fill next open."""
    out, expo = {}, {}
    prev_pos = 0
    for m in range(MOM_LOOKBACK, len(frames)):
        label, i0, il = frames[m]
        now = closes[frames[m - 1][2]]
        then = closes[frames[m - MOM_LOOKBACK][2]]
        pos = 1 if now > then else 0
        spot0 = closes[i0]
        pnl = pos * (closes[il] - opens[i0])
        if pos != prev_pos:
            pnl -= (costs["fut_fee_bps"] / 1e4) * opens[i0]
        prev_pos = pos
        out[label] = pnl / spot0 * 100.0
        expo[label] = pos
    return out, expo


def straddle_same_window(dates, opens, highs, lows, closes, dvol, step,
                         costs, atr_src, rule):
    atr = compute_atr(closes, highs, lows, source=atr_src)
    sig = compute_renko_signal(closes, atr)
    out = {}
    for idxs in monthly_groups(dates):
        idxs = [i for i in idxs if sig[i] is not None]
        if len(idxs) < 2:
            continue
        r = run_month(idxs, dates, opens, closes, sig, dvol, step, rule,
                      None, **costs)
        if r:
            out[r["month"]] = r["ret_pct"]
    return out


def stats(rets):
    n = len(rets)
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (n - 1))
    cum = peak = mdd = 0.0
    for x in rets:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return {"n": n, "win": 100 * sum(1 for x in rets if x > 0) / n,
            "avg": m, "sd": sd,
            "sharpe": m / sd * math.sqrt(12) if sd else float("nan"),
            "mdd": mdd, "worst": min(rets), "total": cum}


def fmt(label, s):
    return (f"{label:24} win={s['win']:3.0f}% avg={s['avg']:+5.2f}%/mo "
            f"sd={s['sd']:5.2f} Sharpe={s['sharpe']:5.2f} "
            f"maxDD={s['mdd']:6.1f}pp worst={s['worst']:+6.1f}% "
            f"total={s['total']:+7.1f}pp")


if __name__ == "__main__":
    per_asset, straddle, colors = {}, {}, {}
    for name, ohlc_csv, dvol_csv, step, costs, atr_src, rule, color in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(dvol_csv)
        frames = month_frames(dates, opens, closes)
        tr, expo = trend(frames, dates, opens, closes, costs)
        per_asset[name] = {
            "PW-ATM": put_write(frames, dates, closes, dvol, step, costs),
            "PW-30d": put_write(frames, dates, closes, dvol, step, costs, 0.30),
            "TREND": tr, "expo": expo,
        }
        straddle[name] = straddle_same_window(
            dates, opens, highs, lows, closes, dvol, step, costs,
            atr_src, rule)
        colors[name] = color

    months = None
    for name in per_asset:
        keys = (set(per_asset[name]["TREND"]) & set(per_asset[name]["PW-30d"])
                & set(straddle[name]))
        months = keys if months is None else months & keys
    months = sorted(months)
    print(f"common window: {months[0]} -> {months[-1]} ({len(months)} months)\n")

    rows_csv = []
    ew = {"PW-ATM": [], "PW-30d": [], "TREND": [], "COMBO": [], "STRADDLE": []}
    for name in per_asset:
        a = per_asset[name]
        pw_atm = [a["PW-ATM"][m] for m in months]
        pw30 = [a["PW-30d"][m] for m in months]
        tr = [a["TREND"][m] for m in months]
        combo = [(p + t) / 2 for p, t in zip(pw30, tr)]
        strad = [straddle[name][m] for m in months]
        long_frac = 100 * sum(a["expo"][m] for m in months) / len(months)
        print(f"=== {name} === (trend long {long_frac:.0f}% of months)")
        for lbl, series in [("put-write ATM", pw_atm),
                            ("put-write 30d", pw30),
                            ("trend 12m L/F", tr),
                            ("COMBO (30d+trend)/2", combo),
                            ("hedged straddle (ref)", strad)]:
            s = stats(series)
            print("  " + fmt(lbl, s))
            rows_csv.append([name, lbl] + [round(s[k], 3) for k in
                            ("n", "win", "avg", "sd", "sharpe", "mdd",
                             "worst", "total")])
        print()
        ew["PW-ATM"].append(pw_atm)
        ew["PW-30d"].append(pw30)
        ew["TREND"].append(tr)
        ew["COMBO"].append(combo)
        ew["STRADDLE"].append(strad)

    print("=== EQUAL-WEIGHT across the five assets ===")
    ew_series = {}
    for lbl, mat in ew.items():
        series = [sum(col) / len(col) for col in zip(*mat)]
        ew_series[lbl] = series
        s = stats(series)
        print("  " + fmt(lbl, s))
        rows_csv.append(["EW", lbl] + [round(s[k], 3) for k in
                        ("n", "win", "avg", "sd", "sharpe", "mdd",
                         "worst", "total")])

    with open("results_pwt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "sleeve", "n", "win%", "avg%/mo", "sd",
                    "sharpe", "maxDD_pp", "worst", "total_pp"])
        w.writerows(rows_csv)

    # chart: EW cumulative curves + combo-vs-straddle drawdown
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

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.6],
                                          "hspace": 0.13})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax = axes[0]
    for lbl, color, lw in [("PW-30d", "#2a78d6", 1.4),
                           ("TREND", "#eb6834", 1.4),
                           ("STRADDLE", "#898781", 1.6),
                           ("COMBO", INK, 2.6)]:
        cum = cumsum(ew_series[lbl])
        ax.plot(dts, cum, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {cum[-1]:+.0f}pp", (dts[-1], cum[-1]),
                    color=color, fontsize=8.5, fontweight="bold",
                    va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Equal-weight cumulative return — put-write + trend vs "
                 "hedged straddle (same window)", color=INK, fontsize=11.5,
                 loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    ax.plot(dts, dd(cumsum(ew_series["COMBO"])), color=INK, linewidth=2.2,
            label="COMBO")
    ax.plot(dts, dd(cumsum(ew_series["STRADDLE"])), color="#898781",
            linewidth=1.6, label="STRADDLE")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.savefig("report_pwt.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote results_pwt.csv, report_pwt.png")
