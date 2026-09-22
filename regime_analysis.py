"""When is it worth selling the straddle or the put? Conditional analysis
on the extended 2020-2026 window (data/ext), pooled across the five
assets (~400 asset-months).

Each asset-month is bucketed by state variables OBSERVABLE AT INITIATION
(no lookahead in the signal itself):

- IV tercile: the asset's vol index at init, low/mid/high (tercile edges
  are full-sample per asset — a mild in-sample convenience for the
  diagnostic; a live rule would use expanding percentiles).
- VRP tercile: IV minus trailing-21-day realized vol (annualized) — the
  richness of the premium vs how much the asset is actually moving. The
  classic short-vol conditioner.
- IV direction: vol index vs its level ~3 months (63 trading days) ago.
- Price vs 200dma: above/below at init.

For each bucket: pooled avg %/mo, win rate and a t-stat per sleeve
(PW-30d put-write, hedged STRADDLE on the tearsheet configs, TREND,
COMBO). Diagnostic, not a fitted trading rule.
"""
import datetime
import math

from backtest_v2 import load_dvol
from backtest_v3 import load_ohlc, SPY_COSTS, GOLD_COSTS, ETF_COSTS
from backtest_putwrite_trend import (month_frames, put_write, trend,
                                     straddle_same_window)

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
SLEEVES = ("PW-30d", "STRADDLE", "TREND", "COMBO")


def sigma_at(dvol, day):
    for back in range(6):
        v = dvol.get((day - datetime.timedelta(days=back)).isoformat())
        if v is not None:
            return v
    return None


def rv_ann(closes, i, lookback=21):
    if i < lookback + 1:
        return None
    rets = [math.log(closes[j] / closes[j - 1])
            for j in range(i - lookback + 1, i + 1)]
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var * 252) * 100


def terciles(vals):
    s = sorted(vals)
    return s[len(s) // 3], s[2 * len(s) // 3]


if __name__ == "__main__":
    obs = []   # one record per asset-month
    for name, ohlc_csv, vol_csv, step, costs, atr_src, rule in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(vol_csv)
        frames = month_frames(dates, opens, closes)
        pw = put_write(frames, dates, closes, dvol, step, costs, 0.30)
        tr, _ = trend(frames, dates, opens, closes, costs)
        st = straddle_same_window(dates, opens, highs, lows, closes, dvol,
                                  step, costs, atr_src, rule)
        day_index = {d: i for i, d in enumerate(dates)}
        recs = []
        for label, i0, il in frames:
            if not (label in pw and label in tr and label in st):
                continue
            iv = sigma_at(dvol, dates[i0])
            rv = rv_ann(closes, i0)
            iv_old = sigma_at(dvol, dates[max(0, i0 - 63)])
            if None in (iv, rv, iv_old) or i0 < 200:
                continue
            ma200 = sum(closes[i0 - 199:i0 + 1]) / 200
            recs.append({
                "asset": name, "month": label, "iv": iv, "vrp": iv - rv,
                "iv_rising": iv > iv_old, "above_ma": closes[i0] > ma200,
                "PW-30d": pw[label], "STRADDLE": st[label],
                "TREND": tr[label],
                "COMBO": (pw[label] + tr[label]) / 2,
            })
        lo, hi = terciles([r["iv"] for r in recs])
        vlo, vhi = terciles([r["vrp"] for r in recs])
        for r in recs:
            r["iv_b"] = ("low" if r["iv"] <= lo else
                         "high" if r["iv"] > hi else "mid")
            r["vrp_b"] = ("cheap" if r["vrp"] <= vlo else
                          "rich" if r["vrp"] > vhi else "mid")
        obs.extend(recs)

    print(f"{len(obs)} asset-months, "
          f"{min(r['month'] for r in obs)} -> {max(r['month'] for r in obs)}")

    def report(title, groups):
        print(f"\n=== {title} ===")
        print(f"{'bucket':22}{'n':>5}" + "".join(
            f"{s:>10}{'win%':>6}" for s in SLEEVES))
        for gname, sel in groups:
            rows = [r for r in obs if sel(r)]
            if len(rows) < 10:
                continue
            line = f"{gname:22}{len(rows):5d}"
            for s in SLEEVES:
                v = [r[s] for r in rows]
                m = sum(v) / len(v)
                sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
                t = m / (sd / math.sqrt(len(v))) if sd else 0
                mark = "*" if abs(t) > 2 else " "
                win = 100 * sum(1 for x in v if x > 0) / len(v)
                line += f"{m:+9.2f}{mark}{win:5.0f}%"
            print(line)

    report("IV level at init (per-asset terciles)", [
        ("IV low", lambda r: r["iv_b"] == "low"),
        ("IV mid", lambda r: r["iv_b"] == "mid"),
        ("IV high", lambda r: r["iv_b"] == "high")])
    report("Vol risk premium (IV - RV21) at init", [
        ("VRP cheap (IV<~RV)", lambda r: r["vrp_b"] == "cheap"),
        ("VRP mid", lambda r: r["vrp_b"] == "mid"),
        ("VRP rich (IV>>RV)", lambda r: r["vrp_b"] == "rich")])
    report("IV direction (vs 63 trading days ago)", [
        ("IV falling", lambda r: not r["iv_rising"]),
        ("IV rising", lambda r: r["iv_rising"])])
    report("Price vs 200dma at init", [
        ("above 200dma", lambda r: r["above_ma"]),
        ("below 200dma", lambda r: not r["above_ma"])])
    report("Composites", [
        ("rich VRP & above ma", lambda r: r["vrp_b"] == "rich" and r["above_ma"]),
        ("rich VRP & below ma", lambda r: r["vrp_b"] == "rich" and not r["above_ma"]),
        ("cheap VRP & below ma", lambda r: r["vrp_b"] == "cheap" and not r["above_ma"]),
        ("IV high & falling", lambda r: r["iv_b"] == "high" and not r["iv_rising"]),
        ("IV high & rising", lambda r: r["iv_b"] == "high" and r["iv_rising"]),
        ("IV low & rising", lambda r: r["iv_b"] == "low" and r["iv_rising"])])
    print("\n(* = |t| > 2; avg is %/mo of spot, pooled asset-months)")
