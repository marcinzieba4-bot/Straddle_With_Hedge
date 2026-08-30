"""Correlation & diversification of v3 strategy returns across the
surviving assets: SPY, XLK, XLV, XLY, GOLD.

One UNIFORM config for all five — cc ATR, itm rule (strike-anchored
hysteresis: a Renko flip while the hedge is in profit vs the strike is
ignored until price crosses back through it), no wings — so nothing is
cherry-picked per asset. Prints the monthly-return correlation matrix,
per-asset stats, and equal-weight portfolio stats vs SPY alone.
"""
import math

from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, run_variant,
                         SPY_COSTS, GOLD_COSTS, ETF_COSTS)

CONFIG = ("cc", "itm", None)

ASSETS = [
    ("SPY", "data/spy_ohlc.csv", "data/vix_daily.csv", 1, SPY_COSTS),
    ("XLK", "data/xlk_ohlc.csv", "data/xlk_ivproxy.csv", 1, ETF_COSTS),
    ("XLV", "data/xlv_ohlc.csv", "data/xlv_ivproxy.csv", 1, ETF_COSTS),
    ("XLY", "data/xly_ohlc.csv", "data/xly_ivproxy.csv", 1, ETF_COSTS),
    ("GOLD", "data/gold_ohlc.csv", "data/gvz_daily.csv", 5, GOLD_COSTS),
]


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def stats(rets):
    n = len(rets)
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (n - 1))
    cum = peak = mdd = 0.0
    for x in rets:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    srt = sorted(rets)
    k = max(1, round(0.05 * n))
    return {"win": 100 * sum(1 for x in rets if x > 0) / n, "mean": m,
            "sd": sd, "sharpe": m / sd * math.sqrt(12) if sd else float("nan"),
            "mdd": mdd, "worst": min(rets), "cvar": sum(srt[:k]) / k,
            "total": cum}


def show(label, s):
    print(f"{label:14} win={s['win']:3.0f}% avg={s['mean']:+5.2f}%/mo "
          f"sd={s['sd']:5.2f} Sharpe={s['sharpe']:5.2f} "
          f"maxDD={s['mdd']:6.1f}pp worst={s['worst']:+6.1f}% "
          f"CVaR95={s['cvar']:+6.1f}% total={s['total']:+7.1f}pp")


if __name__ == "__main__":
    atr_source, rule, wing = CONFIG
    series = {}
    for name, ohlc_csv, dvol_csv, step, costs in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(dvol_csv)
        res = run_variant(dates, opens, highs, lows, closes, dvol, step,
                          atr_source, rule, wing, **costs)
        series[name] = {r["month"]: r["ret_pct"] for r in res}

    months = sorted(set.intersection(*(set(s) for s in series.values())))
    names = [a[0] for a in ASSETS]
    rets = {n: [series[n][m] for m in months] for n in names}
    print(f"config: {atr_source}/{rule}/no wings, {len(months)} common "
          f"months {months[0]} -> {months[-1]}\n")

    print("monthly strategy-return correlations:")
    print(" " * 6 + "".join(f"{n:>7}" for n in names))
    for a in names:
        row = "".join(f"{corr(rets[a], rets[b]):7.2f}" for b in names)
        print(f"{a:6}" + row)
    pair_cs = [corr(rets[a], rets[b])
               for i, a in enumerate(names) for b in names[i + 1:]]
    print(f"average pairwise correlation: {sum(pair_cs)/len(pair_cs):+.2f}\n")

    for n in names:
        show(n, stats(rets[n]))
    ew = [sum(rets[n][i] for n in names) / len(names)
          for i in range(len(months))]
    print()
    show("EQUAL-WEIGHT", stats(ew))
