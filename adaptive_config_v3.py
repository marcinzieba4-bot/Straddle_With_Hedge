"""Per-asset configuration without hindsight.

Question: gold's grid winner was cc/flip and oil's was ohlc/flip — can we
'stick to the internal characteristic of the asset' instead of picking the
best row of an in-sample grid?

Two parts:

1. Characterize each asset: daily-return lag-1 autocorrelation and a
   20-day variance ratio (VR > 1 = trending, < 1 = mean-reverting), plus
   which config each asset 'wants' (bigger bricks for choppier assets —
   note cc-ATR is just ~half-size bricks vs ohlc-ATR, so the atr axis IS
   a brick-size knob).

2. Walk-forward selection: each month, trade the config that had the best
   trailing-12-month Sharpe for THIS asset (candidates: {cc,ohlc} x
   {flip,cross,entry}, no wings — wings lose on every asset; 'itm' is
   identical to 'cross' so it is dropped). The first 12 months are burn-in;
   evaluation covers the remaining months. This adapts to the asset's own
   character with zero lookahead, and is compared against each fixed
   config over the SAME evaluation window.
"""
import math

from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, run_variant,
                         CRYPTO_COSTS, SPY_COSTS, GOLD_COSTS, OIL_COSTS)

CANDIDATES = [(a, r) for a in ("cc", "ohlc") for r in ("flip", "cross", "entry")]
BURN_IN = 12

ASSETS = [
    ("SPY", "data/spy_ohlc.csv", "data/vix_daily.csv", 1, SPY_COSTS),
    ("GOLD", "data/gold_ohlc.csv", "data/gvz_daily.csv", 5, GOLD_COSTS),
    ("OIL", "data/oil_ohlc.csv", "data/ovx_daily.csv", 0.5, OIL_COSTS),
]


def characterize(closes):
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(rets)
    m = sum(rets) / n
    var = sum((x - m) ** 2 for x in rets) / n
    ac1 = (sum((rets[i] - m) * (rets[i - 1] - m) for i in range(1, n))
           / n / var)
    k = 20
    kday = [sum(rets[i:i + k]) for i in range(0, n - k)]
    mk = sum(kday) / len(kday)
    vark = sum((x - mk) ** 2 for x in kday) / len(kday)
    vr = vark / (k * var)
    return {"ann_vol%": math.sqrt(var * 365) * 100, "ac1": ac1, "vr20": vr}


def sharpe(rets):
    n = len(rets)
    if n < 2:
        return float("nan")
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (n - 1))
    return m / sd * math.sqrt(12) if sd > 0 else float("nan")


def summarize(rets):
    n = len(rets)
    m = sum(rets) / n
    cum = peak = mdd = 0.0
    for x in rets:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return (f"n={n:2d} win={100*sum(1 for x in rets if x>0)/n:3.0f}% "
            f"avg={m:+5.2f}%/mo Sharpe={sharpe(rets):5.2f} "
            f"maxDD={mdd:6.1f}pp worst={min(rets):+6.1f}% total={cum:+7.1f}pp")


if __name__ == "__main__":
    for name, ohlc_csv, dvol_csv, step, costs in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(dvol_csv)
        ch = characterize(closes)
        print(f"\n=== {name} ===  ann.vol {ch['ann_vol%']:.0f}%  "
              f"daily AC1 {ch['ac1']:+.3f}  VR(20d) {ch['vr20']:.2f} "
              f"({'trending' if ch['vr20'] > 1.05 else 'choppy/mean-reverting' if ch['vr20'] < 0.95 else 'neutral'})")

        per_cfg = {}
        months = None
        for atr_source, rule in CANDIDATES:
            res = run_variant(dates, opens, highs, lows, closes, dvol, step,
                              atr_source, rule, None, **costs)
            per_cfg[(atr_source, rule)] = {r["month"]: r["ret_pct"] for r in res}
            ms = [r["month"] for r in res]
            months = ms if months is None else [m for m in months if m in ms]

        # walk-forward: month i traded with the best trailing-12m Sharpe cfg
        wf_rets, picks = [], []
        for i in range(BURN_IN, len(months)):
            window = months[i - BURN_IN:i]
            best = max(per_cfg, key=lambda c: sharpe(
                [per_cfg[c][m] for m in window]))
            wf_rets.append(per_cfg[best][months[i]])
            picks.append(best)

        eval_months = months[BURN_IN:]
        print(f"  evaluation window: {eval_months[0]} -> {eval_months[-1]}")
        for cfg in CANDIDATES:
            fixed = [per_cfg[cfg][m] for m in eval_months]
            print(f"  fixed {cfg[0]:4}/{cfg[1]:5} : {summarize(fixed)}")
        print(f"  WALK-FORWARD     : {summarize(wf_rets)}")
        counts = {}
        for p in picks:
            counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print("  picks: " + ", ".join(f"{a}/{r} x{c}" for (a, r), c in top))
