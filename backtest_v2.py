"""
Backtest v2: monthly ATM short straddle + Renko-directed 1-unit futures hedge.

Improvements over backtest.py:
- Premium can be derived from Deribit's DVOL index at each month's initiation
  (Black-Scholes ATM approximation: per-side premium = 0.3989 * sigma * sqrt(T)
  * spot), instead of a fixed % of spot.
- Transaction costs: futures taker fee (bps of traded notional per fill) and
  option trade fee (bps of spot per leg, 2 legs at initiation).
- Honest metrics: per-month returns are normalized by THAT month's spot
  (not the first month's), plus stdev, Sharpe, max drawdown, worst month.

Hedge rule is the crossing-required variant: a Renko flip alone is ignored;
the hedge flips only when price is on the far side of the strike AND Renko
agrees with the new direction.
"""
import csv
import datetime
import math

from backtest import (load_prices, compute_atr_proxy, compute_renko_signal,
                      nearest_strike, monthly_groups, ATR_LEN)

BS_ATM_COEF = 1.0 / math.sqrt(2.0 * math.pi)  # 0.3989: per-side ATM premium coef


def load_dvol(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["date"]] = float(row["dvol_close"])
    return out


def run_backtest_v2(dates, closes, signal, *,
                    premium_rate=None, dvol=None, strike_step=50,
                    fut_fee_bps=0.0, opt_fee_bps=0.0):
    """premium_rate: fixed per-side rate; dvol: {date: vol%} used instead if given."""
    months = monthly_groups(dates)
    results = []

    for idxs in months:
        idxs = [i for i in idxs if signal[i] is not None]
        if len(idxs) < 2:
            continue

        i0 = idxs[0]
        spot0 = closes[i0]
        strike = nearest_strike(spot0, strike_step)
        init_date, expiry_date = dates[i0], dates[idxs[-1]]
        days_T = max((expiry_date - init_date).days, 1)

        if dvol is not None:
            sigma = dvol.get(init_date.isoformat())
            if sigma is None:
                continue  # no vol data for this month
            per_side = BS_ATM_COEF * (sigma / 100.0) * math.sqrt(days_T / 365.0) * spot0
        else:
            per_side = premium_rate * spot0

        premium_income = (2 * per_side
                          + max(0.0, spot0 - strike)   # call intrinsic from rounding
                          + max(0.0, strike - spot0))  # put intrinsic from rounding

        fees = 2 * (opt_fee_bps / 1e4) * spot0          # sell call + sell put
        fees += (fut_fee_bps / 1e4) * spot0             # open hedge

        hedge_dir = signal[i0]
        hedge_entry = spot0
        hedge_pnl = 0.0
        n_switches = 0

        for i in idxs[1:]:
            price, sig = closes[i], signal[i]
            if hedge_dir == 1 and price < strike and sig == -1:
                hedge_pnl += (price - hedge_entry)
                hedge_dir, hedge_entry = -1, price
                n_switches += 1
                fees += 2 * (fut_fee_bps / 1e4) * price  # close long + open short
            elif hedge_dir == -1 and price > strike and sig == 1:
                hedge_pnl += -(price - hedge_entry)
                hedge_dir, hedge_entry = 1, price
                n_switches += 1
                fees += 2 * (fut_fee_bps / 1e4) * price

        settle = closes[idxs[-1]]
        hedge_pnl += hedge_dir * (settle - hedge_entry)
        fees += (fut_fee_bps / 1e4) * settle             # close hedge at expiry

        option_pnl = (premium_income
                      - max(0.0, settle - strike)
                      - max(0.0, strike - settle))

        month_pnl = option_pnl + hedge_pnl - fees
        results.append({
            "month": init_date.strftime("%Y-%m"),
            "init_date": init_date, "expiry_date": expiry_date,
            "spot0": spot0, "strike": strike, "settle": settle,
            "premium_income": premium_income, "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl, "fees": fees, "n_switches": n_switches,
            "month_pnl": month_pnl, "ret_pct": month_pnl / spot0 * 100.0,
        })
    return results


def summarize(results, label):
    n = len(results)
    rets = [r["ret_pct"] for r in results]
    mean = sum(rets) / n
    var = sum((x - mean) ** 2 for x in rets) / (n - 1)
    sd = math.sqrt(var)
    sharpe = mean / sd * math.sqrt(12) if sd > 0 else float("nan")
    # max drawdown on cumulative % equity
    cum, peak, mdd = 0.0, 0.0, 0.0
    for x in rets:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    wins = sum(1 for x in rets if x > 0)
    worst = min(rets); best = max(rets)
    total = sum(rets)
    print(f"{label:34} n={n:2d} win={wins:2d}({wins/n*100:3.0f}%) "
          f"avg={mean:+6.2f}%/mo sd={sd:5.2f} Sharpe={sharpe:5.2f} "
          f"maxDD={mdd:6.1f}pp worst={worst:+6.1f}% best={best:+6.1f}% total={total:+7.1f}pp")
    return {"label": label, "n": n, "win_rate": wins / n * 100, "mean": mean,
            "sd": sd, "sharpe": sharpe, "max_dd": mdd, "worst": worst,
            "best": best, "total": total}


def run_asset(name, price_csv, dvol_csv, fixed_rate, strike_step,
              fut_fee_bps=5.0, opt_fee_bps=3.0):
    dates, closes = load_prices(price_csv)
    atr = compute_atr_proxy(closes, ATR_LEN)
    signal = compute_renko_signal(dates, closes, atr)
    dvol = load_dvol(dvol_csv)

    print(f"\n=== {name} ===")
    cfgs = [
        (f"A fixed {fixed_rate*100:.2f}%/side, no fees",
         dict(premium_rate=fixed_rate)),
        ("B DVOL-implied premium, no fees",
         dict(dvol=dvol)),
        ("C DVOL premium + fees(5bp fut,3bp opt)",
         dict(dvol=dvol, fut_fee_bps=fut_fee_bps, opt_fee_bps=opt_fee_bps)),
    ]
    out = {}
    for label, kw in cfgs:
        res = run_backtest_v2(dates, closes, signal, strike_step=strike_step, **kw)
        summarize(res, label)
        out[label[0]] = res
    return out


if __name__ == "__main__":
    eth = run_asset("ETHUSD", "data/eth_usd_daily.csv", "data/dvol_eth.csv", 0.068, 50)
    btc = run_asset("BTCUSD", "data/btc_usd_daily.csv", "data/dvol_btc.csv", 0.05, 500)

    # dump per-month detail for the realistic config (C) for the report
    for name, res in (("eth", eth["C"]), ("btc", btc["C"])):
        with open(f"results_{name}_v2.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["month", "spot0", "strike", "settle", "premium_income",
                        "option_pnl", "hedge_pnl", "fees", "n_switches",
                        "month_pnl", "ret_pct"])
            for r in res:
                w.writerow([r["month"], round(r["spot0"], 2), r["strike"],
                            round(r["settle"], 2), round(r["premium_income"], 2),
                            round(r["option_pnl"], 2), round(r["hedge_pnl"], 2),
                            round(r["fees"], 2), r["n_switches"],
                            round(r["month_pnl"], 2), round(r["ret_pct"], 3)])
