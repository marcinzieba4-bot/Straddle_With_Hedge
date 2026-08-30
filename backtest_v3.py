"""
Backtest v3: deploys the audit conclusions. All variants run under realistic
trading conditions:
- short options collect 90% of Black-Scholes mark (bid-side haircut),
  long wings pay 110% of mark
- fees: 5bp of notional per futures fill, 3bp of spot per option leg
- perp funding drag: 0.5bp/day on hedge notional
- hedge fills execute at the NEXT day's open (signal on close t, fill at
  open t+1), not the triggering close

Variant axes:
- ATR source: 'cc' = close-to-close proxy (v2 behaviour) vs 'ohlc' = true
  range from OHLC data (matches TradingView's ta.atr, ~2x larger bricks)
- flip rule:
    'flip'  = hedge follows every Renko flip
    'cross' = flip only when price is past the strike AND Renko agrees
    'itm'   = ITM-hysteresis vs the strike: flip immediately on Renko flip
              while the hedge is NOT in profit vs the neutral point (strike);
              if in the money, hold until price crosses back through the
              strike. (Mathematically equivalent to 'cross'.)
    'entry' = ITM-hysteresis vs the hedge's own entry price (true
              break-even): flip on Renko flip only while the hedge is at or
              below break-even; a profitable hedge is held until price comes
              back through its entry.
- wings: None, or target delta (0.15 / 0.20) for a long OTM put + call,
  turning the short straddle into put/call spreads
"""
import csv
import datetime
import math
from statistics import NormalDist

from backtest import nearest_strike, monthly_groups
from backtest_v2 import load_dvol

ATR_LEN = 14
RENKO_MULT = 0.15
TICK_TREND = 2
TICK_REVERSAL = 4

SHORT_MARK_FRAC = 0.90   # collect 90% of mark on short legs
LONG_MARK_FRAC = 1.10    # pay 110% of mark on long wings
FUT_FEE_BPS = 5.0
OPT_FEE_BPS = 3.0
FUNDING_BPS_PER_DAY = 0.5

_N = NormalDist()


def load_ohlc(path):
    dates, opens, highs, lows, closes = [], [], [], [], []
    import datetime
    with open(path) as f:
        for row in csv.DictReader(f):
            dates.append(datetime.date.fromisoformat(row["date"]))
            opens.append(float(row["open"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            closes.append(float(row["close"]))
    return dates, opens, highs, lows, closes


def compute_atr(closes, highs=None, lows=None, length=ATR_LEN, source="ohlc"):
    """Wilder-smoothed ATR; true range if highs/lows given, else |dC| proxy."""
    n = len(closes)
    tr = [None] * n
    for i in range(1, n):
        if source == "ohlc":
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        else:
            tr[i] = abs(closes[i] - closes[i - 1])
    atr = [None] * n
    if length >= n:
        return atr
    atr[length] = sum(tr[1:length + 1]) / length
    for i in range(length + 1, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def compute_renko_signal(closes, atr):
    """Universal Renko port (close-based bricks, matching the Pine script):
    tick = ATR14 * RENKO_MULT, continuation 2 ticks, reversal 4 ticks."""
    n = len(closes)
    signal = [None] * n
    start = next((i for i, a in enumerate(atr) if a is not None), None)
    if start is None:
        return signal
    tick = atr[start] * RENKO_MULT
    bar_max = closes[start] + TICK_TREND * tick
    bar_min = closes[start] - TICK_TREND * tick
    direction = 0
    for i in range(start + 1, n):
        price = closes[i]
        if price > bar_max or price < bar_min:
            direction = 1 if price > bar_max else -1
            this_close = bar_max if direction == 1 else bar_min
            tick = (atr[i] if atr[i] else tick / RENKO_MULT) * RENKO_MULT
            if direction > 0:
                bar_max = this_close + TICK_TREND * tick
                bar_min = this_close - TICK_REVERSAL * tick
            else:
                bar_max = this_close + TICK_REVERSAL * tick
                bar_min = this_close - TICK_TREND * tick
        signal[i] = direction if direction != 0 else None
    return signal


def bs_price(S, K, sigma, T, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if kind == "c" else max(0.0, K - S)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if kind == "c":
        return S * _N.cdf(d1) - K * _N.cdf(d2)
    return K * _N.cdf(-d2) - S * _N.cdf(-d1)


def delta_strike(S, sigma, T, target_delta, kind, step):
    """Strike whose BS delta magnitude equals target (r=0)."""
    v = sigma * math.sqrt(T)
    nd1 = target_delta if kind == "c" else 1.0 - target_delta
    d1 = _N.inv_cdf(nd1)
    K = S * math.exp(0.5 * v * v - d1 * v)
    return nearest_strike(K, step)


def run_month(idxs, dates, opens, closes, signal, dvol, strike_step,
              rule, wing_delta, fut_fee_bps=FUT_FEE_BPS,
              opt_fee_bps=OPT_FEE_BPS, funding_bps=FUNDING_BPS_PER_DAY):
    i0 = idxs[0]
    spot0 = closes[i0]
    strike = nearest_strike(spot0, strike_step)
    init_date, expiry_date = dates[i0], dates[idxs[-1]]
    days_T = max((expiry_date - init_date).days, 1)
    T = days_T / 365.0
    # FX trades on days the vol index doesn't print (e.g. Jan 2); fall back
    # to the most recent quote within 5 days. No-op when calendars align.
    sigma = None
    for back in range(6):
        sigma = dvol.get((init_date - datetime.timedelta(days=back)).isoformat())
        if sigma is not None:
            break
    if sigma is None:
        return None
    sigma /= 100.0

    # short ATM straddle at 90% of mark, plus rounding intrinsic
    per_side_mark = (1.0 / math.sqrt(2.0 * math.pi)) * sigma * math.sqrt(T) * spot0
    premium_income = (2 * SHORT_MARK_FRAC * per_side_mark
                      + max(0.0, spot0 - strike) + max(0.0, strike - spot0))
    fees = 2 * (opt_fee_bps / 1e4) * spot0

    # long wings at 110% of mark
    wing_cost = 0.0
    put_wing = call_wing = None
    if wing_delta:
        call_wing = delta_strike(spot0, sigma, T, wing_delta, "c", strike_step)
        put_wing = delta_strike(spot0, sigma, T, wing_delta, "p", strike_step)
        wing_cost = LONG_MARK_FRAC * (bs_price(spot0, call_wing, sigma, T, "c")
                                      + bs_price(spot0, put_wing, sigma, T, "p"))
        fees += 2 * (opt_fee_bps / 1e4) * spot0

    # hedge: direction decided on close, filled at NEXT open
    hedge_dir = signal[i0]
    if i0 + 1 < len(opens) and (i0 + 1) <= idxs[-1]:
        hedge_entry = opens[i0 + 1]
    else:
        hedge_entry = spot0
    hedge_pnl = 0.0
    n_switches = 0
    fees += (fut_fee_bps / 1e4) * hedge_entry

    for i in idxs[1:]:
        price, sig = closes[i], signal[i]
        want_flip = False
        if sig is not None and sig != hedge_dir:
            if rule == "flip":
                want_flip = True
            elif rule == "cross":
                want_flip = (hedge_dir == 1 and price < strike) or \
                            (hedge_dir == -1 and price > strike)
            elif rule == "itm":
                itm = (hedge_dir == 1 and price > strike) or \
                      (hedge_dir == -1 and price < strike)
                want_flip = not itm
            elif rule == "entry":
                itm = (hedge_dir == 1 and price > hedge_entry) or \
                      (hedge_dir == -1 and price < hedge_entry)
                want_flip = not itm
        if want_flip and i + 1 < len(opens) and i < idxs[-1]:
            fill = opens[i + 1]
            hedge_pnl += hedge_dir * (fill - hedge_entry)
            hedge_dir = sig
            hedge_entry = fill
            n_switches += 1
            fees += 2 * (fut_fee_bps / 1e4) * fill

    settle = closes[idxs[-1]]
    hedge_pnl += hedge_dir * (settle - hedge_entry)
    fees += (fut_fee_bps / 1e4) * settle
    fees += (funding_bps / 1e4) * spot0 * days_T

    option_pnl = (premium_income
                  - max(0.0, settle - strike)
                  - max(0.0, strike - settle))
    wing_pnl = 0.0
    if wing_delta:
        wing_pnl = (max(0.0, settle - call_wing) + max(0.0, put_wing - settle)
                    - wing_cost)

    month_pnl = option_pnl + wing_pnl + hedge_pnl - fees
    return {
        "month": init_date.strftime("%Y-%m"), "spot0": spot0, "strike": strike,
        "settle": settle, "option_pnl": option_pnl, "wing_pnl": wing_pnl,
        "hedge_pnl": hedge_pnl, "fees": fees, "n_switches": n_switches,
        "month_pnl": month_pnl, "ret_pct": month_pnl / spot0 * 100.0,
    }


def run_variant(dates, opens, highs, lows, closes, dvol, strike_step,
                atr_source, rule, wing_delta, **costs):
    atr = compute_atr(closes, highs, lows, source=atr_source)
    signal = compute_renko_signal(closes, atr)
    results = []
    for idxs in monthly_groups(dates):
        idxs = [i for i in idxs if signal[i] is not None]
        if len(idxs) < 2:
            continue
        r = run_month(idxs, dates, opens, closes, signal, dvol, strike_step,
                      rule, wing_delta, **costs)
        if r:
            results.append(r)
    return results


def stats(results):
    n = len(results)
    rets = [r["ret_pct"] for r in results]
    mean = sum(rets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / (n - 1))
    sharpe = mean / sd * math.sqrt(12) if sd > 0 else float("nan")
    cum = peak = mdd = 0.0
    for x in rets:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return {"n": n, "win": sum(1 for x in rets if x > 0) / n * 100,
            "mean": mean, "sd": sd, "sharpe": sharpe, "mdd": mdd,
            "worst": min(rets), "total": sum(rets),
            "switches": sum(r["n_switches"] for r in results)}


CRYPTO_COSTS = dict(fut_fee_bps=FUT_FEE_BPS, opt_fee_bps=OPT_FEE_BPS,
                    funding_bps=FUNDING_BPS_PER_DAY)
# SPY hedged with ES/MES index futures: ~1bp per fill covers commission +
# half-spread + slippage; no perp funding (carry sits in the futures basis
# and roughly nets out for a hedge that flips long/short). Options on SPY
# are penny-wide ATM, so 1bp of spot per leg on top of the 90/110% marks.
SPY_COSTS = dict(fut_fee_bps=1.0, opt_fee_bps=1.0, funding_bps=0.0)
# EURUSD hedged with CME 6E futures: ~0.5 pip half-spread + fees on €125k
# notional is ~0.5bp per fill; no funding (the EUR/USD rate differential
# sits in the forward points and largely nets out for a flipping hedge).
FX_COSTS = dict(fut_fee_bps=0.5, opt_fee_bps=1.0, funding_bps=0.0)


if __name__ == "__main__":
    for name, ohlc_csv, dvol_csv, step, costs in [
            ("ETHUSD", "data/eth_ohlc.csv", "data/dvol_eth.csv", 50, CRYPTO_COSTS),
            ("BTCUSD", "data/btc_ohlc.csv", "data/dvol_btc.csv", 500, CRYPTO_COSTS),
            ("SPY", "data/spy_ohlc.csv", "data/vix_daily.csv", 1, SPY_COSTS),
            ("EURUSD", "data/eurusd_ohlc.csv", "data/evz_daily.csv", 0.005,
             FX_COSTS)]:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(dvol_csv)
        print(f"\n=== {name} === (all realistic: 90/110% marks, fees, funding, next-open fills)")
        print(f"{'atr':5} {'rule':6} {'wing':5} {'win%':>5} {'avg%/mo':>8} "
              f"{'sd':>6} {'Sharpe':>7} {'maxDD':>7} {'worst':>7} {'total':>8} {'flips':>6}")
        for atr_source in ("cc", "ohlc"):
            for rule in ("flip", "cross", "itm", "entry"):
                for wing in (None, 0.20, 0.15):
                    res = run_variant(dates, opens, highs, lows, closes, dvol,
                                      step, atr_source, rule, wing, **costs)
                    s = stats(res)
                    wl = f"{wing:.2f}" if wing else "none"
                    print(f"{atr_source:5} {rule:6} {wl:5} {s['win']:5.0f} "
                          f"{s['mean']:+8.2f} {s['sd']:6.2f} {s['sharpe']:7.2f} "
                          f"{s['mdd']:7.1f} {s['worst']:+7.1f} {s['total']:+8.1f} "
                          f"{s['switches']:6d}")
        if name in ("SPY", "EURUSD"):
            res = run_variant(dates, opens, highs, lows, closes, dvol, step,
                              "ohlc", "entry", None, **costs)
            with open(f"results_{name.lower()}_v3.csv", "w", newline="") as f:
                w = csv.writer(f)
                cols = ["month", "spot0", "strike", "settle", "option_pnl",
                        "wing_pnl", "hedge_pnl", "fees", "n_switches",
                        "month_pnl", "ret_pct"]
                w.writerow(cols)
                for r in res:
                    w.writerow([r["month"]] + [round(r[c], 4) if isinstance(r[c], float)
                                               else r[c] for c in cols[1:]])
