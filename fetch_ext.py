"""Extended-window data (2019-01 -> present) into data/ext/, for the
long-window put-write/trend/straddle comparison (backtest_pwt_ext.py).
Kept separate from data/ so the published 2023+ results stay reproducible.

- OHLC from Yahoo for SPY, XLK, XLV, XLY, gold (GC=F).
- VIX from Yahoo, GVZ from CBOE (both real, full window).
- XLK/XLV/XLY vol: HYBRID — the real CBOE sector index where it existed
  (VXXLK/VXXLV/VXXLY, 2020-11-27 -> 2022-11-07), else the calibrated
  proxy k * VIX * RV63_etf/RV63_spy (k = mean(real/proxy) on the overlap,
  as in make_sector_vol.py). The proxy's known weakness: RV63 lags vol
  spikes, so COVID-month premiums for the sector sleeves are the least
  reliable part of the extension (SPY/GOLD use real indices throughout).
"""
import csv
import datetime
import io
import json
import math
import os
import urllib.request

P1 = 1546300800   # 2019-01-01
P2 = 1790812800   # 2026-10-01
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
RV_LEN = 63
START = datetime.date(2019, 1, 1)

os.makedirs("data/ext", exist_ok=True)


def yahoo(sym, ohlc=False):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?"
           f"period1={P1}&period2={P2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    res = d["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    out = {}
    for i, t in enumerate(res["timestamp"]):
        day = datetime.datetime.utcfromtimestamp(t).date()
        if ohlc:
            v = (q["open"][i], q["high"][i], q["low"][i], q["close"][i])
            if None in v or day in out:
                continue
            out[day] = v
        else:
            if q["close"][i] is None or day in out:
                continue
            out[day] = q["close"][i]
    return out


def cboe(idx):
    req = urllib.request.Request(CBOE.format(idx),
                                 headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=45).read().decode()
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        day = datetime.datetime.strptime(row["DATE"], "%m/%d/%Y").date()
        out[day] = float(row.get(idx) or row["CLOSE"])
    return out


def rv_series(closes):
    days = sorted(closes)
    rets, rv = [], {}
    for i in range(1, len(days)):
        rets.append(math.log(closes[days[i]] / closes[days[i - 1]]))
        if len(rets) >= RV_LEN:
            w = rets[-RV_LEN:]
            m = sum(w) / RV_LEN
            var = sum((x - m) ** 2 for x in w) / (RV_LEN - 1)
            rv[days[i]] = math.sqrt(var * 252) * 100
    return rv


def write_vol(path, series):
    with open(path, "w") as f:
        f.write("date,dvol_close\n")
        for day in sorted(series):
            if day >= START:
                f.write(f"{day},{series[day]:.4f}\n")
    print(path, len(series), "rows")


if __name__ == "__main__":
    for sym, name in [("SPY", "spy"), ("XLK", "xlk"), ("XLV", "xlv"),
                      ("XLY", "xly"), ("GC=F", "gold")]:
        o = yahoo(sym, ohlc=True)
        with open(f"data/ext/{name}_ohlc.csv", "w") as f:
            f.write("date,open,high,low,close\n")
            for day in sorted(o):
                op, hi, lo, cl = o[day]
                f.write(f"{day},{op:.6f},{hi:.6f},{lo:.6f},{cl:.6f}\n")
        print(f"data/ext/{name}_ohlc.csv", len(o), "rows")

    write_vol("data/ext/vix_daily.csv", yahoo("%5EVIX"))
    write_vol("data/ext/gvz_daily.csv", cboe("GVZ"))

    spy = yahoo("SPY")
    vix = yahoo("%5EVIX")
    rv_spy = rv_series(spy)
    for sym, idx in [("XLK", "VXXLK"), ("XLV", "VXXLV"), ("XLY", "VXXLY")]:
        etf = yahoo(sym)
        rv_etf = rv_series(etf)
        proxy = {d: vix[d] * rv_etf[d] / rv_spy[d]
                 for d in rv_etf if d in rv_spy and d in vix and rv_spy[d] > 0}
        real = cboe(idx)
        overlap = sorted(set(proxy) & set(real))
        k = sum(real[d] / proxy[d] for d in overlap) / len(overlap)
        hybrid = {d: real[d] if d in real else k * v
                  for d, v in proxy.items()}
        hybrid.update({d: real[d] for d in real if d >= START})
        print(f"{sym}: k={k:.3f} on {len(overlap)} overlap days, "
              f"{sum(1 for d in hybrid if d in real)} real / "
              f"{len(hybrid)} total")
        write_vol(f"data/ext/{sym.lower()}_vol.csv", hybrid)
