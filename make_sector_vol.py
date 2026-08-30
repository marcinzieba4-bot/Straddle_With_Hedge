"""Build implied-vol series for sector ETFs (XLE, XLV, XLI).

CBOE's sector vol indices were discontinued before this repo's backtest
window opens (VXXLE ends 2022-02, VXXLV/VXXLI end 2022-11), so no real
30-day IV series exists for 2023+. Proxy instead:

    proxy(t) = VIX(t) * RV63_etf(t) / RV63_spy(t)

i.e. scale the live VIX by the ETF's realized-vol ratio to SPY (63
trading days, annualized, log returns). The VIX term carries the IV-index
dynamics (vol spikes, term premium); the ratio carries the sector's own
vol level.

Because the discontinued indices DO overlap our raw data on 2020-2022,
the proxy is validated against each real index on that overlap (bias,
MAE, level correlation) and CALIBRATED: the raw proxy systematically
overstates high-vol sectors (XLE: +35-40% even in calm 2021 — the VIX
term carries SPX's variance-risk premium and the full RV ratio
double-counts it), so a per-ETF scale factor k = mean(real/proxy) from
the overlap is applied. The calibration data ends 2022, entirely before
the backtest window — no lookahead. Output:
data/{xle,xlv,xli}_ivproxy.csv (date,dvol_close; 2023-01-01 onward).
"""
import csv
import datetime
import io
import json
import math
import urllib.request

P1 = 1577836800   # 2020-01-01, so RV63 has history before 2023 and the
P2 = 1782086400   # validation overlaps the discontinued indices
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
RV_LEN = 63


def yahoo_closes(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?"
           f"period1={P1}&period2={P2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    res = d["chart"]["result"][0]
    out = {}
    for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]):
        if c is not None:
            out[datetime.datetime.utcfromtimestamp(t).date()] = c
    return out


def cboe_closes(idx):
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
    rv = {}
    rets = []
    for i in range(1, len(days)):
        rets.append(math.log(closes[days[i]] / closes[days[i - 1]]))
        if len(rets) >= RV_LEN:
            window = rets[-RV_LEN:]
            m = sum(window) / RV_LEN
            var = sum((x - m) ** 2 for x in window) / (RV_LEN - 1)
            rv[days[i]] = math.sqrt(var * 252) * 100
    return rv


if __name__ == "__main__":
    spy = yahoo_closes("SPY")
    vix = yahoo_closes("%5EVIX")
    rv_spy = rv_series(spy)

    for sym, idx in [("XLE", "VXXLE"), ("XLV", "VXXLV"), ("XLI", "VXXLI")]:
        etf = yahoo_closes(sym)
        rv_etf = rv_series(etf)
        proxy = {}
        for day in sorted(rv_etf):
            if day in rv_spy and day in vix and rv_spy[day] > 0:
                proxy[day] = vix[day] * rv_etf[day] / rv_spy[day]

        real = cboe_closes(idx)
        overlap = sorted(set(proxy) & set(real))
        k = 1.0
        if overlap:
            diffs = [proxy[d] - real[d] for d in overlap]
            bias = sum(diffs) / len(diffs)
            mae = sum(abs(x) for x in diffs) / len(diffs)
            mp = sum(proxy[d] for d in overlap) / len(overlap)
            mr = sum(real[d] for d in overlap) / len(overlap)
            num = sum((proxy[d] - mp) * (real[d] - mr) for d in overlap)
            den = math.sqrt(sum((proxy[d] - mp) ** 2 for d in overlap)
                            * sum((real[d] - mr) ** 2 for d in overlap))
            k = sum(real[d] / proxy[d] for d in overlap) / len(overlap)
            print(f"{sym}: validation vs {idx} on {len(overlap)} days "
                  f"({overlap[0]} -> {overlap[-1]}): real mean {mr:.1f}, "
                  f"proxy mean {mp:.1f}, bias {bias:+.1f} vol pts, "
                  f"MAE {mae:.1f}, corr {num/den:.2f}, calib k={k:.3f}")
        proxy = {d: v * k for d, v in proxy.items()}

        with open(f"data/{sym.lower()}_ivproxy.csv", "w") as f:
            f.write("date,dvol_close\n")
            n = 0
            for day in sorted(proxy):
                if day >= datetime.date(2023, 1, 1):
                    f.write(f"{day},{proxy[day]:.4f}\n")
                    n += 1
        print(f"{sym}: {n} proxy rows written for 2023+")
