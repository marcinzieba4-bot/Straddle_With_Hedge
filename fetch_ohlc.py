"""Fetch ETH-USD, BTC-USD, SPY, EURUSD, gold (COMEX GC front month) and
WTI crude (NYMEX CL front month) daily OHLC from Yahoo Finance into
data/{eth,btc,spy,eurusd,gold,oil}_ohlc.csv, used by backtest_v3.py."""
import json
import datetime
import urllib.request

for sym, name in [("ETH-USD", "eth"), ("BTC-USD", "btc"), ("SPY", "spy"),
                  ("EURUSD=X", "eurusd"), ("GC=F", "gold"), ("CL=F", "oil"),
                  ("XLE", "xle"), ("XLV", "xlv"), ("XLI", "xli"),
                  ("XLP", "xlp"), ("XLU", "xlu")]:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?"
           f"period1=1672531200&period2=1782086400&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows, seen = [], set()
    for i, t in enumerate(ts):
        day = datetime.datetime.utcfromtimestamp(t).date()
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c) or day in seen:
            continue
        seen.add(day)
        rows.append(f"{day},{o:.6f},{h:.6f},{l:.6f},{c:.6f}")
    with open(f"data/{name}_ohlc.csv", "w") as f:
        f.write("date,open,high,low,close\n" + "\n".join(rows) + "\n")
    print(name, len(rows), "rows written")
