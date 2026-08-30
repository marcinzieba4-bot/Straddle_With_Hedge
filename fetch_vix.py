"""Fetch CBOE VIX daily closes from Yahoo Finance into data/vix_daily.csv.

VIX is the SPY-world analog of Deribit's DVOL: a 30-day implied-volatility
index quoted in annualized %, so it drops straight into load_dvol() /
the v2-v3 premium repricing. Written in the same date,dvol_close format.
"""
import json
import datetime
import urllib.request

url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?"
       "period1=1672531200&period2=1782086400&interval=1d")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
d = json.load(urllib.request.urlopen(req, timeout=20))
res = d["chart"]["result"][0]
ts = res["timestamp"]
closes = res["indicators"]["quote"][0]["close"]
rows, seen = [], set()
for t, c in zip(ts, closes):
    day = datetime.datetime.utcfromtimestamp(t).date()
    if c is None or day in seen:
        continue
    seen.add(day)
    rows.append(f"{day},{c:.4f}")
with open("data/vix_daily.csv", "w") as f:
    f.write("date,dvol_close\n" + "\n".join(rows) + "\n")
print("vix", len(rows), "rows written")
