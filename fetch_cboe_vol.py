"""Fetch CBOE volatility index daily histories into data/ (replaces the
EVZ-only fetch_evz.py).

- EVZ (EuroCurrency Volatility Index): EURUSD analog of DVOL/VIX,
  discontinued 2025-03-11 — the series simply ends there.
- GVZ (Gold Volatility Index): 30-day IV of gold options, still published.

Both are annualized %, written in the same date,dvol_close format as the
other vol files so they drop straight into load_dvol().
"""
import csv
import datetime
import io
import urllib.request

BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"

for idx, out in [("EVZ", "data/evz_daily.csv"), ("GVZ", "data/gvz_daily.csv")]:
    req = urllib.request.Request(BASE.format(idx),
                                 headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=45).read().decode()
    rows = []
    for row in csv.DictReader(io.StringIO(txt)):
        day = datetime.datetime.strptime(row["DATE"], "%m/%d/%Y").date()
        if day >= datetime.date(2023, 1, 1):
            rows.append(f"{day},{float(row[idx]):.4f}")
    with open(out, "w") as f:
        f.write("date,dvol_close\n" + "\n".join(rows) + "\n")
    print(idx.lower(), len(rows), "rows written")
