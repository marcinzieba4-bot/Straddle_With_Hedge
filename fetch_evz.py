"""Fetch CBOE EVZ (EuroCurrency Volatility Index) daily history into
data/evz_daily.csv.

EVZ is the EURUSD analog of DVOL/VIX: 30-day implied volatility of euro
(FXE) options, annualized %. CBOE discontinued the index on 2025-03-11,
so the series covers only part of the backtest window — months whose
initiation date has no EVZ quote are skipped by backtest_v3.
Written in the same date,dvol_close format as the other vol files.
"""
import csv
import datetime
import io
import urllib.request

URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/EVZ_History.csv"

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
txt = urllib.request.urlopen(req, timeout=45).read().decode()
rows = []
for row in csv.DictReader(io.StringIO(txt)):
    day = datetime.datetime.strptime(row["DATE"], "%m/%d/%Y").date()
    if day >= datetime.date(2023, 1, 1):
        rows.append(f"{day},{float(row['EVZ']):.4f}")
with open("data/evz_daily.csv", "w") as f:
    f.write("date,dvol_close\n" + "\n".join(rows) + "\n")
print("evz", len(rows), "rows written")
