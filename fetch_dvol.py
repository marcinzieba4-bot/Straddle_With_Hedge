"""Fetch Deribit DVOL (30-day implied volatility index) daily history into
data/dvol_eth.csv and data/dvol_btc.csv, used by backtest_v2.py."""
import json
import time
import datetime
import urllib.request


def fetch_all(currency, start=datetime.datetime(2023, 1, 1),
              end=datetime.datetime(2026, 6, 22)):
    seen = {}
    d0 = start
    while d0 < end:
        d1 = min(d0 + datetime.timedelta(days=240), end)
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
               f"currency={currency}&start_timestamp={int(d0.timestamp()*1000)}"
               f"&end_timestamp={int(d1.timestamp()*1000)}&resolution=86400")
        data = json.load(urllib.request.urlopen(url, timeout=20))["result"]["data"]
        for ts, o, h, l, c in data:
            day = datetime.datetime.utcfromtimestamp(ts / 1000).date().isoformat()
            seen[day] = c
        d0 = d1
        time.sleep(0.25)
    return seen


if __name__ == "__main__":
    for cur in ["ETH", "BTC"]:
        seen = fetch_all(cur)
        with open(f"data/dvol_{cur.lower()}.csv", "w") as f:
            f.write("date,dvol_close\n")
            for day in sorted(seen):
                f.write(f"{day},{seen[day]:.4f}\n")
        print(cur, len(seen), "days written")
