"""Risk report for the v3 headline variant (ohlc ATR, entry-hysteresis
flip, no wings) across SPY, GOLD and OIL, plus an equal-weight combined
portfolio (1/3 of capital per asset, monthly ret = mean of the three).

Reads results_{spy,gold,oil}_v3.csv, prints a risk-stats table, writes
risk_stats_v3.csv and risk_report_v3.png (cumulative return, drawdown,
combined monthly bars). Returns are % of that month's spot, summed
arithmetically (repo convention), so curves are in percentage points.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ASSETS = [("SPY", "results_spy_v3.csv"),
          ("GOLD", "results_gold_v3.csv"),
          ("OIL", "results_oil_v3.csv")]

# reference dataviz palette, light mode: slots 1-3 + chrome
COLORS = {"SPY": "#2a78d6", "GOLD": "#eb6834", "OIL": "#1baf7a",
          "COMBINED": "#0b0b0b"}
POS, NEG = "#2a78d6", "#e34948"   # diverging pair for +/- bars
SURFACE, GRID, MUTED, INK = "#fcfcfb", "#e1e0d9", "#898781", "#0b0b0b"
BASELINE = "#c3c2b7"


def load_rets(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["month"]] = float(row["ret_pct"])
    return out


def drawdown(cum):
    peak, dd = -1e18, []
    for x in cum:
        peak = max(peak, x)
        dd.append(x - peak)
    return dd


def stats(rets):
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / (n - 1))
    downside = [min(0.0, x) for x in rets]
    dsd = math.sqrt(sum(d * d for d in downside) / n)
    cum = []
    run = 0.0
    for x in rets:
        run += x
        cum.append(run)
    mdd = min(drawdown(cum))
    srt = sorted(rets)
    k = max(1, round(0.05 * n))
    var95 = srt[k - 1]
    cvar95 = sum(srt[:k]) / k
    m3 = sum((x - mean) ** 3 for x in rets) / n
    m4 = sum((x - mean) ** 4 for x in rets) / n
    psd = math.sqrt(sum((x - mean) ** 2 for x in rets) / n)
    return {
        "n": n, "win%": 100.0 * sum(1 for x in rets if x > 0) / n,
        "avg%/mo": mean, "sd": sd,
        "Sharpe": mean / sd * math.sqrt(12) if sd else float("nan"),
        "Sortino": mean / dsd * math.sqrt(12) if dsd else float("nan"),
        "maxDD_pp": mdd,
        "Calmar": (mean * 12) / abs(mdd) if mdd else float("nan"),
        "worst": min(rets), "best": max(rets),
        "VaR95": var95, "CVaR95": cvar95,
        "skew": m3 / psd ** 3 if psd else float("nan"),
        "kurt_ex": m4 / psd ** 4 - 3 if psd else float("nan"),
        "total_pp": sum(rets),
    }


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb)


if __name__ == "__main__":
    series = {name: load_rets(path) for name, path in ASSETS}
    months = sorted(set.intersection(*(set(s) for s in series.values())))
    rets = {name: [series[name][m] for m in months] for name in series}
    rets["COMBINED"] = [sum(series[n][m] for n in series) / len(series)
                        for m in months]
    dates = [datetime.date(int(m[:4]), int(m[5:]), 1) for m in months]

    order = ["SPY", "GOLD", "OIL", "COMBINED"]
    all_stats = {name: stats(rets[name]) for name in order}

    cols = ["n", "win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "best", "VaR95", "CVaR95", "skew",
            "kurt_ex", "total_pp"]
    print(f"{'':9}" + "".join(f"{c:>9}" for c in cols))
    for name in order:
        s = all_stats[name]
        print(f"{name:9}" + "".join(
            f"{s[c]:9.0f}" if c == "n" else f"{s[c]:9.2f}" for c in cols))
    print("\nmonthly return correlations:")
    for i, a in enumerate(order[:3]):
        for b in order[i + 1:3]:
            print(f"  {a:5} vs {b:5}: {corr(rets[a], rets[b]):+.2f}")

    with open("risk_stats_v3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["portfolio"] + cols)
        for name in order:
            s = all_stats[name]
            w.writerow([name] + [s["n"]] + [round(s[c], 3) for c in cols[1:]])

    # ---- chart ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 2],
                                          "hspace": 0.16})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)

    cums = {}
    for name in order:
        run, cum = 0.0, []
        for x in rets[name]:
            run += x
            cum.append(run)
        cums[name] = cum

    ax = axes[0]
    for name in order:
        lw = 2.6 if name == "COMBINED" else 1.6
        ax.plot(dates, cums[name], color=COLORS[name], linewidth=lw,
                label=name, solid_capstyle="round")
        ax.annotate(f" {name} {cums[name][-1]:+.0f}pp",
                    (dates[-1], cums[name][-1]), color=COLORS[name],
                    fontsize=9, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Cumulative return — v3 headline variant "
                 "(ohlc ATR / entry rule / no wings)",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("cumulative % of spot (pp)", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

    ax = axes[1]
    for name in order:
        lw = 2.6 if name == "COMBINED" else 1.6
        ax.plot(dates, drawdown(cums[name]), color=COLORS[name],
                linewidth=lw, label=name, solid_capstyle="round")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Drawdown from running peak", color=INK, fontsize=12,
                 loc="left", pad=8)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)

    ax = axes[2]
    colors = [POS if x >= 0 else NEG for x in rets["COMBINED"]]
    ax.bar(dates, rets["COMBINED"], width=22, color=colors)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Combined portfolio — monthly return "
                 "(equal weight SPY / GOLD / OIL)", color=INK,
                 fontsize=12, loc="left", pad=8)
    ax.set_ylabel("% of spot / month", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.savefig("risk_report_v3.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote risk_stats_v3.csv, risk_report_v3.png")
