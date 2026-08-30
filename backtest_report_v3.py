"""Tearsheet for the five surviving assets, each on its best full-window
v3 configuration (no wings everywhere — wings lose on every asset):

    SPY  cc/flip     XLK  cc/flip     XLV  cc/cross
    XLY  ohlc/itm    GOLD cc/flip

Per asset: equity curve, drawdown, monthly bars (report_<asset>.png) and
full risk stats. Then the equal-weight portfolio of the five
(report_combined.png) with the same stats plus the correlation matrix.
Also writes report_v3.html — a self-contained page with the charts
embedded — and report_stats_v3.csv.

NOTE: 'best configuration' is in-sample selection over the 42-month
window; the walk-forward study (adaptive_config_v3.py) is the honest
check on which of these choices were learnable in real time.
"""
import base64
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, run_variant,
                         SPY_COSTS, GOLD_COSTS, ETF_COSTS)
from risk_report_v3 import stats, drawdown, corr

ASSETS = [
    ("SPY", "cc", "flip", "data/spy_ohlc.csv", "data/vix_daily.csv", 1,
     SPY_COSTS, "#2a78d6", "S&P 500 · ES futures hedge · VIX premium"),
    ("XLK", "cc", "flip", "data/xlk_ohlc.csv", "data/xlk_ivproxy.csv", 1,
     ETF_COSTS, "#eb6834", "Technology · share hedge · VXXLK-validated proxy"),
    ("XLV", "cc", "cross", "data/xlv_ohlc.csv", "data/xlv_ivproxy.csv", 1,
     ETF_COSTS, "#1baf7a", "Health care · share hedge · VXXLV-validated proxy"),
    ("XLY", "ohlc", "itm", "data/xly_ohlc.csv", "data/xly_ivproxy.csv", 1,
     ETF_COSTS, "#eda100", "Consumer discretionary · share hedge · VXXLY-validated proxy"),
    ("GOLD", "cc", "flip", "data/gold_ohlc.csv", "data/gvz_daily.csv", 5,
     GOLD_COSTS, "#e87ba4", "COMEX GC futures · GC hedge · GVZ premium"),
]
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"

STAT_COLS = ["n", "win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
             "Calmar", "worst", "best", "VaR95", "CVaR95", "skew", "total_pp"]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def cumsum(rets):
    out, run = [], 0.0
    for x in rets:
        run += x
        out.append(run)
    return out


def month_dates(months):
    return [datetime.date(int(m[:4]), int(m[5:]), 1) for m in months]


def asset_chart(name, months, rets, color, path):
    dates = month_dates(months)
    cum = cumsum(rets)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.4, 1.6],
                                          "hspace": 0.14})
    for ax in axes:
        style_ax(ax)
    ax = axes[0]
    ax.plot(dates, cum, color=color, linewidth=2.2, solid_capstyle="round")
    ax.fill_between(dates, cum, 0, color=color, alpha=0.08, linewidth=0)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.annotate(f" {cum[-1]:+.0f}pp", (dates[-1], cum[-1]), color=color,
                fontsize=10, fontweight="bold", va="center")
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax = axes[1]
    ax.plot(dates, drawdown(cum), color=color, linewidth=1.8)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax = axes[2]
    ax.bar(dates, rets, width=22, color=[POS if x >= 0 else NEG for x in rets])
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("% / month", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def combined_chart(months, per_asset, ew, path):
    dates = month_dates(months)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.4, 1.6],
                                          "hspace": 0.14})
    for ax in axes:
        style_ax(ax)
    ax = axes[0]
    for name, rets, color in per_asset:
        ax.plot(dates, cumsum(rets), color=color, linewidth=1.3,
                alpha=0.85, label=name)
    cum_ew = cumsum(ew)
    ax.plot(dates, cum_ew, color=INK, linewidth=2.6, label="EQUAL-WEIGHT",
            solid_capstyle="round")
    ax.annotate(f" EW {cum_ew[-1]:+.0f}pp", (dates[-1], cum_ew[-1]),
                color=INK, fontsize=10, fontweight="bold", va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3,
              labelcolor=INK)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax = axes[1]
    ax.plot(dates, drawdown(cum_ew), color=INK, linewidth=1.8)
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("EW drawdown (pp)", color=MUTED, fontsize=8.5)
    ax = axes[2]
    ax.bar(dates, ew, width=22, color=[POS if x >= 0 else NEG for x in ew])
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("EW % / month", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def img64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def stats_rows_html(s, flips=None):
    fmt = {"n": "{:.0f}", "win%": "{:.0f}%"}
    cells = []
    for c in STAT_COLS:
        v = fmt.get(c, "{:+.2f}").format(s[c]) if c in ("n", "win%") \
            else f"{s[c]:+.2f}"
        cells.append((c, v))
    if flips is not None:
        cells.append(("flips", str(flips)))
    return "".join(f"<div class='stat'><span>{k}</span><b>{v}</b></div>"
                   for k, v in cells)


if __name__ == "__main__":
    series, colors, flips, descs, cfgs = {}, {}, {}, {}, {}
    for name, atr, rule, ohlc_csv, dvol_csv, step, costs, color, desc in ASSETS:
        dates, opens, highs, lows, closes = load_ohlc(ohlc_csv)
        dvol = load_dvol(dvol_csv)
        res = run_variant(dates, opens, highs, lows, closes, dvol, step,
                          atr, rule, None, **costs)
        series[name] = {r["month"]: r["ret_pct"] for r in res}
        flips[name] = sum(r["n_switches"] for r in res)
        colors[name], descs[name] = color, desc
        cfgs[name] = f"{atr} / {rule} / no wings"

    months = sorted(set.intersection(*(set(s) for s in series.values())))
    names = list(series)
    rets = {n: [series[n][m] for m in months] for n in names}
    ew = [sum(rets[n][i] for n in names) / len(names)
          for i in range(len(months))]

    all_stats = {n: stats(rets[n]) for n in names}
    ew_stats = stats(ew)

    with open("report_stats_v3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["portfolio", "config"] + STAT_COLS + ["flips"])
        for n in names:
            w.writerow([n, cfgs[n]] + [round(all_stats[n][c], 3)
                                       for c in STAT_COLS] + [flips[n]])
        w.writerow(["EQUAL-WEIGHT", "1/5 each"] +
                   [round(ew_stats[c], 3) for c in STAT_COLS] + [""])

    for n in names:
        asset_chart(n, months, rets[n], colors[n],
                    f"report_{n.lower()}.png")
    combined_chart(months, [(n, rets[n], colors[n]) for n in names], ew,
                   "report_combined.png")

    corr_head = "".join(f"<th>{n}</th>" for n in names)
    corr_rows = ""
    for a in names:
        cells = "".join(
            f"<td>{corr(rets[a], rets[b]):+.2f}</td>" if a != b else
            "<td class='diag'>1.00</td>" for b in names)
        corr_rows += f"<tr><th>{a}</th>{cells}</tr>"

    sections = ""
    for n in names:
        s = all_stats[n]
        sections += f"""
<section>
  <div class="head">
    <span class="dot" style="background:{colors[n]}"></span>
    <h2>{n}</h2><span class="cfg">{cfgs[n]}</span>
    <span class="desc">{descs[n]}</span>
  </div>
  <div class="tiles">
    <div class="tile hero"><span>Sharpe</span><b>{s['Sharpe']:.2f}</b></div>
    <div class="tile"><span>avg / month</span><b>{s['avg%/mo']:+.2f}%</b></div>
    <div class="tile"><span>max drawdown</span><b>{s['maxDD_pp']:.1f}pp</b></div>
    <div class="tile"><span>total</span><b>{s['total_pp']:+.1f}pp</b></div>
  </div>
  <div class="stats">{stats_rows_html(s, flips[n])}</div>
  <img src="{img64(f'report_{n.lower()}.png')}" alt="{n} equity, drawdown and monthly returns">
</section>"""

    html = f"""<title>Straddle Hedge Tearsheet</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ background:#f4f4f1; color:#0b0b0b; margin:0;
         font-family:'IBM Plex Sans',system-ui,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:40px 20px 64px; }}
  header h1 {{ font-size:30px; font-weight:700; margin:0 0 4px;
               letter-spacing:-0.01em; text-wrap:balance; }}
  header p {{ color:#52514e; margin:0; font-size:14.5px; max-width:64ch;
              line-height:1.55; }}
  .meta {{ color:#898781; font-size:12px; text-transform:uppercase;
           letter-spacing:0.08em; margin-top:10px; }}
  section {{ background:#fcfcfb; border:1px solid rgba(11,11,11,0.10);
             border-radius:6px; padding:22px 24px 16px; margin-top:26px; }}
  .head {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
  .head h2 {{ font-size:20px; margin:0; }}
  .dot {{ width:11px; height:11px; border-radius:50%; align-self:center; }}
  .cfg {{ font-family:'IBM Plex Mono',monospace; font-size:12px;
          background:#f0efec; border-radius:4px; padding:2px 8px;
          color:#0b0b0b; }}
  .desc {{ color:#898781; font-size:12.5px; }}
  .tiles {{ display:flex; gap:10px; margin:16px 0 6px; flex-wrap:wrap; }}
  .tile {{ background:#f4f4f1; border-radius:5px; padding:10px 16px;
           display:flex; flex-direction:column; gap:2px; min-width:104px; }}
  .tile span {{ font-size:11px; color:#898781; text-transform:uppercase;
                letter-spacing:0.06em; }}
  .tile b {{ font-size:21px; font-variant-numeric:tabular-nums; }}
  .tile.hero {{ background:#0b0b0b; }}
  .tile.hero span {{ color:#c3c2b7; }}
  .tile.hero b {{ color:#fcfcfb; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(96px,1fr));
            gap:6px 14px; margin:12px 0 14px; }}
  .stat {{ display:flex; justify-content:space-between; font-size:12.5px;
           border-bottom:1px solid #e1e0d9; padding-bottom:3px; }}
  .stat span {{ color:#898781; }}
  .stat b {{ font-family:'IBM Plex Mono',monospace; font-weight:500;
             font-variant-numeric:tabular-nums; }}
  img {{ max-width:100%; border-radius:4px; }}
  table.corr {{ border-collapse:collapse; margin:14px 0 6px;
                font-family:'IBM Plex Mono',monospace; font-size:12.5px;
                font-variant-numeric:tabular-nums; }}
  table.corr th, table.corr td {{ padding:5px 12px; text-align:right;
                border-bottom:1px solid #e1e0d9; color:#0b0b0b; }}
  table.corr th {{ color:#52514e; font-weight:600; }}
  table.corr td.diag {{ color:#c3c2b7; }}
  .note {{ background:#f0efec; border-radius:5px; padding:12px 16px;
           font-size:13px; color:#52514e; line-height:1.55; margin-top:18px; }}
  .note b {{ color:#0b0b0b; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<div class="wrap">
<header>
  <h1>Straddle Hedge Tearsheet</h1>
  <p>Monthly ATM short straddle, premium repriced off each asset's
     implied-vol index, hedged by one unit of future/underlying whose side
     follows a daily Renko signal with per-asset flip hysteresis. Realistic
     conditions throughout: 90%/110% bid-ask marks, per-fill fees,
     next-open hedge fills. Returns are % of each month's spot, summed
     arithmetically.</p>
  <div class="meta">{months[0]} → {months[-1]} · {len(months)} months ·
     best full-window config per asset (in-sample selection — see note)</div>
</header>
{sections}
<section>
  <div class="head"><span class="dot" style="background:#0b0b0b"></span>
    <h2>Equal-weight portfolio</h2><span class="cfg">1/5 per asset</span>
    <span class="desc">rebalanced monthly across the five sleeves</span></div>
  <div class="tiles">
    <div class="tile hero"><span>Sharpe</span><b>{ew_stats['Sharpe']:.2f}</b></div>
    <div class="tile"><span>avg / month</span><b>{ew_stats['avg%/mo']:+.2f}%</b></div>
    <div class="tile"><span>max drawdown</span><b>{ew_stats['maxDD_pp']:.1f}pp</b></div>
    <div class="tile"><span>worst month</span><b>{ew_stats['worst']:+.1f}%</b></div>
  </div>
  <div class="stats">{stats_rows_html(ew_stats)}</div>
  <img src="{img64('report_combined.png')}" alt="Combined equity curves, equal-weight drawdown and monthly returns">
  <h2 style="font-size:16px;margin:18px 0 0">Monthly strategy-return correlations</h2>
  <table class="corr"><tr><th></th>{corr_head}</tr>{corr_rows}</table>
</section>
<div class="note"><b>Read before trading:</b> each sleeve runs its best
  configuration chosen on this same 42-month window — in-sample selection
  that flatters every number here. The walk-forward study in the repo
  found SPY robust to the choice, GOLD's cc/flip learnable in real time,
  and XLY/XLV/XLK's exact configs untested out-of-window. All five sleeves
  are short unhedged-tail volatility: 2023–2026 contains no true vol
  crisis, correlations would converge toward 1 in one, and the equal-weight
  worst month would not hold. Premiums for XLK/XLV/XLY come from
  RV-scaled VIX proxies validated against CBOE's discontinued sector vol
  indices on 2020–2022; SPY/GOLD use live indices (VIX, GVZ).</div>
</div>
"""
    with open("report_v3.html", "w") as f:
        f.write(html)

    print(f"{'':13}" + "".join(f"{c:>9}" for c in STAT_COLS))
    for n in names + ["EW"]:
        s = ew_stats if n == "EW" else all_stats[n]
        print(f"{n:6} {cfgs.get(n,'1/5 each'):>6}" if False else
              f"{n:13}" + "".join(
                  f"{s[c]:9.0f}" if c == "n" else f"{s[c]:9.2f}"
                  for c in STAT_COLS))
    print("wrote report_{spy,xlk,xlv,xly,gold,combined}.png, "
          "report_stats_v3.csv, report_v3.html")
