"""Post-drawdown burst put-writing on SPY, 2007-2026 (data/ext2007).

The user's rule, formalized with minimal knobs:
  ARMED     : wait until close is >=10% below the trailing 252d high.
  TRIGGERED : wait for the first POSITIVE calendar month (confirmation).
  ACTIVE    : sell one 30-delta weekly put per week for 10 weeks
              (90% of Black-Scholes mark at VIX, opt fee per leg,
              cash-secured; the rest of the time fully in cash).
  DORMANT   : after the burst, re-arm only on a FRESH episode — either
              drawdown heals to <5% and a new >=10% drawdown forms, or
              price closes below the previous episode's trough (new leg
              down -> back to TRIGGERED, waiting for a positive month).

Economic basis (from regime_analysis.py): post-panic months are the
premium seller's best regime (rich VRP after the crash, vol crush), and
the positive-month confirmation avoids selling into the falling knife.
Parameters are the user's spec + canonical values (10% correction, 5%
heal, 30-delta, 10 weeks) — NOT tuned; a sensitivity grid is printed so
the fragility is visible instead of hidden.

Outputs: per-burst log, full stats (monthly-aggregated so Calmar etc.
compare with earlier tables), unconditional weekly PW-30d reference,
sensitivity grid, results_burst.csv, report_burst.png.
"""
import csv
import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_v2 import load_dvol
from backtest_v3 import (load_ohlc, bs_price, delta_strike,
                         SHORT_MARK_FRAC, SPY_COSTS)
from regime_analysis import sigma_at
from risk_report_v3 import stats as full_stats

DIR = "data/ext2007"
DD_ENTER, DD_REARM, BURST_WEEKS, PUT_DELTA = 0.10, 0.05, 10, 0.30
START = datetime.date(2007, 1, 1)
INK, SURFACE, GRID, MUTED, BASELINE = ("#0b0b0b", "#fcfcfb", "#e1e0d9",
                                       "#898781", "#c3c2b7")


def weekly_frames(dates):
    """[(i_first, i_last)] per ISO week."""
    out, cur = [], []
    for i, d in enumerate(dates):
        if cur and d.isocalendar()[:2] != dates[cur[0]].isocalendar()[:2]:
            out.append((cur[0], cur[-1]))
            cur = []
        cur.append(i)
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def weekly_put(dates, closes, i0, il, dvol, costs, delta=PUT_DELTA):
    spot0, settle = closes[i0], closes[il]
    days = max((dates[il] - dates[i0]).days, 1)
    T = days / 365.0
    sigma = sigma_at(dvol, dates[i0])
    if sigma is None:
        return None
    sigma /= 100.0
    K = delta_strike(spot0, sigma, T, delta, "p", 1)
    prem = SHORT_MARK_FRAC * bs_price(spot0, K, sigma, T, "p")
    fees = (costs["opt_fee_bps"] / 1e4) * spot0
    return (prem - max(0.0, K - settle) - fees) / spot0 * 100.0


def run(dates, closes, dvol, costs, dd_enter=DD_ENTER, burst_n=BURST_WEEKS,
        delta=PUT_DELTA, log=None):
    """Weekly return series (0 when idle) + active flags, from START."""
    frames = weekly_frames(dates)
    # month-over-month confirmation, known at each month's last close
    month_end = {}
    for i, d in enumerate(dates):
        month_end[(d.year, d.month)] = i
    me = sorted(month_end)
    pos_month_at = {}   # day index of month end -> was that month positive
    for k in range(1, len(me)):
        i_now, i_prev = month_end[me[k]], month_end[me[k - 1]]
        pos_month_at[i_now] = closes[i_now] > closes[i_prev]

    state, trough, burst_left = "ARMED", None, 0
    rets, active, week_dates = [], [], []
    burst_start, burst_pnl = None, 0.0
    for (i0, il) in frames:
        if dates[i0] < START:
            continue
        # state transitions use info up to the PREVIOUS close (i0-1)
        j = i0 - 1
        hi = max(closes[max(0, j - 251):j + 1])
        dd = 1 - closes[j] / hi
        if state == "ARMED" and dd >= dd_enter:
            state, trough = "TRIGGERED", closes[j]
        elif state == "TRIGGERED":
            trough = min(trough, closes[j])
            conf = [pos_month_at[k] for k in pos_month_at
                    if j - 5 <= k <= j]  # a month closed positive last week
            if conf and conf[-1]:
                state, burst_left = "ACTIVE", burst_n
                burst_start, burst_pnl = dates[i0], 0.0
        elif state == "DORMANT":
            if closes[j] < trough:
                state = "TRIGGERED"      # new leg down, wait for pos month
            elif dd < DD_REARM:
                state = "ARMED"

        if state == "ACTIVE":
            r = weekly_put(dates, closes, i0, il, dvol, costs, delta)
            r = 0.0 if r is None else r
            rets.append(r)
            active.append(True)
            burst_pnl += r
            burst_left -= 1
            if burst_left == 0:
                if log is not None:
                    log.append((burst_start, dates[il], burst_pnl))
                state = "DORMANT"
                trough = min(trough, min(closes[i0:il + 1]))
        else:
            rets.append(0.0)
            active.append(False)
        week_dates.append(dates[il])
    return week_dates, rets, active


def to_monthly(week_dates, rets):
    out, cur_key, acc = [], None, 0.0
    months = []
    for d, r in zip(week_dates, rets):
        key = (d.year, d.month)
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            out.append(acc)
            months.append(cur_key)
            cur_key, acc = key, 0.0
        acc += r
    out.append(acc)
    months.append(cur_key)
    return months, out


if __name__ == "__main__":
    dates, opens, highs, lows, closes = load_ohlc(f"{DIR}/spy_ohlc.csv")
    dvol = load_dvol(f"{DIR}/vix_daily.csv")
    costs = SPY_COSTS

    log = []
    wd, rets, active = run(dates, closes, dvol, costs, log=log)
    n_active = sum(active)
    print(f"{len(wd)} weeks 2007-2026, active {n_active} "
          f"({100*n_active/len(wd):.0f}%), {len(log)} bursts")
    print("\nburst log (start -> end, 10-week P&L in % of spot):")
    for s, e, p in log:
        print(f"  {s} -> {e}  {p:+6.2f}%")
    act = [r for r, a in zip(rets, active) if a]
    print(f"\nper active week: avg {sum(act)/len(act):+.3f}%, "
          f"win {100*sum(1 for x in act if x>0)/len(act):.0f}%, "
          f"worst {min(act):+.2f}%")

    # unconditional weekly PW-30d reference on the same engine
    frames = weekly_frames(dates)
    ref = []
    for (i0, il) in frames:
        if dates[i0] < START:
            continue
        r = weekly_put(dates, closes, i0, il, dvol, costs)
        ref.append(0.0 if r is None else r)

    mth, m_burst = to_monthly(wd, rets)
    _, m_ref = to_monthly(wd, ref)
    cols = ["n", "win%", "avg%/mo", "sd", "Sharpe", "Sortino", "maxDD_pp",
            "Calmar", "worst", "VaR95", "CVaR95", "skew", "total_pp"]
    print(f"\n{'':18}" + "".join(f"{c:>9}" for c in cols))
    rows_csv = []
    for lbl, series in [("BURST (as spec'd)", m_burst),
                        ("weekly PW always", m_ref)]:
        s = full_stats(series)
        print(f"{lbl:18}" + "".join(
            f"{s[c]:9.0f}" if c == "n" else f"{s[c]:9.2f}" for c in cols))
        rows_csv.append([lbl] + [round(s[c], 3) for c in cols])

    print("\nsensitivity (total pp over 2007-2026 / n bursts):")
    print(f"{'':12}" + "".join(f"{f'{b}wk':>12}" for b in (6, 10, 14)))
    for dd in (0.08, 0.10, 0.15):
        line = f"DD {dd:.0%}   "
        for b in (6, 10, 14):
            _, r2, a2 = run(dates, closes, dvol, costs, dd_enter=dd,
                            burst_n=b)
            lg = []
            run(dates, closes, dvol, costs, dd_enter=dd, burst_n=b, log=lg)
            line += f"{sum(r2):+8.1f}/{len(lg):<3d}"
        print(line)

    with open("results_burst.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
        w.writerows(rows_csv)
        w.writerow([])
        w.writerow(["burst_start", "burst_end", "pnl_pct"])
        for s, e, p in log:
            w.writerow([s, e, round(p, 3)])

    def cumsum(r):
        out, run_ = [], 0.0
        for x in r:
            run_ += x
            out.append(run_)
        return out

    def dd_series(cum):
        peak, out = -1e18, []
        for x in cum:
            peak = max(peak, x)
            out.append(x - peak)
        return out

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.6, 0.7],
                                          "hspace": 0.11})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax = axes[0]
    for lbl, series, color, lw in [
            ("weekly PW always-on", cumsum(ref), "#86b6ef", 1.3),
            ("BURST", cumsum(rets), INK, 2.4)]:
        ax.plot(wd, series, color=color, linewidth=lw, label=lbl,
                solid_capstyle="round")
        ax.annotate(f" {lbl} {series[-1]:+.0f}pp", (wd[-1], series[-1]),
                    color=color, fontsize=8.5, fontweight="bold",
                    va="center")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_title("Post-drawdown burst put-writing, SPY 2007 → 2026 "
                 "(active only after a confirmed recovery)", color=INK,
                 fontsize=11.5, loc="left", pad=10)
    ax.set_ylabel("cumulative (pp of spot)", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    ax.plot(wd, dd_series(cumsum(ref)), color="#86b6ef", linewidth=1.3,
            label="weekly PW always-on")
    ax.plot(wd, dd_series(cumsum(rets)), color=INK, linewidth=2.0,
            label="BURST")
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.set_ylabel("drawdown (pp)", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[2]
    ax.fill_between(wd, [1 if a else 0 for a in active], step="mid",
                    color="#1baf7a", alpha=0.7, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("active", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig("report_burst.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    print("\nwrote results_burst.csv, report_burst.png")
