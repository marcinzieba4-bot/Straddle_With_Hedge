# Straddle With Hedge — ETHUSD / BTCUSD / SPY / CAT / SHW / TSLA backtest

Monthly ATM short straddle (call + put), hedged with 1 unit of the
underlying future whose side (long/short) is driven by a daily Renko
signal.

## Rules

- Each calendar month, sell ATM call + ATM put. Strike = spot rounded to
  the nearest strike step. Premium per side = a fixed % of spot, plus/minus
  the intrinsic moneyness created by rounding spot to that strike.
- On the initiation day, open 1 unit of future: **long** if the Renko signal
  is up, **short** if down. This is the starting hedge.
- During the month, a Renko flip alone is ignored. The hedge only flips
  when BOTH hold: price crosses the strike (neutral point) AND Renko
  agrees with the new direction.
- At expiry, options are cash-settled against intrinsic value and any open
  hedge leg is closed at the settlement price.
- Renko signal ported from the "Universal Renko Bars by SiddWolf" Pine
  indicator (ATR Based method, multiplier 0.15, tick_trend=2,
  tick_reversal=4, open_offset=0). ATR(14) is approximated from
  close-to-close ranges with Wilder's RMA smoothing (true OHLC ATR isn't
  available from the daily close-only price feed used here).

## Data

Daily closes pulled from Yahoo Finance's chart API, covering 2023-01-01 to
2026-06-21 (~42 months):

- `data/eth_usd_daily.csv` — ETHUSD
- `data/btc_usd_daily.csv` — BTCUSD
- `data/spy_daily.csv` — SPY
- `data/cat_daily.csv` — CAT
- `data/shw_daily.csv` — SHW
- `data/tsla_daily.csv` — TSLA

## Running

```
pip install matplotlib
python3 backtest.py --data data/eth_usd_daily.csv --premium-rate 0.068 --strike-step 50 --asset ETHUSD --out-csv results_eth.csv --out-png equity_curve_eth.png
python3 backtest.py --data data/btc_usd_daily.csv --premium-rate 0.05 --strike-step 500 --asset BTCUSD --out-csv results_btc.csv --out-png equity_curve_btc.png
python3 backtest.py --data data/spy_daily.csv --premium-rate 0.0166 --strike-step 1 --asset SPY --out-csv results_spy.csv --out-png equity_curve_spy.png
python3 backtest.py --data data/cat_daily.csv --premium-rate 0.0464 --strike-step 5 --asset CAT --out-csv results_cat.csv --out-png equity_curve_cat.png
python3 backtest.py --data data/shw_daily.csv --premium-rate 0.025 --strike-step 5 --asset SHW --out-csv results_shw.csv --out-png equity_curve_shw.png
python3 backtest.py --data data/tsla_daily.csv --premium-rate 0.05 --strike-step 5 --asset TSLA --out-csv results_tsla.csv --out-png equity_curve_tsla.png
```

Outputs a `--out-csv` (per-month P&L breakdown) and `--out-png`
(cumulative P&L chart), and prints a summary table to stdout.

## Result snapshots (42 months, 1 unit notional)

**ETHUSD** (6.8% premium per side, $50 strike step)
- 23/42 winning months
- Total P&L: ~$2,943 (~4.44% of spot per month on average)

**BTCUSD** (5% premium per side, $500 strike step)
- 31/42 winning months
- Total P&L: ~$87,888 (~9.88% of spot per month on average)

**SPY** (1.66% premium per side, $1 strike step)
- 27/42 winning months
- Total P&L: ~$179.1 (~1.05% of spot per month on average)

**CAT** (4.64% premium per side, $5 strike step)
- 35/42 winning months
- Total P&L: ~$794.4 (~7.22% of spot per month on average)

**SHW** (2.5% premium per side, $5 strike step)
- 24/42 winning months
- Total P&L: ~$42.7 (~0.41% of spot per month on average)

**TSLA** (5% premium per side, $5 strike step)
- 30/42 winning months
- Total P&L: ~$204.8 (~3.04% of spot per month on average)

This is a simplified, fixed-premium model (not real options market prices)
intended to illustrate the hedge-switching mechanics, not as a production
P&L estimate.

## v3 (realistic conditions) — now including SPY

`backtest_v3.py` reprices the straddle off an implied-vol index at each
month's initiation (Black-Scholes ATM approximation), applies 90%/110%
bid/ask marks, per-fill fees, and next-open hedge fills, and sweeps a grid
of variants (ATR source × flip rule × optional long wings). For crypto the
vol index is Deribit DVOL; for SPY it is the **CBOE VIX** (same convention:
30-day IV, annualized %), fetched by `fetch_vix.py` into
`data/vix_daily.csv`. SPY OHLC comes from `fetch_ohlc.py` into
`data/spy_ohlc.csv` (both pinned in `data/` for reproducibility).

The SPY hedge is 1 unit of an **ES/MES index future** (delta ≈ SPY x 10
per MES; the model works in 1-SPY-share units). Costs differ from the
crypto perp: ~1bp per fill covers commission + half-spread + slippage, and
there is **no funding drag** — cost of carry sits in the futures basis and
roughly nets out for a hedge that flips long/short (`SPY_COSTS` in
`backtest_v3.py`; crypto keeps 5bp fills + 0.5bp/day funding).

```
python3 fetch_ohlc.py && python3 fetch_vix.py
python3 backtest_v3.py   # prints the grid for ETHUSD, BTCUSD, SPY
```

**SPY v3 snapshot (42 months, 2023-01 → 2026-06, 1-share units):**

- Headline variant `ohlc / entry / no wings` (true-range ATR,
  break-even-hysteresis flip, dumped to `results_spy_v3.csv`):
  27/42 winning months, +0.71%/mo avg, sd 2.82, **Sharpe 0.87**,
  max DD -8.9pp, worst month -6.7%, total +29.7pp (~$155 per share unit).
- Best variant on this window is `cc / flip / no wings`: +0.99%/mo,
  **Sharpe 1.26**, max DD -10.2pp.
- Long wings (0.15/0.20-delta put+call) cut SPY returns to ~0 — with VIX
  this low, the 110%-of-mark wing cost eats most of the straddle premium.
- Unlike ETH (which degrades to ~0 Sharpe under v3 realism), SPY stays
  positive across all no-wing variants (Sharpe 0.74–1.26): lower vol means
  fewer Renko whipsaws and far cheaper hedge fills relative to premium.

### EURUSD

Same pipeline, hedged with CME 6E futures (`FX_COSTS`: 0.5bp per fill,
no funding — the EUR/USD rate differential sits in the forward points and
largely nets out for a flipping hedge). Strike step 0.005 (6E option
grid). Vol index: **CBOE EVZ** (EuroCurrency Volatility Index, 30-day IV,
annualized %), fetched by `fetch_cboe_vol.py` into `data/evz_daily.csv`.
CBOE discontinued EVZ on 2025-03-11, so the run covers **27 months,
2023-01 → 2025-03** (months without a vol quote at initiation are
skipped; the vol lookup tolerates up to 5 days of calendar mismatch since
FX trades on CBOE holidays).

**EURUSD v3 snapshot (27 months, 1-EUR units):** essentially flat.
Headline `ohlc / entry / no wings` (→ `results_eurusd_v3.csv`):
17/27 winning months, +0.09%/mo, Sharpe 0.18, worst month -5.3%.
Best variant `cc / cross / no wings`: +0.21%/mo, Sharpe 0.63. All wing
variants are negative. With EVZ at 7–10%, the straddle collects only
~1.6% of spot per month, and Renko whipsaws in a ranging currency eat
most of it — the premium-to-noise ratio that carries SPY isn't there.

### GOLD

Same pipeline on COMEX front-month gold futures (`GC=F` from Yahoo into
`data/gold_ohlc.csv`) — a clean fit, since GC options are options on that
same future, so carry/contango is already inside the price series. Vol
index: **CBOE GVZ** (Gold Volatility Index, 30-day IV, annualized %),
still published, fetched by `fetch_cboe_vol.py` (which now pins both EVZ
and GVZ) into `data/gvz_daily.csv`. Hedge: GC/MGC futures (`GOLD_COSTS`:
0.5bp per fill, no funding). Strike step $5.

**GOLD v3 snapshot (42 months, 2023-01 → 2026-06, 1-oz units):** the
verdict depends on the ATR source, which is a robustness warning in
itself.

- Headline `ohlc / entry / no wings` (→ `results_gold_v3.csv`): flat —
  25/42 winning months, -0.02%/mo, Sharpe -0.02, worst month -15.5%.
- All `cc`-ATR flip/entry variants: +1.2–1.3%/mo, Sharpe ~1.3. The
  smaller close-to-close bricks track gold's relentless 2024–2026 uptrend
  (roughly $2,000 → $4,200), so the hedge leg captures trend; the larger
  true-range bricks flip more (78 vs 57–62) and give it all back.
- A spread this wide between two ATR definitions on the same rules means
  the gold result is signal-parameter luck, not a robust edge — treat the
  cc Sharpe 1.3 as in-sample selection on a historic bull run.

### OIL (WTI)

Same pipeline on NYMEX front-month WTI futures (`CL=F` from Yahoo into
`data/oil_ohlc.csv`; CL options trade on the same future). Vol index:
**CBOE OVX** (Crude Oil Volatility Index, 30-day IV of USO options,
annualized %), still published, pinned by `fetch_cboe_vol.py` into
`data/ovx_daily.csv`. Hedge: CL/MCL futures (`OIL_COSTS`: 1bp per fill,
no funding). Strike step $0.50.

**OIL v3 snapshot (42 months, 2023-01 → 2026-06, 1-bbl units):**
marginal at best, with ugly tails.

- Headline `ohlc / entry / no wings` (→ `results_oil_v3.csv`): 24/42
  winning months, +0.82%/mo, sd 6.96, Sharpe 0.41, max DD -44pp, worst
  month -16.3%.
- Every `cc`-ATR variant is *negative* (down to -0.7%/mo) — the exact
  mirror image of gold, where cc won and ohlc lost. Which ATR definition
  "works" flips per asset, i.e. neither is an edge.
- All wing variants are negative despite OVX's 30–45 vol: crude's fat
  tails make the wings genuinely expensive.
- With OVX high, the straddle collects ~7–10% of spot per month — but
  choppy trends burn 69–123 hedge flips and monthly sd is ~7%, so the
  premium is fair compensation for the risk, not free income. The
  front-month series also rolls mid-month (contango/backwardation jumps
  land in the P&L as noise a real single-expiry position wouldn't see).

### Per-asset configuration without hindsight (walk-forward)

`adaptive_config_v3.py` asks whether each asset's "best" grid variant was
learnable in real time: every month it trades the config (out of
{cc,ohlc} × {flip,cross,entry}, no wings) with the best *trailing*
12-month Sharpe for that asset — zero lookahead — and compares against
each fixed config over the same evaluation window (2024-01 → 2026-06).
The cc-vs-ohlc ATR axis is effectively a brick-size knob (cc ATR is
roughly half of true-range ATR, so cc bricks are ~2x smaller).

- **GOLD: the cc preference was learnable.** Walk-forward earns
  +1.07%/mo, Sharpe 1.00 (vs 1.56 for the hindsight-best fixed cc/flip,
  -0.09 for the headline ohlc/entry). The adaptive picker converges on
  cc/flip within the burn-in and stays near it. Gold legitimately wants
  small bricks + follow-every-flip: its 2024–2026 drift was persistent
  enough that the hedge should chase every signal.
- **OIL: no learnable config exists.** Walk-forward is *negative*
  (-1.01%/mo, Sharpe -0.47) even though it mostly picks ohlc/flip — the
  retrospective winner. Trailing performance simply doesn't predict
  next-month performance for crude; the picker is always one regime
  late. Oil's positive full-window grid rows are selection artifacts,
  not a tradeable characteristic.
- **SPY: robust to the choice.** Walk-forward Sharpe 1.26, within the
  1.0–1.7 band of every decent fixed config — consistent with SPY being
  the only asset where all no-wing variants agree.
- Realized-price diagnostics (daily autocorrelation ~0, 20-day variance
  ratio ~0.7 for all three) do NOT separate the assets — gold's cc edge
  comes from multi-month drift, not daily trendiness, which is why it
  must be learned from strategy P&L (as above) rather than read off a
  price statistic.
