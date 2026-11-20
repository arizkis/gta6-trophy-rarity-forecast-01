# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arizkis/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arizkis/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## Description

A quantile forecaster for the make.fun bounty "Forecast how many GTA VI players earn each
trophy in week one". For every trophy of Grand Theft Auto VI on PlayStation 5 it reads the
house's recording of the game's Exophase trophy page (every valid read before the case's
checkpoint) and writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the trophy's earn rate
at the case's target instant. Python 3.13, standard library only, no network, deterministic.

The model blends two views of the same history on the logit scale:

- **Trend** — least squares of logit(rate) against ln(1 + hours since 2026-11-19T00:00Z) over
  the reads of the last 24 hours, read forward at the target; the fit's residual rms feeds the
  spread.
- **Counts** — earner counts per page snapshot (nulls carried forward), a players-tracked
  forecast (the median of a weighted log-linear extrapolation and stretched-exponential
  saturation fits toward several asymptotes), and a band-shrunk earner-growth slope over the
  local 36-hour window, giving a counts-based median rate.

The final median is 0.15 counts + 0.85 trend (the trend clamped to within 2.2 logits of the
counts view); the spread combines horizon, trend residual and component disagreement, clamped
to [0.45, 1.5] logits. Trophies pinned at ≥ 99.5% in every snapshot get a one-sided downside
row near 99.99%. With fewer than 3 distinct page snapshots the program falls back to the last
known rate shaded ×0.8 (launch-window dilution) with a refresh-risk spread. Every per-trophy
computation is guarded: a failure falls back to a wide last-rate row, so a well-formed case
never crashes.

### Run it

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `*.in` case file in the input directory is read in name order; each writes `<id>.out`
to the output directory: exactly `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]},...]}`,
compact JSON, numbers rounded half up to 4 decimals (whole numbers as ints), trailing newline.
The grader runs `python3 -P -s /bot/main.py` with the directories defaulting to `/input` and
`/output`.

### Reading choices where the design was ambiguous

- `t_last` (the anchor for recency weights, the local window and `t_target = t_last + horizon`)
  is the last **read**'s timestamp, hours since the first read — the read closest to the
  checkpoint, rather than the last state's (possibly older) snapshot time.
- The spread's disagreement term uses the trend logit **after** its ±2.2 clamp around the
  counts logit (the same value the blend uses).
- The stretched-exponential profiling keeps the `u` with the best **weighted** residual rms
  in ln y space (the same weights as the fit).
- A null rate fails the pinned test (≥ 99.5 at *every* state); a trophy with no known rate
  anywhere is treated as 0.01% (the logit clamp floor) in fallbacks, and its earner count is 0.
- The trend window is the reads with timestamps in `[at − 24 h, at]`; band-ranking ties on the
  last rate break by trophy order (stable sort).

<sub>Entered on [make.fun](https://make.fun/?ref=8kbmacyg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
