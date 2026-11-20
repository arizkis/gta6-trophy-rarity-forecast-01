"""gta6-trophy-rarity-forecast-01: trend-and-counts quantile forecaster (Python 3.13, stdlib only).

Forecasts each GTA VI trophy's Exophase earn rate at the case's target as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball loss on the logit scale.

Model (design frozen by its designer, validated on synthetic trajectory backtests):
  - A state is a distinct page snapshot: a new state starts whenever (players_tracked, rates)
    changes walking the history in order (null is a sentinel). Fewer than 3 states -> fallback:
    last known rate shaded x0.8 (launch-window dilution), refresh-risk spread.
  - Pinned trophies (rate >= 99.5 at every state) get a one-sided downside row near 99.99.
  - Trend: least squares of logit(rate) on ln(1 + hours since 2026-11-19T00:00Z) over the reads
    of the last 24 h, read at the target; its residual rms feeds the spread.
  - Counts: earner counts per state (nulls carried forward), a players forecast P-hat (median of
    a weighted log-linear fit and stretched-exponential saturation fits), a band-shrunk earner
    slope over the local 36 h window -> a counts median rate.
  - Blend 0.15 counts / 0.85 trend (trend clamped to counts +/- 2.2 logits), spread from horizon,
    trend rms and component disagreement, clamped to [0.45, 1.5] logits.

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs `python3 -P -s /bot/main.py` with one case file in /input and writes /output.
Every *.in file is read in name order; each <id>.out holds exactly
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, compact JSON, 4-decimal half-up
rounding (whole numbers as ints), trailing newline, <= 65536 bytes, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path; keep this file self-contained (single module).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard normal quantiles at the scored levels (0.05, 0.25, 0.5, 0.75, 0.95).
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
# Fixed epoch for the trend component's time axis.
EPOCH_S = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
# The page prints 2 decimals; 0% and 100% have no logit.
RATE_MIN, RATE_MAX = 0.01, 99.99


def clamp_rate(p):
    return min(RATE_MAX, max(RATE_MIN, p))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints (100, not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s):
    """ISO 8601 timestamp -> epoch seconds (naive stamps read as UTC)."""
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wlsq(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, weighted rms) or None if degenerate."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    rms = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return a, b, rms


def median(vals):
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def quantiles(lg, sigma):
    return [inv_logit(lg + z * sigma) for z in Z]


def pinned_row():
    """One-sided downside: the expensive error is a 100% trophy slipping below 100%."""
    lg_max = logit(RATE_MAX)
    return [inv_logit(lg_max - 1.645 * 1.6), inv_logit(lg_max - 0.674 * 1.6),
            RATE_MAX, RATE_MAX, RATE_MAX]


def wide_row(last_rate):
    """Per-trophy failure fallback: last-rate row with a wide spread."""
    return quantiles(logit(last_rate if last_rate is not None else RATE_MIN), 1.5)


def fallback_row(last_rate, days, horizon_hours):
    """Fewer than 3 distinct states: shaded last rate with a refresh-risk spread."""
    if last_rate is not None and last_rate >= 99.5:
        return pinned_row()
    m = last_rate if last_rate is not None else RATE_MIN
    if m < RATE_MAX:
        m *= 0.8  # launch-window dilution shade
    sigma = math.sqrt(0.25 * days + 1.21 * min(1.0, horizon_hours / 1.5))
    return quantiles(logit(m), sigma)


def build_states(reads, n):
    """Distinct page snapshots: (players, rates) change -> new state. t = hours since first read."""
    states = []
    first_s = reads[0]["at_s"]
    prev_key = None
    for r in reads:
        rates = r["rates"]
        key = (r["players"], tuple(rates))
        if key != prev_key:
            states.append({"t": (r["at_s"] - first_s) / 3600.0,
                           "players": r["players"], "rates": rates})
            prev_key = key
    return states


def players_forecast(states, t_last, t_target):
    """Median of a weighted log-linear fit and stretched-exponential saturation fits."""
    ts = [s["t"] for s in states]
    ps = [s["players"] for s in states]
    p_last = ps[-1]
    ws = [0.5 ** ((t_last - t) / 24.0) for t in ts]
    preds = []

    # (i) weighted log P on log1p(t).
    if all(p > 0 for p in ps):
        fit = wlsq([math.log1p(t) for t in ts], [math.log(p) for p in ps], ws)
        if fit is not None:
            a, b, _ = fit
            val = math.exp(a + b * math.log1p(t_target))
            if math.isfinite(val) and val > 0:
                preds.append(val)

    # (ii) stretched exponential P = A*(1 - exp(-(t/u)^b)) toward asymptotes A.
    for mult in (1.6, 2.5, 4.0, 7.0, 12.0):
        a_sym = mult * p_last
        if a_sym <= 0 or any(p >= 0.98 * a_sym for p in ps):
            continue
        best = None
        for u in (0.5, 2.0, 6.0, 14.0, 30.0):
            xs, ys, wu = [], [], []
            for t, p, w in zip(ts, ps, ws):
                if p <= 0:
                    continue
                y = -math.log(1.0 - p / a_sym)
                if y <= 0:
                    continue
                xs.append(math.log(t + u))
                ys.append(math.log(y))
                wu.append(w)
            fit = wlsq(xs, ys, wu)
            if fit is None:
                continue
            a, b, rms = fit
            if best is None or rms < best[0]:
                best = (rms, a, b, u)
        if best is not None:
            _, a, b, u = best
            inner = a + b * math.log(t_target + u)
            if inner < 700:  # keep exp(exp(...)) finite
                val = a_sym * (1.0 - math.exp(-math.exp(inner)))
                if math.isfinite(val) and val > 0:
                    preds.append(val)

    preds = [max(v, p_last) for v in preds]
    p_hat = median(preds) if preds else p_last
    if not math.isfinite(p_hat) or p_hat <= 0:
        raise ValueError("no usable players forecast")
    return p_hat


def earner_counts(states, n):
    """Per-trophy earner count per state; nulls carry forward, leading nulls take the first known."""
    counts = [[0.0] * len(states) for _ in range(n)]
    for i in range(n):
        known = [(j, s["rates"][i] / 100.0 * s["players"])
                 for j, s in enumerate(states) if s["rates"][i] is not None]
        if not known:
            continue  # null everywhere -> count 0
        first_val = known[0][1]
        prev = first_val
        k = 0
        for j in range(len(states)):
            if k < len(known) and known[k][0] == j:
                prev = known[k][1]
                k += 1
            counts[i][j] = prev
    return counts


def trend_component(reads, i, at_s, target_s, last_rate):
    """Logit-rate trend over the last 24 h of reads; returns (median logit, residual rms)."""
    xs, ys = [], []
    for r in reads:
        if r["rates"][i] is None or not (at_s - 86400.0 <= r["at_s"] <= at_s):
            continue
        h = (r["at_s"] - EPOCH_S) / 3600.0
        if 1.0 + h <= 0:
            continue
        xs.append(math.log(1.0 + h))
        ys.append(logit(r["rates"][i]))
    lg_last = logit(last_rate if last_rate is not None else RATE_MIN)
    if len(xs) < 3:
        return lg_last, 0.5
    fit = wlsq(xs, ys, [1.0] * len(xs))
    if fit is None:
        return lg_last, 0.5
    a, b, rms = fit
    h_target = (target_s - EPOCH_S) / 3600.0
    if 1.0 + h_target <= 0:
        return lg_last, 0.5
    lg = a + b * math.log(1.0 + h_target)
    if not math.isfinite(lg):
        return lg_last, 0.5
    return lg, rms


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    horizon_minutes = float(case.get("horizon_minutes") or 0.0)
    horizon_hours = horizon_minutes / 60.0
    days = horizon_minutes / 1440.0
    at_s = parse_ts(case["at"])
    try:
        target_s = parse_ts(case["target"])
    except (KeyError, TypeError, ValueError):
        target_s = at_s + horizon_minutes * 60.0

    reads = []
    for r in case.get("history") or []:
        rates = list(r.get("rates") or [])
        rates = (rates + [None] * n)[:n]
        players = r.get("players_tracked")
        reads.append({"at_s": parse_ts(r["at"]),
                      "players": float(players) if players is not None else 0.0,
                      "rates": rates})

    # Last known rate per trophy, searching back through the reads.
    last_known = [None] * n
    for r in reversed(reads):
        for i in range(n):
            if last_known[i] is None and r["rates"][i] is not None:
                last_known[i] = r["rates"][i]

    rows = []
    if len(reads) == 0:
        for i in range(n):
            rows.append({"id": trophies[i]["id"], "q": wide_row(None)})
        return {"trophies": rows}

    states = build_states(reads, n)
    if len(states) < 3:
        for i in range(n):
            try:
                q = fallback_row(last_known[i], days, horizon_hours)
            except Exception:
                q = wide_row(last_known[i])
            rows.append({"id": trophies[i]["id"], "q": q})
        return {"trophies": rows}

    t_last = (reads[-1]["at_s"] - reads[0]["at_s"]) / 3600.0
    t_target = t_last + horizon_hours
    p_hat = players_forecast(states, t_last, t_target)
    counts = earner_counts(states, n)

    # Local window: states within 36 h of the last read (all states if fewer than 3).
    local_idx = [j for j, s in enumerate(states) if s["t"] >= t_last - 36.0]
    if len(local_idx) < 3:
        local_idx = list(range(len(states)))
    ws_local = [0.5 ** ((t_last - states[j]["t"]) / 24.0) for j in local_idx]
    xs_local = [math.log1p(states[j]["t"]) for j in local_idx]

    # Per-trophy trend and earner slope.
    trends, slopes, anchors, caps = [], [], [], []
    for i in range(n):
        try:
            lg_trend, rms = trend_component(reads, i, at_s, target_s, last_known[i])
        except Exception:
            lg_trend, rms = logit(last_known[i] if last_known[i] is not None else RATE_MIN), 0.5
        trends.append((lg_trend, rms))
        fit = wlsq(xs_local, [math.log(max(counts[i][j], 0.35)) for j in local_idx], ws_local)
        slopes.append(fit[1] if fit is not None else 0.0)
        tail = local_idx[-3:]
        anchors.append(sum(counts[i][j] for j in tail) / len(tail))
        local_rates = [states[j]["rates"][i] for j in local_idx if states[j]["rates"][i] is not None]
        caps.append(2.2 * max([RATE_MIN] + local_rates))

    # Band shrinkage of the earner slopes: rank by last rate, three bands.
    order = sorted(range(n), key=lambda i: (last_known[i] if last_known[i] is not None else -1.0, i))
    band_of = [0] * n
    for rank, i in enumerate(order):
        band_of[i] = min(2, rank * 3 // n)
    band_slopes = []
    for b in range(3):
        members = [slopes[i] for i in range(n) if band_of[i] == b]
        band_slopes.append(sum(members) / len(members) if members else 0.0)

    delta_x = math.log1p(t_target) - math.log1p(t_last)
    for i in range(n):
        try:
            shrunk = 0.72 * slopes[i] + 0.28 * band_slopes[band_of[i]]
            if abs(slopes[i]) < 0.02:
                growth = 0.0
            else:
                growth = min(1.4, max(-0.7, shrunk * delta_x * 0.6))
            e_hat = min(anchors[i] * math.exp(growth), p_hat)
            counts_rate = min(100.0 * e_hat / p_hat, caps[i])
            lg_counts = logit(counts_rate)
            lg_trend, rms = trends[i]
            lg_trend = min(lg_counts + 2.2, max(lg_counts - 2.2, lg_trend))
            lg = 0.15 * lg_counts + 0.85 * lg_trend
            sigma = math.sqrt(0.70 ** 2 * (0.25 * days + rms ** 2)
                              + (0.25 * abs(lg_counts - lg_trend)) ** 2)
            sigma = min(1.5, max(0.45, sigma))
            q = quantiles(lg, sigma)
        except Exception:
            q = wide_row(last_known[i])
        rows.append({"id": trophies[i]["id"], "q": q})
    return {"trophies": rows}


def sanitize(rows):
    """Finite, in [0, 100], non-decreasing, rounded half up to 4 decimals."""
    out = []
    for row in rows:
        q, prev = [], 0.0
        for v in row["q"]:
            v = float(v)
            if not math.isfinite(v):
                v = prev
            v = min(100.0, max(prev, max(0.0, v)))
            q.append(round4(v))
            prev = v
        out.append({"id": row["id"], "q": q})
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = {"trophies": sanitize(forecast_case(case)["trophies"])}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
