"""Drift detection for the text2text endpoint, in plain Python (no numpy), so it runs in Lambda.

How it works
------------
1. Each request/response pair is turned into simple, explainable text features
   (lengths, question vs. task prompt, non-ASCII share, empty answers, ...).
2. A *baseline* is built from the training data: prompts from `input = target` lines,
   with the targets standing in for "expected answers". For every feature we store
   bin edges and the share of examples in each bin.
3. Live traffic (SageMaker data capture) is bucketed the same way and compared with the
   baseline using the Population Stability Index (PSI):
       PSI < 0.1  no meaningful change
       0.1-0.2    moderate change, keep an eye on it
       >= 0.2     significant drift
   A feature only counts if the shift is also statistically significant
   (chi-square test, Bonferroni-corrected), so small samples don't raise false alarms.
"""
import base64
import bisect
import json
import math
from datetime import datetime, timezone

# name -> kind. "numeric" features get quantile bins; "binary" features have two buckets (0/1).
FEATURES = {
    "input_words": "numeric",
    "input_chars": "numeric",
    "input_non_ascii_ratio": "numeric",
    "input_digit_ratio": "numeric",
    "input_is_question": "binary",
    "input_has_task_prefix": "binary",
    "output_words": "numeric",
    "output_chars": "numeric",
    "output_to_input_ratio": "numeric",
    "output_empty": "binary",
}

SMOOTHING = 0.5  # extra count per bin on both sides (keeps empty bins from dominating)
ALPHA = 0.01     # false-alarm rate for the significance test, shared across all features


# --------------------------------------------------------------------------- features
def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def text_features(prompt: str, answer: str) -> dict:
    """Numbers that describe one request (prompt) and its response (answer)."""
    prompt = prompt or ""
    answer = answer or ""
    p_words, a_words = len(prompt.split()), len(answer.split())
    head = prompt[:60]
    return {
        "input_words": p_words,
        "input_chars": len(prompt),
        "input_non_ascii_ratio": _ratio(sum(not c.isascii() for c in prompt), len(prompt)),
        "input_digit_ratio": _ratio(sum(c.isdigit() for c in prompt), len(prompt)),
        "input_is_question": int(prompt.rstrip().endswith("?")),
        # T5-style task prompts, e.g. "translate English to French: Hello"
        "input_has_task_prefix": int(":" in head and not head.lower().startswith("http")),
        "output_words": a_words,
        "output_chars": len(answer),
        "output_to_input_ratio": _ratio(a_words, max(p_words, 1)),
        "output_empty": int(not answer.strip()),
    }


# --------------------------------------------------------------------------- parsing
def parse_training_text(text: str) -> list:
    """`input = target` lines -> [(input, target)], split on the first '=' like train.py."""
    pairs = []
    for line in text.splitlines():
        if "=" in line:
            src, tgt = line.split("=", 1)
            if src.strip() and tgt.strip():
                pairs.append((src.strip(), tgt.strip()))
    return pairs


def _decode(event_data: dict) -> str:
    data = event_data.get("data", "")
    if str(event_data.get("encoding", "")).upper() == "BASE64":
        data = base64.b64decode(data).decode("utf-8", errors="replace")
    return data


def _prompt_from(payload) -> str:
    if isinstance(payload, dict):
        value = payload.get("inputs", payload.get("prompt", ""))
        return value if isinstance(value, str) else json.dumps(value)
    return payload if isinstance(payload, str) else json.dumps(payload)


def _answer_from(payload) -> str:
    # The Hugging Face container returns [{"generated_text": "..."}].
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        value = payload.get("generated_text", payload.get("answer", ""))
        return value if isinstance(value, str) else json.dumps(value)
    return payload if isinstance(payload, str) else json.dumps(payload)


def parse_capture_line(line: str):
    """One SageMaker data-capture JSON line -> (prompt, answer), or None if unusable."""
    try:
        record = json.loads(line)
        capture = record["captureData"]
        req_raw = _decode(capture["endpointInput"])
        res_raw = _decode(capture.get("endpointOutput", {}))
        try:
            request = json.loads(req_raw)
        except json.JSONDecodeError:
            request = req_raw
        try:
            response = json.loads(res_raw) if res_raw else ""
        except json.JSONDecodeError:
            response = res_raw
        return _prompt_from(request), _answer_from(response)
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- baseline + PSI
def _quantile_edges(values: list, bins: int) -> list:
    """Bin edges for a numeric feature: quantile cuts, plus an edge at the baseline minimum
    and just above the baseline maximum. Values outside the range seen in training then
    land in their own bins and show up as drift, even when the baseline is narrow or constant."""
    ordered = sorted(values)
    n = len(ordered)
    lo, hi = ordered[0], ordered[-1]
    edges = [lo]
    for i in range(1, bins):
        cut = ordered[min(n - 1, int(i * n / bins))]
        if cut > edges[-1]:
            edges.append(cut)
    top = hi + max(abs(hi), 1.0) * 1e-9
    if top > edges[-1]:
        edges.append(top)
    return edges


def _counts(values: list, kind: str, edges: list) -> list:
    if kind == "binary":
        counts = [0, 0]
        for v in values:
            counts[1 if v else 0] += 1
    else:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
    return counts


def build_baseline(rows: list, bins: int = 10, source: str = "") -> dict:
    """rows: list of feature dicts. Returns a JSON-serialisable baseline with per-feature bin counts."""
    if not rows:
        raise ValueError("Cannot build a baseline from zero rows")
    feats = {}
    for name, kind in FEATURES.items():
        values = [r[name] for r in rows]
        edges = [] if kind == "binary" else _quantile_edges(values, bins)
        counts = _counts(values, kind, edges)
        feats[name] = {
            "kind": kind,
            "edges": edges,
            "counts": counts,
            "proportions": [c / len(values) for c in counts],
            "mean": sum(values) / len(values),
        }
    return {"created": datetime.now(timezone.utc).isoformat(), "source": source, "count": len(rows), "features": feats}


def psi(expected_counts: list, actual_counts: list, smoothing: float = SMOOTHING):
    """Population Stability Index between two binned samples -> (psi, bins_used).

    Bins empty on both sides are dropped; every remaining bin gets `smoothing` extra counts
    on both sides, so a bin that is empty in one sample cannot blow the score up."""
    used = [i for i, (e, a) in enumerate(zip(expected_counts, actual_counts)) if e + a > 0] or [0]
    k = len(used)
    ne, na = sum(expected_counts), sum(actual_counts)
    total = 0.0
    for i in used:
        e = (expected_counts[i] + smoothing) / (ne + smoothing * k)
        a = (actual_counts[i] + smoothing) / (na + smoothing * k)
        total += (a - e) * math.log(a / e)
    return total, k


def chi2_sf(x: float, df: int) -> float:
    """P(Chi-square(df) >= x): regularized upper incomplete gamma Q(df/2, x/2)."""
    a, x = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series for the lower tail P, then Q = 1 - P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300  # continued fraction for Q (Lentz's method)
    b = x + 1 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return min(1.0, h * math.exp(log_prefix))


def compare(baseline: dict, rows: list, alpha: float = ALPHA) -> dict:
    """Per feature: PSI, chi-square p-value, and whether the shift is statistically significant.

    For two samples of sizes n and m, n_eff * PSI (n_eff = 1 / (1/n + 1/m)) is approximately
    chi-square with (bins - 1) degrees of freedom when nothing changed. A feature counts as
    drifted only if this test is significant at alpha / number-of-features (Bonferroni), so
    small or noisy samples do not raise false alarms."""
    per_feature_alpha = alpha / len(baseline["features"])
    n_base = baseline["count"]
    result = {}
    for name, spec in baseline["features"].items():
        values = [r[name] for r in rows]
        base_counts = spec.get("counts") or [round(p * n_base) for p in spec["proportions"]]
        live_counts = _counts(values, spec["kind"], spec["edges"])
        score, k = psi(base_counts, live_counts)
        n_eff = 1.0 / (1.0 / max(len(values), 1) + 1.0 / max(n_base, 1))
        p_value = chi2_sf(score * n_eff, max(k - 1, 1))
        result[name] = {
            "psi": round(score, 6),
            "p_value": float(f"{p_value:.6g}"),
            "significant": p_value < per_feature_alpha,
            "baseline_mean": round(spec["mean"], 6),
            "live_mean": round(sum(values) / len(values), 6) if values else None,
        }
    return result


def drift_score(features: dict) -> tuple:
    """(highest PSI among statistically significant features, that feature); (0.0, None) if none."""
    sig = {n: s["psi"] for n, s in features.items() if s["significant"]}
    if not sig:
        return 0.0, None
    worst = max(sig, key=sig.get)
    return sig[worst], worst


def severity(score: float) -> str:
    """For a drift_score: >= 0.2 significant, >= 0.1 moderate, otherwise none."""
    return "significant" if score >= 0.2 else "moderate" if score >= 0.1 else "none"
