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

PSI_EPSILON = 1e-4  # avoids log(0) when a bin is empty on one side


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
    """Inner cut points that split `values` into roughly equal-sized bins (duplicates removed)."""
    ordered = sorted(values)
    n = len(ordered)
    edges = []
    for i in range(1, bins):
        cut = ordered[min(n - 1, int(i * n / bins))]
        if not edges or cut > edges[-1]:
            edges.append(cut)
    return edges


def _proportions(values: list, kind: str, edges: list) -> list:
    if kind == "binary":
        counts = [0, 0]
        for v in values:
            counts[1 if v else 0] += 1
    else:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def build_baseline(rows: list, bins: int = 10, source: str = "") -> dict:
    """rows: list of feature dicts (from text_features). Returns a JSON-serialisable baseline."""
    if not rows:
        raise ValueError("Cannot build a baseline from zero rows")
    features = {}
    for name, kind in FEATURES.items():
        values = [r[name] for r in rows]
        edges = [] if kind == "binary" else _quantile_edges(values, bins)
        features[name] = {
            "kind": kind,
            "edges": edges,
            "proportions": _proportions(values, kind, edges),
            "mean": sum(values) / len(values),
        }
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "count": len(rows),
        "features": features,
    }


def psi(expected: list, actual: list, eps: float = PSI_EPSILON) -> float:
    """Population Stability Index between two lists of bin proportions."""
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def compare(baseline: dict, rows: list) -> dict:
    """PSI and mean shift per feature for live `rows` against `baseline`."""
    result = {}
    for name, spec in baseline["features"].items():
        values = [r[name] for r in rows]
        actual = _proportions(values, spec["kind"], spec["edges"])
        result[name] = {
            "psi": round(psi(spec["proportions"], actual), 6),
            "baseline_mean": round(spec["mean"], 6),
            "live_mean": round(sum(values) / len(values), 6) if values else None,
        }
    return result


def severity(score: float) -> str:
    if score >= 0.2:
        return "significant"
    if score >= 0.1:
        return "moderate"
    return "none"
