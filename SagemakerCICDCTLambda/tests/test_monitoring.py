"""Tests for drift monitoring: run with `pytest SagemakerCICDCTLambda/tests` (needs pytest + boto3).

The Lambdas are exercised end to end against in-memory fakes of S3 and CloudWatch,
using records shaped like real SageMaker data capture.
"""
import base64
import importlib
import io
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "monitoring"))
sys.path.insert(0, str(ROOT / "lambda"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import drift  # noqa: E402


# --------------------------------------------------------------------------- fakes
class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.exceptions = type("E", (), {"NoSuchKey": NoSuchKey})

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name):
        s3 = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for b, k in s3.objects if b == Bucket and k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]} if keys else {}
        return Paginator()


class FakeCloudWatch:
    def __init__(self):
        self.calls = []

    def put_metric_data(self, Namespace, MetricData):
        self.calls.append((Namespace, MetricData))


# --------------------------------------------------------------------------- helpers
TRAINING_LINES = [
    "translate English to French: Hello = Bonjour",
    "translate English to French: Thank you very much = Merci beaucoup",
    "translate English to German: Good morning = Guten Morgen",
    "What is the capital of France? = Paris",
    "What is the capital of Japan? = Tokyo",
    "summarize: The meeting covered budget, hiring and the product roadmap = Budget, hiring, roadmap",
]


def training_text(n=300, seed=0):
    rnd = random.Random(seed)
    return "\n".join(rnd.choice(TRAINING_LINES) for _ in range(n)) + "\nnot a pair line\n"


def capture_line(prompt, answer, base64_encode=False):
    req, res = json.dumps({"inputs": prompt}), json.dumps([{"generated_text": answer}])
    if base64_encode:
        enc = lambda s: {"observedContentType": "application/json", "mode": "X", "encoding": "BASE64",
                         "data": base64.b64encode(s.encode()).decode()}
        inp, out = enc(req), enc(res)
    else:
        inp = {"observedContentType": "application/json", "mode": "INPUT", "encoding": "JSON", "data": req}
        out = {"observedContentType": "application/json", "mode": "OUTPUT", "encoding": "JSON", "data": res}
    return json.dumps({"captureData": {"endpointInput": inp, "endpointOutput": out},
                       "eventMetadata": {"eventId": "x", "inferenceTime": "2026-10-30T12:00:00Z"},
                       "eventVersion": "0"})


def load_lambda(module, env, monkeypatch, s3, cw=None):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.reload(importlib.import_module(module))
    monkeypatch.setattr(mod, "s3", s3)
    if cw is not None:
        monkeypatch.setattr(mod, "cloudwatch", cw)
    return mod


ENV = {"BUCKET": "b", "ENDPOINT_NAME": "llm-endpoint", "MIN_SAMPLES": "30", "WINDOW_HOURS": "3"}


def put_capture(s3, lines, hours_ago=0):
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    key = f"datacapture/llm-endpoint/AllTraffic/{t:%Y/%m/%d/%H}/{t:%M-%S}-abc.jsonl"
    s3.put_object(Bucket="b", Key=key, Body="\n".join(lines) + "\n")


# --------------------------------------------------------------------------- unit tests
def test_text_features():
    f = drift.text_features("What is the capital of France?", "Paris")
    assert f["input_words"] == 6 and f["input_is_question"] == 1 and f["input_has_task_prefix"] == 0
    assert f["output_words"] == 1 and f["output_empty"] == 0
    f = drift.text_features("translate English to French: Hello", "")
    assert f["input_has_task_prefix"] == 1 and f["output_empty"] == 1 and f["output_to_input_ratio"] == 0
    assert set(f) == set(drift.FEATURES)


def test_parse_training_text_matches_train_py_rules():
    pairs = drift.parse_training_text("a = b\nx = y = z\nno equals\n = empty\nonly = \n")
    assert pairs == [("a", "b"), ("x", "y = z")]


@pytest.mark.parametrize("b64", [False, True])
def test_parse_capture_line(b64):
    assert drift.parse_capture_line(capture_line("Hi there", "Salut", b64)) == ("Hi there", "Salut")


def test_parse_capture_line_bad_input():
    assert drift.parse_capture_line("not json") is None
    assert drift.parse_capture_line(json.dumps({"other": 1})) is None


def test_psi_identical_is_zero_and_shift_is_large():
    assert drift.psi([50, 50], [50, 50])[0] == pytest.approx(0)
    assert drift.psi([90, 10], [10, 90])[0] > 1
    assert drift.psi([100, 0, 0], [0, 0, 100])[1] == 2  # bins empty on both sides are dropped


def test_same_distribution_has_low_psi():
    rows = [drift.text_features(s, t) for s, t in drift.parse_training_text(training_text(400, 1))]
    base = drift.build_baseline(rows[:200])
    scores = drift.compare(base, rows[200:])
    assert max(s["psi"] for s in scores.values()) < 0.1


def test_values_outside_training_range_count_as_drift():
    # Regression: with a constant baseline, larger/smaller live values used to share the
    # baseline's bin and score PSI 0.
    rows = [drift.text_features("What is the capital of France?", "Paris")] * 100
    base = drift.build_baseline(rows)
    same = drift.compare(base, rows)["input_words"]["psi"]
    longer = drift.compare(base, [drift.text_features("word " * 40, "Paris")] * 50)["input_words"]["psi"]
    shorter = drift.compare(base, [drift.text_features("Hi?", "Paris")] * 50)["input_words"]["psi"]
    assert same == pytest.approx(0) and longer >= 0.2 and shorter >= 0.2


def test_chi2_sf_matches_known_critical_values():
    assert drift.chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    assert drift.chi2_sf(11.070, 5) == pytest.approx(0.05, abs=1e-3)
    assert drift.chi2_sf(23.209, 10) == pytest.approx(0.01, abs=1e-3)
    assert drift.chi2_sf(0, 3) == 1.0


def _sample(rnd, n, shift=False):
    rows = []
    for _ in range(n):
        words = rnd.randint(40, 120) if shift else rnd.randint(3, 15)
        prompt = ("translate English to French: " if rnd.random() < 0.5 else "What is ") + "word " * words
        rows.append(drift.text_features(prompt.strip() + ("?" if rnd.random() < 0.3 else ""), "x " * rnd.randint(1, 6)))
    return rows


def test_low_false_alarm_rate_and_high_detection():
    # Unchanged traffic: random samples of 30-200 should almost never be flagged.
    alarms = 0
    for seed in range(60):
        rnd = random.Random(seed)
        base = drift.build_baseline(_sample(rnd, 300))
        score, _ = drift.drift_score(drift.compare(base, _sample(rnd, rnd.choice([30, 60, 200]))))
        alarms += score >= 0.2
    assert alarms <= 2, alarms
    # Clearly shifted traffic (much longer prompts) is always flagged, even with 30 samples.
    for seed in range(20):
        rnd = random.Random(1000 + seed)
        base = drift.build_baseline(_sample(rnd, 300))
        score, feature = drift.drift_score(drift.compare(base, _sample(rnd, 30, shift=True)))
        assert score >= 0.2 and feature is not None


# --------------------------------------------------------------------------- Lambda tests
def test_baseline_lambda_writes_baseline(monkeypatch):
    s3 = FakeS3()
    s3.put_object(Bucket="b", Key="input/data.txt", Body=training_text())
    s3.put_object(Bucket="b", Key="input/ignore.csv", Body="x = y")
    mod = load_lambda("baseline", ENV, monkeypatch, s3)
    out = mod.lambda_handler({"detail": {"TrainingJobName": "llm-train-1"}}, None)
    assert out["baseline_updated"] and out["examples"] == 300
    base = json.loads(s3.objects[("b", "monitoring/baseline.json")])
    assert base["count"] == 300 and set(base["features"]) == set(drift.FEATURES)
    assert "llm-train-1" in base["source"]


def test_baseline_lambda_no_data(monkeypatch):
    mod = load_lambda("baseline", ENV, monkeypatch, FakeS3())
    assert mod.lambda_handler({}, None) == {"baseline_updated": False}


def _setup_check(monkeypatch):
    s3, cw = FakeS3(), FakeCloudWatch()
    s3.put_object(Bucket="b", Key="input/data.txt", Body=training_text())
    load_lambda("baseline", ENV, monkeypatch, s3).lambda_handler({}, None)
    return s3, cw, load_lambda("check_drift", ENV, monkeypatch, s3, cw)


def _metrics(cw):
    return {(m["MetricName"], tuple(d["Value"] for d in m["Dimensions"])): m["Value"]
            for _, data in cw.calls for m in data}


def test_check_no_baseline(monkeypatch):
    mod = load_lambda("check_drift", ENV, monkeypatch, FakeS3(), FakeCloudWatch())
    assert mod.lambda_handler({}, None)["status"] == "no_baseline"


def test_check_insufficient_data(monkeypatch):
    s3, cw, mod = _setup_check(monkeypatch)
    put_capture(s3, [capture_line("What is the capital of Spain?", "Madrid")] * 5)
    out = mod.lambda_handler({}, None)
    assert out["status"] == "insufficient_data" and out["samples"] == 5
    assert _metrics(cw)[("SampleCount", ("llm-endpoint",))] == 5
    assert ("DriftScore", ("llm-endpoint",)) not in _metrics(cw)


def test_check_similar_traffic_no_drift(monkeypatch):
    s3, cw, mod = _setup_check(monkeypatch)
    rnd = random.Random(7)
    lines = []
    for _ in range(120):
        src, tgt = rnd.choice(TRAINING_LINES).split(" = ")
        lines.append(capture_line(src, tgt, base64_encode=rnd.random() < 0.3))
    put_capture(s3, lines[:60], hours_ago=0)
    put_capture(s3, lines[60:], hours_ago=2)
    put_capture(s3, lines, hours_ago=5)  # outside the 3h window: must be ignored
    out = mod.lambda_handler({}, None)
    assert out["samples"] == 120 and out["status"] == "none", out
    m = _metrics(cw)
    assert m[("DriftScore", ("llm-endpoint",))] == 0
    assert ("PSI", ("llm-endpoint", "input_words")) in m
    report = json.loads(s3.objects[("b", out["report"])])
    assert report["status"] == "none" and len(report["features"]) == len(drift.FEATURES)


def test_check_detects_drift(monkeypatch):
    s3, cw, mod = _setup_check(monkeypatch)
    # Users now paste long, non-English customer emails and the model answers with nothing.
    long_prompt = "Bonjour, je voudrais être remboursé pour ma commande numéro 48213, livrée cassée. " * 4
    put_capture(s3, [capture_line(long_prompt, "") for _ in range(80)])
    out = mod.lambda_handler({}, None)
    assert out["status"] == "significant" and out["drift_score"] >= 0.2, out
    report = json.loads(s3.objects[("b", out["report"])])
    assert report["features"]["output_empty"]["psi"] >= 0.2 and report["features"]["output_empty"]["significant"]
    assert report["features"]["input_words"]["live_mean"] > report["features"]["input_words"]["baseline_mean"]


def test_deploy_lambda_adds_data_capture(monkeypatch):
    calls = {}

    class FakeSM:
        exceptions = type("E", (), {"ClientError": Exception})

        def describe_training_job(self, **_):
            return {"ModelArtifacts": {"S3ModelArtifacts": "s3://b/model.tar.gz"}}

        def create_model(self, **kw):
            calls["model"] = kw

        def create_endpoint_config(self, **kw):
            calls["config"] = kw

        def describe_endpoint(self, **_):
            raise Exception("Could not find endpoint")

        def create_endpoint(self, **kw):
            calls["endpoint"] = kw

    env = {"SAGEMAKER_ROLE": "arn:role", "ENDPOINT_NAME": "llm-endpoint", "INFERENCE_IMAGE": "img",
           "DATA_CAPTURE_S3_URI": "s3://b/datacapture", "DATA_CAPTURE_PERCENT": "50"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.reload(importlib.import_module("deploy_model"))
    monkeypatch.setattr(mod, "sagemaker", FakeSM())
    out = mod.lambda_handler({"detail": {"TrainingJobName": "llm-train-1", "TrainingJobStatus": "Completed"}}, None)
    assert out["action"] == "created"
    cap = calls["config"]["DataCaptureConfig"]
    assert cap["EnableCapture"] and cap["InitialSamplingPercentage"] == 50
    assert cap["DestinationS3Uri"] == "s3://b/datacapture"
    assert cap["CaptureContentTypeHeader"] == {"JsonContentTypes": ["application/json"]}

    monkeypatch.setenv("DATA_CAPTURE_S3_URI", "")
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "sagemaker", FakeSM())
    mod.lambda_handler({"detail": {"TrainingJobName": "llm-train-2", "TrainingJobStatus": "Completed"}}, None)
    assert "DataCaptureConfig" not in calls["config"]
