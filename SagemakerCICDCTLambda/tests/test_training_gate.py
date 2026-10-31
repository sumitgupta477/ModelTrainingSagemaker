"""Tests for held-out evaluation (train.py) and the champion/challenger deploy gate.

    pip install pytest boto3                                    # metric, split, Lambda tests
    pip install torch transformers tokenizers                   # + the real tiny-T5 training test
    pytest SagemakerCICDCTLambda/tests

The training test builds a tiny randomly initialised T5 (no downloads), trains it on CPU,
evaluates it against a second tiny model acting as the deployed "champion", and checks that
the trigger Lambda's MetricDefinitions regexes actually match what train.py prints.
"""
import importlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda"))
sys.path.insert(0, str(ROOT / "training"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

CAPITALS = {"france": "paris", "japan": "tokyo", "italy": "rome", "spain": "madrid", "egypt": "cairo",
            "kenya": "nairobi", "peru": "lima", "chile": "santiago", "india": "delhi", "china": "beijing",
            "canada": "ottawa", "mexico": "mexico city", "cuba": "havana", "greece": "athens", "norway": "oslo",
            "sweden": "stockholm", "poland": "warsaw", "austria": "vienna", "portugal": "lisbon", "ireland": "dublin",
            "russia": "moscow", "turkey": "ankara", "iran": "tehran", "iraq": "baghdad", "nepal": "kathmandu",
            "ghana": "accra", "qatar": "doha", "fiji": "suva", "laos": "vientiane", "chad": "ndjamena"}


def train_module():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return importlib.import_module("train")


# --------------------------------------------------------------------------- metrics + split
def test_metrics():
    t = train_module()
    assert t.exact_match("The Paris.", "paris") == 1.0
    assert t.exact_match("Paris, France", "Paris") == 0.0
    assert t.token_f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert t.token_f1("", "") == 1.0 and t.token_f1("x", "") == 0.0
    assert t.score(["paris", "rome"], ["paris", "madrid"]) == {"exact_match": 0.5, "token_f1": 0.5}


def test_split_is_stable_and_has_no_leakage():
    t = train_module()
    pairs = [(f"what is the capital of {c}?", cap) for c, cap in CAPITALS.items()]
    train1, eval1, on_train = t.split_pairs(pairs, eval_percent=20)
    assert eval1 and train1 and not on_train
    # New data never moves existing examples between train and eval.
    more = pairs + [("what is the capital of brazil?", "brasilia"), ("what is the capital of japan?", "tokyo")]
    train2, eval2, _ = t.split_pairs(more, eval_percent=20)
    assert set(eval1) <= set(eval2) and set(train1) <= set(train2)
    # The duplicate of a held-out input is held out too: no leakage into training.
    assert not {s for s, _ in train2} & {s for s, _ in eval2}


def test_split_tiny_datasets():
    t = train_module()
    two = [("a?", "x"), ("b?", "y")]
    tr, ev, on_train = t.split_pairs(two, eval_percent=0)
    assert len(tr) == 1 and len(ev) == 1 and not on_train
    one = [("only?", "z"), ("only?", "z")]
    tr, ev, on_train = t.split_pairs(one, eval_percent=0)
    assert on_train and tr == ev == one


# --------------------------------------------------------------------------- trigger Lambda
class FakeSM:
    def __init__(self, endpoint=True):
        self.endpoint, self.jobs = endpoint, []

    def describe_endpoint(self, EndpointName):
        if not self.endpoint:
            raise Exception("Could not find endpoint")
        return {"EndpointConfigName": "cfg-1"}

    def describe_endpoint_config(self, EndpointConfigName):
        return {"ProductionVariants": [{"ModelName": "llm-train-1"}]}

    def describe_model(self, ModelName):
        return {"PrimaryContainer": {"ModelDataUrl": "s3://b/llm-model/output/llm-train-1/output/model.tar.gz"}}

    def create_training_job(self, **kw):
        self.jobs.append(kw)


def load_trigger(monkeypatch, sm):
    for k, v in {"SAGEMAKER_ROLE": "arn:role", "BUCKET": "b", "TRAINING_IMAGE": "img",
                 "ENDPOINT_NAME": "llm-endpoint"}.items():
        monkeypatch.setenv(k, v)
    mod = importlib.reload(importlib.import_module("trigger_training"))
    monkeypatch.setattr(mod, "sagemaker", sm)
    return mod


S3_EVENT = {"Records": [{"s3": {"object": {"key": "input/new.txt"}}}]}


@pytest.mark.parametrize("endpoint", [True, False])
def test_trigger_passes_champion_and_metric_definitions(monkeypatch, endpoint):
    sm = FakeSM(endpoint)
    load_trigger(monkeypatch, sm).lambda_handler(S3_EVENT, None)
    job = sm.jobs[0]
    hp = job["HyperParameters"]
    if endpoint:
        assert json.loads(hp["champion_model_s3"]).endswith("llm-train-1/output/model.tar.gz")
    else:
        assert "champion_model_s3" not in hp
    names = {m["Name"] for m in job["AlgorithmSpecification"]["MetricDefinitions"]}
    assert {"train:loss", "eval:candidate_token_f1", "eval:champion_token_f1"} <= names
    assert job["EnableManagedSpotTraining"] is True


# --------------------------------------------------------------------------- deploy gate
class FakeDeploySM:
    exceptions = type("E", (), {"ClientError": Exception})

    def __init__(self, metrics, endpoint=True):
        self.metrics, self.endpoint, self.calls = metrics, endpoint, []

    def describe_training_job(self, TrainingJobName):
        return {"TrainingJobArn": f"arn:aws:sagemaker:us-east-1:1:training-job/{TrainingJobName}",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/model.tar.gz"},
                "FinalMetricDataList": [{"MetricName": f"eval:{k}", "Value": v} for k, v in self.metrics.items()]
                + [{"MetricName": "train:loss", "Value": 0.1}]}

    def __getattr__(self, name):
        def record(**kw):
            self.calls.append((name, kw))
            if name == "describe_endpoint" and not self.endpoint:
                raise Exception("Could not find endpoint")
            return {}
        return record


class Recorder:
    def __init__(self):
        self.calls = []

    def publish(self, **kw):
        self.calls.append(kw)

    def invoke(self, **kw):
        self.calls.append(kw)


def run_deploy(monkeypatch, metrics, endpoint=True, **env):
    base = {"SAGEMAKER_ROLE": "arn:role", "ENDPOINT_NAME": "llm-endpoint", "INFERENCE_IMAGE": "img",
            "ALERT_TOPIC_ARN": "arn:aws:sns:us-east-1:1:alerts", "BASELINE_FUNCTION": "llm-chat-drift-baseline",
            "MIN_TOKEN_F1": "0.0", "DEPLOY_TOLERANCE": "0.0", "DEPLOY_WITHOUT_METRICS": "false"}
    base.update(env)
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    mod = importlib.reload(importlib.import_module("deploy_model"))
    sm, sns, lam = FakeDeploySM(metrics, endpoint), Recorder(), Recorder()
    monkeypatch.setattr(mod, "sagemaker", sm)
    monkeypatch.setattr(mod, "sns", sns)
    monkeypatch.setattr(mod, "lambda_client", lam)
    out = mod.lambda_handler({"detail": {"TrainingJobName": "llm-train-2", "TrainingJobStatus": "Completed"}}, None)
    tags = {t["Key"]: t["Value"] for name, kw in sm.calls if name == "add_tags" for t in kw["Tags"]}
    return out, sm, sns, lam, tags


def test_gate_deploys_better_model_and_rebuilds_baseline(monkeypatch):
    out, sm, sns, lam, tags = run_deploy(monkeypatch, {"candidate_token_f1": 0.8, "champion_token_f1": 0.7})
    assert out["deployed"] and out["action"] == "updated"
    assert [n for n, _ in sm.calls if n.startswith("create")] == ["create_model", "create_endpoint_config"]
    assert tags["deployment-decision"] == "deployed" and not sns.calls
    assert json.loads(lam.calls[0]["Payload"])["detail"]["TrainingJobName"] == "llm-train-2"
    assert lam.calls[0]["InvocationType"] == "Event"


def test_gate_rejects_worse_model(monkeypatch):
    out, sm, sns, lam, tags = run_deploy(monkeypatch, {"candidate_token_f1": 0.6, "champion_token_f1": 0.7})
    assert not out["deployed"] and "worse than the current model" in out["reason"]
    assert not [n for n, _ in sm.calls if n.startswith(("create", "update"))]
    assert tags["deployment-decision"] == "rejected" and not lam.calls
    assert sns.calls[0]["Subject"].startswith("Model not deployed") and "0.6000" in sns.calls[0]["Message"]


def test_gate_tolerance_floor_and_first_deploy(monkeypatch):
    out, *_ = run_deploy(monkeypatch, {"candidate_token_f1": 0.68, "champion_token_f1": 0.7}, DEPLOY_TOLERANCE="0.05")
    assert out["deployed"]
    out, *_ = run_deploy(monkeypatch, {"candidate_token_f1": 0.3}, MIN_TOKEN_F1="0.5")
    assert not out["deployed"] and "below MIN_TOKEN_F1" in out["reason"]
    out, *_ = run_deploy(monkeypatch, {"candidate_token_f1": 0.3}, endpoint=False)
    assert out["deployed"] and out["action"] == "created" and "no current model" in out["reason"]


def test_gate_missing_metrics(monkeypatch):
    out, _, sns, *_ = run_deploy(monkeypatch, {})
    assert not out["deployed"] and "no evaluation metrics" in out["reason"] and sns.calls
    out, *_ = run_deploy(monkeypatch, {}, DEPLOY_WITHOUT_METRICS="true")
    assert out["deployed"]


# --------------------------------------------------------------------------- real tiny T5 run
def make_tiny_t5(path: Path, seed: int):
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    words = sorted({w for c, cap in CAPITALS.items() for w in f"what is the capital of {c} ? {cap}".split()})
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    torch.manual_seed(seed)
    cfg = T5Config(vocab_size=len(vocab), d_model=32, d_ff=64, d_kv=16, num_layers=1, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    T5ForConditionalGeneration(cfg).save_pretrained(path)
    fast.save_pretrained(path)


def test_train_evaluates_candidate_and_champion(tmp_path):
    pytest.importorskip("torch")
    make_tiny_t5(tmp_path / "base", seed=0)
    make_tiny_t5(tmp_path / "champion", seed=1)
    with tarfile.open(tmp_path / "champion.tar.gz", "w:gz") as tar:  # shaped like SageMaker's model.tar.gz
        for f in (tmp_path / "champion").iterdir():
            tar.add(f, arcname=f.name)
    data = tmp_path / "data"
    data.mkdir()
    (data / "capitals.txt").write_text("".join(f"what is the capital of {c} ? = {cap}\n" for c, cap in CAPITALS.items()))

    env = dict(os.environ, SM_CHANNEL_TRAINING=str(data), SM_MODEL_DIR=str(tmp_path / "model"),
               CHECKPOINT_DIR=str(tmp_path / "ckpt"), BASE_MODEL=str(tmp_path / "base"), EPOCHS="3",
               LEARNING_RATE="1e-2", EVAL_PERCENT="20", SM_HP_CHAMPION_MODEL_S3=f'"{tmp_path / "champion.tar.gz"}"')
    run = subprocess.run([sys.executable, str(ROOT / "training" / "train.py")], env=env,
                         capture_output=True, text=True, check=True)

    report = json.loads((tmp_path / "model" / "evaluation.json").read_text())
    assert report["eval_samples"] >= 1 and not report["eval_on_train"]
    for who in ("candidate", "champion"):
        assert set(report[who]) == {"exact_match", "token_f1"}
        assert 0.0 <= report[who]["token_f1"] <= 1.0
    assert "champion" in report["examples"][0]

    # The trigger Lambda's MetricDefinitions must match what train.py prints, or the gate sees nothing.
    os.environ.update({"SAGEMAKER_ROLE": "r", "BUCKET": "b", "TRAINING_IMAGE": "i"})
    defs = importlib.reload(importlib.import_module("trigger_training")).METRIC_DEFINITIONS
    parsed = {}
    for d in defs:
        found = re.findall(d["Regex"], run.stdout)
        if found:
            parsed[d["Name"]] = float(found[-1])
    assert parsed["eval:candidate_token_f1"] == pytest.approx(report["candidate"]["token_f1"], abs=1e-4)
    assert parsed["eval:champion_token_f1"] == pytest.approx(report["champion"]["token_f1"], abs=1e-4)
    assert parsed["eval:eval_samples"] == report["eval_samples"] and "train:loss" in parsed
