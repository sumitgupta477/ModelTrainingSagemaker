"""Fine-tune a small Hugging Face seq2seq model (T5) on SageMaker, and evaluate it.

Training data: every *.txt file in the "training" channel (s3://<bucket>/input/),
one example per line in the form  `input text = target text`.

Evaluation (champion vs challenger)
-----------------------------------
* About EVAL_PERCENT% of the examples are held out. The split is decided by a hash of the
  input text, so an example always lands on the same side across retrains: new data never
  reshuffles old data, and duplicate inputs can't leak from training into evaluation.
* After training, the new model (the "candidate") answers the held-out inputs and is scored
  against the targets with exact match and token F1 (word-overlap, SQuAD-style).
* If the pipeline passes the currently deployed model (hyperparameter champion_model_s3),
  that "champion" is scored on the SAME held-out examples, so the comparison is fair.
* Scores are printed as `EVAL name=value` lines. SageMaker records them as training-job
  metrics (MetricDefinitions in trigger_training.py), and the deploy Lambda only deploys
  the candidate if it is good enough.

Supports managed spot training: a checkpoint is written to /opt/ml/checkpoints
after every epoch (SageMaker syncs it to S3), and training resumes from the
latest checkpoint if the spot instance is interrupted.
"""

import glob
import hashlib
import json
import os
import re
import string
import tarfile
import tempfile
from collections import Counter

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# SageMaker paths
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")

MODEL_NAME = os.environ.get("BASE_MODEL", "t5-small")  # lightweight text2text model
MAX_LEN = 32
BATCH_SIZE = 2
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
EVAL_PERCENT = int(os.environ.get("EVAL_PERCENT", "10"))  # share of examples held out
MAX_EVAL = int(os.environ.get("MAX_EVAL", "200"))         # cap on evaluated examples


# --------------------------------------------------------------------------- data
def read_pairs(data_dir):
    """`input = target` pairs from every .txt file in the data directory."""
    pairs = []
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    # Split on the first "=" only, so targets may contain "=".
                    src, tgt = line.split("=", 1)
                    if src.strip() and tgt.strip():
                        pairs.append((src.strip(), tgt.strip()))
    print(f"Loaded {len(pairs)} training pairs from {len(files)} file(s)")
    if not pairs:
        raise ValueError(f"No 'input = target' lines found in {data_dir}")
    return pairs


def _hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def split_pairs(pairs, eval_percent=EVAL_PERCENT):
    """(train, eval, eval_on_train). Stable: depends only on each input's text."""
    if len({p[0] for p in pairs}) < 2:
        # A single distinct input: train on it and evaluate on it, and say so.
        return list(pairs), list(pairs), True
    held_out = [p for p in pairs if _hash(p[0]) % 100 < eval_percent]
    train = [p for p in pairs if _hash(p[0]) % 100 >= eval_percent]
    if not held_out:
        # Tiny datasets: hold out the input with the smallest hash so there is something to score.
        first = min({p[0] for p in pairs}, key=_hash)
        held_out = [p for p in pairs if p[0] == first]
        train = [p for p in pairs if p[0] != first]
    if not train:
        # Everything hashed into the eval split: keep the smallest-hash input for training.
        keep = min({p[0] for p in pairs}, key=_hash)
        train = [p for p in held_out if p[0] == keep]
        held_out = [p for p in held_out if p[0] != keep]
    return train, held_out, False


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.tokenizer = tokenizer
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        x = self.tokenizer(src, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
        y = self.tokenizer(tgt, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
        labels = y["input_ids"].squeeze()
        # Ignore padding tokens in the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": x["input_ids"].squeeze(),
            "attention_mask": x["attention_mask"].squeeze(),
            "labels": labels,
        }


# --------------------------------------------------------------------------- metrics
def normalize(text):
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference):
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score(predictions, references):
    n = len(references)
    return {
        "exact_match": sum(exact_match(p, r) for p, r in zip(predictions, references)) / n,
        "token_f1": sum(token_f1(p, r) for p, r in zip(predictions, references)) / n,
    }


def generate_answers(model, tokenizer, inputs, device, batch_size=8):
    model.eval()
    answers = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = tokenizer(inputs[i:i + batch_size], max_length=MAX_LEN, padding=True,
                              truncation=True, return_tensors="pt").to(device)
            out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 max_new_tokens=MAX_LEN)
            answers.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return answers


# --------------------------------------------------------------------------- champion
def champion_uri():
    """Currently deployed model's artifact (s3://.../model.tar.gz), passed by the pipeline."""
    value = os.environ.get("SM_HP_CHAMPION_MODEL_S3") or os.environ.get("CHAMPION_MODEL_S3", "")
    return value.strip().strip('"')


def load_champion(uri):
    """Download (if s3://) and unpack the champion's model.tar.gz; return its directory."""
    work = tempfile.mkdtemp(prefix="champion-")
    if uri.startswith("s3://"):
        import boto3  # available in the SageMaker training containers

        bucket, _, key = uri[len("s3://"):].partition("/")
        archive = os.path.join(work, "model.tar.gz")
        boto3.client("s3").download_file(bucket, key, archive)
    else:
        archive = uri
    if os.path.isdir(archive):
        return archive
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(work)
    return work


# --------------------------------------------------------------------------- checkpoints
def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    state = torch.load(CHECKPOINT_FILE, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"Resumed from checkpoint after epoch {state['epoch']}")
    return state["epoch"]


def save_checkpoint(model, optimizer, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp = CHECKPOINT_FILE + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch}, tmp)
    os.replace(tmp, CHECKPOINT_FILE)


# --------------------------------------------------------------------------- main
def main():
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_pairs, eval_pairs, eval_on_train = split_pairs(read_pairs(DATA_DIR))
    eval_pairs = eval_pairs[:MAX_EVAL]
    print(f"Training on {len(train_pairs)} pairs, holding out {len(eval_pairs)} for evaluation"
          + (" (dataset too small: evaluating on training data)" if eval_on_train else ""))
    loader = DataLoader(PairDataset(train_pairs, tokenizer), batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoint(model, optimizer)

    print("Training started...")
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        # SageMaker parses this line into the training job's metrics.
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")
        save_checkpoint(model, optimizer, epoch + 1)

    print("Saving model...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Evaluate the candidate (and the champion, if any) on the same held-out examples.
    inputs = [src for src, _ in eval_pairs]
    targets = [tgt for _, tgt in eval_pairs]
    candidate_answers = generate_answers(model, tokenizer, inputs, device)
    report = {"eval_samples": len(eval_pairs), "eval_on_train": eval_on_train,
              "candidate": score(candidate_answers, targets), "champion": None,
              "examples": [{"input": i, "target": t, "candidate": c}
                           for i, t, c in list(zip(inputs, targets, candidate_answers))[:20]]}

    uri = champion_uri()
    if uri:
        try:
            champ_dir = load_champion(uri)
            champ_tok = AutoTokenizer.from_pretrained(champ_dir)
            champ_model = AutoModelForSeq2SeqLM.from_pretrained(champ_dir).to(device)
            champion_answers = generate_answers(champ_model, champ_tok, inputs, device)
            report["champion"] = score(champion_answers, targets)
            report["champion_model"] = uri
            for example, answer in zip(report["examples"], champion_answers):
                example["champion"] = answer
        except Exception as err:  # noqa: BLE001 - never fail training because the old model is unreadable
            print(f"WARNING: could not evaluate champion {uri}: {err!r}")

    with open(os.path.join(MODEL_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # SageMaker turns these lines into training-job metrics (FinalMetricDataList).
    print(f"EVAL eval_samples={len(eval_pairs)}")
    print(f"EVAL eval_on_train={int(eval_on_train)}")
    for who in ("candidate", "champion"):
        if report[who]:
            for metric, value in report[who].items():
                print(f"EVAL {who}_{metric}={value:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
