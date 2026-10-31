"""Continuous-monitoring step 1: build the drift baseline.

Triggered by the same EventBridge rule as the deploy Lambda (a training job from this
pipeline completed), so the baseline always describes the data the live model was
trained on. Reads every `input = target` line under s3://<bucket>/input/ (exactly what
the training job read), turns each pair into text features, and writes
s3://<bucket>/monitoring/baseline.json.
"""
import json
import logging
import os

import boto3

import drift

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET = os.environ["BUCKET"]
INPUT_PREFIX = os.environ.get("INPUT_PREFIX", "input/")
BASELINE_KEY = os.environ.get("BASELINE_KEY", "monitoring/baseline.json")
BINS = int(os.environ.get("BASELINE_BINS", "10"))


def read_training_pairs() -> list:
    pairs, files = [], 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=INPUT_PREFIX):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".txt"):
                continue
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
            pairs.extend(drift.parse_training_text(body.decode("utf-8", errors="replace")))
            files += 1
    logger.info("Read %d training pairs from %d file(s) under s3://%s/%s", len(pairs), files, BUCKET, INPUT_PREFIX)
    return pairs


def lambda_handler(event, context):
    job_name = event.get("detail", {}).get("TrainingJobName", "manual")
    pairs = read_training_pairs()
    if not pairs:
        logger.warning("No training pairs found; baseline not updated")
        return {"baseline_updated": False}

    # Training targets stand in for "expected answers" when describing normal output.
    rows = [drift.text_features(src, tgt) for src, tgt in pairs]
    baseline = drift.build_baseline(rows, bins=BINS, source=f"training job {job_name}")
    s3.put_object(Bucket=BUCKET, Key=BASELINE_KEY, Body=json.dumps(baseline, indent=2).encode(),
                  ContentType="application/json")
    logger.info("Wrote baseline (%d examples) to s3://%s/%s", len(rows), BUCKET, BASELINE_KEY)
    return {"baseline_updated": True, "examples": len(rows), "key": BASELINE_KEY, "training_job": job_name}
