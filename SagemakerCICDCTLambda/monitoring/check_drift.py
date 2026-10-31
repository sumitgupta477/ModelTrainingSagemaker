"""Continuous-monitoring step 2: compare live traffic with the baseline (runs on a schedule).

Reads the endpoint's data capture for the last WINDOW_HOURS hours
(s3://<bucket>/datacapture/<endpoint>/<variant>/YYYY/MM/DD/HH/*.jsonl), turns each
request/response into text features, scores drift per feature with PSI, and:
  * publishes CloudWatch metrics (namespace METRIC_NAMESPACE, dimension Endpoint):
      MaxPSI, SampleCount, and PSI per feature (extra dimension Feature)
  * writes a JSON report to s3://<bucket>/monitoring/reports/YYYY/MM/DD/HH.json
A CloudWatch alarm on MaxPSI (see terraform/monitoring.tf) emails the alerts topic.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3

import drift

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")

BUCKET = os.environ["BUCKET"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
VARIANT = os.environ.get("VARIANT_NAME", "AllTraffic")
CAPTURE_PREFIX = os.environ.get("CAPTURE_PREFIX", "datacapture/")
BASELINE_KEY = os.environ.get("BASELINE_KEY", "monitoring/baseline.json")
REPORT_PREFIX = os.environ.get("REPORT_PREFIX", "monitoring/reports/")
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "24"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "30"))
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "LLMChat/ModelDrift")
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "50000"))  # keeps one run inside Lambda limits


def load_baseline():
    try:
        body = s3.get_object(Bucket=BUCKET, Key=BASELINE_KEY)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(body)


def hour_prefixes(now: datetime, hours: int) -> list:
    base = f"{CAPTURE_PREFIX.rstrip('/')}/{ENDPOINT_NAME}/{VARIANT}/"
    return [base + (now - timedelta(hours=h)).strftime("%Y/%m/%d/%H/") for h in range(hours)]


def read_captured_rows(now: datetime) -> list:
    rows = []
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in hour_prefixes(now, WINDOW_HOURS):
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read().decode("utf-8", errors="replace")
                for line in body.splitlines():
                    parsed = drift.parse_capture_line(line) if line.strip() else None
                    if parsed:
                        rows.append(drift.text_features(*parsed))
                        if len(rows) >= MAX_RECORDS:
                            return rows
    return rows


def publish(metrics: list) -> None:
    for i in range(0, len(metrics), 500):  # PutMetricData accepts up to 1000 items per call
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metrics[i:i + 500])


def lambda_handler(event, context):
    now = datetime.now(timezone.utc)
    endpoint_dim = {"Name": "Endpoint", "Value": ENDPOINT_NAME}

    baseline = load_baseline()
    if baseline is None:
        logger.warning("No baseline at s3://%s/%s yet (it is built when a training job completes)", BUCKET, BASELINE_KEY)
        return {"status": "no_baseline"}

    rows = read_captured_rows(now)
    report = {
        "checked_at": now.isoformat(),
        "endpoint": ENDPOINT_NAME,
        "window_hours": WINDOW_HOURS,
        "samples": len(rows),
        "baseline": {"created": baseline.get("created"), "source": baseline.get("source"), "count": baseline.get("count")},
    }
    metrics = [{"MetricName": "SampleCount", "Dimensions": [endpoint_dim], "Value": len(rows), "Unit": "Count"}]

    if len(rows) < MIN_SAMPLES:
        report["status"] = "insufficient_data"
        logger.info("Only %d samples in the last %dh (need %d); skipping drift scores", len(rows), WINDOW_HOURS, MIN_SAMPLES)
    else:
        features = drift.compare(baseline, rows)
        max_feature = max(features, key=lambda f: features[f]["psi"])
        max_psi = features[max_feature]["psi"]
        report.update(
            status=drift.severity(max_psi),
            max_psi=max_psi,
            max_psi_feature=max_feature,
            features=features,
        )
        metrics.append({"MetricName": "MaxPSI", "Dimensions": [endpoint_dim], "Value": max_psi, "Unit": "None"})
        metrics.extend(
            {"MetricName": "PSI", "Dimensions": [endpoint_dim, {"Name": "Feature", "Value": name}],
             "Value": scores["psi"], "Unit": "None"}
            for name, scores in features.items()
        )
        logger.info("Drift %s: max PSI %.4f on %s over %d samples", report["status"], max_psi, max_feature, len(rows))

    publish(metrics)
    key = f"{REPORT_PREFIX.rstrip('/')}/{now.strftime('%Y/%m/%d/%H')}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(report, indent=2).encode(), ContentType="application/json")
    return {"status": report["status"], "samples": len(rows), "max_psi": report.get("max_psi"), "report": key}
