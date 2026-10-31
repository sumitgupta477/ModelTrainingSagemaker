"""Continuous-deployment step, with a quality gate.

Triggered by an EventBridge rule when a SageMaker training job from this pipeline
completes. Before deploying, it reads the job's evaluation metrics (recorded from
train.py's `EVAL name=value` lines) and only deploys the new model ("candidate") if:

  1. it was evaluated (eval:candidate_token_f1 is present), unless DEPLOY_WITHOUT_METRICS;
  2. its token F1 on held-out data is at least MIN_TOKEN_F1; and
  3. when a model is already serving ("champion"), the candidate's token F1 on the SAME
     held-out examples is at least champion - DEPLOY_TOLERANCE.

If it passes, the model is registered, a new endpoint configuration is created and the
endpoint is created (first run) or updated in place (SageMaker's blue/green rollout keeps
the previous model serving until the new one is healthy), and the drift baseline is rebuilt.
If it fails, nothing changes on the endpoint and an alert is sent to the SNS topic.
Either way the decision is tagged on the training job (deployment-decision).
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client("sagemaker")
sns = boto3.client("sns")
lambda_client = boto3.client("lambda")

ROLE = os.environ["SAGEMAKER_ROLE"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
INFERENCE_IMAGE = os.environ["INFERENCE_IMAGE"]
ENDPOINT_INSTANCE_TYPE = os.environ.get("ENDPOINT_INSTANCE_TYPE", "ml.t2.medium")
HF_TASK = os.environ.get("HF_TASK", "text2text-generation")
JOB_PREFIX = os.environ.get("JOB_PREFIX", "llm-train")
# Data capture: every request/response is saved to S3 for drift monitoring (monitoring/).
DATA_CAPTURE_S3_URI = os.environ.get("DATA_CAPTURE_S3_URI", "")
DATA_CAPTURE_PERCENT = int(os.environ.get("DATA_CAPTURE_PERCENT", "100"))
# Quality gate
MIN_TOKEN_F1 = float(os.environ.get("MIN_TOKEN_F1", "0.0"))
DEPLOY_TOLERANCE = float(os.environ.get("DEPLOY_TOLERANCE", "0.0"))
DEPLOY_WITHOUT_METRICS = os.environ.get("DEPLOY_WITHOUT_METRICS", "false").lower() == "true"
ALERT_TOPIC_ARN = os.environ.get("ALERT_TOPIC_ARN", "")
# Drift baseline Lambda, rebuilt only when a model is actually deployed
BASELINE_FUNCTION = os.environ.get("BASELINE_FUNCTION", "")


def data_capture_config():
    """Capture requests and responses as plain JSON (not base64) so they're easy to analyse."""
    if not DATA_CAPTURE_S3_URI:
        return None
    return {
        "EnableCapture": True,
        "InitialSamplingPercentage": DATA_CAPTURE_PERCENT,
        "DestinationS3Uri": DATA_CAPTURE_S3_URI,
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
    }


def eval_metrics(job):
    """{'candidate_token_f1': 0.82, ...} from the job's final metric values."""
    return {
        m["MetricName"].split(":", 1)[1]: float(m["Value"])
        for m in job.get("FinalMetricDataList", [])
        if m.get("MetricName", "").startswith("eval:")
    }


def gate(metrics):
    """(deploy?, reason) for the candidate's evaluation metrics."""
    candidate = metrics.get("candidate_token_f1")
    if candidate is None:
        if DEPLOY_WITHOUT_METRICS:
            return True, "no evaluation metrics; deploying because DEPLOY_WITHOUT_METRICS=true"
        return False, "no evaluation metrics recorded (eval:candidate_token_f1 missing)"
    if candidate < MIN_TOKEN_F1:
        return False, f"candidate token F1 {candidate:.4f} is below MIN_TOKEN_F1 {MIN_TOKEN_F1:.4f}"
    champion = metrics.get("champion_token_f1")
    if champion is None:
        return True, f"candidate token F1 {candidate:.4f} (no current model to compare with)"
    if candidate < champion - DEPLOY_TOLERANCE:
        return False, (f"candidate token F1 {candidate:.4f} is worse than the current model's "
                       f"{champion:.4f} on the same held-out data (tolerance {DEPLOY_TOLERANCE})")
    return True, f"candidate token F1 {candidate:.4f} vs current model {champion:.4f}"


def tag_decision(job_arn, decision, reason):
    try:
        sagemaker.add_tags(ResourceArn=job_arn, Tags=[
            {"Key": "deployment-decision", "Value": decision},
            {"Key": "deployment-reason", "Value": reason[:256]},
        ])
    except Exception as err:  # noqa: BLE001 - tagging is informational only
        logger.warning("Could not tag %s: %s", job_arn, err)


def alert(job_name, reason, metrics):
    if not ALERT_TOPIC_ARN:
        return
    sns.publish(
        TopicArn=ALERT_TOPIC_ARN,
        Subject=f"Model not deployed: {job_name}"[:100],
        Message=json.dumps({"training_job": job_name, "endpoint": ENDPOINT_NAME,
                            "decision": "rejected", "reason": reason, "metrics": metrics}, indent=2),
    )


def rebuild_baseline(job_name):
    if BASELINE_FUNCTION:
        lambda_client.invoke(
            FunctionName=BASELINE_FUNCTION,
            InvocationType="Event",  # asynchronous
            Payload=json.dumps({"detail": {"TrainingJobName": job_name}}).encode(),
        )


def lambda_handler(event, context):
    detail = event.get("detail", {})
    job_name = detail.get("TrainingJobName", "")
    status = detail.get("TrainingJobStatus")

    if status != "Completed" or not job_name.startswith(JOB_PREFIX):
        logger.info("Ignoring job %s with status %s", job_name, status)
        return {"deployed": False}

    job = sagemaker.describe_training_job(TrainingJobName=job_name)
    model_data = job["ModelArtifacts"]["S3ModelArtifacts"]
    metrics = eval_metrics(job)
    deploy, reason = gate(metrics)
    logger.info("Gate for %s: %s (%s) metrics=%s", job_name, "DEPLOY" if deploy else "REJECT", reason, metrics)

    if not deploy:
        tag_decision(job["TrainingJobArn"], "rejected", reason)
        alert(job_name, reason, metrics)
        return {"deployed": False, "reason": reason, "metrics": metrics}

    # Resource names are limited to 63 characters.
    model_name = job_name[:63]
    config_name = f"{job_name}-cfg"[:63]

    sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE,
        PrimaryContainer={
            "Image": INFERENCE_IMAGE,
            "ModelDataUrl": model_data,
            "Environment": {"HF_TASK": HF_TASK},
        },
    )

    endpoint_config = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": ENDPOINT_INSTANCE_TYPE,
            }
        ],
    }
    capture = data_capture_config()
    if capture:
        endpoint_config["DataCaptureConfig"] = capture
    sagemaker.create_endpoint_config(**endpoint_config)

    if endpoint_exists():
        sagemaker.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        action = "updated"
    else:
        sagemaker.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        action = "created"

    tag_decision(job["TrainingJobArn"], "deployed", reason)
    rebuild_baseline(job_name)
    logger.info("Endpoint %s %s with model %s (%s)", ENDPOINT_NAME, action, model_name, model_data)
    return {"deployed": True, "action": action, "model": model_name, "model_data": model_data,
            "reason": reason, "metrics": metrics}


def endpoint_exists():
    try:
        sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return True
    except sagemaker.exceptions.ClientError as err:
        if "Could not find endpoint" in str(err):
            return False
        raise
