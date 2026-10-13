"""Continuous-training trigger.

Invoked by S3 when new training data lands under input/. Starts a SageMaker
managed spot training job that fine-tunes the model on everything in input/.
When the job completes, the deploy_model Lambda (EventBridge) rolls the new
model out to the endpoint.
"""

import json
import logging
import os
import time
import urllib.parse

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client("sagemaker")

ROLE = os.environ["SAGEMAKER_ROLE"]
BUCKET = os.environ["BUCKET"]
JOB_PREFIX = os.environ.get("JOB_PREFIX", "llm-train")
TRAINING_IMAGE = os.environ["TRAINING_IMAGE"]
INSTANCE_TYPE = os.environ.get("TRAIN_INSTANCE_TYPE", "ml.m5.large")
INSTANCE_COUNT = int(os.environ.get("TRAIN_INSTANCE_COUNT", "1"))
VOLUME_SIZE = int(os.environ.get("TRAIN_VOLUME_SIZE", "50"))
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME_SECONDS", str(4 * 60 * 60)))
# For spot training, the wait time must exceed the run time so interrupted jobs can resume.
MAX_WAIT = int(os.environ.get("MAX_WAIT_SECONDS", str(8 * 60 * 60)))


def lambda_handler(event, context):
    s3_files = [
        urllib.parse.unquote_plus(r["s3"]["object"]["key"])
        for r in event.get("Records", [])
        if "s3" in r
    ]
    if not s3_files:
        logger.warning("No S3 records in event; nothing to do")
        return {"statusCode": 400, "body": json.dumps({"error": "No S3 files detected"})}

    logger.info("New training data: %s", s3_files)

    # Job names must be unique; milliseconds avoid collisions when several files land at once.
    job_name = f"{JOB_PREFIX}-{int(time.time() * 1000)}"

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE,
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
            # Publishes the loss printed by train.py as a CloudWatch metric.
            "MetricDefinitions": [{"Name": "train:loss", "Regex": r"Loss: ([0-9\.]+)"}],
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/input/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/plain",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/llm-model/output"},
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": INSTANCE_COUNT,
            "VolumeSizeInGB": VOLUME_SIZE,
        },
        # Managed spot training: up to ~90% cheaper than on-demand. Checkpoints in
        # S3 let SageMaker resume the job if the spot capacity is reclaimed.
        EnableManagedSpotTraining=True,
        CheckpointConfig={
            "S3Uri": f"s3://{BUCKET}/checkpoints/{job_name}",
            "LocalPath": "/opt/ml/checkpoints",
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": MAX_RUNTIME,
            "MaxWaitTimeInSeconds": MAX_WAIT,
        },
        HyperParameters={
            "sagemaker_program": json.dumps("train.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/code/training.tar.gz"),
            "sagemaker_container_log_level": "20",
        },
        Tags=[{"Key": "pipeline", "Value": "continuous-training"}],
    )

    logger.info("Started spot training job %s", job_name)
    return {
        "statusCode": 200,
        "body": json.dumps({"job_name": job_name, "files": s3_files, "mode": "managed-spot"}),
    }
