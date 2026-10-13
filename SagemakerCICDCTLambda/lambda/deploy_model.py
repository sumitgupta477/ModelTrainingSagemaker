"""Continuous-deployment step.

Triggered by an EventBridge rule when a SageMaker training job from this
pipeline completes. Registers the new model artifact, creates a new endpoint
configuration and creates the endpoint (first run) or updates it in place.
SageMaker updates endpoints with a blue/green rollout, so the chat API keeps
serving the previous model until the new one is healthy.
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client("sagemaker")

ROLE = os.environ["SAGEMAKER_ROLE"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
INFERENCE_IMAGE = os.environ["INFERENCE_IMAGE"]
ENDPOINT_INSTANCE_TYPE = os.environ.get("ENDPOINT_INSTANCE_TYPE", "ml.t2.medium")
HF_TASK = os.environ.get("HF_TASK", "text2text-generation")
JOB_PREFIX = os.environ.get("JOB_PREFIX", "llm-train")


def lambda_handler(event, context):
    detail = event.get("detail", {})
    job_name = detail.get("TrainingJobName", "")
    status = detail.get("TrainingJobStatus")

    if status != "Completed" or not job_name.startswith(JOB_PREFIX):
        logger.info("Ignoring job %s with status %s", job_name, status)
        return {"deployed": False}

    model_data = sagemaker.describe_training_job(TrainingJobName=job_name)[
        "ModelArtifacts"
    ]["S3ModelArtifacts"]

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

    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": ENDPOINT_INSTANCE_TYPE,
            }
        ],
    )

    if endpoint_exists():
        sagemaker.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        action = "updated"
    else:
        sagemaker.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        action = "created"

    logger.info("Endpoint %s %s with model %s (%s)", ENDPOINT_NAME, action, model_name, model_data)
    return {"deployed": True, "action": action, "model": model_name, "model_data": model_data}


def endpoint_exists():
    try:
        sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return True
    except sagemaker.exceptions.ClientError as err:
        if "Could not find endpoint" in str(err):
            return False
        raise
