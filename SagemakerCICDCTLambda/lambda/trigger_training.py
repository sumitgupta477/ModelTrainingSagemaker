import boto3
import os
import time
import logging
import json

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create SageMaker client (outside handler for reuse)
sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):

    try:

        # Read environment variables
        role = os.environ.get("SAGEMAKER_ROLE")
        bucket = os.environ.get("BUCKET")

        instance_type = os.environ.get(
            "TRAIN_INSTANCE_TYPE",
            "ml.m5.large"
        )

        instance_count = int(
            os.environ.get("TRAIN_INSTANCE_COUNT", "1")
        )

        volume_size = int(
            os.environ.get("TRAIN_VOLUME_SIZE", "50")
        )

        endpoint_instance_type = os.environ.get("ENDPOINT_INSTANCE_TYPE", "t2.medium")

        if not role or not bucket:
            raise Exception("Missing environment variables: SAGEMAKER_ROLE or BUCKET")

        records = event.get("Records", [])

        s3_files = [
            r["s3"]["object"]["key"]
            for r in records
            if "s3" in r
        ]

        if not s3_files:
            raise Exception("No S3 files detected")

        logger.info(f"Files: {s3_files}")

        # Create unique job name
        job_name = f"llm-train-{int(time.time())}"

        logger.info(f"Starting training job: {job_name}")

        # Start SageMaker training job
        sagemaker.create_training_job(

            TrainingJobName=job_name,
            RoleArn=role,

            AlgorithmSpecification={
                "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker",
                "TrainingInputMode": "File"
            },

            InputDataConfig=[
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{bucket}/input/",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/plain"
                }
            ],

            OutputDataConfig={
                "S3OutputPath": f"s3://{bucket}/llm-model/output"
            },

            ResourceConfig={
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size
            },

            StoppingCondition={
                "MaxRuntimeInSeconds": 4*60*60
            },

            EnableManagedSpotTraining=True,
            CheckpointConfig={
                "S3Uri": f"s3://{bucket}/checkpoints/{job_name}",
                "LocalPath": "/opt/ml/checkpoints"
            },

            # IMPORTANT: WAIT LONGER THAN RUNTIME
            StoppingCondition={

                # Actual training time
                "MaxRuntimeInSeconds": 4 * 60 * 60,

                # Must be HIGHER than runtime
                # Allows retries on interruption
                "MaxWaitTimeInSeconds": 8 * 60 * 60
            },

            HyperParameters={
                "sagemaker_program": "train.py",
                "sagemaker_submit_directory": f"s3://{bucket}/code/training.tar.gz",
                "sagemaker_container_log_level": "20"
            }
        )

        # Success response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Spot training started (safe mode)",
                "job_name": job_name,
                "files": s3_files,
                "mode": "spot-only"
            })
        }

    except Exception as e:

        logger.exception("Training failed")

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }