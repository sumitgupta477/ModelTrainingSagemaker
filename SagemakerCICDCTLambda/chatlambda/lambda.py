import json
import boto3
import os

runtime = boto3.client("sagemaker-runtime")

ENDPOINT = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):

    try:

        body = json.loads(event.get("body", "{}"))

        prompt = body.get("prompt", "")

        if not prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing prompt"})
            }

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({
                "inputs": prompt
            })
        )

        result = response["Body"].read().decode()

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "answer": result
            })
        }

    except Exception as e:

        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
