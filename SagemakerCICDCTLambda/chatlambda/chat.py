import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")

ENDPOINT = os.environ["ENDPOINT_NAME"]

HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
}


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return respond(400, {"error": "Request body must be valid JSON"})

    prompt = body.get("prompt", "")
    if not prompt:
        return respond(400, {"error": "Missing prompt"})

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"inputs": prompt}),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        # The Hugging Face container returns [{"generated_text": "..."}].
        if isinstance(result, list) and result and "generated_text" in result[0]:
            answer = result[0]["generated_text"]
        else:
            answer = result
        return respond(200, {"answer": answer})

    except runtime.exceptions.ValidationError as e:
        # Usually means the endpoint hasn't been deployed yet (first training run still going).
        print(f"Endpoint not ready: {e!r}")
        return respond(503, {"error": "Model endpoint is not available yet"})
    except Exception as e:
        # Log details to CloudWatch, but don't expose internals to API callers.
        print(f"Inference failed: {e!r}")
        return respond(500, {"error": "Inference failed"})


def respond(status, body):
    return {"statusCode": status, "headers": HEADERS, "body": json.dumps(body)}
