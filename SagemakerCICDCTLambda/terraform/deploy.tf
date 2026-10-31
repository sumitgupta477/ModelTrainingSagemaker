# -----------------------------
# Continuous deployment: when a training job completes, deploy the new model,
# but only if its held-out evaluation passes the quality gate (see deploy_model.py)
# -----------------------------
data "archive_file" "deploy_zip" {
  type        = "zip"
  source_file = "../lambda/deploy_model.py"
  output_path = "${path.module}/build/deploy_model.zip"
}

resource "aws_lambda_function" "deployer" {
  function_name = "llm-deployer"

  filename         = data.archive_file.deploy_zip.output_path
  source_code_hash = data.archive_file.deploy_zip.output_base64sha256

  role        = aws_iam_role.lambda_role.arn
  handler     = "deploy_model.lambda_handler"
  runtime     = "python3.12"
  timeout     = 60
  memory_size = 256

  environment {
    variables = {
      SAGEMAKER_ROLE         = aws_iam_role.sagemaker_role.arn
      ENDPOINT_NAME          = var.endpoint_name
      INFERENCE_IMAGE        = var.inference_image
      ENDPOINT_INSTANCE_TYPE = var.endpoint_instance_type
      HF_TASK                = "text2text-generation"
      JOB_PREFIX             = local.job_prefix
      DATA_CAPTURE_S3_URI    = "s3://${aws_s3_bucket.ml_bucket.bucket}/${local.capture_prefix}"
      DATA_CAPTURE_PERCENT   = tostring(var.data_capture_percent)
      MIN_TOKEN_F1           = tostring(var.min_token_f1)
      DEPLOY_TOLERANCE       = tostring(var.deploy_tolerance)
      DEPLOY_WITHOUT_METRICS = tostring(var.deploy_without_metrics)
      ALERT_TOPIC_ARN        = aws_sns_topic.pipeline_alerts.arn
      BASELINE_FUNCTION      = aws_lambda_function.drift_baseline.function_name
    }
  }
}

resource "aws_cloudwatch_log_group" "deployer" {
  name              = "/aws/lambda/${aws_lambda_function.deployer.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_event_rule" "training_completed" {
  name        = "${var.project_name}-training-completed"
  description = "SageMaker training jobs from this pipeline that completed successfully"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Training Job State Change"]
    detail = {
      TrainingJobStatus = ["Completed"]
      TrainingJobName   = [{ prefix = local.job_prefix }]
    }
  })
}

resource "aws_cloudwatch_event_target" "deployer" {
  rule = aws_cloudwatch_event_rule.training_completed.name
  arn  = aws_lambda_function.deployer.arn
}

resource "aws_lambda_permission" "eventbridge_invoke" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.deployer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.training_completed.arn
}

# -----------------------------
# Alert when a training job fails
# -----------------------------
resource "aws_cloudwatch_event_rule" "training_failed" {
  name        = "${var.project_name}-training-failed"
  description = "SageMaker training jobs from this pipeline that failed"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Training Job State Change"]
    detail = {
      TrainingJobStatus = ["Failed"]
      TrainingJobName   = [{ prefix = local.job_prefix }]
    }
  })
}

resource "aws_sns_topic" "pipeline_alerts" {
  name = "${var.project_name}-pipeline-alerts"
}

resource "aws_sns_topic_policy" "pipeline_alerts" {
  arn = aws_sns_topic.pipeline_alerts.arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "events.amazonaws.com" }
        Action    = "sns:Publish"
        Resource  = aws_sns_topic.pipeline_alerts.arn
      },
      {
        # Drift and monitoring-failure alarms (monitoring.tf) publish here too.
        Effect    = "Allow"
        Principal = { Service = "cloudwatch.amazonaws.com" }
        Action    = "sns:Publish"
        Resource  = aws_sns_topic.pipeline_alerts.arn
        Condition = {
          StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        }
      }
    ]
  })
}

resource "aws_cloudwatch_event_target" "training_failed_alert" {
  rule = aws_cloudwatch_event_rule.training_failed.name
  arn  = aws_sns_topic.pipeline_alerts.arn
}
