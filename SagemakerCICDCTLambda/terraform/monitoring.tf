# -----------------------------
# Continuous monitoring: drift detection on the live endpoint
#
#   deploy Lambda    -> endpoint data capture  -> s3://<bucket>/datacapture/
#   model deployed   -> llm-drift-baseline     -> s3://<bucket>/monitoring/baseline.json
#   every hour       -> llm-drift-check        -> CloudWatch metrics + s3://<bucket>/monitoring/reports/
#   DriftScore high  -> CloudWatch alarm       -> SNS pipeline alerts topic
#
# Built with Lambda + CloudWatch rather than SageMaker Model Monitor, which AWS has
# closed to new customers.
# -----------------------------
data "archive_file" "monitoring_zip" {
  type        = "zip"
  source_dir  = "../monitoring"
  excludes    = ["__pycache__"]
  output_path = "${path.module}/build/monitoring.zip"
}

resource "aws_iam_role" "monitoring_lambda_role" {
  name = "${var.project_name}-monitoring-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "monitoring_basic" {
  role       = aws_iam_role.monitoring_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "monitoring_lambda_policy" {
  name = "${var.project_name}-monitoring-lambda-policy"
  role = aws_iam_role.monitoring_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.ml_bucket.arn
        Condition = {
          StringLike = { "s3:prefix" = ["input/*", "${local.capture_prefix}/*", "monitoring/*"] }
        }
      },
      {
        # Read training data, captured traffic and the baseline
        Effect = "Allow"
        Action = "s3:GetObject"
        Resource = [
          "${aws_s3_bucket.ml_bucket.arn}/input/*",
          "${aws_s3_bucket.ml_bucket.arn}/${local.capture_prefix}/*",
          "${aws_s3_bucket.ml_bucket.arn}/monitoring/*",
        ]
      },
      {
        # Write the baseline and drift reports
        Effect   = "Allow"
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.ml_bucket.arn}/monitoring/*"
      },
      {
        Effect   = "Allow"
        Action   = "cloudwatch:PutMetricData"
        Resource = "*" # PutMetricData has no resource ARNs; scoped by namespace below
        Condition = {
          StringEquals = { "cloudwatch:namespace" = var.drift_metric_namespace }
        }
      }
    ]
  })
}

# -----------------------------
# Baseline: rebuilt by the deploy Lambda after a model passes the deploy gate and is
# deployed, so it always describes the model that is actually serving
# -----------------------------
resource "aws_lambda_function" "drift_baseline" {
  function_name = "${var.project_name}-drift-baseline"

  filename         = data.archive_file.monitoring_zip.output_path
  source_code_hash = data.archive_file.monitoring_zip.output_base64sha256

  role        = aws_iam_role.monitoring_lambda_role.arn
  handler     = "baseline.lambda_handler"
  runtime     = "python3.12"
  timeout     = 300
  memory_size = 512

  environment {
    variables = {
      BUCKET       = aws_s3_bucket.ml_bucket.bucket
      INPUT_PREFIX = "input/"
      BASELINE_KEY = "monitoring/baseline.json"
    }
  }
}

resource "aws_cloudwatch_log_group" "drift_baseline" {
  name              = "/aws/lambda/${aws_lambda_function.drift_baseline.function_name}"
  retention_in_days = 14
}

# -----------------------------
# Drift check: on a schedule, compare captured traffic with the baseline
# -----------------------------
resource "aws_lambda_function" "drift_check" {
  function_name = "${var.project_name}-drift-check"

  filename         = data.archive_file.monitoring_zip.output_path
  source_code_hash = data.archive_file.monitoring_zip.output_base64sha256

  role        = aws_iam_role.monitoring_lambda_role.arn
  handler     = "check_drift.lambda_handler"
  runtime     = "python3.12"
  timeout     = 600
  memory_size = 1024

  environment {
    variables = {
      BUCKET           = aws_s3_bucket.ml_bucket.bucket
      ENDPOINT_NAME    = var.endpoint_name
      VARIANT_NAME     = "AllTraffic"
      CAPTURE_PREFIX   = "${local.capture_prefix}/"
      BASELINE_KEY     = "monitoring/baseline.json"
      REPORT_PREFIX    = "monitoring/reports/"
      WINDOW_HOURS     = tostring(var.drift_window_hours)
      MIN_SAMPLES      = tostring(var.drift_min_samples)
      METRIC_NAMESPACE = var.drift_metric_namespace
    }
  }
}

resource "aws_cloudwatch_log_group" "drift_check" {
  name              = "/aws/lambda/${aws_lambda_function.drift_check.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_event_rule" "drift_schedule" {
  name                = "${var.project_name}-drift-check"
  description         = "Compare live endpoint traffic with the training baseline"
  schedule_expression = var.drift_check_schedule
}

resource "aws_cloudwatch_event_target" "drift_check" {
  rule = aws_cloudwatch_event_rule.drift_schedule.name
  arn  = aws_lambda_function.drift_check.arn
}

resource "aws_lambda_permission" "drift_check_schedule" {
  statement_id  = "AllowScheduleInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.drift_check.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.drift_schedule.arn
}

# -----------------------------
# Alarms -> SNS pipeline alerts topic
# -----------------------------
resource "aws_cloudwatch_metric_alarm" "model_drift" {
  alarm_name          = "${var.project_name}-model-drift"
  alarm_description   = "Live traffic drifted from the training baseline (highest PSI among statistically significant features). See s3://${aws_s3_bucket.ml_bucket.bucket}/monitoring/reports/ for which feature."
  namespace           = var.drift_metric_namespace
  metric_name         = "DriftScore"
  dimensions          = { Endpoint = var.endpoint_name }
  statistic           = "Maximum"
  period              = 3600
  evaluation_periods  = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = var.drift_psi_threshold
  treat_missing_data  = "notBreaching" # no traffic / not enough samples is not drift
  alarm_actions       = [aws_sns_topic.pipeline_alerts.arn]
  ok_actions          = [aws_sns_topic.pipeline_alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "drift_check_errors" {
  alarm_name          = "${var.project_name}-drift-check-errors"
  alarm_description   = "The drift-check Lambda is failing, so drift is not being monitored."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.drift_check.function_name }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.pipeline_alerts.arn]
}
