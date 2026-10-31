output "bucket_name" {
  value = aws_s3_bucket.ml_bucket.bucket
}

output "endpoint_name" {
  description = "Created by the deploy Lambda after the first training job completes"
  value       = var.endpoint_name
}

output "drift_reports" {
  description = "Hourly drift reports written by the drift-check Lambda"
  value       = "s3://${aws_s3_bucket.ml_bucket.bucket}/monitoring/reports/"
}

output "drift_alarm" {
  value = aws_cloudwatch_metric_alarm.model_drift.alarm_name
}

output "api_url" {
  value = "${aws_apigatewayv2_api.http_api.api_endpoint}/chat"
}
