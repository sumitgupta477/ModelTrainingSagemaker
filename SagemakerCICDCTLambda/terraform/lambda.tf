data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "../lambda/trigger_training.py"
  output_path = "../lambda/trigger_training.zip"
}

resource "aws_lambda_function" "trainer" {
  function_name = "llm-trainer"

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  role = aws_iam_role.lambda_role.arn
  handler = "trigger_training.lambda_handler"
  runtime = "python3.10"
  timeout       = 900
  memory_size   = 1024

  environment {
    variables = {
      BUCKET         = aws_s3_bucket.ml_bucket.bucket
      SAGEMAKER_ROLE = aws_iam_role.sagemaker_role.arn
      TRAIN_INSTANCE_TYPE   = "ml.m5.large"
      TRAIN_INSTANCE_COUNT  = "1"
      TRAIN_VOLUME_SIZE     = "50"
      ENDPOINT_INSTANCE_TYPE= "t2.medium"
    }
  }
}

# -----------------------------
# 4. S3 Event Notification
# -----------------------------
resource "aws_lambda_permission" "s3_invoke" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trainer.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.ml_bucket.arn
}

resource "aws_s3_bucket_notification" "bucket_event" {
  bucket = aws_s3_bucket.ml_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.trainer.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".txt" # optional: trigger only on .txt files
  }

  depends_on = [aws_lambda_permission.s3_invoke]
}