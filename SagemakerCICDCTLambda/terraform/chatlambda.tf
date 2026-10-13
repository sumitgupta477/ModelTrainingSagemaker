# -------------------------------
# IAM Role for Lambda
# -------------------------------
resource "aws_iam_role" "chat_lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# -------------------------------
# Lambda Policy
# -------------------------------
resource "aws_iam_policy" "chat_lambda_policy" {
  name = "${var.project_name}-lambda-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [

      # CloudWatch logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },

      # Invoke SageMaker
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:endpoint/${var.endpoint_name}"
      }
    ]
  })
}

# -------------------------------
# Attach Policy
# -------------------------------
resource "aws_iam_role_policy_attachment" "lambda_attach" {
  role       = aws_iam_role.chat_lambda_role.name
  policy_arn = aws_iam_policy.chat_lambda_policy.arn
}

# -------------------------------
# Lambda ZIP
# -------------------------------
data "archive_file" "chat_lambda_zip" {
  type        = "zip"
  source_dir  = "../chatlambda"
  output_path = "${path.module}/build/chat_lambda.zip"
}

# -------------------------------
# Lambda Function
# -------------------------------
resource "aws_lambda_function" "chat_lambda" {

  function_name = "${var.project_name}-invoke"

  filename         = data.archive_file.chat_lambda_zip.output_path
  source_code_hash = data.archive_file.chat_lambda_zip.output_base64sha256

  runtime = "python3.12"
  handler = "chat.lambda_handler"

  role = aws_iam_role.chat_lambda_role.arn

  timeout = 30
  memory_size = 512

  environment {
    variables = {
      ENDPOINT_NAME = var.endpoint_name
    }
  }
}