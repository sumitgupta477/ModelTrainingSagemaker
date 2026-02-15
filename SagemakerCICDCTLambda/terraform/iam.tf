# ------------------------------
# 1️⃣ IAM Role for SageMaker
# ------------------------------
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-llm-role"

  # Allow SageMaker to assume this role
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# ------------------------------
# 2️⃣ List of Managed Policies
# ------------------------------
locals {
  sagemaker_policies = [
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/CloudWatchFullAccess"
  ]
}

# ------------------------------
# 3️⃣ Attach Multiple Policies Using for_each
# ------------------------------
resource "aws_iam_role_policy_attachment" "sagemaker_attach" {
  for_each  = toset(local.sagemaker_policies)
  role      = aws_iam_role.sagemaker_role.name
  policy_arn = each.value
}

# ------------------------------
# 4️⃣ (Optional) Inline Policy for Custom Permissions
# ------------------------------
resource "aws_iam_role_policy" "sagemaker_inline" {
  name = "sagemaker-llm-inline"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "s3:ListBucket"
        Effect = "Allow"
        # ListBucket requires the bucket ARN
        Resource = "arn:aws:s3:::mlops-llm-bucket-12345"
      },
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Effect = "Allow"
        # Objects require the bucket ARN followed by /*
        Resource = "arn:aws:s3:::mlops-llm-bucket-12345/*"
      }
    ]
  })
}


resource "aws_iam_role" "lambda_role" {
  name = "llm-lambda-role"

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

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_sagemaker" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_s3" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Attach policies: S3, SageMaker, CloudWatch Logs
resource "aws_iam_role_policy" "lambda_policy" {
  name = "llm-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:ListTrainingJobs"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.ml_bucket.arn,
          "${aws_s3_bucket.ml_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}