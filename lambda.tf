data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "backend"
  output_path = "lambda.zip"
}

resource "aws_lambda_function" "api" {

  function_name = "ai-case-api"

  filename = data.archive_file.lambda_zip.output_path

  handler = "submit.handler"
  runtime = "nodejs20.x"

  role   = aws_iam_role.lambda_role.arn
  timeout = 30

  environment {
    variables = {
      CASES_TABLE = aws_dynamodb_table.cases.name
      // MODEL_ID    = "anthropic.claude-v2"
      ADMIN_EMAIL = "sumitgupta477@gmail.com"
      SES_FROM    = "sumitgupta477@gmail.com"
    }
  }
}
