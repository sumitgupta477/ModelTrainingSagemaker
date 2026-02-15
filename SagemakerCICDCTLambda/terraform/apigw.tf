# -------------------------------
# API Gateway HTTP API
# -------------------------------
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["*"]
  }
}

# -------------------------------
# API Integration
# -------------------------------
resource "aws_apigatewayv2_integration" "lambda_integration" {

  api_id = aws_apigatewayv2_api.http_api.id

  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.chat_lambda.invoke_arn

  payload_format_version = "2.0"
}

# -------------------------------
# API Route
# -------------------------------
resource "aws_apigatewayv2_route" "chat_route" {

  api_id = aws_apigatewayv2_api.http_api.id

  route_key = "POST /chat"

  target = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# -------------------------------
# API Stage
# -------------------------------
resource "aws_apigatewayv2_stage" "default" {

  api_id = aws_apigatewayv2_api.http_api.id

  name = "$default"
  auto_deploy = true
}

# -------------------------------
# Lambda Permission
# -------------------------------
resource "aws_lambda_permission" "api_permission" {

  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chat_lambda.function_name

  principal = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}