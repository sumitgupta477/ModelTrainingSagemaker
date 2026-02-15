output "endpoint_name" {
  value = aws_sagemaker_endpoint.llm.name
}

output "api_url" {
  value = aws_apigatewayv2_api.http_api.api_endpoint
}

