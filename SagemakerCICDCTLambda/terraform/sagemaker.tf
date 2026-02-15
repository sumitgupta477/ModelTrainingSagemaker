resource "aws_sagemaker_model" "llm" {

  depends_on = [aws_lambda_function.trainer]

  name               = "llm-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"

    model_data_url = "s3://mlops-llm-bucket-12345/llm-model/output/llm-train-1771038993/output/model.tar.gz"

    environment = {
      "HF_TASK": "text-generation"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "llm" {
  name = "llm-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.llm.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }
}

resource "aws_sagemaker_endpoint" "llm" {
  name = "llm-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.llm.name
}
