variable "bucket_name" {
  description = "S3 bucket for training data (input/), training code (code/), checkpoints and model artifacts. Must be globally unique."
  type        = string
  default     = "mlops-llm-bucket-12345"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "llm-chat"
}

variable "endpoint_name" {
  description = "SageMaker endpoint that the deploy Lambda creates and keeps updated"
  type        = string
  default     = "llm-endpoint"
}

variable "train_instance_type" {
  type    = string
  default = "ml.m5.large"
}

variable "endpoint_instance_type" {
  description = "SageMaker hosting instance type (SageMaker instance types use the ml. prefix)"
  type        = string
  default     = "ml.t2.medium"
}

variable "training_image" {
  description = "AWS Deep Learning Container used for training"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker"
}

variable "inference_image" {
  description = "Hugging Face inference container used to serve the fine-tuned model"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
}
