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

variable "data_capture_percent" {
  description = "Percentage of endpoint requests saved to S3 for drift monitoring (1-100)"
  type        = number
  default     = 100

  validation {
    condition     = var.data_capture_percent >= 1 && var.data_capture_percent <= 100
    error_message = "data_capture_percent must be between 1 and 100."
  }
}

variable "data_capture_retention_days" {
  description = "Days to keep captured requests/responses in S3"
  type        = number
  default     = 30
}

variable "drift_check_schedule" {
  description = "How often the drift check runs (EventBridge schedule expression)"
  type        = string
  default     = "rate(1 hour)"
}

variable "drift_window_hours" {
  description = "How many hours of captured traffic each drift check analyses"
  type        = number
  default     = 24
}

variable "drift_min_samples" {
  description = "Minimum captured requests in the window before drift is scored"
  type        = number
  default     = 30
}

variable "drift_psi_threshold" {
  description = "Alarm when any feature's PSI reaches this (0.1 = moderate, 0.2 = significant drift)"
  type        = number
  default     = 0.2
}

variable "drift_metric_namespace" {
  description = "CloudWatch namespace for drift metrics"
  type        = string
  default     = "LLMChat/ModelDrift"
}

variable "inference_image" {
  description = "Hugging Face inference container used to serve the fine-tuned model"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
}
