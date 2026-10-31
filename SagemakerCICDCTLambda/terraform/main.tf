terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  # Tag Terraform-managed resources so costs can be filtered in Cost Explorer
  default_tags {
    tags = {
      Project   = "sagemaker-continuous-training"
      ManagedBy = "terraform"
    }
  }
}

locals {
  job_prefix     = "llm-train"
  capture_prefix = "datacapture" # endpoint data capture lands under s3://<bucket>/datacapture/
}
