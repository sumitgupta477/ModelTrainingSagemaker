resource "aws_s3_bucket" "ml_bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "ml_bucket" {
  bucket                  = aws_s3_bucket.ml_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_bucket" {
  bucket = aws_s3_bucket.ml_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Spot-training checkpoints are only needed while a job runs; expire them to save storage cost.
resource "aws_s3_bucket_lifecycle_configuration" "ml_bucket" {
  bucket = aws_s3_bucket.ml_bucket.id
  rule {
    id     = "expire-checkpoints"
    status = "Enabled"
    filter {
      prefix = "checkpoints/"
    }
    expiration {
      days = 7
    }
  }
}
