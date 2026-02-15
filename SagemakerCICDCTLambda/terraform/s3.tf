resource "aws_s3_bucket" "ml_bucket" {
  bucket = var.bucket_name
}
