#!/usr/bin/env bash
# Package train.py + requirements.txt and upload them to the pipeline bucket.
# SageMaker installs requirements.txt before running train.py.
#
# Usage: ./scripts/package_training_code.sh <bucket-name>
set -euo pipefail

BUCKET="${1:?usage: $0 <bucket-name>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)/training.tar.gz"

tar -czf "$OUT" -C "$HERE/training" train.py requirements.txt
aws s3 cp "$OUT" "s3://$BUCKET/code/training.tar.gz"
echo "Uploaded training code to s3://$BUCKET/code/training.tar.gz"
