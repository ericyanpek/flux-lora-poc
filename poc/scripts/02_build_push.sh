#!/usr/bin/env bash
set -euo pipefail

# Load config from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a && source "$SCRIPT_DIR/../.env" && set +a

ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/flux-poc-training"
TAG="latest"

echo "-> ECR auth via docker-credential-ecr-login (credHelpers configured)..."
# No explicit docker login needed — ~/.docker/config.json uses credHelpers for ECR

echo "-> Building image (this takes 10-20 minutes)..."
docker build \
  --platform linux/amd64 \
  -t "${ECR_URI}:${TAG}" \
  "$SCRIPT_DIR/../docker/"

echo "-> Pushing image to ECR..."
docker push "${ECR_URI}:${TAG}"

echo "Image pushed: ${ECR_URI}:${TAG}"
