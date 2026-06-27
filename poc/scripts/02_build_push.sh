#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="984072314535"
REGION="us-east-1"
ECR_REPO="flux-poc-training"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
TAG="latest"

echo "-> ECR auth via docker-credential-ecr-login (credHelpers configured)..."
# No explicit docker login needed — ~/.docker/config.json uses credHelpers for ECR

echo "-> Building image (this takes 10-20 minutes)..."
docker build \
  --platform linux/amd64 \
  -t "${ECR_URI}:${TAG}" \
  /Users/yabolin/claude-code/flux/poc/docker/

echo "-> Pushing image to ECR..."
docker push "${ECR_URI}:${TAG}"

echo "Image pushed: ${ECR_URI}:${TAG}"
