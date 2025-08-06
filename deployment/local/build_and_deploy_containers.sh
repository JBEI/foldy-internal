#!/bin/bash

set -e
set -o pipefail  # Also fail if any part of a pipe fails

if [ "$#" -gt 1 ]; then
  echo "Illegal number of parameters."
  echo ""
  echo "Example invocation:"
  echo "build_and_deploy_containers.sh [version_tag]"
  echo ""
  echo "If no version_tag is provided, 'latest' will be used."
  exit 2
fi

# Default version tag
VERSION=${1:-latest}

# DockerHub configuration - change these to your DockerHub username/organization
DOCKERHUB_USER=${DOCKERHUB_USER:-jbrlbl}
DOCKERHUB_REPO_PREFIX=${DOCKERHUB_REPO_PREFIX:-foldy}

# Docker image tags
FRONTEND_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-frontend:${VERSION}
BACKEND_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-backend:${VERSION}
WORKER_ESM_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-esm:${VERSION}
WORKER_BOLTZ_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-boltz:${VERSION}

# Build arguments for frontend
BACKEND_URL=${BACKEND_URL:-http://localhost:8080}
INSTITUTION=${INSTITUTION:-Foldy Local}

echo "Building and deploying Foldy containers to DockerHub..."
echo "Using the following configuration:"
echo "  DOCKERHUB_USER: $DOCKERHUB_USER"
echo "  VERSION: $VERSION"
echo "  BACKEND_URL: $BACKEND_URL"
echo "  INSTITUTION: $INSTITUTION"
echo ""

# Navigate to project root (assuming script is in deployment/local/)
cd "$(dirname "$0")/../.."

echo "Building backend..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $BACKEND_TAG -f backend/Dockerfile .

echo "Building worker ESM..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_ESM_TAG -f worker/Dockerfile.esm .

echo "Building worker BOLTZ..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_BOLTZ_TAG -f worker/Dockerfile.boltz .

echo "Building frontend..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FRONTEND_TAG \
  --build-arg BACKEND_URL=$BACKEND_URL \
  --build-arg INSTITUTION="$INSTITUTION" \
  frontend

echo ""
echo "Pushing images to DockerHub..."
echo "Note: Make sure you're logged in to DockerHub (docker login)"

docker push $BACKEND_TAG
docker push $WORKER_ESM_TAG
docker push $WORKER_BOLTZ_TAG
docker push $FRONTEND_TAG

echo ""
echo "Successfully built and pushed all images!"
echo ""
echo "Images pushed:"
echo "  - $FRONTEND_TAG"
echo "  - $BACKEND_TAG"
echo "  - $WORKER_ESM_TAG"
echo "  - $WORKER_BOLTZ_TAG"
