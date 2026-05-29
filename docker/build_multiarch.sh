#! /usr/bin/env bash

# Build multiarch script

# Platform variants: - Raspberry Pi 3B (linux/arm/v7); - Dev PC (linux/amd64)

# Prerequisites (run once on dev PC):
#   1. Docker Desktop installed and running
#   2. docker buildx version >= 0.10
#   3. docker login   (your Docker Hub credentials)
#   4. QEMU binfmt registered (step below does it automatically)

# Usage:
#   chmod +x build_multiarch.sh
#   ./build_multiarch.sh                 # builds + pushes latest
#   ./build_multiarch.sh v1.2.0          # builds + pushes a tag

set -euo pipefail # -e: stop if fails; -u: treat unset variables as errors; -o pipefail: a pipe fails if ANY command in it fails


# ── Configuration ────────────────────────────────────────────
DOCKER_HUB_USER="your_dockerhub_username"   # ← change this
IMAGE_NAME="edgeia-person-detector"
TAG="${1:-latest}"                           # first CLI arg, default = "latest"
FULL_IMAGE="${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"
BUILDER_NAME="multiarch-builder"

echo "EdgeIA PFE - Multi-Architecture Docker Build"
echo "Image : ${FULL_IMAGE}"
echo "Platforms : linux/amd64, linux/arm/v7"

# Step 1: Register QEMU binfmt handlers
echo ""
echo "Register QEMU binfmt handlers"
docker run --rm --privileged \
    multiarch/qemu-user-static \
    --reset -p yes
echo "QEMU registred for arm/v7 emulation"

# Step 2: Create or reuse buildx builder instance
echo ""
echo "Setup buildx builder instance"
if docker buildx inspect "${BUILDER_NAME}" > /dev/null 2>&1; then
    echo "Builder '${BUILDER_NAME}' already exists - reusing"
else
    docker buildx create \
        --name "${BUILDER_NAME}" \
        --driver docker-container \
        --bootstrap
    
    echo "builder '${BUILDER_NAME}' created"
fi
docker buildx use "${BUILDER_NAME}"

# Step 3: Build and push the multi-arch image
echo ""
echo "Building for amd64 + arm/v7 and pushing"
docker buildx build \
    --platform linux/amd64,linux/arm/v7 \
    --tag "${FULL_IMAGE}" \
    --push \
    -f Dockerfile \
    ..
echo "Build complete and pushing to Docker Hub"

# Step 4: Verify the manifest
echo ""
echo "Verify the manifest on Docker Hub"
docker buildx imagetools inspect "${FULL_IMAGE}"

# Step 5: Tag as 'latest' if a version tag was given
# Good practice: always keep 'latest' pointing to the most recent

if [ "${TAG}" != "latest" ]; then
    echo ""
    echo "tagging as 'latest'"
    docker buildx build \
        --platform linux/amd64,linux/arm/v7 \
        --tag "${DOCKER_HUB_USER}/${IMAGE_NAME}:latest" \
        --push \
        -f Dockerfile \
        ..
    echo "'latest' tag updated"
fi

echo ""
echo "Image available on Docker Hub: '${FULL_IMAGE}'"