#!/bin/bash

# This script is used to build a docker image for LBM ECS pipeline:
# https://us-east-1.console.aws.amazon.com/ecs/v2/clusters/lbm-ecs-pipeline/services?region=us-east-1   # noqa
#
# Usage:
#   AWS_PROFILE=manip-cluster ./build_and_push_docker.sh

set -euo pipefail

# Build the latest docker image.
echo "Building the docker image..."

TMP_DOCKER_BUILD_DIR="/tmp/roboverse_build"
# Clean up a temp folder if exits from the previously failed build.
if [ -d ${TMP_DOCKER_BUILD_DIR} ]; then
  rm -rf ${TMP_DOCKER_BUILD_DIR}
fi

# Create a temp folder and copy the local lbm into it.
REQUIREMENTS_FILE=requirements-312.txt
mkdir -p ${TMP_DOCKER_BUILD_DIR}/lbm
cd $(dirname $0)/../../ && \
  cp ./* ${TMP_DOCKER_BUILD_DIR}/lbm -r && \
  cp ./.git ${TMP_DOCKER_BUILD_DIR}/lbm/ -r
cd ${TMP_DOCKER_BUILD_DIR}
cp roboverse/Dockerfile ./Dockerfile

# NOTE: If there is any change, please sync the same logic to
# ./setup/docker/build_and_push_image.sh
cp lbm/setup/venv/${REQUIREMENTS_FILE} .
sed -i "/^torch==/d" ${REQUIREMENTS_FILE}
# Copy all the pip wheels into here and modify the paths in `requirements.in`
# so that Dockerfile can handle the wheels properly.
cp lbm/setup/venv/wheels/*.whl .
sed -i -e "s/setup\/venv\/wheels/\/opt\/ml\/code/g" ${REQUIREMENTS_FILE}

# Build docker image.
docker build -t lbm-ecs-pipeline:latest \
  --build-arg REQUIREMENTS_FILE=${REQUIREMENTS_FILE} \
  --squash .
rm -rf ${TMP_DOCKER_BUILD_DIR}

# Login ECR.
ECR_ADDR=682769330988.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_ADDR

# Tag & push the built docker image.
docker tag lbm-ecs-pipeline:latest $ECR_ADDR/lbm-ecs-pipeline:latest
docker push $ECR_ADDR/lbm-ecs-pipeline:latest
