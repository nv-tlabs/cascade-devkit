# CASCADE Challenge common base image.
#
# This image is the ABI contract shared by BOTH sides of evaluation:
#   - Participants build their submission `FROM` this image.
#   - The official evaluator runs on this SAME image and reapplies the layers
#     the participant added on top of it.
#
# Because both sides use an identical base, an environment built here
# (venv/conda, compiled CUDA extensions, weights — installed anywhere) runs
# without relocation or glibc/CUDA-ABI surprises at evaluation time.
#
# The CUDA toolkit that participants install must be compatible with the GPU
# driver on the evaluation host (toolkit version <= driver's supported version).
# Pin and publish this image by digest before a challenge run.

ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

# Submissions install wherever is natural (a venv, conda, system packages,
# ~/.cache, ...). There is no required prefix: every layer the build adds on top
# of this base is captured and reapplied at evaluation time.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
        curl \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
