# Build your submission FROM the pinned CASCADE base (a stock Python image). The
# build tool ships every layer you add on top of it, so install things wherever
# is natural — a venv, conda, system packages (apt), weights in ~/.cache, etc.
# There is NO required prefix; whatever your build adds is captured.
#
# GPU/CUDA comes from your framework's pip wheels (e.g. `pip install torch` pulls
# CUDA + cuDNN) plus the GPU driver provided by the evaluation host. You do NOT
# need a CUDA base image. The base already includes git/gcc/make/curl.
#
# Two rules still matter:
#   1. Download EVERYTHING at build time (deps + weights). The evaluator runs your
#      system with the network DISABLED.
#   2. Target the single NVIDIA A100 80 GB evaluation GPU. Its compute capability
#      is 8.0, so framework wheels and compiled CUDA extensions must include
#      sm_80 kernels. A CUDA version string alone does not prove compatibility.
#
# The build tool passes the pinned base via --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146
FROM ${BASE_IMAGE}

# Example: an isolated environment (location is up to you).
RUN python -m venv /opt/app/.venv
ENV PATH=/opt/app/.venv/bin:$PATH

# Install your dependencies. Your GPU framework brings its own CUDA, e.g. add
# `torch` to requirements.txt (it pulls the matching CUDA + cuDNN wheels).
# If this step builds PyTorch CUDA extensions, compile them for the evaluation GPU:
#   ENV TORCH_CUDA_ARCH_LIST="8.0"
COPY requirements.txt /opt/app/requirements.txt
RUN /opt/app/.venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt

# Bake your code in.
COPY run.py /opt/app/run.py

# Bake model weights at build time. They can live anywhere (HF's default cache is
# ~/.cache/huggingface) — it will be captured. Example:
#   RUN /opt/app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('your/model')"
# For gated/private weights, download them on the host with the cached `hf auth
# login`, place only the weights in this build context, and COPY them. Never use
# ARG/ENV for a token or COPY a token file: every resulting layer is published.

# No ENTRYPOINT needed: the evaluator declares how to launch you in submission.yaml.
