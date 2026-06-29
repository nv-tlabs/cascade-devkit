# Build your submission FROM the CASCADE base image. The build tool ships every
# layer you add on top of this base, so install things wherever is natural — a
# venv, conda, system packages via apt, model weights in ~/.cache, etc. There is
# NO required prefix; whatever your build adds is captured.
#
# Two rules still matter:
#   1. Download EVERYTHING at build time (deps + weights). The evaluator runs your
#      system with the network DISABLED.
#   2. Keep the CUDA toolkit you install compatible with the base's CUDA line.
#
# The build tool passes the pinned base via --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=ghcr.io/nv-tlabs/cascade-base:cuda13.0-py312
FROM ${BASE_IMAGE}

# Example: an isolated environment (location is up to you).
RUN python -m venv /opt/app/.venv
ENV PATH=/opt/app/.venv/bin:$PATH

# Install your dependencies (into the venv here, but system/conda also work).
COPY requirements.txt /opt/app/requirements.txt
RUN /opt/app/.venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt

# Bake your code in.
COPY run.py /opt/app/run.py

# Bake model weights at build time. They can live anywhere (HF's default cache is
# ~/.cache/huggingface) — it will be captured. Example:
#   RUN /opt/app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('your/model')"

# No ENTRYPOINT needed: the evaluator declares how to launch you in submission.yaml.
