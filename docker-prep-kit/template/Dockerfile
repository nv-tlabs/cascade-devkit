# Build your submission FROM the CASCADE common base image so your environment is
# ABI-compatible with the evaluator.
#
# CRITICAL: everything your system needs (Python env, dependencies, compiled
# extensions, model weights, code) MUST be installed under /opt/submission.
# Files outside that prefix are NOT shipped to the evaluator.
#
# The build tool passes the pinned base image via --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=ghcr.io/nv-tlabs/cascade-base:cuda13.0-py312
FROM ${BASE_IMAGE}

# Create an isolated environment inside the prefix.
RUN python -m venv /opt/submission/env
ENV PATH=/opt/submission/env/bin:$PATH

# Install your dependencies INTO the prefix.
COPY requirements.txt /opt/submission/requirements.txt
RUN /opt/submission/env/bin/pip install --no-cache-dir -r /opt/submission/requirements.txt

# Bake your code and weights into the prefix. Download weights HERE, at build
# time — the evaluator runs your system with the network disabled.
COPY run.py /opt/submission/run.py
# COPY model/ /opt/submission/model/

# No ENTRYPOINT is needed: the evaluator never runs this image. It restores
# /opt/submission on the base and launches the command from submission.yaml.
