# Minimal submission image for the AV Causal Scenario Retrieval Challenge.
#
# Bake all code, config, and weights into this image. The official evaluator
# may run without internet access and will mount inputs at /input.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY model/ ./model/
COPY run.py .

ENTRYPOINT ["python", "run.py"]
