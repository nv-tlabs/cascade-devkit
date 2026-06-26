# Docker Preparation Kit

Use this kit to prepare the Docker image that you submit to the AV Causal
Scenario Retrieval Challenge.

The evaluator runs your image as a batch program. Your Space only needs to build
successfully; it does not need to serve a web app.

## Submission Flow

1. Copy `template/` into a new repository or Hugging Face Space.
2. Replace the baseline ranking logic in `run.py` with your retrieval system.
3. Bake model code, configuration, and weights into the image. Do not rely on
   runtime downloads during evaluation.
4. Build and test locally:

   ```bash
   docker build -t my-crc-submission ./template
   docker run --rm \
     -v /path/to/input:/input:ro \
     -v /tmp/crc-output:/output \
     my-crc-submission
   python validate_submission.py \
     --predictions /tmp/crc-output/predictions.jsonl \
     --queries /path/to/input/queries.jsonl \
     --videos /path/to/input/videos.jsonl
   ```

5. Create a private Hugging Face Space with `sdk: docker`.
6. Push your Docker Space repository.
7. Add the challenge evaluation bot as a read collaborator when instructed.
8. Submit the Space ref, for example `your-team/your-submission`, in the
   challenge frontend.

## Runtime Contract

Your image entrypoint must:

- Read `/input/queries.jsonl`, one JSON object per line:
  `{"query_id": "...", "text": "..."}`.
- Read `/input/videos.jsonl`, one JSON object per line:
  `{"video_id": "...", "path": "videos/<video_id>.mp4"}`.
- Read video files from `/input/videos/<video_id>.mp4`.
- Write `/output/predictions.jsonl`, one JSON object per query:
  `{"query_id": "...", "video_ids": ["...ranked best-first..."]}`.
- Exit with status code `0` after writing predictions.

Every query should appear exactly once. Returned `video_ids` must come from the
provided corpus and should be ordered from best match to worst match. Duplicate
IDs are counted once by the scorer, preserving the first occurrence.

## Evaluation Constraints

The official evaluator runs participant inference separately from trusted
scoring. Participant containers do not receive labels or service tokens. Treat
`/input` as read-only and write only under `/output`.
