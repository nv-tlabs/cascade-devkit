---
title: My CRC Submission
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# My CRC Submission

This is a minimal Docker Space template for the AV Causal Scenario Retrieval
Challenge.

The evaluator runs this image with challenge data mounted at `/input` and
collects predictions from `/output/predictions.jsonl`. Replace the baseline in
`run.py` with your text-to-video retrieval model. For each query, return only
videos your system predicts are matches, ranked strongest match first.

Your Space should be private. Add the challenge evaluation bot as a read
collaborator when instructed, then submit this Space ref in the challenge
frontend.
