# Phase 1 — Baseline Pipeline

## Overview
Phase 1 establishes the baseline processing pipeline for LexiTrace, connecting Automatic Speech Recognition (ASR), English summarisation, and cross-lingual translation.

## Pipeline Architecture

Lecture Audio → ASR → English Transcript → English Summarisation → Urdu Translation

## Models Used

| Component | Model |
|---|---|
| ASR | Faster-Whisper large-v3 |
| Summarisation | google/flan-t5-base |
| Translation | facebook/nllb-200-distilled-600M |

## Test Lecture

- ID: L01
- Topic: Gradient Descent
- Source: Andrew Ng — Machine Learning Specialization
- Source language: English
- Target language: Urdu
- Duration: Approximately 8 minutes

## Folder Structure

```text
Phase1_Baseline/
├── code/
│   └── pipeline.py
├── config/
│   └── config.yaml
├── documentation/
│   └── phase1_observations.md
└── results/
    └── L01/
        ├── asr.json
        ├── summary_en.json
        └── translation_ur.json
```

## Execution Status

- [x] Audio acquisition
- [x] Automatic Speech Recognition
- [x] English summarisation
- [x] Urdu translation
- [x] End-to-end pipeline execution

## Research Notes

The initial experiment revealed qualitative issues involving technical terminology, concept preservation, repetition, and semantic distortion.

These observations motivate the controlled investigations planned for Phase 2.

See [Phase 1 Observations](documentation/phase1_observations.md) for details.

## Limitations

This phase represents a baseline proof-of-concept using one lecture. It does not establish quantitative performance or generalisation across multiple lectures and languages.