# LexiTrace
### Tracking Technical Concepts Through ASR, Summarisation and Translation in Cross-Lingual Lectures

LexiTrace is a research-oriented project that investigates how technical concepts are preserved, altered, omitted, or mistranslated as educational lecture content passes through multiple language-processing stages.

The project focuses on the interaction between Automatic Speech Recognition (ASR), text summarisation, and machine translation, particularly in technical and educational contexts.

The broader goal is to understand how errors introduced at one stage affect the representation and understanding of technical concepts in subsequent stages, and to use these findings to develop a research-informed interactive lecture companion.

---

## 1. Motivation

Recorded lectures are an important source of educational content. However, transforming spoken lectures into concise notes in another language involves several potentially error-prone stages.

For example:

- ASR may incorrectly recognise technical terminology.
- Summarisation may omit or alter important technical concepts.
- Machine translation may mistranslate domain-specific expressions.
- Errors introduced in earlier stages may propagate through subsequent stages.

A transcript can have a relatively low overall word error rate while still containing errors that significantly affect the meaning of an important technical concept.

LexiTrace investigates this problem by examining concept-level changes throughout the processing pipeline rather than evaluating each component only in isolation.

---

## 2. Research Objectives

The primary objectives are:

1. Investigate the preservation of technical concepts across ASR, summarisation, and translation.
2. Study how different types of ASR errors affect downstream summaries and translations.
3. Examine whether summarisation preserves, removes, or recovers technical information.
4. Analyse how technical terminology changes during cross-lingual translation.
5. Identify concepts that are particularly vulnerable to information loss or semantic distortion.
6. Use research findings to inform the development of an interactive lecture companion.

---

## 3. System Architecture

The initial experimental pipeline follows three sequential stages:

```text
             Lecture Audio
                   |
                   v
       Automatic Speech Recognition
             Faster-Whisper
                   |
                   v
          English Transcript
                   |
                   v
             Summarisation
              FLAN-T5
                   |
                   v
            English Notes
                   |
                   v
           Machine Translation
                 NLLB
                   |
                   v
            Urdu Translation
```

Each stage produces structured outputs that can be inspected and used in subsequent experiments.

The initial implementation uses English lecture audio, generates English summaries, and translates the summaries into Urdu.

---

## 4. Research Methodology

The project is organised into four phases.

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Baseline Pipeline Development | Completed |
| Phase 2 | Controlled Experiments and Concept Tracking | Planned |
| Phase 3 | Research-Informed Application Development | Planned |
| Phase 4 | Evaluation and Research Documentation | Planned |

### Phase 1 — Baseline Pipeline

Develop a working ASR → summarisation → translation pipeline.

The initial implementation establishes the experimental foundation and produces structured outputs for subsequent analysis.

### Phase 2 — Core Research

Investigate how technical concepts change across the processing pipeline.

Planned research activities include:

- Controlled ASR error injection.
- Comparison of technical and non-technical recognition errors.
- Concept preservation and omission analysis.
- Error propagation across summarisation and translation.
- Investigation of semantic distortion and potential information recovery.

### Phase 3 — Application Development

Develop an interactive lecture companion informed by the findings of the research.

Potential capabilities include:

- Lecture transcription and multilingual notes.
- Timestamp-linked lecture navigation.
- Editable notes.
- Translation and concept clarification.
- Identification of potentially unreliable technical concepts.

Application features will be guided by the research findings rather than developed independently of them.

### Phase 4 — Evaluation and Documentation

Conduct systematic evaluation and document the two complementary contributions:

1. The research contribution: analysis of technical-concept changes across language-processing stages.
2. The software contribution: development and evaluation of the interactive lecture companion.

---

## 5. Technologies and Models

| Component | Technology / Model |
|---|---|
| Programming Language | Python |
| Speech Recognition | Faster-Whisper — `large-v3` |
| Summarisation | `google/flan-t5-base` |
| Machine Translation | `facebook/nllb-200-distilled-600M` |
| Deep Learning Framework | PyTorch |
| Model Framework | Hugging Face Transformers |
| Configuration | YAML |
| Experimental Outputs | JSON |
| Version Control | Git and GitHub |

The baseline models are locally executed rather than accessed through paid inference APIs.

**Note:** Model licences and usage restrictions must be checked before redistribution or commercial deployment.

---

## 6. Repository Structure

```text
LexiTrace/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
└── Phase1_Baseline/
    │
    ├── README.md
    │
    ├── code/
    │   └── pipeline.py
    │
    ├── config/
    │   └── config.yaml
    │
    ├── data/
    │   └── L01/
    │       ├── audio.wav
    │       └── source.json
    │
    ├── results/
    │   └── L01/
    │       └── asr_natural/
    │           ├── asr.json
    │           ├── summary_en.json
    │           └── translation_ur.json
    │
    └── documentation/
        └── phase1_observations.md
```

As development progresses, additional phase-specific directories will be introduced.

Each phase will maintain its own README and relevant code, configuration, results, and documentation.

---

## 7. Current Progress

**Current milestone: Phase 1 — Baseline Pipeline Completed**

The initial implementation supports:

- Lecture audio processing.
- Automatic speech recognition.
- English transcript generation.
- Chunk-based summarisation.
- Urdu translation.
- Structured JSON output storage.

The first baseline experiment uses an English machine-learning lecture on Gradient Descent.

Initial qualitative observations indicate potential issues involving technical-term recognition, information loss during summarisation, and inconsistent terminology during translation.

These observations are preliminary and are not presented as statistically validated research findings.

The next milestone is Phase 2: controlled experimentation and systematic concept-level analysis.

---

## 8. Project Vision

LexiTrace aims to move beyond conventional pipeline-level evaluation by examining whether the technical meaning of educational content survives successive transformations.

Ultimately, the project seeks to connect empirical research on language-processing errors with practical tools that help learners identify, inspect, and understand potentially altered technical information.

---

## 9. Project Status

This repository is under active development.

The current implementation represents the baseline experimental stage. Research experiments, application development, and systematic evaluation are planned as subsequent milestones.