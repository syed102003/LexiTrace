# LexiTrace — Phase 1 Baseline Observations

## 1. Objective

The objective of Phase 1 was to establish a working baseline pipeline for processing cross-lingual lecture content through Automatic Speech Recognition (ASR), English summarisation, and machine translation.

## 2. Test Lecture

- Lecture ID: L01
- Topic: Gradient Descent
- Source: Andrew Ng — Machine Learning Specialization
- Duration: Approximately 8 minutes
- Source language: English
- Target language: Urdu

## 3. Baseline Configuration

| Stage | Model |
|---|---|
| ASR | Faster-Whisper large-v3 |
| Summarisation | google/flan-t5-base |
| Translation | facebook/nllb-200-distilled-600M |

The pipeline was executed using CPU inference.

## 4. Pipeline Execution

The following stages were completed successfully:

1. Lecture audio acquisition.
2. English speech recognition and transcript generation.
3. English summarisation of the recognised transcript.
4. Translation of the generated English summary into Urdu.

## 5. Preliminary Qualitative Observations

### ASR
The ASR system generated a transcript containing the main lecture content. Some possible recognition errors involving technical terminology and mathematical expressions were observed.

### Summarisation
The generated summaries contained repetition, incomplete statements, and some semantic distortions. In one instance, an incorrectly recognised technical expression appeared to influence the generated summary.

### Translation
The Urdu translation was generated successfully. However, technical terminology was translated inconsistently, and some semantic problems present in the English summaries were retained in the translated output.

## 6. Limitations

- Only one lecture was processed during this baseline demonstration.
- Evaluation was qualitative rather than based on quantitative metrics.
- No controlled error injection or comparison experiments were conducted.
- No conclusions about generalisation across lectures or languages can yet be drawn.

## 7. Conclusion

Phase 1 successfully established an end-to-end baseline pipeline connecting ASR, English summarisation, and Urdu translation.

The observed issues motivate the controlled experiments planned for Phase 2, particularly the investigation of technical concept preservation, omission, alteration, and error propagation across processing stages.