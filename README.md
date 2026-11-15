# LexiTrace — Phase 1: basic pipeline

Minimal ASR → summarisation → translation pipeline. Its only job is to give the
Phase 2 experiments something to run on, with every stage's output saved.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # Windows: set ANTHROPIC_API_KEY=...
# yt-dlp also needs ffmpeg installed on your system
```

## Run
```bash
python pipeline.py download  --lecture L01 --url "https://www.youtube.com/watch?v=XXXX"
python pipeline.py run-all   --lecture L01 --lang ur ks
```
Or stage by stage: `asr`, `summarise`, `translate --lang ur`.

## Output layout
```
data/L01/
  audio.wav, source.json
  asr_natural/            <- condition = natural Whisper output
    asr.json              (text, segments, word timestamps + confidences)
    summary_en.json       (per-chunk notes, source chunk kept alongside)
    translation_ur.json   (per-chunk English notes + translation)
  human_ref/              <- other conditions, added via `transcript`
```

## Using other transcript conditions (needed for Phase 2)
```bash
python pipeline.py transcript --lecture L01 --condition human_ref --file ref.txt
python pipeline.py summarise  --lecture L01 --condition human_ref
python pipeline.py translate  --lecture L01 --condition human_ref --lang ur
```
Later you will use the same commands for `ordinary_err`, `term_err`, `both_err`.

## Design decisions (write these down in your report)
- **Neutral baseline prompts.** No instruction to fix ASR errors and none about
  handling technical terms. This keeps research question C (implicit recovery)
  measurable. Term-handling instructions are a Phase 3 intervention.
- **Temperature 0, fixed model, prompt hash logged** in every output for reproducibility.
- **Chunked processing** (`chunk_words`) so each note stays linked to its source
  segment, which makes concept tracing possible.
- **Same LLM for summarising and translating** for simplicity. Swapping the
  translator (e.g. IndicTrans2) later is an easy ablation.
- Chunk-by-chunk summaries are concatenated, with no final merge step.

## Phase 1 done when
- [ ] 1 lecture runs end to end (ASR → notes → Urdu)
- [ ] Outputs inspected by hand (are the notes sensible? any obvious failures?)
- [ ] Settings frozen in `config.yaml`
- [ ] 10–15 lectures chosen for Phase 2
