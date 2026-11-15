# LexiTrace — Phase 1: local basic pipeline

Minimal local **ASR → summarisation → translation** pipeline. Its purpose is to
provide a consistent experimental substrate for Phase 2, with every stage's
output saved for concept tracing.

No paid API key is required. The first run downloads the selected model weights
from Hugging Face; subsequent runs use the local model cache.

## Models

- **ASR:** Faster-Whisper (`large-v3`)
- **Summarisation:** `google/flan-t5-base`
- **Translation:** `facebook/nllb-200-distilled-600M`

Model downloads require an internet connection. Inference runs locally after
the models have been downloaded. Runtime and memory requirements depend on
your computer; CPU inference may be slow.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Install **FFmpeg** separately and make sure it is available on your PATH.
`yt-dlp` uses it to extract audio.

There is no `ANTHROPIC_API_KEY` or other paid API key to configure.

## Run the full pipeline

```bash
python pipeline.py download --lecture L01 --url "https://www.youtube.com/watch?v=XXXX"
python pipeline.py run-all --lecture L01 --lang ur ks
```

The first execution downloads the model weights. This can take time and requires
sufficient disk space.

You can also run stages separately:

```bash
python pipeline.py asr --lecture L01
python pipeline.py summarise --lecture L01
python pipeline.py translate --lecture L01 --lang ur
```

Supported configured language codes:

- `ur` — Urdu
- `ks` — Kashmiri (Perso-Arabic script; verify model support before relying on it)
- `hi` — Hindi
- `kn` — Kannada

## Output layout

```text
data/L01/
  audio.wav
  source.json
  asr_natural/
    asr.json
    summary_en.json
    translation_ur.json
    translation_ks.json
```

- `asr.json` stores transcript text, segments, word timestamps and confidence.
- `summary_en.json` stores each source transcript chunk beside its generated notes.
- `translation_<language>.json` stores each English notes chunk beside its translation.

## Using other transcript conditions (needed for Phase 2)

Import a human-reference or error-injected transcript as a separate condition:

```bash
python pipeline.py transcript --lecture L01 --condition human_ref --file ref.txt
python pipeline.py summarise --lecture L01 --condition human_ref
python pipeline.py translate --lecture L01 --condition human_ref --lang ur
```

Use the same process for `ordinary_err`, `term_err` and `both_err`.

For `run-all` with a non-default condition, import that condition's transcript
first. The ASR command writes the natural Whisper transcript to `asr_natural`.

## Experimental design notes

- **Neutral baseline summarisation prompt:** it does not instruct the model to
  repair ASR errors or specially protect technical terms. This keeps implicit
  recovery measurable. Any term-handling instruction should be a later,
  separately controlled intervention.
- **Fixed model and settings:** keep the model IDs, prompts, chunking and
  generation settings unchanged across experimental conditions.
- **Reproducibility metadata:** outputs record model/configuration details and
  a prompt or configuration hash.
- **Chunk-level traceability:** each summary and translation chunk retains its
  corresponding source text or English notes.
- **No final merge step:** chunk-level outputs are concatenated for convenience,
  while the individual chunk records remain available for analysis.

Changing a model, prompt, chunk size, decoding setting or language mapping can
change results. Record and freeze such changes before the controlled experiments.

## Important model-license note

The project code's `LICENSE` does not automatically cover downloaded model
weights. Check each model's current model card and licence before research
redistribution or commercial use. In particular, review the NLLB model's
non-commercial-use restrictions before using it in a commercial application.

## Phase 1 completion checklist

- [ ] One lecture runs end to end (ASR → notes → Urdu).
- [ ] Outputs are inspected manually for obvious failures.
- [ ] Settings are frozen in `config.yaml`.
- [ ] Kashmiri translation support is tested rather than assumed.
- [ ] 10–15 lectures are selected for Phase 2.
