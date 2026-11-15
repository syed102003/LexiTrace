#!/usr/bin/env python3
"""LexiTrace Phase 1: minimal ASR -> summarisation -> translation pipeline.

Every stage reads/writes JSON under data/<lecture>/<condition>/ so concepts can
be traced stage by stage in Phase 2.

Examples
  python pipeline.py download  --lecture L01 --url https://youtube.com/watch?v=XXXX
  python pipeline.py asr       --lecture L01
  python pipeline.py summarise --lecture L01
  python pipeline.py translate --lecture L01 --lang ur
  python pipeline.py run-all   --lecture L01 --lang ur ks

  # feed a different transcript (human reference / error-injected) as a new condition
  python pipeline.py transcript --lecture L01 --condition human_ref --file ref.txt
  python pipeline.py summarise  --lecture L01 --condition human_ref
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DEFAULT_CONDITION = "asr_natural"


# ---------- helpers ----------
def lecture_dir(lecture):
    d = ROOT / "data" / lecture
    d.mkdir(parents=True, exist_ok=True)
    return d


def cond_dir(lecture, condition):
    d = lecture_dir(lecture) / condition
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}")


def load_json(path):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}; run the previous stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_hash(name):
    return hashlib.sha256(CFG["prompts"][name].encode()).hexdigest()[:10]


def chunk_words(text, n):
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [""]


def call_llm(prompt):
    """Single LLM call (Anthropic API). Reads ANTHROPIC_API_KEY from the environment."""
    import anthropic

    client = anthropic.Anthropic()
    kwargs = dict(
        model=CFG["llm"]["model"],
        max_tokens=CFG["llm"]["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    if CFG["llm"].get("temperature") is not None:
        kwargs["temperature"] = CFG["llm"]["temperature"]
    for attempt in range(3):
        try:
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:  # simple retry; fine for a research prototype
            print(f"LLM error (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    raise SystemExit("LLM call failed 3 times.")


# ---------- stages ----------
def cmd_download(a):
    out = lecture_dir(a.lecture) / "audio.wav"
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "-o", str(out.with_suffix(".%(ext)s")), a.url],
        check=True,
    )
    save_json(lecture_dir(a.lecture) / "source.json", {"url": a.url, "audio": out.name})


def cmd_asr(a):
    from faster_whisper import WhisperModel

    audio = lecture_dir(a.lecture) / "audio.wav"
    if not audio.exists():
        raise SystemExit("audio.wav not found; run `download` first (or place it there).")
    c = CFG["asr"]
    model = WhisperModel(c["model"], device=c["device"], compute_type=c["compute_type"])
    segments, info = model.transcribe(
        str(audio), language=c["language"], beam_size=c["beam_size"],
        word_timestamps=True, vad_filter=True,
    )
    segs = []
    for s in segments:
        segs.append({
            "id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "words": [{"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2),
                       "p": round(w.probability, 3)} for w in (s.words or [])],
        })
    text = " ".join(s["text"] for s in segs)
    save_json(cond_dir(a.lecture, DEFAULT_CONDITION) / "asr.json", {
        "meta": {"stage": "asr", "model": c["model"], "language": info.language,
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "text": text, "segments": segs,
    })


def cmd_transcript(a):
    """Register an external transcript (human reference, error-injected, ...) as a condition."""
    text = Path(a.file).read_text(encoding="utf-8").strip()
    save_json(cond_dir(a.lecture, a.condition) / "asr.json", {
        "meta": {"stage": "imported_transcript", "source_file": a.file,
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "text": text,
    })


def cmd_summarise(a):
    d = cond_dir(a.lecture, a.condition)
    text = load_json(d / "asr.json")["text"]
    chunks = []
    for i, ch in enumerate(chunk_words(text, CFG["llm"]["chunk_words"])):
        print(f"summarising chunk {i + 1}")
        notes = call_llm(CFG["prompts"]["summarise"].format(text=ch))
        chunks.append({"index": i, "source_text": ch, "notes": notes})
    save_json(d / "summary_en.json", {
        "meta": {"stage": "summarise", "model": CFG["llm"]["model"],
                 "temperature": CFG["llm"].get("temperature"),
                 "prompt_hash": prompt_hash("summarise"),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "chunks": chunks,
        "notes": "\n\n".join(c["notes"] for c in chunks),
    })


def cmd_translate(a):
    d = cond_dir(a.lecture, a.condition)
    summ = load_json(d / "summary_en.json")
    language = CFG["target_languages"][a.lang]
    chunks = []
    for c in summ["chunks"]:
        print(f"translating chunk {c['index'] + 1} -> {language}")
        tr = call_llm(CFG["prompts"]["translate"].format(language=language, text=c["notes"]))
        chunks.append({"index": c["index"], "english_notes": c["notes"], "translation": tr})
    save_json(d / f"translation_{a.lang}.json", {
        "meta": {"stage": "translate", "language": language, "model": CFG["llm"]["model"],
                 "temperature": CFG["llm"].get("temperature"),
                 "prompt_hash": prompt_hash("translate"),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "chunks": chunks,
        "notes": "\n\n".join(c["translation"] for c in chunks),
    })


def cmd_run_all(a):
    if not (cond_dir(a.lecture, a.condition) / "asr.json").exists():
        cmd_asr(a)
    cmd_summarise(a)
    for lang in a.lang:
        cmd_translate(argparse.Namespace(lecture=a.lecture, condition=a.condition, lang=lang))


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **flags):
        sp = sub.add_parser(name)
        sp.add_argument("--lecture", required=True)
        if flags.get("cond", True):
            sp.add_argument("--condition", default=DEFAULT_CONDITION)
        sp.set_defaults(fn=fn)
        return sp

    add("download", cmd_download, cond=False).add_argument("--url", required=True)
    add("asr", cmd_asr)
    t = add("transcript", cmd_transcript)
    t.add_argument("--file", required=True)
    add("summarise", cmd_summarise)
    add("translate", cmd_translate).add_argument("--lang", required=True, choices=CFG["target_languages"])
    add("run-all", cmd_run_all).add_argument("--lang", nargs="+", default=["ur"], choices=CFG["target_languages"])

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
