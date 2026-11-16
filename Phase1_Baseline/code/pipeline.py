import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"

CFG = yaml.safe_load(
    CONFIG_PATH.read_text(encoding="utf-8")
)

DEFAULT_CONDITION = "asr_natural"

_SUMMARISER = None
_TRANSLATOR = None

def lecture_dir(lecture):

    directory = ROOT / "data" / lecture
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def cond_dir(lecture, condition):

    directory = lecture_dir(lecture) / condition
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def save_json(path, obj):

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {path.relative_to(ROOT)}")

def load_json(path):

    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(ROOT)}. "
            "Run the previous pipeline stage first."
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )

def stable_hash(value):

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:10]

def prompt_hash(name):

    prompt = CFG["prompts"][name]

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()[:10]

def chunk_words(text, words_per_chunk):

    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be greater than zero.")

    words = text.split()

    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ] or [""]

def choose_device(setting):

    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if setting == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but PyTorch cannot access a GPU. "
            "Change the device setting to 'auto' or 'cpu' "
            "in config.yaml."
        )

    return setting

def get_summariser():

    global _SUMMARISER

    if _SUMMARISER is None:

        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

        settings = CFG["summarisation"]

        device = choose_device(
            settings["device"]
        )

        model_name = settings["model"]

        print(
            f"\nLoading summarisation model: "
            f"{model_name} on {device}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        model.to(device)
        model.eval()

        _SUMMARISER = (
            tokenizer,
            model,
            device,
        )

    return _SUMMARISER

def get_translator():

    global _TRANSLATOR

    if _TRANSLATOR is None:

        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

        settings = CFG["translation"]

        device = choose_device(
            settings["device"]
        )

        model_name = settings["model"]

        print(
            f"\nLoading translation model: "
            f"{model_name} on {device}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        model.to(device)
        model.eval()

        _TRANSLATOR = (
            tokenizer,
            model,
            device,
        )

    return _TRANSLATOR

def summarise_chunk(text):

    import torch

    settings = CFG["summarisation"]

    tokenizer, model, device = get_summariser()

    prompt = CFG["prompts"]["summarise"].format(
        text=text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings["max_input_tokens"],
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=settings["max_new_tokens"],
            do_sample=False,
        )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result.strip()

def translate_chunk(text, target_code):

    import torch

    settings = CFG["translation"]

    tokenizer, model, device = get_translator()

    tokenizer.src_lang = settings[
        "source_language_code"
    ]

    target_token_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

    if (
        target_token_id is None
        or target_token_id == tokenizer.unk_token_id
    ):
        raise SystemExit(
            f"Language code '{target_code}' is not "
            "supported by the loaded NLLB tokenizer. "
            "Check the model's supported languages."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=settings["max_input_tokens"],
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=settings["max_new_tokens"],
            do_sample=False,
        )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result.strip()

def cmd_download(args):

    output = lecture_dir(args.lecture) / "audio.wav"

    output_template = str(
        output.with_suffix(".%(ext)s")
    )

    print("\nDownloading lecture audio...")

    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            output_template,
            args.url,
        ],
        check=True,
    )

    save_json(
        lecture_dir(args.lecture) / "source.json",
        {
            "url": args.url,
            "audio": output.name,
        },
    )

    print("\nAudio download completed.")

def cmd_asr(args):

    from faster_whisper import WhisperModel

    audio = lecture_dir(args.lecture) / "audio.wav"

    if not audio.exists():
        raise SystemExit(
            "audio.wav not found. "
            "Run the download command first or place "
            "the audio file in the lecture directory."
        )

    settings = CFG["asr"]

    print(
        f"\nLoading ASR model: {settings['model']}"
    )

    model = WhisperModel(
        settings["model"],
        device=settings["device"],
        compute_type=settings["compute_type"],
    )

    print("\nTranscribing lecture...")

    segments, info = model.transcribe(
        str(audio),
        language=settings["language"],
        beam_size=settings["beam_size"],
        word_timestamps=True,
        vad_filter=True,
    )

    segment_records = []

    for segment in segments:

        segment_records.append({

            "id": segment.id,

            "start": round(
                segment.start,
                2,
            ),

            "end": round(
                segment.end,
                2,
            ),

            "text": segment.text.strip(),

            "words": [

                {
                    "w": word.word,

                    "start": round(
                        word.start,
                        2,
                    ),

                    "end": round(
                        word.end,
                        2,
                    ),

                    "p": round(
                        word.probability,
                        3,
                    ),
                }

                for word in (segment.words or [])
            ],
        })

    transcript = " ".join(
        item["text"]
        for item in segment_records
    )

    save_json(
        cond_dir(
            args.lecture,
            args.condition,
        ) / "asr.json",

        {
            "meta": {

                "stage": "asr",

                "model": settings["model"],

                "language": info.language,

                "condition": args.condition,

                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },

            "text": transcript,

            "segments": segment_records,
        },
    )

    print("\nASR completed.")

def cmd_transcript(args):

    source_file = Path(args.file)

    if not source_file.exists():
        raise SystemExit(
            f"Transcript file not found: {source_file}"
        )

    transcript = source_file.read_text(
        encoding="utf-8"
    ).strip()

    save_json(
        cond_dir(
            args.lecture,
            args.condition,
        ) / "asr.json",

        {
            "meta": {

                "stage": "imported_transcript",

                "source_file": str(source_file),

                "condition": args.condition,

                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },

            "text": transcript,
        },
    )

    print(
        f"\nTranscript registered under "
        f"condition: {args.condition}"
    )

def cmd_summarise(args):

    directory = cond_dir(
        args.lecture,
        args.condition,
    )

    asr_data = load_json(
        directory / "asr.json"
    )

    transcript = asr_data["text"]

    settings = CFG["summarisation"]

    chunks = []

    transcript_chunks = chunk_words(
        transcript,
        settings["chunk_words"],
    )

    for index, source_chunk in enumerate(
        transcript_chunks
    ):

        print(
            f"\nSummarising chunk "
            f"{index + 1}/{len(transcript_chunks)}"
        )

        notes = summarise_chunk(
            source_chunk
        )

        chunks.append({

            "index": index,

            "source_text": source_chunk,

            "notes": notes,
        })

    save_json(
        directory / "summary_en.json",

        {
            "meta": {

                "stage": "summarise",

                "model": settings["model"],

                "device": choose_device(
                    settings["device"]
                ),

                "max_input_tokens": settings[
                    "max_input_tokens"
                ],

                "max_new_tokens": settings[
                    "max_new_tokens"
                ],

                "chunk_words": settings[
                    "chunk_words"
                ],

                "prompt_hash": prompt_hash(
                    "summarise"
                ),

                "condition": args.condition,

                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },

            "chunks": chunks,

            "notes": "\n\n".join(
                item["notes"]
                for item in chunks
            ),
        },
    )

    print("\nSummarisation completed.")

def cmd_translate(args):

    directory = cond_dir(
        args.lecture,
        args.condition,
    )

    summary = load_json(
        directory / "summary_en.json"
    )

    language_settings = CFG[
        "target_languages"
    ][args.lang]

    translation_settings = CFG[
        "translation"
    ]

    chunks = []

    for item in summary["chunks"]:

        print(
            f"\nTranslating chunk "
            f"{item['index'] + 1} -> "
            f"{language_settings['name']}"
        )

        translated = translate_chunk(
            item["notes"],
            language_settings["code"],
        )

        chunks.append({

            "index": item["index"],

            "english_notes": item["notes"],

            "translation": translated,
        })

    translation_config_hash = stable_hash({

        "model": translation_settings["model"],

        "source_language_code":
            translation_settings["source_language_code"],

        "target_language_code":
            language_settings["code"],

        "max_input_tokens":
            translation_settings["max_input_tokens"],

        "max_new_tokens":
            translation_settings["max_new_tokens"],
    })

    save_json(
        directory / f"translation_{args.lang}.json",

        {
            "meta": {

                "stage": "translate",

                "language": language_settings["name"],

                "language_code": language_settings["code"],

                "model": translation_settings["model"],

                "device": choose_device(
                    translation_settings["device"]
                ),

                "config_hash": translation_config_hash,

                "condition": args.condition,

                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },

            "chunks": chunks,

            "notes": "\n\n".join(
                item["translation"]
                for item in chunks
            ),
        },
    )

    print("\nTranslation completed.")

def cmd_run_all(args):

    condition_transcript = (
        cond_dir(
            args.lecture,
            args.condition,
        ) / "asr.json"
    )

    if (
        args.condition == DEFAULT_CONDITION
        and not condition_transcript.exists()
    ):

        cmd_asr(args)

    elif not condition_transcript.exists():

        raise SystemExit(
            f"No transcript found for condition "
            f"'{args.condition}'.\n"
            "Import a transcript first using the "
            "`transcript` command."
        )

    cmd_summarise(args)

    for language_code in args.lang:

        cmd_translate(
            argparse.Namespace(
                lecture=args.lecture,
                condition=args.condition,
                lang=language_code,
            )
        )

    print("\nComplete pipeline execution finished.")

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    def add_stage(
        name,
        function,
        include_condition=True,
    ):

        stage_parser = subparsers.add_parser(name)

        stage_parser.add_argument(
            "--lecture",
            required=True,
        )

        if include_condition:

            stage_parser.add_argument(
                "--condition",
                default=DEFAULT_CONDITION,
            )

        stage_parser.set_defaults(
            function=function
        )

        return stage_parser

    download_parser = add_stage(
        "download",
        cmd_download,
        include_condition=False,
    )

    download_parser.add_argument(
        "--url",
        required=True,
    )

    add_stage(
        "asr",
        cmd_asr,
    )

    transcript_parser = add_stage(
        "transcript",
        cmd_transcript,
    )

    transcript_parser.add_argument(
        "--file",
        required=True,
    )

    add_stage(
        "summarise",
        cmd_summarise,
    )

    translate_parser = add_stage(
        "translate",
        cmd_translate,
    )

    translate_parser.add_argument(
        "--lang",
        required=True,
        choices=CFG["target_languages"],
    )

    run_all_parser = add_stage(
        "run-all",
        cmd_run_all,
    )

    run_all_parser.add_argument(
        "--lang",
        nargs="+",
        default=["ur"],
        choices=CFG["target_languages"],
    )

    args = parser.parse_args()

    args.function(args)

if __name__ == "__main__":
    main()
