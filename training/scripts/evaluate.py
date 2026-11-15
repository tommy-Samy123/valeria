#!/usr/bin/env python3
"""Score a model (base or fine-tuned) on Valeria's test sets.

  # Baseline: the untouched base model with the same system prompt
  python training/scripts/evaluate.py --model unsloth/gemma-4-E2B-it \
      --data training/data/text/test.jsonl training/data/images/test.jsonl --out training/outputs/eval_base

  # Fine-tuned LoRA adapter (or a merged folder)
  python training/scripts/evaluate.py --model training/outputs/mix_r16/lora \
      --data training/data/text/test.jsonl training/data/images/test.jsonl --out training/outputs/eval_mix

Reports (report.json): valid-JSON rate, category accuracy (overall, per kind, 4 core classes on photos),
emergency recall and false-alarm rate, exact red-flag match, clarify-question rate on "unclear",
a confusion matrix (confusion.csv) and every prediction (predictions.jsonl).
`--score-only predictions.jsonl` recomputes the report without a GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CORE = ["abrasion", "laceration", "burn", "sprain"]


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def score(preds: list[dict]) -> tuple[dict, list[list]]:
    n = len(preds)
    valid = [p for p in preds if p["pred"] is not None]
    def cat(p):
        return (p["pred"] or {}).get("category")

    correct = [p for p in preds if cat(p) == p["label"]["category"]]
    by_kind = defaultdict(lambda: [0, 0])
    for p in preds:
        by_kind[p["kind"]][1] += 1
        by_kind[p["kind"]][0] += cat(p) == p["label"]["category"]

    emerg = [p for p in preds if p["label"]["emergency"]]
    non_emerg = [p for p in preds if not p["label"]["emergency"]]
    def says_emerg(p):
        return bool((p["pred"] or {}).get("emergency"))

    missed = [p for p in emerg if not says_emerg(p)]
    false_alarm = [p for p in non_emerg if says_emerg(p)]
    flags_exact = sum(set((p["pred"] or {}).get("red_flags") or []) == set(p["label"]["red_flags"]) for p in emerg)
    unclear = [p for p in preds if p["label"]["category"] == "unclear"]
    asks = sum(bool((p["pred"] or {}).get("clarify")) for p in unclear)
    core_img = [p for p in preds if p["kind"] == "image" and p["label"]["category"] in CORE]

    labels = sorted({p["label"]["category"] for p in preds} | {str(cat(p)) for p in preds})
    idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for p in preds:
        matrix[idx[p["label"]["category"]]][idx[str(cat(p))]] += 1

    report = {
        "examples": n,
        "valid_json_rate": round(len(valid) / n, 4) if n else None,
        "category_accuracy": round(len(correct) / n, 4) if n else None,
        "category_accuracy_by_kind": {k: round(c / t, 4) for k, (c, t) in sorted(by_kind.items())},
        "core4_photo_accuracy": round(sum(cat(p) == p["label"]["category"] for p in core_img) / len(core_img), 4) if core_img else None,
        "emergency_recall": round(1 - len(missed) / len(emerg), 4) if emerg else None,
        "emergency_missed": len(missed),
        "emergency_false_alarm_rate": round(len(false_alarm) / len(non_emerg), 4) if non_emerg else None,
        "red_flags_exact_match_on_emergencies": round(flags_exact / len(emerg), 4) if emerg else None,
        "unclear_asks_question_rate": round(asks / len(unclear), 4) if unclear else None,
        "mean_latency_s": round(sum(p.get("latency_s", 0) for p in preds) / n, 3) if n else None,
        "missed_emergency_examples": [p["user"] for p in missed[:25]],
    }
    true_labels = {p["label"]["category"] for p in preds}
    rows = [[labels[i]] + row for i, row in enumerate(matrix) if labels[i] in true_labels]
    return report, [["true \\ predicted"] + labels] + rows


def user_text(ex: dict) -> str:
    parts = []
    for c in ex["messages"][1]["content"]:
        parts.append(c["text"] if c["type"] == "text" else f"<image:{c['image']}>")
    return " ".join(parts)


def run_model(args, rows: list[dict]) -> list[dict]:
    from unsloth import FastVisionModel, get_chat_template
    import torch
    from PIL import Image

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=args.load_in_4bit)
    processor = get_chat_template(processor, args.chat_template)
    FastVisionModel.for_inference(model)
    tok = getattr(processor, "tokenizer", processor)

    preds = []
    for i, ex in enumerate(rows):
        msgs = ex["messages"][:-1]
        images = []
        for m in msgs:
            for c in m["content"]:
                if c["type"] == "image":
                    p = Path(c["image"])
                    images.append(Image.open(p if p.is_absolute() else DATA_ROOT / p).convert("RGB"))
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kwargs["images"] = [images]
        enc = processor(**kwargs).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append({
            "id": ex.get("id"), "kind": ex.get("kind"), "user": user_text(ex), "label": ex["label"],
            "raw": raw, "pred": parse_json(raw), "latency_s": round(time.time() - t0, 3),
        })
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF id, LoRA adapter folder, or merged model folder")
    ap.add_argument("--data", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples per file")
    ap.add_argument("--chat-template", default="gemma-4")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--score-only", type=Path, help="re-score an existing predictions.jsonl")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        preds = [json.loads(l) for l in open(args.score_only)]
    else:
        rows = []
        for p in args.data:
            file_rows = [json.loads(l) for l in open(p) if l.strip()]
            rows += file_rows[: args.limit] if args.limit else file_rows
        preds = run_model(args, rows)
        with open(args.out / "predictions.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

    report, matrix = score(preds)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with open(args.out / "confusion.csv", "w") as f:
        f.write("\n".join(",".join(map(str, r)) for r in matrix) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "missed_emergency_examples"}, indent=2))


if __name__ == "__main__":
    main()
