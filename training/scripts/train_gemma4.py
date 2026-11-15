#!/usr/bin/env python3
"""Fine-tune Gemma 4 E2B-it into Valeria's triage model with LoRA (Unsloth).

Runs on one GPU with >= 15 GB (free Colab/Kaggle T4 works with --load-in-4bit; L4/A100 is faster).

  # Run 1: text only, vision encoder frozen
  python training/scripts/train_gemma4.py --train training/data/text/train.jsonl \
      --val training/data/text/val.jsonl --out training/outputs/text_r16

  # Run 2: text + photos
  python training/scripts/train_gemma4.py \
      --train training/data/text/train.jsonl training/data/images/train.jsonl \
      --val training/data/text/val.jsonl training/data/images/val.jsonl --out training/outputs/mix_r16 --merge

Loss is computed only on the assistant's JSON answer (system + user tokens and image tokens are masked).
Examples are packed into micro-batches of one modality (all-text or all-image) so text and photo
examples never need padding together.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"  # image paths in the jsonl files are relative to this folder


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def has_image(ex: dict) -> bool:
    return any(c["type"] == "image" for m in ex["messages"] for c in m["content"])


def make_packs(rows: list[dict], micro_batch: int, seed: int) -> list[list[dict]]:
    """Group examples into single-modality packs, then shuffle the packs."""
    rng = random.Random(seed)
    text = [r for r in rows if not has_image(r)]
    imgs = [r for r in rows if has_image(r)]
    rng.shuffle(text)
    rng.shuffle(imgs)
    packs = [text[i:i + micro_batch] for i in range(0, len(text), micro_batch)]
    packs += [imgs[i:i + micro_batch] for i in range(0, len(imgs), micro_batch)]
    rng.shuffle(packs)
    return packs


class PackDataset:
    def __init__(self, packs):
        self.packs = packs

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        return self.packs[i]


class Collator:
    """Tokenizes one pack; labels are -100 everywhere except the assistant answer."""

    def __init__(self, processor, max_len: int):
        self.processor = processor
        self.max_len = max_len
        tok = getattr(processor, "tokenizer", processor)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        tok.padding_side = "right"

    @staticmethod
    def _images(messages):
        from PIL import Image

        out = []
        for m in messages:
            for c in m["content"]:
                if c["type"] == "image":
                    p = c["image"]
                    path = Path(p) if Path(p).is_absolute() else DATA_ROOT / p
                    out.append(Image.open(path).convert("RGB"))
        return out

    def _encode(self, text: str, images):
        kwargs = {"text": [text], "return_tensors": "pt"}
        if images:
            kwargs["images"] = [images]
        return self.processor(**kwargs)

    def __call__(self, batch):
        import torch

        pack = batch[0]  # Trainer batch size is 1; the pack is the real micro-batch
        encs, labels = [], []
        for ex in pack:
            msgs = ex["messages"]
            images = self._images(msgs)
            full = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt = self.processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            enc = self._encode(full, images)
            n_prompt = self._encode(prompt, images)["input_ids"].shape[1]
            lab = enc["input_ids"].clone()
            lab[:, :n_prompt] = -100
            if enc["input_ids"].shape[1] > self.max_len:
                raise ValueError(f"example {ex.get('id')} is {enc['input_ids'].shape[1]} tokens > --max-len {self.max_len}")
            encs.append(enc)
            labels.append(lab)

        # Right-pad every sequence-shaped tensor to the longest in the pack.
        longest = max(e["input_ids"].shape[1] for e in encs)
        out: dict = {}
        seq_keys = [k for k, v in encs[0].items() if hasattr(v, "shape") and v.dim() == 2 and v.shape[1] == encs[0]["input_ids"].shape[1]]
        for k in encs[0].keys():
            vals = [e[k] for e in encs]
            if k in seq_keys:
                pad_val = self.pad_id if k == "input_ids" else 0
                vals = [torch.nn.functional.pad(v, (0, longest - v.shape[1]), value=pad_val) for v in vals]
            out[k] = torch.cat(vals, dim=0)
        out["labels"] = torch.cat(
            [torch.nn.functional.pad(l, (0, longest - l.shape[1]), value=-100) for l in labels], dim=0
        )
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--val", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--chat-template", default="gemma-4", help="non-thinking template recommended for E2B")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--finetune-vision", action="store_true", help="also train vision layers (try in a later run)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs, e.g. 30 for a smoke test")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--val-limit", type=int, default=300, help="val examples used for eval loss")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--merge", action="store_true", help="also save a merged 16-bit model for export")
    args = ap.parse_args()

    from unsloth import FastVisionModel, get_chat_template  # import first so Unsloth can patch transformers
    import torch
    from transformers import Trainer, TrainingArguments

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_len,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        target_modules="all-linear",
    )
    processor = get_chat_template(processor, args.chat_template)
    FastVisionModel.for_training(model)

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    random.Random(args.seed).shuffle(val_rows)
    val_rows = val_rows[: args.val_limit]
    print(f"train examples: {len(train_rows)} ({sum(map(has_image, train_rows))} with images); val: {len(val_rows)}")

    train_ds = PackDataset(make_packs(train_rows, args.micro_batch, args.seed))
    val_ds = PackDataset(make_packs(val_rows, args.micro_batch, args.seed)) if val_rows else None
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor, args.max_len),
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.001,
            max_grad_norm=0.3,
            optim="adamw_8bit",
            bf16=bf16,
            fp16=not bf16,
            logging_steps=5,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            prediction_loss_only=True,
            dataloader_num_workers=2,
            report_to="none",
            seed=args.seed,
        ),
    )
    stats = trainer.train()
    print(stats)

    lora_dir = args.out / "lora"
    model.save_pretrained(str(lora_dir))
    processor.save_pretrained(str(lora_dir))
    (args.out / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"saved LoRA adapter to {lora_dir}")
    if args.merge:
        merged = args.out / "merged"
        model.save_pretrained_merged(str(merged), processor, save_method="merged_16bit")
        print(f"saved merged 16-bit model to {merged}")


if __name__ == "__main__":
    main()
