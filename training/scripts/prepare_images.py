#!/usr/bin/env python3
"""Turn raw Roboflow exports into Valeria's image training set.

Steps:
  1. Read every image and its class from training/data/raw/<source>/ ("folder" or "coco" exports).
  2. Map source classes to Valeria categories (training/config/image_sources.json); drop unmapped ones.
  3. Group near-duplicates (perceptual hash, including mirrored copies). Roboflow exports often contain
     augmented copies of one photo; a whole group always goes to the same split so test images are unseen.
  4. Split groups 80/10/10, resize to <= 768 px, and write
       training/data/images/<split>/<category>/<file>.jpg
       training/data/images/<split>.jsonl   (chat-format examples, same JSON target as the text set)
       training/data/images/ATTRIBUTION.md  (required by CC BY 4.0)
       training/data/images/stats.json

Usage:
  python3 training/scripts/prepare_images.py
  python3 training/scripts/prepare_images.py --max-per-class 600   # cap classes to limit imbalance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
import imagehash

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (ROOT / "config" / "system_prompt.txt").read_text().strip()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DUP_DISTANCE = 6  # phash Hamming distance at or below which two images count as the same photo

PROMPTS = [
    "What injury is this?",
    "Valeria, look at this. What happened?",
    "Can you tell what this is?",
    "Look at my injury.",
    "What kind of injury is this?",
    "Is this bad?",
    "valeria check this",
    "",
]
# Optional spoken hint that agrees with the photo (teaches the model to combine image + words).
HINTS = {
    "abrasion": ["I fell on the sidewalk", "I fell off my bike", "I slid on the ground"],
    "laceration": ["I cut it on some glass", "the knife slipped", "I cut it on something sharp"],
    "burn": ["I touched the stove", "hot water spilled on it", "I grabbed a hot pan"],
    "sprain": ["I twisted it playing soccer", "I rolled it on the stairs", "I landed on it wrong"],
    "bruise": ["I bumped it on the table", "I got hit with a ball", "I banged it on the door"],
    "none": ["does this look okay", "is something wrong here", "it feels a little sore"],
}
CLARIFY = {
    "none": "I don't see an injury in this photo. What happened, and where does it hurt?",
    "sprain": "Did you twist it, and can you put weight on it or move it?",
}


def norm(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def map_class(name: str, class_map: dict) -> str | None:
    if "*" in class_map:
        return class_map["*"]
    lookup = {norm(k): v for k, v in class_map.items()}
    return lookup.get(norm(name))


def collect(src_dir: Path, cfg: dict) -> list[tuple[Path, str]]:
    """Return (image path, valeria category) pairs for one raw source."""
    items: list[tuple[Path, str]] = []
    if cfg["format"] == "folder":
        for split_dir in [d for d in src_dir.iterdir() if d.is_dir()]:
            for class_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
                label = map_class(class_dir.name, cfg["class_map"])
                if label is None:
                    continue
                items += [(p, label) for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    elif cfg["format"] == "coco":
        for ann_file in src_dir.rglob("_annotations.coco.json"):
            coco = json.loads(ann_file.read_text())
            cats = {c["id"]: c["name"] for c in coco["categories"]}
            labels_per_image: dict[int, set[str]] = defaultdict(set)
            for a in coco["annotations"]:
                lab = map_class(cats[a["category_id"]], cfg["class_map"])
                if lab:
                    labels_per_image[a["image_id"]].add(lab)
            for img in coco["images"]:
                labs = labels_per_image.get(img["id"], set())
                if len(labs) == 1:  # skip unlabeled images and images showing several injury types
                    items.append((ann_file.parent / img["file_name"], next(iter(labs))))
    else:
        raise ValueError(f"unsupported format {cfg['format']}")
    return items


def phashes(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return int(str(imagehash.phash(im)), 16), int(str(imagehash.phash(ImageOps.mirror(im))), 16)
    except Exception as e:  # corrupt file
        print(f"  unreadable {path}: {e}")
        return None


def group_duplicates(hashes: list[tuple[int, int]]) -> list[int]:
    """Union-find over near-duplicate pairs; returns a group id per item."""
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        hi, hi_m = hashes[i]
        for j in range(i + 1, len(hashes)):
            hj = hashes[j][0]
            if (hi ^ hj).bit_count() <= DUP_DISTANCE or (hi_m ^ hj).bit_count() <= DUP_DISTANCE:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(hashes))]


def split_for(group_key: str) -> str:
    h = int(hashlib.sha1(group_key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def make_example(rel_path: str, label: str, rng: random.Random) -> dict:
    prompt = rng.choice(PROMPTS)
    if rng.random() < 0.3:
        prompt = (rng.choice(HINTS[label]) + ". " + prompt).strip()
    target = {
        "category": label,
        "also": [],
        "emergency": False,
        "red_flags": [],
        "body_part": None,
        "clarify": CLARIFY.get(label),
    }
    user = [{"type": "image", "image": rel_path}]
    if prompt:
        user.append({"type": "text", "text": prompt})
    return {
        "kind": "image",
        "label": target,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, separators=(",", ":"))}]},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "images")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--max-per-class", type=int, default=0, help="cap training groups per class (0 = no cap)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sources = {s["name"]: s for s in json.loads((ROOT / "config" / "image_sources.json").read_text())["sources"]}
    records: list[dict] = []
    used_sources = []
    for src_dir in sorted(d for d in args.raw.iterdir() if d.is_dir()) if args.raw.exists() else []:
        cfg = sources.get(src_dir.name)
        if not cfg:
            print(f"skip {src_dir.name}: not in image_sources.json")
            continue
        items = collect(src_dir, cfg)
        print(f"{src_dir.name}: {len(items)} labeled images {dict(Counter(l for _, l in items))}")
        used_sources.append(cfg)
        records += [{"path": p, "label": l, "source": src_dir.name} for p, l in items]
    if not records:
        raise SystemExit(f"No images found under {args.raw}. Run download_images.py first.")

    print("hashing images ...")
    keep, hashes = [], []
    for r in records:
        h = phashes(r["path"])
        if h:
            keep.append(r)
            hashes.append(h)
    print("grouping near-duplicates ...")
    groups = group_duplicates(hashes)

    # One group = one real photo; drop exact repeats inside a group that share the same label.
    by_group: dict[int, list[dict]] = defaultdict(list)
    for r, g in zip(keep, groups):
        by_group[g].append(r)
    conflicts = sum(1 for rs in by_group.values() if len({r["label"] for r in rs}) > 1)

    if args.out.exists():
        shutil.rmtree(args.out)
    split_rows: dict[str, list[dict]] = defaultdict(list)
    per_class_groups: Counter = Counter()
    stats: Counter = Counter()
    for gid, rs in sorted(by_group.items()):
        labels = {r["label"] for r in rs}
        if len(labels) > 1:
            continue  # the same photo labeled differently by different sources: unreliable, skip
        label = labels.pop()
        rep = rs[0]
        split = split_for(f"{rep['source']}/{rep['path'].name}")
        if split == "train" and args.max_per_class and per_class_groups[label] >= args.max_per_class:
            continue
        per_class_groups[label] += split == "train"
        # Train keeps augmented copies (up to 3 per photo); val/test keep one image per photo.
        chosen = rs[:3] if split == "train" else rs[:1]
        for k, r in enumerate(chosen):
            dest = args.out / split / label / f"{r['source']}_{gid}_{k}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(r["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                im.save(dest, "JPEG", quality=90)
            rel = dest.relative_to(args.out.parent).as_posix()  # e.g. images/train/burn/x.jpg
            ex = make_example(rel, label, rng)
            ex["id"] = f"img-{split}-{len(split_rows[split]):05d}"
            ex["source"] = r["source"]
            split_rows[split].append(ex)
            stats[f"{split}/{label}"] += 1

    for split, rows in split_rows.items():
        rng.shuffle(rows)
        with open(args.out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    lines = ["# Image dataset attribution", "", "Images were adapted (resized, relabeled) from these datasets:", ""]
    for s in used_sources:
        lines.append(f"- **{s['project']}** by {s['author']}, {s['url']}, licensed {s['license']}.")
    (args.out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")
    summary = {
        "images_read": len(keep),
        "photo_groups": len(by_group),
        "groups_with_conflicting_labels_skipped": conflicts,
        "written": dict(sorted(stats.items())),
    }
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
