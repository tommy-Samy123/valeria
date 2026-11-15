#!/usr/bin/env python3
"""Download the image datasets listed in training/config/image_sources.json from Roboflow Universe.

Needs a free Roboflow account and API key (https://app.roboflow.com -> Settings -> API Keys):
  export ROBOFLOW_API_KEY=...        # never commit this
  pip install -r training/requirements-data.txt
  python3 training/scripts/download_images.py               # all enabled sources
  python3 training/scripts/download_images.py --only ankle_sprain

Raw exports land in training/data/raw/<source name>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_version(project, explicit: int | None):
    """Return the Version object to export: the explicit one, or the newest that exists."""
    if explicit:
        return project.version(explicit)
    for n in range(20, 0, -1):
        try:
            v = project.version(n)
            return v
        except Exception:
            continue
    raise RuntimeError("no published dataset version found (fork it and generate a version)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="source names to download")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY first (Roboflow -> Settings -> API Keys).")

    from roboflow import Roboflow  # imported late so --help works without the package

    rf = Roboflow(api_key=key)
    sources = json.loads((ROOT / "config" / "image_sources.json").read_text())["sources"]
    for src in sources:
        if args.only and src["name"] not in args.only:
            continue
        if not src.get("enabled") and not args.only:
            print(f"skip {src['name']}: disabled ({src.get('_why_disabled', '')[:80]}...)")
            continue
        dest = args.out / src["name"]
        if dest.exists() and any(dest.iterdir()):
            print(f"skip {src['name']}: already downloaded at {dest}")
            continue
        print(f"downloading {src['name']} ({src['license']}) from {src['url']}")
        project = rf.workspace(src["workspace"]).project(src["project"])
        version = latest_version(project, src.get("version"))
        version.download(src["format"], location=str(dest), overwrite=True)
        (dest / "SOURCE.json").write_text(json.dumps(src, indent=2))
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
