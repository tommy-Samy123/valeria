# Valeria model training

Fine-tunes **Gemma 4 E2B-it** (Apache-2.0, text + image input) into Valeria's *triage* model. The model reads
what the user says, and optionally a photo, and returns one JSON object:

```json
{"category":"burn","also":[],"emergency":false,"red_flags":[],"body_part":"hand","clarify":null}
```

The model never writes treatment steps. The app turns `category` into verified Red Cross/AHA steps, and the
rule-based red-flag check runs before and after the model (see `docs/LLM_FINETUNE_PLAN.md` §3).

## Layout
| Path | What it is |
|---|---|
| `config/labels.json` | Categories, red flags, output schema |
| `config/system_prompt.txt` | System prompt used in training **and** in the app (keep them identical) |
| `config/image_sources.json` | Photo datasets: URL, author, license, class mapping |
| `scripts/build_text_dataset.py` | Generates ~5,000 spoken-style text examples (no downloads needed) |
| `scripts/download_images.py` | Downloads the photo datasets from Roboflow (needs a free API key) |
| `scripts/prepare_images.py` | Relabels, de-duplicates, splits, and resizes photos; writes the attribution file |
| `scripts/train_gemma4.py` | LoRA fine-tuning with Unsloth (GPU) |
| `scripts/evaluate.py` | Scores the base or fine-tuned model; writes report, confusion matrix, predictions |
| `notebooks/valeria_finetune_colab.ipynb` | Runs everything above on Colab |
| `EXPORT.md` | Converting the merged model to `.litertlm` for Android |

Generated data (`training/data/`) and model outputs (`training/outputs/`) are git-ignored. Regenerate them with the scripts.

## Datasets
**Text (synthetic, generated here):** 3,740 train / 644 val / 635 test examples. They cover the 7 injury types,
unclear descriptions (the model asks a question back), red-flag add-ons ("...and it won't stop bleeding"),
stand-alone emergencies, out-of-scope requests, and two injuries in one sentence. Simulated speech-recognition errors
("sprained my angle") are included. Every phrasing template belongs to exactly one split, so val/test wording is never
seen during training.

**Photos (Roboflow Universe, all CC BY 4.0, checked 2026-11-14):**
| Dataset | Images | Classes used |
|---|---|---|
| [basic-wound-classify](https://universe.roboflow.com/clinicvision/basic-wound-classify-mpoys) (clinicvision) | ~1,515 | abrasion, bruise, burn, laceration, normal skin → none |
| [Ankle Sprain](https://universe.roboflow.com/yolo-ujzf8/ankle-sprain-wikyk) (yolo-ujzf8) | ~682 | sprain |
| [Classify wounds](https://universe.roboflow.com/clinicvision/classify-wounds-pgnkj) (clinicvision) | ~131 | abrasion, bruise, burn, laceration |
| [Master-Wound-Classifier](https://universe.roboflow.com/tan-keqin-a23cs0184/master-wound-classifier) (optional) | ~2,458 | adds more of the same classes. **Fork it first** (see `image_sources.json`) |

Cautions:
- These are community uploads. Look through some images per class before training, and delete anything mislabeled or inappropriate.
- Roboflow exports often contain augmented copies. `prepare_images.py` groups near-duplicates so copies of one photo never land in both train and test.
- The sprain photos come from one source, so the model may learn that source's style instead of swelling. Add your own consented photos if you can.
- **The judged science-fair test sets (60 text scenarios, 20 emergencies, 60 photos) must be built separately and never added to training.**

## Run it
Local data prep (Mac):
```bash
python3 -m venv .venv-data && source .venv-data/bin/activate
pip install -r training/requirements-data.txt
python3 training/scripts/build_text_dataset.py
export ROBOFLOW_API_KEY=...            # from app.roboflow.com → Settings → API Keys
python3 training/scripts/download_images.py
python3 training/scripts/prepare_images.py --max-per-class 600
```

Training (GPU): open `training/notebooks/valeria_finetune_colab.ipynb` in Colab and run the cells, or run the
commands in the docstrings of `train_gemma4.py` and `evaluate.py`.

## Suggested order
1. **Smoke test** (`--max-steps 10`) to check that everything loads.
2. **Baseline:** evaluate the base model. This is the "before" number for the science-fair comparison.
3. **Run 1:** text only, vision frozen. Evaluate.
4. **Run 2:** text + photos. Evaluate. Check that text accuracy didn't drop.
5. Optional **run 3:** `--finetune-vision` if photo accuracy is below 85%.
6. Pick the run with the best **emergency recall**, then the best category accuracy. Export it (see `EXPORT.md`).

Unsloth notes that a training loss of 13–15 can be normal for Gemma 4 E2B. Judge runs by `evaluate.py`, not by loss.
