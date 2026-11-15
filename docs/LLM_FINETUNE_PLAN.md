# Valeria — On-Device LLM Fine-Tuning Plan

_Drafted 2026-11-14. Inputs: this repo (`main` @ 6e91413) and the "Valeria Research Plan" Google Doc (F33-Research-Plan tab, dated Nov 9, 2026)._

---

## 1. Where the project is today

| Area | Research plan says | Code does |
|---|---|---|
| Understanding | Fine-tuned, quantized Gemma-family model | Keyword `contains()` matching in `FirstAidResponse.kt` |
| Safety | Separate rule-based red-flag module, 20/20 emergencies caught | None. Emergencies are only mentioned inside response text |
| Knowledge | Local verified protocol files (Red Cross / AHA) | Hard-coded strings |
| Image input | Classify abrasion / laceration / burn / sprain at ≥85% | No camera code |
| Wake word | Picovoice Porcupine | Tap-to-talk only |
| Coaching | One step at a time, "repeat" / "next" | Whole response spoken at once |
| Scope | 4 injury types | 6 types (adds choking, nosebleed, bruise); none are abrasions as their own type |

### Safety bugs to fix now, regardless of the LLM work
1. `sprainKeywords` includes `break`, `broke`, `broken`, `fracture`, so a suspected fracture gets RICE advice. A suspected fracture should go to "seek care" / red flag.
2. `can't breathe` is matched as choking. Trouble breathing is one of the red flags the plan lists, and it should go to "call 911" first.
3. `fell`, `fall`, `bike`, `accident` all go to bleeding. A head injury from a fall gets wound-care steps.
4. Plain substring matching misfires: `"hi"` matches inside "t**hi**s"/"ws**hi**", `"hit"` matches "w**hit**e", and `"cut"` matches "exe**cut**e".
5. Match order means "I burned my hand and it's bleeding" only ever gets the bleeding response.

### Gaps in the research plan to fix before judging
- **Memory limit:** the F33 tab says < 3 GB, the older draft says < 4 GB. The Gemma 4 E2B LiteRT-LM bundle alone is ~2.6 GB, so 3 GB *peak app memory* is very tight. Pick one number and define what's measured (PSS from Android Profiler, with the model memory-mapped).
- **Noise conditions:** criterion 5 says up to 60 dB, but test 3 uses 65 dB. Make them match.
- **Test set size:** 15 images per class is too few to report 85% with confidence (±9% at 95% CI). Aim for at least 40 per class, or report the confidence interval.
- **HIPAA:** "HIPAA standards are inherently preserved" is not accurate. HIPAA covers covered entities (providers, insurers) and their business associates, not a consumer app. Say instead that the design *follows HIPAA-style privacy principles: no data leaves the device and no audio or images are stored*.
- **Sprains from photos:** a sprain often looks normal in a photo. Expect this class to be the weakest, and plan for the model to answer "can't tell from the image, ask a question" (see §3).

---

## 2. Model choice

**Recommendation: Gemma 4 E2B-it.** It has open weights under **Apache-2.0**, takes text and image input (plus audio on the small models), and Google's LiteRT-LM supports it directly. This fits the research plan's "compact fine-tuned Gemma-family model" and LiteRT-LM bibliography entry as written.

| Candidate | License | Input | Why / why not |
|---|---|---|---|
| **Gemma 4 E2B-it** (primary) | Apache-2.0 | text, image, audio | Official LiteRT-LM path for Android; ~2.6 GB int4 bundle |
| Gemma 3n E2B-it (fallback) | Gemma Terms of Use | text, image, audio | Older, mature mobile tooling; use it if the Gemma 4 vision export gives trouble |
| Qwen3-VL-2B-Instruct | Apache-2.0 | text, image | Strong small vision-language model; Android path is llama.cpp/MLC, not LiteRT. More integration work |
| SmolVLM2 (500M / 2.2B) | Apache-2.0 | text, image | Very small and light on RAM; weaker reasoning |

**Risk to check in week 1:** some community LiteRT-LM conversions of Gemma 4 fine-tunes are **text-only** (the vision weights were removed). Before spending time on data, export the *base* E2B model with the official `litert-torch` tool and confirm image input works on the phone.

**Parallel baseline for criterion 4 (strongly recommended):** a small image classifier (MobileNetV3 or EfficientNet-Lite, fine-tuned and exported to LiteRT, ~5–15 MB). It will likely beat a 2B vision-language model on 4-way classification, use almost no RAM, and run in <100 ms. Comparing *LLM vision vs. dedicated classifier* is also a strong science-fair result.

---

## 3. What the LLM should (and should not) do

To reach 95% checklist agreement, **the model should not write first-aid steps freely.** Give it a narrow job:

```
Speech/text ─┐
             ├─► [Red-flag rules] ──hit──► 911 script (bypasses the LLM)
Image ───────┘          │ no hit
                        ▼
              [Fine-tuned Gemma 4 E2B]  → strict JSON
                        ▼
  { "category": "burn|laceration|abrasion|sprain|out_of_scope|unclear",
    "red_flags": ["large_area", ...],
    "details": {"location":"hand","cause":"hot water"},
    "clarify": "Is the burned area bigger than your palm?" | null,
    "confidence": 0.0-1.0 }
                        ▼
  [Red-flag rules again on the JSON] ──hit──► 911 script
                        ▼
  [Protocol card lookup]  (verified Red Cross/AHA steps, stored as JSON in assets/)
                        ▼
  Step-by-step TTS with "next / repeat / back"
```

- The protocol cards are the **only** source of treatment steps. The model fills in personalization slots at most (e.g., "Cool your **hand** under running water").
- If the JSON fails to parse, or the confidence is low, ask the clarify question or fall back to the current keyword engine. The app never goes silent.
- This *is* the "with vs. without retrieval/safety" comparison from the research plan: run the same 60 scenarios with free-form generation and compare.

---

## 4. Training data

Keep **train / validation / test strictly separate**. The 60 + 20 judged scenarios and the 60 judged images must never be used for training.

### 4a. Text (target ~3,000–5,000 examples)
- **Seed:** write ~40 hand-written scenarios per category, with varied phrasing, panic speech, and kid vocabulary ("I got an owie").
- **Expand:** paraphrase with a large LLM *offline*, then **review every label by hand**.
- **STT noise augmentation:** run part of the set through the actual Android recognizer, or simulate its errors (dropped words, homophones like "burn" → "bern"). The model sees STT output, not clean text.
- **Multi-injury and ambiguous cases** → `unclear` with a clarify question.
- **Out of scope** (headache, bee sting, "what's the weather") → `out_of_scope`.
- **Red-flag examples** (~20%): heavy bleeding that won't stop, spurting blood, burns on face/hands/genitals or larger than the palm, chemical/electrical burns, bone visible, deformed limb, numbness, head injury, trouble breathing, passing out. **Write these in different words from the 20 test emergencies.**

### 4b. Images (target ≥150 per class for training)
- Look for publicly licensed wound/burn image datasets (Kaggle, Roboflow Universe, Medetec). **Check each dataset's license**, and record the source in the bibliography.
- Abrasions and lacerations are easy to find. Burns are moderately easy. Sprains are hard (photos of swollen ankles/wrists), so supplement with consented photos of moulage (makeup) injuries.
- Add a `not_an_injury / unclear` class (normal skin, blurry photos, pets) so the model can say "I can't tell."
- Pair each image with a short instruction-style prompt and the same JSON target as the text data.
- Ask the SRC/teacher whether consented photos of people (even moulage) need the Human Participants form.

---

## 5. Fine-tuning recipe

**Hardware:** free Colab/Kaggle T4 (16 GB) works for QLoRA on E2B. Colab Pro L4/A100 is faster.
**Stack:** Hugging Face `transformers` + `peft` + `trl` (SFTTrainer), or **Unsloth**, which has ready-made Gemma vision fine-tuning notebooks and uses less memory.

| Setting | Start value |
|---|---|
| Method | QLoRA (4-bit base), LoRA on language attention + MLP |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Vision encoder | **Frozen** in run 1. In run 2, also unfreeze the vision→text projector |
| LR / epochs | 2e-4, 2–3 epochs, cosine schedule, warmup 3% |
| Batch | effective 16 (grad accumulation) |
| Max seq len | 1,024 (inputs are short) |
| Format | Gemma chat template. The system prompt says "Output only JSON matching this schema" |

**Run order:**
1. **Baseline:** score the *base* E2B model with a good prompt on the validation set. If it's already >90%, fine-tuning just needs to tighten the JSON format and the red-flag recall.
2. **Text-only LoRA:** train, then evaluate.
3. **Text + image LoRA:** train, then evaluate (check whether text accuracy dropped).
4. Pick the best checkpoint by **red-flag recall first**, then category accuracy.

**Export:** merge the LoRA into the base → `litert-torch` export (`export_hf`, base `google/gemma-4-E2B-it`) → int4 `.litertlm` → copy to the phone. **Re-run the validation set on the phone**, because quantization can change answers.

---

## 6. Android integration (code changes)

1. Add the LiteRT-LM Android dependency and a `llm/ValeriaModel.kt` wrapper (load once, keep warm, run on a background dispatcher, time each call).
2. Add `safety/RedFlagRules.kt`: regex/phrase lists with word boundaries, run *before* and *after* the model. Unit-test it with the 20 emergency prompts.
3. Add `firstaid/protocols/*.json` (one card per category, citing the Red Cross/AHA source), plus a step-by-step coaching state machine (`next`, `repeat`, `back`, `stop`).
4. Add CameraX capture. Images stay in memory only and are never written to disk.
5. Keep `FirstAidResponse` (with the bug fixes above) as the fallback when the model isn't loaded or the JSON fails.
6. Model delivery: a 2.6 GB file is too large for the APK. Sideload it to app storage for the prototype, or use Play Asset Delivery later.
7. Stretch: Gemma 4 E2B accepts audio. A later experiment could feed audio straight to the model and compare WER/latency with Android SpeechRecognizer.

---

## 7. Evaluation, mapped to the design criteria

| Criterion | How the model work is measured |
|---|---|
| 1. ≥95% checklist agreement | Mostly guaranteed by protocol cards; measure the *category routing* error that causes wrong checklists |
| 2. 20/20 emergencies | Rules + model; report recall for rules alone, model alone, and both combined |
| 3. ≤15 s wake→speech | Log time-to-first-token and total model time per request on the target phone |
| 4. ≥85% image accuracy | Confusion matrix on held-out images for (a) Gemma vision and (b) the small CNN |
| Memory constraint | Android Profiler peak PSS during an image + text request |
| Experimental comparison | Rules-only vs. base Gemma vs. fine-tuned Gemma vs. fine-tuned + retrieval + safety |

---

## 8. Suggested timeline (8 weeks)

| Week | Work |
|---|---|
| 1 | Fix the keyword-engine safety bugs; export **base** Gemma 4 E2B to LiteRT-LM; confirm text + image input and RAM on the phone |
| 2 | Write protocol cards + red-flag rules + unit tests; integrate the base model with the JSON prompt |
| 3 | Build the text dataset + baseline eval |
| 4 | Collect images (license log); train the small-CNN baseline |
| 5 | Text LoRA, then text + image LoRA |
| 6 | Export, on-device eval, latency/memory tuning |
| 7 | Run the full test plan (latency, WER, accuracy, emergencies, offline) |
| 8 | Analysis, graphs, write-up; fix failures and re-test |

---

## Sources
- Google AI Edge — LiteRT-LM: Gemma 4 — https://ai.google.dev/edge/litert-lm/models/gemma-4
- Qualcomm AI Hub — Gemma 4 E2B-it — https://aihub.qualcomm.com/iot/models/gemma_4_e2b_it
- Datature — Gemma 4 for CV engineers — https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know
- Example community LiteRT-LM conversion (text-only fine-tune) — https://huggingface.co/PeppX/gemma-4-e2b-uncensored-litertlm
- 2024 AHA/Red Cross First Aid Guidelines — https://doi.org/10.1161/CIR.0000000000001281
