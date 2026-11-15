# Exporting the fine-tuned model to the phone (LiteRT-LM)

Train with `--merge` so `training/outputs/<run>/merged/` holds a full 16-bit Hugging Face checkpoint.

```bash
uv tool install litert-torch-nightly

litert-torch export_hf \
  --model=training/outputs/mix_r16/merged \
  --output_dir=training/outputs/mix_r16/litert \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm

# Quick desktop check before copying to the phone
litert-lm run training/outputs/mix_r16/litert/model.litertlm \
  --prompt="i cut my finger on a can lid"
```

This is Google's published command for Gemma 4, taken from
https://developers.google.com/edge/litert-lm/models/gemma-4. Their example passes the hub id
`google/gemma-4-E2B-it` to `--model`.

## Check these before relying on the export
1. **Does `--model` accept a local folder?** If it doesn't, push the merged model to a *private* Hugging Face repo and pass that id.
2. **Is the vision encoder kept?** Some community Gemma 4 `.litertlm` conversions are text-only. Export the **base** model first, then send it a photo with `litert-lm` or on the phone. If images are dropped, either check for an export flag for vision, or use the small CNN classifier for photos (see `docs/LLM_FINETUNE_PLAN.md` §2).
3. **Does quantization change answers?** Re-run the test set on the device (or with `litert-lm`) and compare with `evaluate.py` results from the GPU.
4. **Size and RAM:** expect a file of about 2.6 GB. Measure peak memory with Android Studio Profiler against the 3 GB constraint.
