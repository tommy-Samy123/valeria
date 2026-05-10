import os
import shutil
from unsloth import FastVisionModel
from mediapipe.tasks.python.genai import converter

MODEL_DIR = "gemma_4_lora"
MERGED_DIR = "merged_model"
OUTPUT_TASK = "firstaid_gemma_q8.task"

def main():
    # 1. Load the model and merge LoRA weights
    print(f"Loading LoRA adapter from {MODEL_DIR}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_DIR,
        load_in_4bit=False, # We want to save full precision weights before quantization
    )
    
    print(f"Merging LoRA into base model and saving to {MERGED_DIR}...")
    # This saves the full Hugging Face model (safetensors format)
    model.save_pretrained_merged(MERGED_DIR, processor, save_method="merged_16bit")
    
    # 2. Convert to MediaPipe task format
    print(f"Converting merged model to MediaPipe .task format (8-bit quantization)...")
    config = converter.ConversionConfig(
        input_ckpt=MERGED_DIR,
        ckpt_format="safetensors",
        model_type="GEMMA",
        backend="cpu",
        output_dir=os.path.dirname(os.path.abspath(OUTPUT_TASK)),
        combine_file_only=True,
        vocab_model_file=f"{MERGED_DIR}/tokenizer.model",
        output_tflite_file=OUTPUT_TASK,
    )
    
    try:
        converter.convert_checkpoint(config)
        print(f"\nSuccess! The MediaPipe file is saved as: {OUTPUT_TASK}")
        print("You can now download this file and place it in your Android app's assets folder.")
    except Exception as e:
        print("\nConversion failed. Note: If this fails because the model contains vision layers,")
        print("MediaPipe might not yet support this specific variant. Error:")
        print(e)
    
    # Clean up the intermediate merged directory to save space
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)

if __name__ == "__main__":
    main()
