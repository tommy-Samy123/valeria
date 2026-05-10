import torch
from unsloth import FastVisionModel
from transformers import TextStreamer

MODEL_DIR = "gemma_4_lora"

def main():
    print(f"Loading model from {MODEL_DIR}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_DIR,
        load_in_4bit=True,
    )
    
    # The FastVisionModel supports vision, but right now we are passing text-only prompts
    print("Model loaded. Enter a medical question for the FirstAidQA trained model.")
    while True:
        try:
            question = input("\nUser: ")
            if not question.strip():
                continue
            if question.lower() in ["exit", "quit"]:
                break
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(
                text=input_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")
            
            text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
            
            print("\nAssistant: ", end="")
            _ = model.generate(
                **inputs, 
                streamer=text_streamer, 
                max_new_tokens=128,
                use_cache=True, 
                temperature=1.0, 
                top_p=0.95, 
                top_k=64
            )
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
