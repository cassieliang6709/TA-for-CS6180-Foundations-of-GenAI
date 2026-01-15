import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ==========================================
# Configuration & Cache Setup
# ==========================================
# Ideally, run `export HF_HOME=/scratch/$USER/huggingface` in terminal before running this script.
# If not set, we default to a safe location.

if "HF_HOME" not in os.environ:
    # Try to find a writable scratch directory first
    user = os.environ.get("USER", "user")
    scratch_path = f"/scratch/{user}/huggingface"
    
    if os.path.exists(f"/scratch/{user}"):
        print(f"Setting HF_HOME to {scratch_path}")
        os.environ["HF_HOME"] = scratch_path
    else:
        print("Scratch not found, using default home cache.")

# ==========================================
# 1. Quick Verification (Small Model)
# ==========================================
print("\n--- Running Quick Verification (GPT-2) ---")
try:
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    result = pipe("HPC makes AI faster because", max_new_tokens=40)
    print("Output:", result[0]['generated_text'])
    print("✅ GPU + Hugging Face setup is working correctly.")
except Exception as e:
    print(f"❌ Verification failed: {e}")

# ==========================================
# 2. Larger Model Test (Optional)
# ==========================================
# Uncomment below to test Llama-2 if you have access/token
# Note: You need to run `huggingface-cli login` first!

"""
print("\n--- Running Larger Model Test (Llama-2) ---")
model_id = "meta-llama/Llama-2-7b-chat-hf"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    inputs = tokenizer("Explain HPC briefly.", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("✅ Large model loaded successfully.")
except Exception as e:
    print(f"❌ Large model test failed (Check token/access?): {e}")
"""
