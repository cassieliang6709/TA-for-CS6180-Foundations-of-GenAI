import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# ==========================================
# 3. Hugging Face Cache Configuration
# ==========================================
# On HPC systems, never rely on default cache paths.
# Always force Hugging Face to use a writable directory (your home directory).

# Use user's home directory dynamically
user_home = os.path.expanduser("~")

# Force Hugging Face to use a writable cache directory
os.environ["HF_HOME"] = os.path.join(user_home, ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(user_home, ".cache/huggingface/datasets")

# Stability settings for HPC
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("HF_HOME =", os.environ["HF_HOME"])

# ==========================================
# 9. Verifying GPU Usage (Check early)
# ==========================================
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not detected. Training will be slow.")

# ==========================================
# 4. Model and Tokenizer Initialization
# ==========================================
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for training

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# ==========================================
# 5. Dataset Preparation
# ==========================================
texts = [
    "HPC makes AI faster by providing massive parallel compute.",
    "Large language models require GPUs to train efficiently.",
    "Distributed training is essential for scaling deep learning.",
    "Hugging Face simplifies model training on clusters."
]

dataset = Dataset.from_dict({"text": texts})

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

tokenized_ds = dataset.map(tokenize_fn, remove_columns=["text"])

# ==========================================
# 6. Training Configuration
# ==========================================
training_args = TrainingArguments(
    output_dir="./gpt2-hpc-demo",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),  # Enable fp16 if GPU is available
    logging_steps=1,
    save_strategy="no",
    report_to="none",       # disable external logging on HPC
    disable_tqdm=True       # avoids Jupyter widget errors
)

# ==========================================
# 7. Training with Hugging Face Trainer
# ==========================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Training completed successfully.")
