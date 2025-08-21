# train_custom_q_lora.py
"""
Complete corrected QLoRA + LoRA training script for a custom chat dataset (messages format).
Place train.jsonl (one JSON object per line) in the same folder and run this script.
"""

import os
import gc
import psutil
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# -------------------------
# USER CONFIG - edit as needed
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # base model
TRAIN_FILE = "train.jsonl"                         # your dataset
OUTPUT_DIR = "./finetuned_vishnu_lora"
OFFLOAD_DIR = "./offload"                          # where to store CPU-offloaded shards
MAX_LEN = 256                                      # reduce for 12GB GPU
BATCH_PER_DEVICE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
SAVE_STEPS = 200
R_LORA = 8                                         # lower r to save memory
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "v_proj"]
# -------------------------

os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# reduce fragmentation (suggested by PyTorch error messages)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -------------------------
# 0) Clear/Free GPU memory
# -------------------------
def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

free_gpu()
print("→ GPU cache cleared & GC run")

# -------------------------
# 1) Load dataset and convert messages -> text
# -------------------------
print("→ Loading dataset:", TRAIN_FILE)
dataset = load_dataset("json", data_files={"train": TRAIN_FILE})

def format_messages(example):
    msgs = example.get("messages", [])
    system_msgs, user_msgs, assistant_msgs = [], [], []
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system" or role.startswith("sys"):
            system_msgs.append(content)
        elif role == "user":
            user_msgs.append(content)
        elif role == "assistant":
            assistant_msgs.append(content)
        else:
            # unknown roles appended to assistant by default
            assistant_msgs.append(content)

    system_text = "\n".join(system_msgs).strip()
    instruction_text = "\n".join(user_msgs).strip()
    response_text = "\n".join(assistant_msgs).strip()

    parts = []
    if system_text:
        parts.append(f"### System:\n{system_text}")
    if instruction_text:
        parts.append(f"### Instruction:\n{instruction_text}")
    parts.append(f"### Response:\n{response_text}")

    return {"text": "\n\n".join(parts)}

dataset = dataset.map(format_messages, remove_columns=dataset["train"].column_names)
print("→ Example formatted text:\n", dataset["train"][0]["text"][:400])

# -------------------------
# 2) Tokenizer: ensure pad token (prefer reuse of eos)
# -------------------------
print("→ Loading tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

added_special_token = False
if tokenizer.pad_token is None:
    # Prefer to reuse eos if present (no resize of embeddings required)
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print("→ tokenizer.pad_token set to eos_token (no embedding resize required)")
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_special_token = True
        print("→ Added new pad token '[PAD]' (will resize model embeddings after loading)")

# -------------------------
# 3) Tokenize and build labels (mask pad tokens with -100)
# -------------------------
print("→ Tokenizing and creating labels (pad -> -100). MAX_LEN =", MAX_LEN)

def tokenize_and_mask_labels(batch):
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
    )
    pad_id = tokenizer.pad_token_id
    input_ids = enc["input_ids"]
    labels = []
    for seq in input_ids:
        labels.append([tok if tok != pad_id else -100 for tok in seq])
    enc["labels"] = labels
    return enc

tokenized = dataset["train"].map(tokenize_and_mask_labels, batched=True, remove_columns=["text"])
print("→ Tokenized columns:", tokenized.column_names)
print("→ Example input length:", len(tokenized[0]["input_ids"]))

# -------------------------
# 4) Bits & Bytes (quant) config
# -------------------------
print("→ Preparing BitsAndBytesConfig for 4-bit quantization")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# -------------------------
# 5) Load base model (try auto device_map, else fallback)
# -------------------------
print("→ Loading base model (this may take some time)...")
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        torch_dtype=torch.float16,
    )
    print("→ Model loaded with device_map='auto'")
except Exception as e_auto:
    print("! device_map='auto' failed:", repr(e_auto))
    print("→ Falling back to GPU 0 placement with offload folder")
    # compute heuristic for max_memory to leave small headroom
    if torch.cuda.is_available():
        total_gb = int(torch.cuda.get_device_properties(0).total_memory // (1024 ** 3))
        cuda_mem = f"{max(total_gb - 1, 1)}GiB"
    else:
        cuda_mem = "0GiB"
    cpu_mem = f"{int(psutil.virtual_memory().total // (1024 ** 3))}GiB"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map={"": 0},   # try to place on gpu 0
            offload_folder=OFFLOAD_DIR,
            max_memory={"cpu": cpu_mem, "cuda:0": cuda_mem},
            torch_dtype=torch.float16,
        )
        print("→ Model loaded with device_map={'':0} and offload")
    except Exception as e_fallback:
        print("!! Failed to load model. See error below.")
        raise e_fallback from e_auto

# If we earlier added a new pad token, resize embeddings
if added_special_token:
    print("→ Resizing token embeddings to account for added token")
    model.resize_token_embeddings(len(tokenizer))

# memory optimizations
model.gradient_checkpointing_enable()
model.config.use_cache = False

# -------------------------
# 6) Apply LoRA (PEFT)
# -------------------------
print("→ Applying LoRA adapters")
lora_config = LoraConfig(
    r=R_LORA,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# quick sanity: percent trainable parameters
def print_trainable_parameters(m):
    trainable, total = 0, 0
    for _, p in m.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"→ Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
print_trainable_parameters(model)

# -------------------------
# 7) Prepare Trainer + arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# -------------------------
# 8) Train
# -------------------------
try:
    print("→ Starting training. Monitor with `nvidia-smi` if necessary.")
    trainer.train()
except RuntimeError as e:
    # common OOM or CUDA errors -> give guidance
    print("!! RuntimeError during training:", repr(e))
    if "out of memory" in str(e).lower():
        print("!! OOM: try lowering MAX_LEN, reduce batch size, increase gradient_accumulation_steps, or close other GPU processes.")
    raise

# -------------------------
# 9) Save LoRA adapters + tokenizer
# -------------------------
print("→ Saving LoRA adapters + tokenizer to:", OUTPUT_DIR)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -------------------------
# 10) Quick inference check (optional)
# -------------------------
try:
    from peft import PeftModel
    print("→ Running quick inference test (optional).")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        torch_dtype=torch.float16,
    )
    model_for_infer = PeftModel.from_pretrained(base, OUTPUT_DIR, device_map="auto")
    prompt = "### Instruction:\nWho are you?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model_for_infer.device)
    with torch.no_grad():
        out = model_for_infer.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    print("→ Sample output:\n", tokenizer.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print("→ Optional quick inference failed (non-fatal). Error:", repr(e))

print("✅ Done.")
