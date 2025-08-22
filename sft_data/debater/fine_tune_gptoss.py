#!/usr/bin/env python3
import os
from typing import List, Set
import wandb

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

wandb.init()

MODEL_ID = "openai/gpt-oss-20b"
DATA_CSV = "sft_data/debater/training_dataset_for_gptoss.csv"
#OUTPUT_DIR = "models/trained_models/gpt_oss_20b_lora"
OUTPUT_DIR = "outputs/trained_models/gpt_oss_20b_lora_21.08"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")

# ---------- 0) system knobs ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("high")   # Hopper OK

DTYPE = torch.bfloat16        # GH200 sweet spot
MAX_LEN = 4096
BLOCK_LEN = 1536              # start a bit lower than 2048 for extra headroom
PACKING = True

# ---------- 1) tokenizer ----------
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

# ---------- 2) model (bf16 + flash-attn2 + ckpt) ----------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


# ---------- 3) dataset (tokenize; no labels in dataset) ----------
raw = load_dataset("csv", data_files={"train": DATA_CSV})["train"]

def to_harmony(instr: str, out: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{instr.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{out.strip()}<|im_end|>\n"
    )

def preprocess(example):
    text = to_harmony(example["instruction"], example["output"])
    enc = tok(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_attention_mask=False,
    )
    return {"input_ids": enc["input_ids"]}

tokd = raw.map(preprocess, remove_columns=raw.column_names, batched=False, desc="Tokenizing")

# ---------- 4) constant-length packing (cuts padding) ----------
def pack_examples(ds: Dataset, block_len: int) -> Dataset:
    big_ids: List[int] = []
    for ex in ds:
        big_ids.extend(ex["input_ids"])
    blocks = []
    for i in range(0, len(big_ids) - block_len, block_len):
        blocks.append({"input_ids": big_ids[i : i + block_len]})
    return Dataset.from_list(blocks)

if PACKING:
    tokd = pack_examples(tokd, BLOCK_LEN)
    EFFECTIVE_LEN = BLOCK_LEN
else:
    EFFECTIVE_LEN = MAX_LEN

tokd.set_format(type="torch", columns=["input_ids"])

# ---------- 5) LoRA config (auto-detect target modules) ----------
COMMON_LINEAR_SUFFIXES: Set[str] = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
    # fallbacks seen in some arches:
    "wq", "wk", "wv", "wo", "w1", "w2", "w3",
}

def detect_target_modules(m) -> List[str]:
    found: Set[str] = set()
    for name, mod in m.named_modules():
        # only consider linears to keep things tidy
        if isinstance(mod, torch.nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            if suffix in COMMON_LINEAR_SUFFIXES:
                found.add(suffix)
    # reasonable default if nothing matched
    if not found:
        found = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}
    return sorted(found)

targets = detect_target_modules(model)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                 # try 8/16/32
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules= ["q_proj","k_proj","v_proj","o_proj", "down_proj"] #targets,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------- 6) collator (labels on-the-fly) ----------
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

# ---------- 7) training args ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    fp16=False,
    logging_steps=20,
    save_steps=500,
    save_total_limit=1,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    dataloader_prefetch_factor=2,
    gradient_checkpointing=True,
    optim="adafactor",           # much lower optimizer memory than AdamW
    learning_rate=1e-4,          # typical LoRA LR
    weight_decay=0.0,
    max_grad_norm=0.3,
    report_to=["wandb"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokd,
    data_collator=collator,
    tokenizer=tok,
)

trainer.train()

# ---------- 8) save adapters (small) ----------
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapters saved to: {ADAPTER_DIR} (seq_len={EFFECTIVE_LEN}, packed={PACKING})")

# Optional: merge LoRA into base weights 
#merged = model.merge_and_unload()
#merged.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))

'''
source /lambda/nfs/mars-arnesen-gh/leonidtsyplenkov/.venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128"
python -m pip install "peft>=0.17.0" "transformers==4.55.2" "accelerate>=1.0.0" 

python sft_data/debater/fine_tune_gptoss.py

./cli.sh bg-task start -n fine-tune -- bash -c "source /lambda/nfs/mars-arnesen-gh/leonidtsyplenkov/.venv/bin/activate && python sft_data/debater/fine_tune_gptoss.py"

python -m pip install triton==3.4.0

'''
