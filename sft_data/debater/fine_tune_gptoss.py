#!/usr/bin/env python3
import os
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any
import html
import wandb
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------------- 0) System knobs ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
SEED = 42
torch.manual_seed(SEED)
MODEL_ID = "openai/gpt-oss-20b"
DATA_CSV =  "sft_data/debater/training_dataset_for_debater_no_judge_speeches.csv"
OUTPUT_DIR = "outputs/trained_models/gpt_oss_20b_lora_11_09"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
DTYPE = torch.bfloat16
MAX_LEN = 128000
BLOCK_LEN = 3072
PACKING = False  # DISABLED - this was causing the zero coverage issue
NUM_EPOCHS = 8
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-5  
WARMUP_R = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0  # INCREASED from 0.3 for better gradient flow
LOG_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
WANDB_PROJECT = "gpt-oss-20b-lora-fixed"
WANDB_RUN_NAME = "debater_sft_fixed_preprocessing"
# ---------------- 1) W&B ----------------
wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
# ---------------- 2) Tokenizer ----------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# Handle padding token
added_pad = False
if tok.pad_token is None:
    try:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        added_pad = True
    except Exception:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
# ---------------- 3) Model ----------------
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
if added_pad:
    model.resize_token_embeddings(len(tok))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
# ---------------- 4) LoRA ----------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    bias="none",
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    target_modules= ["q_proj","k_proj","v_proj","o_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
# ---------------- 5) FIXED Data Preprocessing ----------------
def to_prompt(instr: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{instr.strip()}<|im_end|>\n"
    )
def to_asst(out: str) -> str:
    return f"{ASSISTANT_START}{out.strip()}{IM_END}\n"
import html
from typing import Dict, Any
def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced preprocessing with detailed debugging"""
    
    def safe_unescape(text: str) -> str:
        if any(entity in text for entity in ['&lt;', '&gt;', '&amp;', '&quot;', '&#']):
            return html.unescape(text)
        return text
    
    # Apply safe unescaping
    instruction = safe_unescape(example["instruction"].strip())
    output = safe_unescape(example["output"].strip())
    
    # Concatenate instruction + output + end token
    full_text = instruction + output + "<|eot_id|>"
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    
    if len(full_ids) > MAX_LEN:
        full_ids = full_ids[:MAX_LEN]
    
    labels = [-100] * len(full_ids)
    
    # 🔧 ENHANCED: Try multiple possible assistant headers
    possible_headers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n",
        "assistant<|end_header_id|>",
        "assistant<|end_header_id|>\n"
    ]
    
    asst_start_idx = None
    matched_header = None
    
    for header in possible_headers:
        header_tokens = tok(header, add_special_tokens=False).input_ids
        
        # Search for this header pattern
        for i in range(len(full_ids) - len(header_tokens), -1, -1):
            if i >= 0 and full_ids[i:i+len(header_tokens)] == header_tokens:
                asst_start_idx = i + len(header_tokens)
                matched_header = header
                break
        
        if asst_start_idx is not None:
            break
    
    # 🐛 DEBUG: Add logging for first example
    if not hasattr(preprocess, '_debug_printed'):
        print(f"\n=== PREPROCESSING DEBUG ===")
        print(f"Full text length: {len(full_text)}")
        print(f"Full text preview: {full_text[:200]}...")
        print(f"Full text end: ...{full_text[-200:]}")
        print(f"Token count: {len(full_ids)}")
        
        print(f"\nTrying headers:")
        for i, header in enumerate(possible_headers):
            header_tokens = tok(header, add_special_tokens=False).input_ids
            print(f"  {i+1}. '{header}' -> tokens: {header_tokens[:10]}...")
            
            # Check if tokens exist anywhere in sequence
            header_str = tok.decode(header_tokens)
            if header_str in full_text:
                print(f"     ✅ Found in text at position: {full_text.find(header_str)}")
            else:
                print(f"     ❌ Not found in text")
        
        print(f"\nMatched header: {matched_header}")
        print(f"Assistant start index: {asst_start_idx}")
        preprocess._debug_printed = True
    
    # Label assistant content if found
    if asst_start_idx is not None:
        # Find end position (look for <|eot_id|>)
        eot_tokens = tok("<|eot_id|>", add_special_tokens=False).input_ids
        end_pos = len(full_ids)
        
        for i in range(asst_start_idx, len(full_ids) - len(eot_tokens) + 1):
            if full_ids[i:i+len(eot_tokens)] == eot_tokens:
                end_pos = i
                break
        
        # Label the assistant response tokens
        for i in range(asst_start_idx, end_pos):
            labels[i] = full_ids[i]
        
        # 🐛 DEBUG: Show labeled content for first example
        if not hasattr(preprocess, '_content_printed'):
            labeled_tokens = [full_ids[i] for i in range(asst_start_idx, end_pos)]
            if labeled_tokens:
                labeled_text = tok.decode(labeled_tokens)
                print(f"\n✅ LABELED CONTENT PREVIEW: {labeled_text[:200]}...")
            preprocess._content_printed = True
    
    return {"input_ids": full_ids, "labels": labels}
# Load and preprocess data
raw_all = load_dataset("csv", data_files={"train": DATA_CSV})["train"].shuffle(seed=SEED)
eval_n = min(500, max(1, int(0.02 * len(raw_all))))
raw_eval = raw_all.select(range(eval_n))
raw_train = raw_all.select(range(eval_n, len(raw_all)))
print("Preprocessing training data...")
tok_train = raw_train.map(preprocess, remove_columns=raw_all.column_names, desc="Preprocess (train)")
tok_eval = raw_eval.map(preprocess, remove_columns=raw_all.column_names, desc="Preprocess (eval)")
# Set format (no packing to avoid disrupting labels)
tok_train.set_format(type="torch", columns=["input_ids", "labels"])
tok_eval.set_format(type="torch", columns=["input_ids", "labels"])
# ---------------- 6) FIXED Data Collator ----------------
@dataclass
class SimpleCLMCollator:
    pad_token_id: int
    
    def __call__(self, features):
        # Handle already tensorized data properly
        if isinstance(features[0]["input_ids"], torch.Tensor):
            ids = [f["input_ids"] for f in features]
            lbl = [f["labels"] for f in features]
        else:
            ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            lbl = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(lbl, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
collator = SimpleCLMCollator(pad_token_id=tok.pad_token_id)
# ---------------- 7) Training Arguments ----------------
def filtered_training_args(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    clean = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**clean)
ta_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    bf16=True,
    fp16=False,
    seed=SEED,
    
    # logging / saving
    logging_strategy="steps",
    logging_steps=LOG_STEPS,
    logging_first_step=True,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    
    # dataloader / memory
    dataloader_pin_memory=False, #True,
    dataloader_num_workers=2,
    dataloader_prefetch_factor=2,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    
    # optim & schedule
    optim="adamw_torch",
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_R,
    
    # eval
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    
    report_to=["wandb"],
)
args = filtered_training_args(**ta_kwargs)
class LossFixTrainer(Trainer):
    """Robust trainer with proper loss computation"""
    
    def _get_logits(self, outputs):
        """Extract logits from model outputs"""
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise RuntimeError(f"Forward didn't return logits; got type={type(outputs)}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        fwd_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**fwd_inputs)
        logits = self._get_logits(outputs)
        
        # Shift for causal LM
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # Optional: Log additional metrics
        if hasattr(self, 'state') and self.state.logging_steps > 0 and self.state.global_step % self.state.logging_steps == 0:
            valid_tokens = (shift_labels != -100).sum()
            if valid_tokens > 0:  # Prevent division by zero
                perplexity = torch.exp(loss)
                self.log({
                    "train_perplexity": perplexity,
                    "valid_tokens": valid_tokens,
                    "tokens_per_example": valid_tokens / shift_labels.shape[0]
                })
        
        return (loss, outputs) if return_outputs else loss
trainer = LossFixTrainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_eval if getattr(args, "do_eval", False) else None,
    data_collator=collator,
    tokenizer=tok,
)
# ---------------- 9) Enhanced Debug Section ----------------
print("=== DEBUGGING FIRST BATCH ===")
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(trainer.model.device) for k, v in batch.items()}
with torch.no_grad():
    out = trainer.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
print("Batch keys:", list(batch.keys()))
print("Input shape:", batch["input_ids"].shape)
print("Labels shape:", batch["labels"].shape)
# Calculate label coverage
total_tokens = batch["attention_mask"].sum().item()
labeled_tokens = (batch["labels"] != -100).sum().item()
coverage = labeled_tokens / max(total_tokens, 1)
print(f"Total tokens: {total_tokens}")
print(f"Labeled tokens: {labeled_tokens}")
print(f"Label coverage: {coverage:.4f}")
print("Logits shape:", tuple(out.logits.shape))
# Show sample of labeled content
sample_ids = batch["input_ids"][0]
sample_labels = batch["labels"][0]
labeled_indices = torch.where(sample_labels != -100)[0]
if len(labeled_indices) > 0:
    print("Sample input text:", tok.decode(sample_ids[:200]))
    print("First 10 labeled tokens:", tok.decode(sample_ids[labeled_indices[:10]]))
    print("Sample labeled text:", tok.decode(sample_ids[labeled_indices[:20]]))
else:
    print("NO LABELED TOKENS FOUND!")
    # Debug the preprocessing on a single example
    raw_example = raw_train[0]
    processed = preprocess(raw_example)
    print("Debug single example:")
    print("Input text:", tok.decode(processed["input_ids"])[:300])
    print("Labeled positions:", [i for i, l in enumerate(processed["labels"]) if l != -100][:10])
# Ensure we have good coverage before training
assert coverage > 0.01, f"Label coverage too low: {coverage:.4f}. Check preprocessing!"
print("=== STARTING TRAINING ===")
# ---------------- 10) Train ----------------
trainer.train()
# ---------------- 11) Save ----------------
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapters saved to: {ADAPTER_DIR}")
print(f"Training completed successfully!")

'''
source /lambda/nfs/mars-arnesen-gh/leonidtsyplenkov/.venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

sudo docker pull nvcr.io/nvidia/pytorch:24.12-py3
sudo docker run --rm -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD:/workspace \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:24.12-py3 bash


python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128"
python -m pip install "peft>=0.17.0" "transformers==4.55.2" "accelerate>=1.0.0" 
python -m pip install -U "triton==3.4.0"
python -m pip install -U "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"

python sft_data/debater/fine_tune_gptoss.py
python -m wandb login --relogin c28cfe82b0c7fedde82a0daf207b28e09484ee63
./cli.sh bg-task start -n fine-tune -- bash -c "source /lambda/nfs/mars-arnesen-gh/leonidtsyplenkov/.venv/bin/activate && DEBUG_FIRST_BATCH=1 python -B sft_data/debater/fine_tune_gptoss.py"


python -m pip install -U git+https://github.com/huggingface/transformers



'''