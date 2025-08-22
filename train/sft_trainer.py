from data import DatasetConfig, DataRow, RawDataset, SplitType
from models import LLMInput, LLModel, LLMType, ModelInput, SpeechStructure
from prompts import RoleType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils  # TODO: REMOVE
import utils.constants as constants

from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import pandas as pd
import datasets
import torch

from typing import Any, Optional, Type
import json
import random
import sys

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class SupervisedTrainer:
    """Class for training a model using Supervised Fine Tuning"""

    @classmethod
    def convert_dataset(
        cls, raw_datasets: list[RawDataset], config: TrainingConfig, tokenizer: AutoTokenizer
    ) -> datasets.Dataset:
        """Converts a dataset (abstraction used in this codebase) into a Dataset object (abstraction
        used by huggingface's trainer objects)"""
        # 1) Load the CSV into a HF Dataset
        dataset = datasets.load_dataset(
            "csv",
            #data_files={"train": "/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/training_dataset_for_debater_no_judge_speeches.csv"},
            #data_files={"train": "/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/training_dataset_for_debater_no_judge_speeches_minitron.csv"},
            data_files={"train": "/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/training_dataset_for_debater_no_judge_speeches_qwen.csv"},
            split="train"
        )  

         # 2) Compute token length of your response boundary
        suffix = "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        suffix_len = len(tokenizer.encode(suffix, add_special_tokens=False))
        # 2) Define a predicate to keep only examples within max_length
        def is_within_limit(example):
            inst_len = len(tokenizer(
                example["instruction"],
                truncation=False
            )["input_ids"])
            return inst_len + suffix_len <= config.max_length

        # 3) Apply filter to drop too-long examples
        #dataset = dataset.filter(is_within_limit)  
        # (Optional) Shuffle for training randomness
        dataset = dataset.shuffle(seed=42)

        return dataset

    @classmethod
    def formatting_func(cls, llm_dictionary: dict[str, list[str]]) -> str:
        formatted = []
        for instruction, output in zip(llm_dictionary["instruction"], llm_dictionary["output"]):
            formatted.append(instruction + output.strip())
        return formatted

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_datasets: Optional[list[RawDataset]] = None,
        is_local: bool = False,
        is_test: bool = False,
    ) -> Optional[SFTTrainer]:
        """
        Generates a Trainer object.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training
            is_local: whether this is being run on a cpu
            is_test: whether to actually instantiate the trainer (if true, do not instantiate)

        Returns:
            sft_trainer: One can call dpo_trainer.train() to then run the training loop.
        """
        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()

        if not raw_datasets:
            raw_datasets = TrainUtils.create_datasets(config=config)

        tokenizer = TrainUtils.get_tokenizer(config=config, is_local=is_local)
        model     = TrainUtils.load_model(config=config, is_local=is_local)
        llm_class = TrainUtils.get_llm_class(config=config)

        # ─── FIX: Build response_template_ids robustly ───────────────────────────────
        #eot = "<|eot_id|>"                  # or pull from your constants module
        #suffix = llm_class.INSTRUCTION_SUFFIX
        #response_template = f"{eot}{suffix}"
        response_template = "[/INST]"

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False
        )
        # ────────────────────────────────────────────────────────────────────────────


        training_args = TrainingArguments(
            output_dir=config.logging_and_saving_config.output_dir,
            num_train_epochs=config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim=config.training_hyperparameters.optim,
            logging_steps=config.logging_and_saving_config.logging_steps,
            save_strategy="epoch",
            learning_rate=config.training_hyperparameters.learning_rate,
            max_grad_norm=config.training_hyperparameters.max_grad_norm,
            warmup_ratio=config.training_hyperparameters.warmup_ratio,
            lr_scheduler_type=config.training_hyperparameters.lr_scheduler_type,
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
            use_cpu=is_local,
        )

        train_dataset = SupervisedTrainer.convert_dataset(
            raw_datasets=raw_datasets,
            tokenizer=tokenizer,
            config=config,
        )

        peft_config = TrainUtils.get_peft_config(config) if not is_local else None
        if peft_config:
            # model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
            if FLASH_ATTENTION_AVAILABLE:
                model = upcast_layer_for_flash_attention(model, torch.bfloat16).to("cuda")

        if not is_test:
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                data_collator=collator,
                formatting_func=SupervisedTrainer.formatting_func,
                max_seq_length=config.max_length,
                callbacks=[LoggingCallback],
                args=training_args,
            )

            torch.cuda.empty_cache()

            return trainer
        return None
