from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from typing import Any, Callable, Optional, Type
import json
import os
import re
from data.quality_loader import QualityLoader
import utils.constants as constants
import random


class LojbanDataset(RawDataset):
    def __init__(
        self,
        test_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        override_type: Optional[DatasetType] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
    ):
        
        super().__init__(override_type or DatasetType.LOJBAN)
        if shuffle_deterministically:
            random.seed(a=123456789)
        self.flip_sides = flip_sides
        self.data = {
            SplitType.VAL: self.convert_batch_to_rows(val_data),
            SplitType.TEST: self.convert_batch_to_rows(test_data),
            SplitType.TRAIN: self.convert_batch_to_rows(test_data)  # Use TEST data for TRAIN since no separate training data
        }

        self.idxs = {SplitType.VAL: 0, SplitType.TEST: 0, SplitType.TRAIN: 0}
        
        self.data[SplitType.VAL] = self.__reorder(self.data[SplitType.VAL])
        self.data[SplitType.TEST] = self.__reorder(self.data[SplitType.TEST])
        self.shuffle_deterministically = shuffle_deterministically

    def get_data(self, split: SplitType = SplitType.TEST) -> list[DataRow]:
        """Returns all the data for a given split"""
        # if split != SplitType.TEST:
        #     raise ValueError(f"Split type {split} is not available. Only TEST split is supported for Lojban dataset")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TEST, batch_size: int = 1) -> list[DataRow]:
        """Returns a subset of the data for a given split"""
        # if split != SplitType.TEST:
        #     raise ValueError(f"Split type {split} is not available. Only TEST split is supported for Lojban dataset")
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TEST, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        # if split != SplitType.TEST:
        #     raise ValueError(f"Split type {split} is not available. Only TEST split is supported for Lojban dataset")
        return self.data[split][idx % len(self.data[split])]

    def convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        rows = []
        for entry in batch:
            rows_to_add = self.example_to_row(entry)
            if rows_to_add:
                rows.extend(rows_to_add)
        return rows


    def example_to_row(self, entry: dict[str, Any]) -> list[DataRow]:
        if entry["original_key"] == "A":
            correct_answer = 0
            incorrect_answer = 1
        else:
            correct_answer = 1
            incorrect_answer = 0
        possible_position_pairs = [(correct_answer, incorrect_answer, True), (incorrect_answer, correct_answer, False)]
        random.shuffle(possible_position_pairs)
        
        rows = []
        for first, second, first_correct in possible_position_pairs:
            rows.append(
                DataRow(
                    background_text=entry["prompt_file_content"],
                    question=entry["prompt"],
                    correct_index=0 if first_correct else 1,
                    positions=(
                        entry["answers"][first],
                        entry["answers"][second],
                    ),
                    debate_id=entry["original_id"],
                    ground_truth=entry["original_key"],
                    explanations=entry["original_explanation"]
                )
            )
            if not self.flip_sides:
                break
        return rows 


    def __reorder(self, rows: list[DataRow]) -> list[DataRow]:
        if len(rows) == 0:
            return rows

        random.shuffle(rows)
        prompt_to_rows = {}
        for row in rows:
            if row.story_title not in prompt_to_rows:
                prompt_to_rows[row.story_title] = []
            prompt_to_rows[row.story_title].append(row)

        final_order = []
        max_index = max([len(prompt_to_rows[row.story_title]) for row in rows])
        for index in range(max_index):
            for story in filter(lambda x: len(prompt_to_rows[x]) > index, prompt_to_rows):
                final_order.append(prompt_to_rows[story][index])
        return final_order

class LojbanLoader(RawDataLoader):
    DEFAULT_VAL_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/lojban/lojban_dataset.jsonl"
    DEFAULT_TEST_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/lojban/lojban_dataset_test.jsonl"

    @classmethod
    def get_splits(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> tuple[list[dict]]:
        """Splits the data in train, val, and test sets"""

        def __load_individual_file(filepath: str) -> list[str, Any]:
            entries = []
            if filepath:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        entries.append(json.loads(line))
            return entries

        val_filepath = val_filepath or LojbanLoader.DEFAULT_VAL_PATH
        test_filepath = test_filepath or LojbanLoader.DEFAULT_TEST_PATH

        val_split = __load_individual_file(val_filepath)
        test_split = __load_individual_file(test_filepath)
        return val_split, test_split


    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
        **kwargs
    ) -> LojbanDataset:
        """Constructs a LojbanDataset"""

        val_split, test_split = LojbanLoader.get_splits(
            val_filepath=val_filepath, test_filepath=test_filepath
        )

        return LojbanDataset(
            test_data=test_split,
            val_data=val_split,
            flip_sides=flip_sides,
            shuffle_deterministically=shuffle_deterministically
        )