from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from typing import Any, Callable, Optional, Type
import json
import os
import re
import utils.constants as constants
import random


class LojbanDataset(RawDataset):
    def __init__(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
        override_type: Optional[DatasetType] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
    ):
        
        super().__init__(override_type or DatasetType.LOJBAN)
        if shuffle_deterministically:
            random.seed(a=123456789)
        self.flip_sides = flip_sides
        self.data = {
            SplitType.TRAIN: self.convert_batch_to_rows(train_data),
            SplitType.VAL: self.convert_batch_to_rows(val_data),
            SplitType.TEST: self.convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}
        if not self.data[SplitType.TEST]:  # Adding b/c Quality Test Set does not have gold labels
                    self.__split_validation_and_test_sets()
        
        self.data[SplitType.TRAIN] = self.__reorder(self.data[SplitType.TRAIN])
        self.data[SplitType.VAL] = self.__reorder(self.data[SplitType.VAL])
        self.data[SplitType.TEST] = self.__reorder(self.data[SplitType.TEST])
        self.shuffle_deterministically = shuffle_deterministically

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
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
                    story_title=entry["original_id"],
                )
            )
            if not self.flip_sides:
                break
        return rows 

    def __split_validation_and_test_sets(self):
        second_half = self.data[SplitType.VAL][int(len(self.data[SplitType.VAL]) / 2) :]
        self.data[SplitType.VAL] = self.data[SplitType.VAL][0 : int(len(self.data[SplitType.VAL]) / 2)]
        val_prompts = set([row.story_title for row in self.data[SplitType.VAL]])

        test_data = []
        for row in second_half:
            if row.story_title not in val_prompts:
                test_data.append(row)
            else:
                self.data[SplitType.VAL].append(row)
        self.data[SplitType.TEST] = test_data

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

class LojbanTranscripts:
    DEFAULT_FILE_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality-debates/debates-readable.jsonl"
    @classmethod
    def get_splits(
            cls,
            file_path: str,
            combine_train_and_val: bool = False,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        
        def _load_file(file_path: str):
            entries = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entries.append(json.loads(line))
            return entries
        
        def __create_splits(rows: list[dict], combine_train_and_val: bool = False):
            background_to_row = {}
            background_to_question = {}
            for row in rows:
                if row["original_id"] not in background_to_row:
                    background_to_row[row["original_id"]] = []
                    background_to_question[row["original_id"]] = []
                if row["prompt"] not in background_to_question[row["original_id"]]:
                    background_to_row[row["original_id"]].append(row)
                    background_to_question[row["original_id"]].append(row["prompt"])
            train = []
            val = []
            test = []
            for i, background in enumerate(background_to_row):
                if i < int(0.8 * len(background_to_row)):
                    train.extend(background_to_row[background])
                elif i < int(0.9 * len(background_to_row)):
                    val.extend(background_to_row[background])
                else:
                    test.extend(background_to_row[background])

            if combine_train_and_val:
                train = train + val

            return train, val, test
        
        return __create_splits(rows=_load_file(file_path), combine_train_and_val=combine_train_and_val)
    
    @classmethod
    def load(
        cls,
        constructor_cls: Type[RawDataLoader],
        full_dataset_filepath: Optional[str] = None,
        combine_train_and_val: bool = False,
    ) -> LojbanDataset:
        full_dataset_filepath = full_dataset_filepath or LojbanTranscripts.DEFAULT_FILE_PATH
        train, val, test = constructor_cls.get_splits(
            file_path=full_dataset_filepath,
            combine_train_and_val=combine_train_and_val,
        )
        return LojbanDataset(
            train_data=train,
            val_data=val,
            test_data=test,
        )

class LojbanLoader(RawDataLoader):
    @classmethod
    def get_splits(cls, file_path: str, combine_train_and_val: bool = False):
        return LojbanTranscripts.get_splits(
            file_path=file_path,
            combine_train_and_val=combine_train_and_val,
        )

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        combine_train_and_val: bool = False,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False
    ) -> LojbanDataset:
        return LojbanTranscripts.load(
            constructor_cls=cls,
            full_dataset_filepath=full_dataset_filepath,
            combine_train_and_val=combine_train_and_val,
        )