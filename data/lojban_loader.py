from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from typing import Any, Callable, Optional, Type
import json
import os
import re
import utils.constants as constants


class LojbanDataset(RawDataset):
    def __init__(
        self,
        train_data: list[str, Any],
        val_data: list[str, Any],
        test_data: list[str, Any],
        override_type: Optional[DatasetType] = None,
    ):
        
        super().__init__(override_type or DatasetType.LOJBAN)
        
    
class LojbanTranscripts:
    DEFAULT_FILE_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/quality-debates/debates-readable.jsonl"
    def get_splits(
            cls,
            file_path: str,
            combine_train_and_val: bool = False,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        
        def _load_file(file_path: str):
            entries = []
            with open(file_path, "r") as f:
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
            override_type=DatasetType.LOJBAN,
        )


class LojbanLoader(RawDataLoader):

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        combine_train_and_val: bool = False,
    ) -> LojbanDataset:
        return LojbanTranscripts.load(
            constructor_cls=cls,
            full_dataset_filepath=full_dataset_filepath,
            combine_train_and_val=combine_train_and_val,
        )