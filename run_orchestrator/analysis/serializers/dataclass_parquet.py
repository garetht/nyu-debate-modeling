from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, TypeVar, Union, cast

import pyarrow as pa
import pyarrow.parquet as pq

DataClassT = TypeVar("DataClassT")
SerializableRecord = Mapping[str, Any]


def dataclasses_to_table(dataclasses: Union[DataClassT, Iterable[DataClassT]]) -> pa.Table:
    """
    Convert one or more dataclass instances into a PyArrow table.

    Args:
        dataclasses: A single dataclass instance or an iterable of dataclass instances.

    Returns:
        A ``pyarrow.Table`` containing one row per dataclass instance.

    Raises:
        TypeError: If any provided object is not a dataclass instance.
        ValueError: If the iterable of dataclasses is empty.
    """
    records: List[SerializableRecord] = _coerce_to_records(dataclasses)
    if not records:
        raise ValueError("Cannot create a table from an empty collection of dataclasses.")
    return pa.Table.from_pylist(records)


def write_dataclasses_to_parquet(
    dataclasses: Union[DataClassT, Iterable[DataClassT]],
    destination: Path,
    *,
    compression: str | None = "snappy",
) -> None:
    """
    Serialize dataclass instances to a Parquet file using PyArrow.

    Args:
        dataclasses: A single dataclass instance or an iterable of dataclass instances.
        destination: The Parquet file path to write.
        compression: Optional Parquet compression codec. Defaults to ``"snappy"``.

    Raises:
        TypeError: If any provided object is not a dataclass instance.
        ValueError: If the iterable of dataclasses is empty.
    """
    table: pa.Table = dataclasses_to_table(dataclasses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression=compression)


def _coerce_to_records(dataclasses: Union[DataClassT, Iterable[DataClassT]]) -> List[SerializableRecord]:
    if _is_dataclass_instance(dataclasses):
        return [_to_serializable_record(cast(DataClassT, dataclasses))]
    if isinstance(dataclasses, IterableABC):
        records: List[SerializableRecord] = []
        for item in dataclasses:
            if not _is_dataclass_instance(item):
                raise TypeError(f"Expected dataclass instances, received {type(item)!r}.")
            records.append(_to_serializable_record(item))
        return records
    raise TypeError("Expected a dataclass instance or iterable of dataclass instances.")


def _to_serializable_record(dataclass_instance: DataClassT) -> SerializableRecord:
    data: Dict[str, Any] = _dataclass_to_dict(dataclass_instance)
    return cast(SerializableRecord, _sanitize_value(data))


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        data: Dict[str, Any] = _dataclass_to_dict(value)
        return _sanitize_value(data)
    return value


def _is_dataclass_instance(candidate: object) -> bool:
    return is_dataclass(candidate) and not isinstance(candidate, type)


def _dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    return {field.name: getattr(instance, field.name) for field in fields(instance)}
