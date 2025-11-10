from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, TypeVar, Union, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

ModelT = TypeVar("ModelT")
SerializableRecord = Mapping[str, Any]


def dataclasses_to_table(dataclasses: Union[ModelT, Iterable[ModelT]]) -> pa.Table:
    """
    Convert one or more dataclass or Pydantic model instances into a PyArrow table.

    Args:
        dataclasses: A single dataclass or Pydantic model instance or an iterable of such instances.

    Returns:
        A ``pyarrow.Table`` containing one row per provided instance.

    Raises:
        TypeError: If any provided object is not a supported instance.
        ValueError: If the iterable of instances is empty.
    """
    records: List[SerializableRecord] = _coerce_to_records(dataclasses)
    if not records:
        raise ValueError("Cannot create a table from an empty collection of instances.")
    return pa.Table.from_pylist(records)


def write_dataclasses_to_parquet(
    dataclasses: Union[ModelT, Iterable[ModelT]],
    destination: Path,
    *,
    compression: str | None = "snappy",
) -> None:
    """
    Serialize dataclass or Pydantic model instances to a Parquet file using PyArrow.

    Args:
        dataclasses: A single dataclass or Pydantic model instance or an iterable of such instances.
        destination: The Parquet file path to write.
        compression: Optional Parquet compression codec. Defaults to ``"snappy"``.

    Raises:
        TypeError: If any provided object is not a supported instance.
        ValueError: If the iterable of instances is empty.
    """
    table: pa.Table = dataclasses_to_table(dataclasses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression=compression)


def _coerce_to_records(dataclasses: Union[ModelT, Iterable[ModelT]]) -> List[SerializableRecord]:
    if _is_supported_model_instance(dataclasses):
        return [_to_serializable_record(cast(ModelT, dataclasses))]
    if isinstance(dataclasses, IterableABC):
        records: List[SerializableRecord] = []
        for item in dataclasses:
            if not _is_supported_model_instance(item):
                raise TypeError(f"Expected dataclass or Pydantic model instances, received {type(item)!r}.")
            records.append(_to_serializable_record(item))
        return records
    raise TypeError("Expected a dataclass or Pydantic model instance or iterable of such instances.")


def _to_serializable_record(model_instance: ModelT) -> SerializableRecord:
    data: Dict[str, Any] = _model_to_dict(model_instance)
    return cast(SerializableRecord, _sanitize_value(data))


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Counter):
        return [
            {
                "key": _sanitize_value(key),
                "count": _sanitize_value(count),
            }
            for key, count in value.items()
        ]
    if isinstance(value, dict):
        return {key: _sanitize_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, BaseModel):
        data: Dict[str, Any] = value.model_dump()
        return _sanitize_value(data)
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
        data: Dict[str, Any] = _model_to_dict(value)
        return _sanitize_value(data)
    return value


def _is_dataclass_instance(candidate: object) -> bool:
    return is_dataclass(candidate) and not isinstance(candidate, type)


def _is_supported_model_instance(candidate: object) -> bool:
    return _is_dataclass_instance(candidate) or isinstance(candidate, BaseModel)


def _model_to_dict(instance: Any) -> Dict[str, Any]:
    if _is_dataclass_instance(instance):
        return {field.name: getattr(instance, field.name) for field in fields(instance)}
    if isinstance(instance, BaseModel):
        return instance.model_dump()
    raise TypeError(f"Unsupported instance type {type(instance)!r}.")
