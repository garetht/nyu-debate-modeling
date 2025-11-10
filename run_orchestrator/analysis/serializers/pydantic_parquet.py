from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable as IterableAlias, List, Mapping, Tuple, Type, TypeVar, Union, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pa_types
from pydantic import BaseModel
from pydantic_to_pyarrow import SchemaCreationError, get_pyarrow_schema

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
BaseModelIterable = Union[PydanticModelT, Iterable[PydanticModelT]]


def _coerce_models(models: BaseModelIterable) -> List[PydanticModelT]:
    """
    Normalize the input into a list of Pydantic models.

    Args:
        models: A single Pydantic model instance or an iterable of instances.

    Returns:
        A list of Pydantic model instances.

    Raises:
        TypeError: If any element is not a Pydantic BaseModel instance.
        ValueError: If the iterable is empty.
    """
    if isinstance(models, BaseModel):
        return [cast(PydanticModelT, models)]

    normalized: List[PydanticModelT] = []
    for item in models:
        if not isinstance(item, BaseModel):
            raise TypeError(f"Expected Pydantic BaseModel instances, received {type(item)!r}.")
        normalized.append(item)

    if not normalized:
        raise ValueError("Cannot serialize an empty collection of Pydantic models.")
    return normalized


def _require_homogeneous_models(models: List[PydanticModelT]) -> Type[PydanticModelT]:
    """
    Ensure all models are instances of the same Pydantic class.

    Args:
        models: The list of models to validate.

    Returns:
        The common Pydantic model class.

    Raises:
        TypeError: If the models are not all of the same type.
    """
    base_class: Type[PydanticModelT] = cast(Type[PydanticModelT], type(models[0]))
    for model in models[1:]:
        if type(model) is not base_class:
            raise TypeError("All Pydantic models must be instances of the same class.")
    return base_class


def pydantic_models_to_table(
    models: BaseModelIterable,
    *,
    allow_losing_tz: bool = False,
    exclude_fields: bool = False,
    by_alias: bool = False,
    model_dump_kwargs: Mapping[str, Any] | None = None,
) -> pa.Table:
    """
    Convert Pydantic model instances into a PyArrow table.

    Args:
        models: A single Pydantic model or an iterable of models.
        allow_losing_tz: Whether timezone-aware datetimes may be converted to naive UTC.
        exclude_fields: Whether to respect ``Field(exclude=True)`` markers.
        by_alias: Whether to serialize using field aliases.
        model_dump_kwargs: Additional keyword arguments passed to ``model_dump``.

    Returns:
        A ``pyarrow.Table`` containing one row per model.

    Raises:
        TypeError: If instances are heterogeneous or not Pydantic models.
        ValueError: If no models are provided.
        SchemaCreationError: If the schema cannot be derived from the Pydantic model.
    """
    normalized_models: List[PydanticModelT] = _coerce_models(models)
    model_class: Type[PydanticModelT] = _require_homogeneous_models(normalized_models)

    schema: pa.Schema = get_pyarrow_schema(
        model_class,
        allow_losing_tz=allow_losing_tz,
        exclude_fields=exclude_fields,
        by_alias=by_alias,
    )

    dump_kwargs: Dict[str, Any] = {"mode": "python"}
    if model_dump_kwargs is not None:
        dump_kwargs.update(model_dump_kwargs)
    dump_kwargs.setdefault("by_alias", by_alias)

    sanitized_rows: List[Dict[str, Any]] = [
        _sanitize_row_for_schema(
            cast(Dict[str, Any], model.model_dump(**dump_kwargs)),
            schema,
        )
        for model in normalized_models
    ]

    try:
        return pa.Table.from_pylist(sanitized_rows, schema=schema)
    except pa.ArrowInvalid as error:  # pragma: no cover - pyarrow provides details
        raise SchemaCreationError(f"Failed to convert models to PyArrow table: {error}") from error


def pydantic_to_parquet(
    models: BaseModelIterable,
    destination: Path,
    *,
    compression: str | None = "snappy",
    allow_losing_tz: bool = False,
    exclude_fields: bool = False,
    by_alias: bool = False,
    model_dump_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """
    Serialize Pydantic model instances into a Parquet file.

    Args:
        models: A single Pydantic model or an iterable of models.
        destination: The destination path for the Parquet file.
        compression: Optional Parquet compression codec name.
        allow_losing_tz: Whether timezone-aware datetimes may lose timezone information.
        exclude_fields: Whether to respect ``Field(exclude=True)`` markers.
        by_alias: Whether to serialize using field aliases.
        model_dump_kwargs: Additional keyword arguments passed to ``model_dump``.
    """
    table: pa.Table = pydantic_models_to_table(
        models,
        allow_losing_tz=allow_losing_tz,
        exclude_fields=exclude_fields,
        by_alias=by_alias,
        model_dump_kwargs=model_dump_kwargs,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression=compression)


def _sanitize_row_for_schema(
    row: Mapping[str, Any],
    schema: pa.Schema,
) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for field in schema:
        sanitized[field.name] = _sanitize_value_for_type(
            row.get(field.name),
            field.type,
        )
    return sanitized


def _sanitize_value_for_type(value: Any, data_type: pa.DataType) -> Any:
    if value is None:
        return None

    if isinstance(value, BaseModel):
        model_data: Dict[str, Any] = value.model_dump(mode="python")
        return _sanitize_value_for_type(model_data, data_type)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if pa_types.is_struct(data_type):
        struct_type: pa.StructType = cast(pa.StructType, data_type)
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Expected mapping for struct field, received {type(value)!r}."
            )
        return {
            field.name: _sanitize_value_for_type(value.get(field.name), field.type)
            for field in struct_type
        }

    if pa_types.is_map(data_type):
        map_type: pa.MapType = cast(pa.MapType, data_type)
        items: IterableAlias[Tuple[Any, Any]]
        if isinstance(value, Mapping):
            items = value.items()
        else:
            items = cast(IterableAlias[Tuple[Any, Any]], value)
        entries: List[Dict[str, Any]] = []
        for key, item_value in items:
            entries.append(
                {
                    "key": _sanitize_value_for_type(
                        key,
                        map_type.key_type,
                    ),
                    "value": _sanitize_value_for_type(
                        item_value,
                        map_type.item_type,
                    ),
                }
            )
        return entries

    if pa_types.is_list(data_type) or pa_types.is_large_list(data_type):
        list_type: pa.ListType | pa.LargeListType = cast(
            pa.ListType | pa.LargeListType,
            data_type,
        )
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise TypeError(
                f"Expected iterable for list field, received {type(value)!r}."
            )
        return [
            _sanitize_value_for_type(item, list_type.value_type)
            for item in value
        ]

    if isinstance(value, (set, frozenset)):
        return list(value)

    if isinstance(value, tuple):
        return list(value)

    return value
