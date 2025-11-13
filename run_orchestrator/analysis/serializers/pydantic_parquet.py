from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from enum import Enum, EnumMeta
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable as IterableAlias, List, Mapping, Tuple, Type, TypeVar, Union, cast, get_args, \
    get_origin

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pa_types
from pydantic import BaseModel
from pydantic.fields import ComputedFieldInfo
from pydantic_to_pyarrow import SchemaCreationError, get_pyarrow_schema
from pydantic_to_pyarrow.schema import BaseModelType, Settings, _is_optional, _get_pyarrow_type, FIELD_MAP, \
    _get_uuid_type, LOSING_TZ_TYPES, _get_enum_type, TYPES_WITH_METADATA, FIELD_TYPES

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

def _get_pyarrow_type_with_computed_fields(  # noqa: PLR0911
    field_type: Type[Any],
    metadata: List[Any],
    settings: Settings,
) -> pa.DataType:
    if field_type in FIELD_MAP:
        return FIELD_MAP[field_type]

    if field_type is uuid.UUID:
        return _get_uuid_type()

    if settings.allow_losing_tz and field_type in LOSING_TZ_TYPES:
        return LOSING_TZ_TYPES[field_type]

    if not settings.allow_losing_tz and field_type in LOSING_TZ_TYPES:
        raise SchemaCreationError(
            f"{field_type} only allowed if ok losing timezone information"
        )

    if isinstance(field_type, EnumMeta):
        return _get_enum_type(field_type)

    if field_type in TYPES_WITH_METADATA:
        return TYPES_WITH_METADATA[field_type](metadata)

    if get_origin(field_type) in FIELD_TYPES:
        return FIELD_TYPES[get_origin(field_type)](
            field_type,
            metadata,
            settings,
        )

    # isinstance(filed_type, type) checks whether it's a class
    # otherwise eg Deque[int] would casue an exception on issubclass
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return _get_pyarrow_schema_with_computed_fields(field_type, settings, as_schema=False)

    raise SchemaCreationError(f"Unknown type: {field_type}")


def _get_pyarrow_schema_with_computed_fields(
    pydantic_class: Type[BaseModelType],
    settings: Settings,
    as_schema: bool = True,
) -> pa.Schema:
    fields = []

    for name, field_info in chain(
        pydantic_class.model_fields.items(),
        pydantic_class.model_computed_fields.items(),
    ):
        is_computed = isinstance(field_info, ComputedFieldInfo)

        if not is_computed and field_info.exclude and settings.exclude_fields:
            continue

        if is_computed:
            field_type = field_info.return_type
            metadata: List[Any] = []
        else:
            field_type = field_info.annotation
            metadata = field_info.metadata

        if field_type is None:  # pragma: no cover
            # Not sure how to get here through pydantic, hence nocover
            field_kind = "computed field" if is_computed else "field"
            raise SchemaCreationError(f"Missing type for {field_kind} {name}")

        try:
            nullable = False
            if _is_optional(field_type):
                nullable = True
                types_under_union = list(set(get_args(field_type)) - {type(None)})
                # mypy infers field_type as Type[Any] | None here, hence casting
                field_type = cast(Type[Any], types_under_union[0])

            pa_field = _get_pyarrow_type_with_computed_fields(field_type, metadata, settings)
        except Exception as err:  # noqa: BLE001 - ignore blind exception
            field_kind = "computed field" if is_computed else "field"
            raise SchemaCreationError(
                f"Error processing {field_kind} {name}: {field_type}, {err}"
            ) from err

        serialized_name = name
        if settings.by_alias:
            alias = (
                field_info.alias
                if is_computed
                else field_info.serialization_alias
            )
            if alias is not None:
                serialized_name = alias

        fields.append(pa.field(serialized_name, pa_field, nullable=nullable))

    if as_schema:
        return pa.schema(fields)

    return pa.struct(fields)

def get_pyarrow_schema_with_computed_fields(
    pydantic_class: Type[BaseModelType],
    allow_losing_tz: bool = False,
    exclude_fields: bool = False,
    by_alias: bool = False,
) -> pa.Schema:
    """
    Converts a Pydantic model into a PyArrow schema.

    Args:
        pydantic_class (Type[BaseModelType]): The Pydantic model class to convert.
        allow_losing_tz (bool, optional): Whether to allow losing timezone information
            when converting datetime fields. Defaults to False.
        exclude_fields (bool, optional): If True, will exclude fields in the pydantic
            model that have `Field(exclude=True)`. Defaults to False.
        by_alias (bool, optional): If True, will create the pyarrow schema using the
            (serialization) alias in the pydantic model. Defaults to False.

    Returns:
        pa.Schema: The PyArrow schema representing the Pydantic model.
    """
    settings = Settings(
        allow_losing_tz=allow_losing_tz,
        by_alias=by_alias,
        exclude_fields=exclude_fields,
    )
    return _get_pyarrow_schema_with_computed_fields(pydantic_class, settings)

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

    schema: pa.Schema = get_pyarrow_schema_with_computed_fields(
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
