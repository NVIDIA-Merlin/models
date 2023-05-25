from typing import Callable, Union

from merlin.schema import ColumnSchema, Schema, Tags

Selection = Union[Schema, ColumnSchema, Callable[[Schema], Schema], Tags]


def select_schema(schema: Schema, selection: Selection) -> Schema:
    """
    Select a subset of a schema based on the selection criteria.

    Parameters
    ----------
    schema : Schema
        The original schema to select from.
    selection : Selection
        The selection criteria. Can be a Schema, ColumnSchema,
        a callable that returns a Schema, or Tags.

    Returns
    -------
    Schema
        The selected subset of the schema.

    Raises
    ------
    ValueError
        If the selection criteria is not a valid type.
    """
    if not isinstance(schema, Schema):
        raise ValueError(f"Schema {schema} is not valid")

    if isinstance(selection, Schema):
        selected = selection
    elif isinstance(selection, ColumnSchema):
        selected = schema[selection.name]
    elif callable(selection):
        selected = selection(schema)
    elif isinstance(selection, Tags):
        selected = schema.select_by_tag(selection)
    else:
        raise ValueError(f"Selection {selection} is not valid")

    return selected


def selection_name(selection: Selection) -> str:
    """
    Get the name of the selection.

    Parameters
    ----------
    selection : Selection
        The selection criteria. Can be a Schema, ColumnSchema, a callable
        that returns a Schema, or Tags.

    Returns
    -------
    str
        The name of the selection.

    Raises
    ------
    ValueError
        If the selection criteria is not a valid type.
    """
    if isinstance(selection, ColumnSchema):
        return selection.name
    elif isinstance(selection, Tags):
        return selection.value
    elif isinstance(selection, Schema):
        return "_".join(selection.column_names)
    elif callable(selection):
        return selection.__name__

    raise ValueError(f"Selection {selection} is not valid")
