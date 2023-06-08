from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import nn

from merlin.dispatch.lazy import LazyDispatcher
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import ParallelBlock
from merlin.models.torch.container import BlockContainer
from merlin.models.torch.utils.schema_utils import SchemaTrackingMixin, input_schema, output_schema
from merlin.schema import ColumnSchema, Schema, Tags

Selection = Union[Schema, ColumnSchema, Callable[[Schema], Schema], Tags]
select = LazyDispatcher("selection")


class ExternalizeDispatch(LazyDispatcher):
    def __call__(self, module: nn.Module, selection: Selection) -> Tuple[nn.Module, nn.Module]:
        route = select(module, selection)
        externalized = self.externalize(module, selection, route)

        return externalized, route

    def externalize(self, module: nn.Module, selection: Selection, route: nn.Module, name=None):
        fn = self.dispatch(module)
        return fn(module, selection, route, name=name)


externalize = ExternalizeDispatch("externalize")


@select.register(Schema)
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
        if selection.name not in schema.column_names:
            return Schema()
        selected = Schema([schema[selection.name]])
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


class Selectable:
    """
    A mixin to allow to be selectable by schema.
    """

    def setup_schema(self, schema: Schema):
        """
        Setup the schema for this selectable.

        Parameters
        ----------
        schema : Schema
            The schema to setup.

        Returns
        -------
        Selectable
            Self for chaining.
        """
        self.schema = schema

        return self

    def select(self, selection: Selection) -> "Selectable":
        """
        Select a subset of the schema.

        This method should be overridden by subclasses.

        Parameters
        ----------
        selection : Selection
            The selection criteria. Can be a Schema, ColumnSchema, a callable that
            returns a Schema, or Tags.

        Returns
        -------
        Selectable
            A new selectable with the selected subset of the schema.

        Raises
        ------
        NotImplementedError
            If this method is not overridden by a subclass.
        """
        raise NotImplementedError()


class SelectKeys(nn.Module, Selectable, SchemaTrackingMixin):
    """Filter tabular data based on a defined schema.

    Example usage::

        >>> select_keys = mm.SelectKeys(Schema(["user_id", "item_id"]))
        >>> inputs = {
        ...     "user_id": torch.tensor([1, 2, 3]),
        ...     "item_id": torch.tensor([4, 5, 6]),
        ...     "other_key": torch.tensor([7, 8, 9]),
        ... }
        >>> outputs = select_keys(inputs)
        >>> print(outputs.keys())
        dict_keys(['user_id', 'item_id'])

    Parameters
    ----------
    schema : Schema, optional
        The schema to use for selection. Default is None.

    Attributes
    ----------
    col_names : list
        List of column names in the schema.
    """

    def __init__(self, schema: Optional[Schema] = None):
        super().__init__()
        self.column_names: List[str] = []
        if schema:
            self.setup_schema(schema)

    def setup_schema(self, schema: Schema):
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        self.schema = schema
        self.column_names = schema.column_names

    def select(self, selection: Selection) -> "SelectKeys":
        """Select a subset of the schema based on the provided selection.

        Parameters
        ----------
        selection : Selection
            The selection to apply to the schema.

        Returns
        -------
        SelectKeys
            A new SelectKeys instance with the selected schema.
        """

        return SelectKeys(select_schema(self.schema, selection))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Only keep the inputs that are present in the schema.

        Parameters
        ----------
        inputs : dict
            A dictionary of torch.Tensor objects.

        Returns
        -------
        dict
            A dictionary of torch.Tensor objects after selection.
        """

        outputs = {}

        for key, val in inputs.items():
            _key = key
            if key.endswith("__values"):
                _key = key[: -len("__values")]
            elif key.endswith("__offsets"):
                _key = key[: -len("__offsets")]

            if _key in self.column_names:
                outputs[key] = val

        return outputs

    def extra_repr(self) -> str:
        return f"{', '.join(self.column_names)}"

    def __bool__(self) -> bool:
        return bool(self.column_names)

    def __hash__(self):
        return hash(tuple(sorted(self.column_names)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, SelectKeys):
            return False

        return set(self.column_names) == set(other.column_names)


class SelectFeatures(nn.Module):
    def __init__(self, schema: Optional[Schema] = None):
        super().__init__()
        self.select_keys = SelectKeys(schema=schema)

    def setup_schema(self, schema: Schema):
        self.select_keys.setup_schema(schema)

    def select(self, selection: Selection) -> "SelectFeatures":
        schema = self.select_keys.select(selection).schema

        return SelectFeatures(schema)

    def forward(self, inputs, batch: Batch) -> Dict[str, torch.Tensor]:
        outputs = {}
        embeddings = self.select_keys(inputs, batch.features)

        for key, val in embeddings.items():
            if key.endswith("_embedding"):
                key = key.replace("_embedding", "")
            outputs[key] = val

        return outputs


BlockT = TypeVar("BlockT", bound=BlockContainer)


def _select_block(container: BlockT, selection: Selection) -> BlockT:
    outputs = []

    if not container.values:
        return container.__class__()

    first = container.values[0]
    selected_first = select_module(first, selection)
    if not selected_first:
        return container.__class__()
    if first == selected_first:
        return container

    outputs.append(selected_first)
    if len(container.values) > 1:
        for module in container.values[1:]:
            try:
                selected_module = select_module(module, selection)
            except ValueError:
                selected_module = None

            if not selected_module:
                schema = getattr(first, "schema", None)
                selected_schema = getattr(selected_first, "schema", None)
                if schema and selected_schema:
                    diff = set(schema.column_names) - set(selected_schema.column_names)
                    raise ValueError(
                        f"Cannot remove {diff} from {container} "
                        f"because {module} does not support selection. ",
                        "If it does, please implement a `select` method.",
                    )

                raise ValueError(f"Cannot select from {module}")

            outputs.append(selected_module)

    return container.__class__(*outputs, name=container._name)


SelectT = TypeVar("SelectT", bound=Selectable)
ParallelT = TypeVar("ParallelT", bound=ParallelBlock)


def _select_parallel_block(
    parallel: ParallelT,
    selection: Selection,
    to_return: Optional[ParallelT] = None,
) -> ParallelT:
    branches = {} if to_return is None else to_return.branches

    pre = parallel.pre
    if pre:
        selected = select_module(pre, selection)
        if not selected:
            return ParallelBlock()

        pre = selected

    for key, val in parallel.branches.items():
        selected = select_module(val, selection)
        if selected:
            branches[key] = selected

    output = to_return or parallel.__class__(branches)
    output.pre = pre

    if parallel.branches == output.branches:
        output.post = parallel.post

    return output


@select.register(nn.Module)
def select_module(module: SelectT, selection: Selection) -> SelectT:
    if hasattr(module, "select"):
        output = module.select(selection)
    elif isinstance(module, ParallelBlock):
        output = _select_parallel_block(module, selection)
    elif isinstance(module, BlockContainer):
        output = _select_block(module, selection)
    else:
        raise ValueError(f"Cannot select from {module}")

    try:
        in_schema = select_schema(input_schema(module), selection)
        to_exclude = (input_schema(module) - in_schema).column_names
        output._output_schema = module.output_schema().excluding_by_name(to_exclude)
    except (AttributeError, RuntimeError):
        pass

    return output


def _externalize_parallel(main, selection, route, name=None):
    output_branches = {}

    for branch_name, branch in main.branches.items():
        if branch_name in route:
            out = externalize.externalize(branch, selection, route[branch_name], name=branch_name)
            if out:
                output_branches[branch_name] = out
        else:
            output_branches[branch_name] = branch

    return main.replace(branches=output_branches)


@externalize.register(BlockContainer)
def _externalize_block(main, selection, route, name=None):
    if isinstance(main, ParallelBlock):
        return _externalize_parallel(main, selection, route=route, name=name)

    main_schema = input_schema(main)
    route_schema = input_schema(route)

    if main_schema == route_schema:
        out_schema = output_schema(main)
        if len(out_schema) == 1 and out_schema.first.name == "output":
            out_schema = Schema([out_schema.first.with_name(name)])

        if not out_schema:
            raise ValueError(f"No output schema found in {route}.")

        return SelectFeatures(out_schema)

    output = main.__class__()
    for i, module in enumerate(main):
        output.append(externalize.externalize(module, selection, route[i], name=name))

    return output


@externalize.register(SelectKeys)
def _externalize_select_keys(main, selection, route, name=None):
    main_schema = input_schema(main)
    route_schema = input_schema(route)
    diff = main_schema.excluding_by_name(route_schema.column_names)

    return SelectKeys(diff)
