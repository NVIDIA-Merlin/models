#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import collections
import inspect
from typing import Callable, Dict, Final, Mapping, Optional, Tuple, Type, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import ParallelBlock
from merlin.models.torch.schema import NAMESPACE_TAGS, Selectable, Selection, select
from merlin.models.utils.schema_utils import get_embedding_size_from_cardinality
from merlin.schema import ColumnSchema, Schema, Tags

DimFn = Callable[[Schema], int]
Combiner = Union[str, nn.Module]


@torch.jit.script
class Domain:
    def __init__(self, min: Union[int, float], max: Union[int, float], name: str) -> None:
        self.min: Union[int, float] = min
        self.max: Union[int, float] = max
        self.name: str = name


def infer_embedding_dim(schema: Schema, multiplier: float = 2.0, ensure_multiple_of_8: bool = True):
    cardinality = sum(col.int_domain.max for col in schema) + 1

    return get_embedding_size_from_cardinality(
        cardinality, multiplier=multiplier, ensure_multiple_of_8=ensure_multiple_of_8
    )


class EmbeddingTable(nn.Module, Selectable):
    """Embedding-table module.

    This can hold either a single feature or multiple features.

    When the table holds multiple features from different domains,
    we stack the embeddings of the features on top of each other.
    In that case, we will shift the inputs accordingly.

    Attributes:
        has_module_combiner (bool): Whether a module combiner is used or not.

    Args:
        dim (Union[int, DimFn]): The dimension for the embedding table.
        schema (Optional[Union[ColumnSchema, Schema]]): The schema for the embedding table.
        seq_combiner (Optional[Combiner]): The combiner used to combine the embeddings.
    """

    has_combiner: Final[bool] = False
    has_module_combiner: Final[bool] = False

    def __init__(
        self,
        dim: Union[int, DimFn] = infer_embedding_dim,
        schema: Optional[Union[ColumnSchema, Schema]] = None,
        seq_combiner: Optional[Combiner] = None,
    ):
        super().__init__()
        self.dim = dim
        self.seq_combiner = seq_combiner
        self.domains: Mapping[str, Domain] = collections.OrderedDict()
        self.feature_to_domain: Dict[str, Domain] = {}
        self.has_combiner = self.seq_combiner is not None
        self.has_module_combiner = isinstance(self.seq_combiner, nn.Module)
        self.num_embeddings = 0
        self.input_schema = None
        if schema:
            self.initialize_from_schema(schema or Schema())
            self._initialized_from_schema = True

    def initialize_from_schema(self, schema: Schema):
        """
        Sets up the schema for the embedding table.

        Args:
            schema (Schema): The schema to setup.
        """

        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        self.input_schema = Schema()
        for col in schema:
            self.add_feature(col)
        if self.input_schema:
            self.table = self.create_table()

    def create_table(self) -> nn.Module:
        """
        Creates the table that holds the embeddings.

        Returns:
            nn.Module: The embedding table.
        """

        if callable(self.dim):
            self.dim = self.dim(self.input_schema)

        return nn.Embedding(self.num_embeddings, self.dim)

    def insert_rows(self, num_rows: int, start_index: int = 0):
        """
        Inserts rows into the embedding table.

        Args:
            num_rows (int): The number of rows to insert.
            start_index (int): The index to start the insertion.
        """

        device = self.table.weight.device
        new_embedding = type(self.table)(num_rows, self.dim).to(device)

        with torch.no_grad():
            # Slice the original weight tensor into two parts: before and after the start index
            weight_before = self.table.weight[:start_index]
            weight_after = self.table.weight[start_index:]

            # Concatenate these parts with the new embedding's weight
            new_weight = torch.cat([weight_before, new_embedding.weight, weight_after])
            new_embedding.weight = torch.nn.Parameter(new_weight)

        self.table = new_embedding

    def append_rows(self, num_rows: int):
        """
        Appends rows to the embedding table.

        Args:
            num_rows (int): The number of rows to append.

        Returns:
            EmbeddingTable: Updated EmbeddingTable instance with the appended rows.
        """

        self.insert_rows(num_rows, self.num_embeddings)

        return self

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        batch: Optional[Batch] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Looking up the embeddings for the given input(s).

        Args:
            inputs (Union[torch.Tensor, Dict[str, torch.Tensor]]): The inputs to the EmbeddingTable.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                The embeddings for the given input(s).
        """
        if len(self.domains) == 0:
            raise RuntimeError("Table is not initialized, please add features to the table")

        if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            return self._forward_dict(inputs)
        elif torch.jit.isinstance(inputs, torch.Tensor):
            return self.forward_tensor(inputs)

        raise ValueError(f"Unsupported input type: {type(inputs)}")

    def forward_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.table(ids)
        if len(out.shape) > 2 and self.has_module_combiner:
            out = self.seq_combiner(out)

        return out

    def _forward_dict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with input as a dictionary.

        When multiple features are passed in, we try to minimize
        the number of times we call the table for performance reasons.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output tensors.
        """

        dense_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        sparse_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for feature_name in sorted(list(self.feature_to_domain.keys())):
            if feature_name in inputs:
                tensor = inputs[feature_name]
                domain = self.feature_to_domain[feature_name]
                if domain.min > 1:
                    tensor = tensor + domain.min

                if len(tensor.shape) not in dense_tensors:
                    dense_tensors[len(tensor.shape)] = {}

                dense_tensors[len(tensor.shape)][feature_name] = tensor
            elif feature_name + "__values" in inputs and feature_name + "__offsets" in inputs:
                values = inputs[feature_name + "__values"]
                offsets = inputs[feature_name + "__offsets"]
                sparse_tensors[feature_name] = (values, offsets)

        out = {}

        for num_dims, tensor_dict in dense_tensors.items():
            tensors = torch.cat(list(tensor_dict.values()))

            if num_dims > 1 and self.has_combiner and not self.has_module_combiner:
                stacked = self.forward_bag(tensors)
            else:
                stacked = self.forward_tensor(tensors)

            if len(tensor_dict) == 1:
                out[list(tensor_dict.keys())[0]] = stacked
            else:
                i = 0
                for name, tensor in tensor_dict.items():
                    out[name] = stacked[i : i + tensor.shape[0]]
                    i += tensor.shape[0]

        if sparse_tensors:
            values, offsets = [], []
            extra_offset = 0
            for val in sparse_tensors.values():
                values.append(val[0])
                offsets.append(val[1][:-1] + extra_offset)
                extra_offset += val[0].shape[0]

            if self.has_combiner and not self.has_module_combiner:
                bags = self.forward_bag(torch.cat(values), torch.cat(offsets))

                if len(sparse_tensors) == 1:
                    out[list(sparse_tensors.keys())[0]] = bags
                else:
                    i = 0
                    for feature_name, (_, feat_offsets) in sparse_tensors.items():
                        num = feat_offsets.shape[0] - 1
                        out[feature_name] = bags[i : i + num]
                        i += num
            else:
                raise NotImplementedError("Sparse tensors not supported without combiner")

        return out

    def forward_bag(
        self,
        inputs: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ):
        """
        Computes the forward pass for an embedding bag.

        An embedding bag is a method of compressing sparse matrices in order to
        speed up certain operations. It is particularly useful when dealing with
        sparse matrices where each row contains few non-zero elements.

        Args:
            inputs (torch.Tensor): The input tensor to be transformed by the embedding bag.
                This tensor should be a 2D tensor of shape (num_instances, num_features),
                where each row corresponds to a single instance, and each column corresponds
                to a feature.
            offsets (Optional[torch.Tensor]): A tensor containing the offsets for each bag in
                the input tensor. This tensor should have the same size (num_instances) as the
                first dimension of inputs. If not provided, it is assumed that the input tensor
                only contains a single bag.

        Returns:
            torch.Tensor: The transformed input tensor, with embeddings applied and
                potentially combined according to the mode of operation of the embedding bag.
        """

        return nn.functional.embedding_bag(
            inputs, weight=self.table.weight, offsets=offsets, mode=self.seq_combiner
        )

    def add_feature(self, col_schema: ColumnSchema) -> "EmbeddingTable":
        """Add a feature to the table.

        Args:
            col_schema (ColumnSchema): The column schema of the feature to add.

        Returns:
            EmbeddingTable: The updated module with the added feature.
        """
        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have an int-domain")

        domain = col_schema.int_domain
        to_add = 0
        if domain.name not in self.domains:
            self.domains[domain.name] = Domain(
                min=domain.min + self.num_embeddings,
                max=domain.max + self.num_embeddings,
                name=domain.name,
            )
            to_add += domain.max

        self.feature_to_domain[col_schema.name] = self.domains[domain.name]
        self.input_schema += Schema([col_schema])

        if self.num_embeddings == 0:
            self.num_embeddings = 1
        self.num_embeddings += to_add
        if hasattr(self, "table"):
            self.append_rows(to_add)

        return self

    def update_feature(self, col_schema: ColumnSchema) -> "EmbeddingTable":
        """Update a feature in the table.

        Args:
            col_schema (ColumnSchema): The new column schema of the feature.

        Returns:
            EmbeddingTable: The updated module with the updated feature.
        """
        feature_name = col_schema.name

        # Check if the feature to update exists in the table
        if feature_name not in self.feature_to_domain:
            raise ValueError(f"Feature '{feature_name}' not found in the table")

        if not col_schema.int_domain:
            raise ValueError("`new_col_schema` needs to have an int-domain")

        new_domain = col_schema.int_domain

        # Calculate the difference between the new and old max values
        old_max = self.feature_to_domain[feature_name].max
        diff = new_domain.max - old_max

        if diff < 0:
            raise ValueError("New domain cannot have fewer embeddings than the old domain")

        # Update the domain of the feature
        self.domains[new_domain.name] = Domain(
            min=self.feature_to_domain[feature_name].min,
            max=new_domain.max,
            name=new_domain.name,
        )
        for feature in self.feature_to_domain:
            if self.feature_to_domain[feature].name == new_domain.name:
                self.feature_to_domain[feature] = self.domains[new_domain.name]

        # Shift the min and max values of the subsequent domains
        for domain in self.domains.values():
            if domain.min > old_max:
                domain.min += diff
                domain.max += diff

        self.input_schema[feature_name] = col_schema

        # Update the total number of embeddings
        self.num_embeddings += diff
        if hasattr(self, "table"):
            self.insert_rows(diff, old_max)

        return self

    def feature_weights(self, name: str):
        if name not in self.domains:
            raise ValueError(f"{name} not found in table: {self}")

        domain = self.domains[name]

        return self.table.weight[int(domain.min) : int(domain.max)]

    def select(self, selection: Selection) -> Selectable:
        selected = select(self.input_schema, selection)

        if not selected:
            raise ValueError(f"Selection {selection} not found in the table")

        return self

    @torch.jit.ignore
    def output_schema(self) -> Schema:
        output = Schema()

        for col in self.input_schema:
            dims = (None, self.dim) if self.seq_combiner else (None, None, self.dim)
            tags = [Tags.EMBEDDING]

            # Namespace tags in the input schema should be propagated to the output schema
            for tag in NAMESPACE_TAGS:
                if tag in col.tags:
                    tags.append(tag)

            name = col.name + "_embedding"
            output[name] = ColumnSchema(name, dims=dims, tags=tags)

        return output

    def contains_multiple_domains(self) -> bool:
        """
        Check if the module contains multiple domains.

        Returns:
            bool: True if the module contains multiple domains, False otherwise.
        """
        return len(self.domains) > 1

    def table_name(self) -> str:
        """Get the table name.

        Returns:
            str: The name of the table.
        """
        return "_".join([domain.name for domain in self.domains.values()])

    def extra_repr(self) -> str:
        return f"features: {', '.join(self.feature_to_domain.keys())}"

    def __bool__(self) -> bool:
        """Check if the module is valid.

        Returns:
            bool: True if the table contains features, False otherwise.
        """
        return bool(self.input_schema)


class EmbeddingTables(ParallelBlock, Selectable):
    """Parallel-block of embedding tables.

    Args:
        dim (Optional[Union[Dict[str, int], int, DimFn]]): The dimensions for the embedding tables.
        schema (Optional[Schema]): The schema for the embedding tables.
        table_cls (Type[nn.Module]): The type of embedding table to create.
        seq_combiner (Optional[Combiner]): The combiner used to combine the embeddings.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        dim: Optional[Union[Dict[str, int], int, DimFn]] = infer_embedding_dim,
        schema: Optional[Schema] = None,
        table_cls: Type[nn.Module] = EmbeddingTable,
        seq_combiner: Optional[Combiner] = None,
        **kwargs,
    ):
        super().__init__(kwargs.pop("tables", {}))
        self.dim = dim
        self.table_cls = table_cls
        self.seq_combiner = seq_combiner
        self.kwargs = kwargs
        if isinstance(schema, Schema):
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True

    def initialize_from_schema(self, schema: Schema):
        """Initializes the module from a schema.
        Called during the schema tracing of the model.

        Parameters
        ----------
        schema : Schema
            The schema to initialize with
        """
        self.schema = schema

        for col in schema:
            if col.int_domain.name in self.branches:
                self.branches[col.int_domain.name][0].add_feature(col)
            else:
                raw_kwargs = {
                    "dim": self.dim,
                    "schema": col,
                    "seq_combiner": self.seq_combiner,
                    **self.kwargs,
                }
                kwargs = _forward_kwargs_to_table(col, self.table_cls, raw_kwargs)
                self.branches[col.int_domain.name] = self.table_cls(**kwargs)

        return self

    def select(self, selection: Selection) -> "EmbeddingTables":
        """
        Selects a subset of the embedding tables based on the given selection.

        Args:
            selection (Selection): The selection criteria to apply.

        Returns:
            EmbeddingTables: The selected subset of the embedding tables.
        """
        selected_branches = {}
        for key, val in self.branches.items():
            try:
                selected = select(val, selection)
                if selected:
                    selected_branches[key] = selected
            except ValueError:
                pass

        output = EmbeddingTables(
            dim=self.dim,
            schema=select(self.schema, selection),
            table_cls=self.table_cls,
            seq_combiner=self.seq_combiner,
            tables=selected_branches,
            **self.kwargs,
        )

        return output

    def extra_repr(self) -> str:
        return ", ".join(list(self.branches.keys()))


def _forward_kwargs_to_table(col, table_cls, kwargs):
    arg_spec = inspect.getfullargspec(table_cls.__init__)
    supported_kwargs = arg_spec.kwonlyargs
    if arg_spec.defaults:
        supported_kwargs += arg_spec.args[-len(arg_spec.defaults) :]

    table_kwargs = {}
    for key, val in kwargs.items():
        if key in supported_kwargs:
            if isinstance(val, dict):
                if col.name in val:
                    table_kwargs[key] = val[col.name]
            else:
                table_kwargs[key] = val

    return table_kwargs
