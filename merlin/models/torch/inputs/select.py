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

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.batch import Batch
from merlin.schema import ColumnSchema, Schema, Tags


class SelectKeys(nn.Module, schema.Selectable, schema.LazySchemaModuleMixin):
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

    def __init__(self, schema: Optional[Union[Schema, ColumnSchema]] = None):
        super().__init__()
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])
        if schema:
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True
        else:
            schema = Schema()
            self.schema = schema
            self.column_names: List[str] = schema.column_names

    def initialize_from_schema(self, schema: Schema):
        super().initialize_from_schema(schema)
        self.schema = schema
        self.column_names = schema.column_names
        self.input_schema = schema
        self.output_schema = schema

    def select(self, selection: schema.Selection) -> "SelectKeys":
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

        return SelectKeys(schema.select(self.schema, selection))

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


class SelectFeatures(nn.Module, schema.LazySchemaModuleMixin):
    """Filter tabular data based on a defined schema.

    It operates similarly to SelectKeys, but it uses the features from Batch.
    This is useful when you want to select raw-features from anywhere in the model.

    Example usage::

        >>> select_features = mm.SelectFeatures(Schema(["user_id", "item_id"]))
        >>> inputs = {
        ...     "user_id_embedding": torch.tensor([1.0, 2.0, 3.0]),
        ...     "item_id_embedding": torch.tensor([4.0, 5.0, 6.0]),
        ...     "other_key": torch.tensor([7, 8, 9]),
        ... }
        >>> batch = Batch(inputs)
        >>> outputs = select_features(inputs, batch)
        >>> print(outputs.keys())
        dict_keys(['user_id', 'item_id'])

    Parameters
    ----------
    schema : Schema, optional
        The schema to use for selection. Default is None.
    """

    def __init__(self, schema: Optional[Schema] = None):
        super().__init__()
        self.select_keys = SelectKeys(schema=schema)
        if schema:
            self.initialize_from_schema(schema)

    def initialize_from_schema(self, schema: Schema):
        """Set up the schema for the SelectFeatures.

        Parameters
        ----------
        schema : Schema
            The schema to use for selection.
        """
        super().initialize_from_schema(schema)
        self.select_keys.initialize_from_schema(schema)
        self.embedding_names = schema.select_by_tag(Tags.EMBEDDING).column_names
        self.input_schema = Schema()
        self.output_schema = schema

    def select(self, selection: schema.Selection) -> "SelectFeatures":
        """Select a subset of the schema based on the provided selection.

        Parameters
        ----------
        selection : Selection
            The selection to apply to the schema.

        Returns
        -------
        SelectFeatures
            A new SelectFeatures instance with the selected schema.
        """
        schema = self.select_keys.select(selection).schema

        return SelectFeatures(schema)

    def compute_feature_schema(self, feature_schema: Schema) -> Schema:
        return feature_schema[self.select_keys.column_names]

    def forward(self, inputs, batch: Batch) -> Dict[str, torch.Tensor]:
        outputs = {}
        selected = self.select_keys(batch.features)

        for key, val in selected.items():
            if key in self.embedding_names and key.endswith("_embedding"):
                key = key.replace("_embedding", "")
            outputs[key] = val

        return outputs


@schema.extract.register(SelectKeys)
def _(main, selection, route, name=None):
    main_schema = schema.input_schema(main)
    route_schema = schema.input_schema(route)

    diff = main_schema.excluding_by_name(route_schema.column_names)

    return SelectKeys(diff)
