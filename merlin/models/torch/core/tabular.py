#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import copy
from abc import ABC
from typing import Dict, List, Optional, Union

import torch

from merlin.models.torch.typing import TabularData, TensorOrTabularData
from merlin.models.torch.utils.torch_utils import calculate_batch_size_from_input_size
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.registry import Registry
from merlin.schema import Schema

tabular_aggregation_registry: Registry = Registry.class_registry("torch.tabular_aggregations")


class TabularAggregation(torch.nn.Module, ABC):
    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def forward(self, inputs: TabularData) -> torch.Tensor:
        raise NotImplementedError()

    def _expand_non_sequential_features(self, inputs: TabularData) -> TabularData:
        inputs_sizes = {k: v.shape for k, v in inputs.items()}
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(inputs_sizes)

        if len(seq_features_shapes) > 0:
            non_seq_features = set(inputs.keys()).difference(set(seq_features_shapes.keys()))
            for fname in non_seq_features:
                # Including the 2nd dim and repeating for the sequence length
                inputs[fname] = inputs[fname].unsqueeze(dim=1).repeat(1, sequence_length, 1)

        return inputs

    def _get_seq_features_shapes(self, inputs_sizes: Dict[str, torch.Size]):
        seq_features_shapes = dict()
        for fname, fshape in inputs_sizes.items():
            # Saves the shapes of sequential features
            if len(fshape) >= 3:
                seq_features_shapes[fname] = tuple(fshape[:2])

        sequence_length = 0
        if len(seq_features_shapes) > 0:
            if len(set(seq_features_shapes.values())) > 1:
                raise ValueError(
                    "All sequential features must share the same shape in the first two dims "
                    "(batch_size, seq_length): {}".format(seq_features_shapes)
                )

            sequence_length = list(seq_features_shapes.values())[0][1]

        return seq_features_shapes, sequence_length

    def _check_concat_shapes(self, inputs: TabularData):
        input_sizes = {k: v.shape for k, v in inputs.items()}
        if len(set(list([v[:-1] for v in input_sizes.values()]))) > 1:
            raise Exception(
                "All features dimensions except the last one must match: {}".format(input_sizes)
            )

    def _get_agg_output_size(self, input_size, agg_dim):
        batch_size = calculate_batch_size_from_input_size(input_size)
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(input_size)

        if len(seq_features_shapes) > 0:
            return (
                batch_size,
                sequence_length,
                agg_dim,
            )

        else:
            return (batch_size, agg_dim)

    @classmethod
    def parse(cls, class_or_str):
        return tabular_aggregation_registry.parse(class_or_str)


TabularAggregationType = Union[str, TabularAggregation]


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `forward`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `forward`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `forward`-method to output a single Tensor.
"""


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
class TabularBlock(torch.nn.Module):
    """PyTorch Module that's specialized for tabular-data by integrating many often used operations.

    Parameters
    ----------
    {tabular_module_parameters}
    """

    def __init__(
        self,
        pre: Optional[torch.nn.Module] = None,
        post: Optional[torch.nn.Module] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = None
        self.pre = pre  # type: ignore
        self.post = post  # type: ignore
        self.aggregation = aggregation  # type: ignore

    @classmethod
    def from_schema(cls, schema: Schema, tags=None, **kwargs) -> Optional["TabularBlock"]:
        """Instantiate a TabularModule instance from a DatasetSchema.

        Parameters
        ----------
        schema
        tags
        kwargs

        Returns
        -------
        Optional[TabularModule]
        """
        schema_copy = copy.copy(schema)
        if tags:
            schema_copy = schema_copy.select_by_tag(tags)

        if not schema_copy.column_schemas:
            return None

        return cls.from_features(schema_copy.column_names, schema=schema_copy, **kwargs)

    @classmethod
    @docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING, extra_padding=4)
    def from_features(
        cls,
        features: List[str],
        pre: Optional[torch.nn.Module] = None,
        post: Optional[torch.nn.Module] = None,
        aggregation: Optional[TabularAggregationType] = None,
    ) -> "TabularBlock":
        """Initializes a TabularModule instance where the contents of features will be filtered
            out

        Parameters
        ----------
        features: List[str]
            A list of feature-names that will be used as the first pre-processing op to filter out
            all other features not in this list.
        {tabular_module_parameters}

        Returns
        -------
        TabularModule
        """
        pre = [Filter(features), pre] if pre else Filter(features)  # type: ignore

        return cls(pre=pre, post=post, aggregation=aggregation)

    @property
    def pre(self) -> Optional[torch.nn.Module]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._pre

    @pre.setter
    def pre(self, value: Optional[torch.nn.Module]):
        if value:
            self._pre: value
        else:
            self._pre = None

    @property
    def post(self) -> Optional[torch.nn.Module]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._post

    @post.setter
    def post(self, value: Optional[torch.nn.Module]):
        if value:
            self._post: value
        else:
            self._post = None

    @property
    def aggregation(self) -> Optional[TabularAggregation]:
        """

        Returns
        -------
        TabularAggregation, optional
        """
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Optional[Union[str, TabularAggregation]]):
        """

        Parameters
        ----------
        value
        """
        if value:
            self._aggregation: Optional[TabularAggregation] = TabularAggregation.parse(value)
        else:
            self._aggregation = None

    def pre_forward(
        self, inputs: TabularData, transformations: Optional[torch.nn.Module] = None
    ) -> TabularData:
        """Method that's typically called before the forward method for pre-processing.

        Parameters
        ----------
        inputs: TabularData
             input-data, typically the output of the forward method.
        transformations: TabularAggregationType, optional

        Returns
        -------
        TabularData
        """
        return self._maybe_apply_transformations(
            inputs, transformations=transformations or self.pre
        )

    def forward(self, x: TabularData, *args, **kwargs) -> TabularData:
        return x

    def post_forward(
        self,
        inputs: TabularData,
        transformations: Optional[torch.nn.Module] = None,
        merge_with: Optional[Union["TabularBlock", List["TabularBlock"]]] = None,
        aggregation: Optional[TabularAggregationType] = None,
    ) -> TensorOrTabularData:
        """Method that's typically called after the forward method for post-processing.

        Parameters
        ----------
        inputs: TabularData
            input-data, typically the output of the forward method.
        transformations: TabularTransformationType, optional
            Transformations to apply on the input data.
        merge_with: Union[TabularModule, List[TabularModule]], optional
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        _aggregation: Optional[TabularAggregation]
        if aggregation:
            _aggregation = TabularAggregation.parse(aggregation)
        else:
            _aggregation = getattr(self, "aggregation", None)

        outputs = inputs
        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        outputs = self._maybe_apply_transformations(
            outputs, transformations=transformations or self.post
        )

        if _aggregation:
            schema = getattr(self, "_schema", None)
            _aggregation.set_schema(schema)
            return _aggregation(outputs)

        return outputs

    def __call__(
        self,
        inputs: TabularData,
        *args,
        pre: Optional[torch.nn.Module] = None,
        post: Optional[torch.nn.Module] = None,
        merge_with: Optional[Union["TabularBlock", List["TabularBlock"]]] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwargs,
    ) -> TensorOrTabularData:
        """We overwrite the call method in order to be able to do pre- and post-processing.

        Parameters
        ----------
        inputs: TabularData
            Input TabularData.
        pre: TabularTransformationType, optional
            Transformations to apply before calling the forward method. If pre is None, this method
            will check if `self.pre` is set.
        post: TabularTransformationType, optional
            Transformations to apply after calling the forward method. If post is None, this method
            will check if `self.post` is set.
        merge_with: Union[TabularModule, List[TabularModule]]
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        inputs = self.pre_forward(inputs, transformations=pre)

        # This will call the `forward` method implemented by the super class.
        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if isinstance(outputs, dict):
            outputs = self.post_forward(
                outputs, transformations=post, merge_with=merge_with, aggregation=aggregation
            )

        return outputs

    def _maybe_apply_transformations(
        self,
        inputs: TabularData,
        transformations: Optional[torch.nn.Module] = None,
    ) -> TabularData:
        """Apply transformations to the inputs if these are defined.

        Parameters
        ----------
        inputs
        transformations

        Returns
        -------

        """
        if transformations:
            return transformations(inputs)

        return inputs


class Filter(TabularBlock):
    """Module that filters out certain features from `TabularData`."

    Parameters
    ----------
    to_include: List[str]
        List of features to include in the result of calling the module
    pop: bool
        Boolean indicating whether to pop the features to exclude from the inputs dictionary.
    """

    def __init__(self, to_include: List[str], pop: bool = False):
        super().__init__()
        self.to_include = to_include
        self.pop = pop

    def forward(self, inputs: TabularData, **kwargs) -> TabularData:
        """

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def forward_output_size(self, input_shape):
        """

        Parameters
        ----------
        input_shape

        Returns
        -------

        """
        return {k: v for k, v in input_shape.items() if k in self.to_include}


class AsTabular(TabularBlock):
    """Converts a Tensor to TabularData by converting it to a dictionary.

    Parameters
    ----------
    output_name: str
        Name that should be used as the key in the output dictionary.
    """

    def __init__(self, output_name: str):
        super().__init__()
        self.output_name = output_name

    def forward(self, inputs: torch.Tensor, **kwargs) -> TabularData:  # type: ignore
        return {self.output_name: inputs}

    def forward_output_size(self, input_size):
        return {self.output_name: input_size}
