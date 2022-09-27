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

import logging
from typing import Callable, Dict, Optional, Union

from tensorflow.keras.layers import Layer

from merlin.models.tf.core.base import BlockType
from merlin.models.tf.core.combinators import ParallelBlock, TabularAggregationType
from merlin.models.tf.inputs.continuous import Continuous
from merlin.models.tf.inputs.embedding import Embeddings
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin-models")


INPUT_TAG_TO_BLOCK: Dict[Tags, Callable[[Schema], Layer]] = {
    Tags.CONTINUOUS: Continuous,
    Tags.CATEGORICAL: Embeddings,
}


def InputBlockV2(
    schema: Optional[Schema] = None,
    categorical: Union[Tags, Layer] = Tags.CATEGORICAL,
    continuous: Union[Tags, Layer] = Tags.CONTINUOUS,
    pre: Optional[BlockType] = None,
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = "concat",
    tag_to_block=INPUT_TAG_TO_BLOCK,
    **branches,
) -> ParallelBlock:
    """The entry block of the model to process input features from a schema.

    This is a new version of InputBlock, which is more flexible for accepting
    the external definition of `embeddings` block. After `22.10` this will become the default.

    Simple Usage::
        inputs = InputBlockV2(schema)

    Custom Embeddings::
        inputs = InputBlockV2(
            schema,
            categorical=Embeddings(schema, dim=32)
        )

    Sparse outputs for one-hot::
        inputs = InputBlockV2(
            schema,
            categorical=CategoryEncoding(schema, sparse=True),
            post=ToSparse()
        )

    Add continuous projection::
        inputs = InputBlockV2(
            schema,
            continuous=ContinuousProjection(continuous_schema, MLPBlock([32])),
        )

    Merge 2D and 3D (for session-based)::
        inputs = InputBlockV2(
            schema,
            post=BroadcastToSequence(context_schema, sequence_schema)
        )


    Parameters
    ----------
    schema : Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    categorical : Union[Tags, Layer], defaults to `Tags.CATEGORICAL`
        A block or column-selector to use for categorical-features.
        If a column-selector is provided (either a schema or tags), the selector
        will be passed to `Embeddings` to infer the embedding tables from the column-selector.
    continuous : Union[Tags, Layer], defaults to `Tags.CONTINUOUS`
        A block to use for continuous-features.
        If a column-selector is provided (either a schema or tags), the selector
        will be passed to `Continuous` to infer the features from the column-selector.
    pre : Optional[BlockType], optional
        Transformation block to apply before the embeddings lookup, by default None
    post : Optional[BlockType], optional
        Transformation block to apply after the embeddings lookup, by default None
    aggregation : Optional[TabularAggregationType], optional
        Transformation block to apply for aggregating the inputs, by default "concat"
    tag_to_block : Dict[str, Callable[[Schema], Layer]], optional
        Mapping from tag to block-type, by default:
            Tags.CONTINUOUS -> Continuous
            Tags.CATEGORICAL -> Embeddings
    **branches : dict
        Extra branches to add to the input block.

    Returns
    -------
    ParallelBlock
        Returns a ParallelBlock with a Dict with two branches:
        continuous and embeddings
    """

    if schema:
        input_schema = schema.remove_by_tag(Tags.TARGET)

    unparsed = {"categorical": categorical, "continuous": continuous, **branches}
    parsed = {}
    for name, branch in unparsed.items():
        if isinstance(branch, Layer):
            parsed[name] = branch
        else:
            if not isinstance(schema, Schema):
                raise ValueError(
                    "If you pass a column-selector as a branch, "
                    "you must also pass a `schema` argument."
                )
            if branch not in tag_to_block:
                raise ValueError(f"No default-block provided for {branch}")
            branch_schema: Schema = input_schema.select_by_tag(branch)
            parsed[name] = tag_to_block[branch](branch_schema)

    if not parsed:
        raise ValueError("No columns selected for the input block")

    return ParallelBlock(
        parsed,
        pre=pre,
        post=post,
        aggregation=aggregation,
        is_input=True,
        schema=schema,
    )
