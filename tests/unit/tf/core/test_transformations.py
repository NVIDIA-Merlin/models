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

import tempfile

import tensorflow as tf

import merlin.models.tf as ml
from merlin.models.utils.schema_utils import create_categorical_column, create_continuous_column
from merlin.schema import Schema, Tags


def test_expand_dims_same_axis():
    NUM_ROWS = 100

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    inputs = {
        "cont_feat": tf.random.uniform((NUM_ROWS,)),
        "multi_hot_categ_feat": tf.random.uniform(
            (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
        ),
    }

    expand_dims_op = ml.ExpandDims(expand_dims=-1)
    expanded_inputs = expand_dims_op(inputs)

    assert inputs.keys() == expanded_inputs.keys()
    assert list(expanded_inputs["cont_feat"].shape) == [NUM_ROWS, 1]
    assert list(expanded_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, 4, 1]


def test_expand_dims_axis_as_dict():
    NUM_ROWS = 100

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    inputs = {
        "cont_feat1": tf.random.uniform((NUM_ROWS,)),
        "cont_feat2": tf.random.uniform((NUM_ROWS,)),
        "multi_hot_categ_feat": tf.random.uniform(
            (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
        ),
    }

    expand_dims_op = ml.ExpandDims(expand_dims={"cont_feat2": 0, "multi_hot_categ_feat": 1})
    expanded_inputs = expand_dims_op(inputs)

    assert inputs.keys() == expanded_inputs.keys()

    assert list(expanded_inputs["cont_feat1"].shape) == [NUM_ROWS]
    assert list(expanded_inputs["cont_feat2"].shape) == [1, NUM_ROWS]
    assert list(expanded_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, 1, 4]


def test_categorical_one_hot_encoding():
    NUM_ROWS = 100
    MAX_LEN = 4

    s = Schema(
        [
            create_categorical_column("cat1", num_items=200, tags=[Tags.CATEGORICAL]),
            create_categorical_column("cat2", num_items=1000, tags=[Tags.CATEGORICAL]),
            create_categorical_column("cat3", num_items=50, tags=[Tags.CATEGORICAL]),
            create_continuous_column("cont1", min_value=0, max_value=1, tags=[Tags.CONTINUOUS]),
        ]
    )

    cardinalities = {"cat1": 201, "cat2": 1001, "cat3": 51}
    inputs = {}
    for cat, cardinality in cardinalities.items():
        inputs[cat] = tf.random.uniform((NUM_ROWS, 1), minval=1, maxval=cardinality, dtype=tf.int32)
    inputs["cat3"] = tf.random.uniform(
        (NUM_ROWS, MAX_LEN), minval=1, maxval=cardinalities["cat3"], dtype=tf.int32
    )
    inputs["cont1"] = tf.random.uniform((NUM_ROWS, 1), minval=0, maxval=1, dtype=tf.float32)

    outputs = ml.CategoricalOneHot(schema=s)(inputs)

    assert list(outputs["cat1"].shape) == [NUM_ROWS, 201]
    assert list(outputs["cat2"].shape) == [NUM_ROWS, 1001]
    assert list(outputs["cat3"].shape) == [NUM_ROWS, MAX_LEN, 51]

    assert inputs["cat1"][0].numpy() == tf.where(outputs["cat1"][0, :] == 1).numpy()[0]
    assert list(outputs.keys()) == ["cat1", "cat2", "cat3"]


def test_popularity_logits_correct():
    from merlin.models.tf.core.base import PredictionOutput
    from merlin.models.tf.core.transformations import PopularityLogitsCorrection

    schema = Schema(
        [
            create_categorical_column(
                "item_feature", num_items=100, tags=[Tags.CATEGORICAL, Tags.ITEM_ID]
            ),
        ]
    )

    NUM_ITEMS = 101
    NUM_ROWS = 16
    NUM_SAMPLE = 20

    logits = tf.random.uniform((NUM_ROWS, NUM_SAMPLE))
    negative_item_ids = tf.random.uniform(
        (NUM_SAMPLE - 1,), minval=1, maxval=NUM_ITEMS, dtype=tf.int32
    )
    positive_item_ids = tf.random.uniform((NUM_ROWS,), minval=1, maxval=NUM_ITEMS, dtype=tf.int32)
    item_frequency = tf.sort(tf.random.uniform((NUM_ITEMS,), minval=0, maxval=1000, dtype=tf.int32))

    inputs = PredictionOutput(
        predictions=logits,
        targets=[],
        positive_item_ids=positive_item_ids,
        negative_item_ids=negative_item_ids,
    )

    corrected_logits = PopularityLogitsCorrection(
        item_frequency, reg_factor=0.5, schema=schema
    ).call_outputs(outputs=inputs)

    tf.debugging.assert_less_equal(logits, corrected_logits.predictions)


def test_popularity_logits_correct_from_parquet():
    import numpy as np
    import pandas as pd

    from merlin.models.tf.core.transformations import PopularityLogitsCorrection

    schema = Schema(
        [
            create_categorical_column(
                "item_feature", num_items=100, tags=[Tags.CATEGORICAL, Tags.ITEM_ID]
            ),
        ]
    )
    NUM_ITEMS = 101

    frequency_table = pd.DataFrame(
        {"frequency": list(np.sort(np.random.randint(0, 1000, size=(NUM_ITEMS,))))}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        frequency_table.to_parquet(tmpdir + "/frequency_table.parquet")
        corrected_logits = PopularityLogitsCorrection.from_parquet(
            tmpdir + "/frequency_table.parquet",
            frequencies_probs_col="frequency",
            gpu=False,
            schema=schema,
        )
    assert corrected_logits.get_candidate_probs().shape == (NUM_ITEMS,)


def test_items_weight_tying_with_different_domain_name():
    from merlin.models.tf.core.transformations import ItemsPredictionWeightTying

    NUM_ROWS = 16
    schema = Schema(
        [
            create_categorical_column(
                "item_id",
                domain_name="joint_item_id",
                num_items=100,
                tags=[Tags.CATEGORICAL, Tags.ITEM_ID],
            ),
        ]
    )
    inputs = {
        "item_id": tf.random.uniform((NUM_ROWS, 1), minval=1, maxval=101, dtype=tf.int32),
        "target": tf.random.uniform((NUM_ROWS, 1), minval=0, maxval=10, dtype=tf.int32),
    }

    weight_tying_block = ItemsPredictionWeightTying(schema=schema)
    input_block = ml.InputBlock(schema)
    task = ml.MultiClassClassificationTask("target")
    model = ml.Model(input_block, ml.MLPBlock([64]), weight_tying_block, task)

    _ = model(inputs)
    weight_tying_embeddings = model.blocks[2].context.get_embedding("joint_item_id")
    assert weight_tying_embeddings.shape == (101, 64)
