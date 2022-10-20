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

from merlin.models.tf.utils import testing_utils
from merlin.models.utils.schema_utils import create_categorical_column
from merlin.schema import Schema, Tags


def test_popularity_logits_correct():
    from merlin.models.tf.core.base import PredictionOutput
    from merlin.models.tf.core.prediction import Prediction
    from merlin.models.tf.transforms.bias import PopularityLogitsCorrection

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

    log_q_correction = PopularityLogitsCorrection(item_frequency, reg_factor=0.5, schema=schema)

    inputs = PredictionOutput(
        predictions=logits,
        targets=[],
        positive_item_ids=positive_item_ids,
        negative_item_ids=negative_item_ids,
    )
    corrected_logits = log_q_correction.call_outputs(outputs=inputs)

    inputs_v2 = Prediction(
        outputs=logits,
        targets=[],
        negative_candidate_ids=negative_item_ids,
    )
    corrected_logits_v2 = log_q_correction(
        outputs=inputs_v2, features={"item_feature": positive_item_ids}, training=True
    )

    tf.debugging.assert_less(logits, corrected_logits.predictions)
    tf.debugging.assert_less(logits, corrected_logits_v2.outputs)

    copy_layer = testing_utils.assert_serialization(log_q_correction)
    tf.debugging.assert_equal(copy_layer.candidate_probs, log_q_correction.candidate_probs)


def test_popularity_logits_correct_from_parquet():
    import numpy as np
    import pandas as pd

    from merlin.models.tf.transforms.bias import PopularityLogitsCorrection

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
