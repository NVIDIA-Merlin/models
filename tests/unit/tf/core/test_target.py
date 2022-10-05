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

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils, tf_utils
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.models.tf.core.target import TargetLayer, layer_targets


# @pytest.mark.parametrize("run_eagerly", [False])
# def test_simple_model(ecommerce_data: Dataset, run_eagerly):
#     model = mm.Model(
#         mm.InputBlock(ecommerce_data.schema),
#         mm.MLPBlock([4]),
#         mm.BinaryClassificationTask("click"),
#     )

#     loaded_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

#     features = ecommerce_data.schema.remove_by_tag(Tags.TARGET).column_names
#     testing_utils.test_model_signature(loaded_model, features, ["click/binary_classification_task"])


def flip_target(target):
    dtype = target.dtype

    return tf.cast(tf.math.logical_not(tf.cast(target, tf.bool)), dtype)



@tf.keras.utils.register_keras_serializable(package="merlin.models")
class FlipTarget(mm.BlockV2):
    def call(self, inputs, targets=None, **kwargs):
        if targets: 
            self.compute_target("click", flip_target(targets["click"]))
        
        return inputs
        
        

def test_1(ecommerce_data: Dataset, run_eagerly=False):
    flip = FlipTarget()
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([4]),
        flip,
        mm.BinaryClassificationTask("click"),
    )

    loaded_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)
    
    a = 5