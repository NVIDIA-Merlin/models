

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

import inspect
from typing import Any, Dict, Optional, Type, Union

import tensorflow as tf
import transformers
from transformers import PretrainedConfig, TFPreTrainedModel

from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.models.tf.core.base import Block


TransformerBody = Union[TFPreTrainedModel, PretrainedConfig, tf.keras.layers.Layer]


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class TransformerBlock(Block):
    """
    Class to support HF Transformers for session-based and sequential-based recommendation models.
    Parameters
    ----------
    transformer: TransformerBody
        The T4RecConfig, The pre-trained HF model or the custom keras layer TF*MainLayer,
        related to specific transformer architecture.
    """

    def __init__(
        self,
        transformer: TransformerBody,
        output_fn=lambda model_outputs: model_outputs.last_hidden_state,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(transformer, PretrainedConfig):
            model_cls = transformers.TF_MODEL_MAPPING[transformer.__class__]
            self.transformer = get_tf_main_layer(model_cls(transformer))
        elif isinstance(transformer, TFPreTrainedModel):
            self.transformer = get_tf_main_layer(transformer)
        else:
            self.transformer = transformer

        self.output_fn = output_fn

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, ["transformer"]
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config, ["transformer"]
        )

        return super().from_config(config)

    def call(self, inputs_embeds: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        inputs_embeds: tf.Tensor 
            The 3d tensor of sequence of interaction embeddings.
        """
        if isinstance(inputs_embeds, tf.RaggedTensor):
            # convert to a dense tensor as HF transformers do not support ragged tensors
            inputs_embeds = inputs_embeds.to_tensor()
        # In HF the call accept inputs as a dictionary containing all needed tensors
        model_outputs = self.transformer(dict(inputs_embeds=inputs_embeds))
        outputs = self.output_fn(model_outputs)

        return outputs

def get_tf_main_layer(hf_model):
    """
    Extract serializable custom keras layer `TF*MainLayer` from the HF model
    """
    main_layer = [v for _, v in hf_model.__dict__.items() if isinstance(v, tf.keras.layers.Layer)][
        0
    ]
    return main_layer