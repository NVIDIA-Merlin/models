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
from types import SimpleNamespace
from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Layer
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_tf_utils import TFSequenceSummary

from merlin.models.tf.core.base import Block


@Block.registry.register("last_hidden_state")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class LastHiddenState(Layer):
    """Select the hidden states of the last transform layer
    from the HF dataclass object
    `TFBaseModelOutputWithPoolingAndCrossAttentions`

    Parameters
    ----------
    inputs: TFBaseModelOutputWithPoolingAndCrossAttentions
        The output class returned by the HuggingFace transformer layer
    """

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return inputs.last_hidden_state


@Block.registry.register("pooler_output")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PoolerOutput(Layer):
    """Select the pooled representation of the sequence
    from the HF dataclass object
    `TFBaseModelOutputWithPoolingAndCrossAttentions`

    Parameters
    ----------
    inputs: TFBaseModelOutputWithPoolingAndCrossAttentions
        The output class returned by the HuggingFace transformer layer
    """

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return inputs.pooler_output


@Block.registry.register("hidden_states")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class HiddenStates(Layer):
    """Select the tuple of hidden states of all transformer layers
    from the HF dataclass object
    `TFBaseModelOutputWithPoolingAndCrossAttentions`

    Parameters
    ----------
    inputs: TFBaseModelOutputWithPoolingAndCrossAttentions
        The output class returned by the HuggingFace transformer layer
    """

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return inputs.hidden_states


@Block.registry.register("attentions")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AttentionWeights(Layer):
    """Select the attention weights of all transformer layers
    from the HF dataclass object
    `TFBaseModelOutputWithPoolingAndCrossAttentions`

    Parameters
    ----------
    inputs: TFBaseModelOutputWithPoolingAndCrossAttentions
        The output class returned by the HuggingFace transformer layer
    """

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return inputs.attentions


@Block.registry.register("last_hidden_state_and_attention")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class LastHiddenStateAndAttention(Layer):
    """Select the hidden states and attention weights of the last
    tranfrom the HF dataclass object
    `TFBaseModelOutputWithPoolingAndCrossAttentions`

    Parameters
    ----------
    inputs: TFBaseModelOutputWithPoolingAndCrossAttentions
        The output class returned by the HuggingFace transformer layer
    """

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return (inputs.last_hidden_state, inputs.attentions[-1])


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PrepareTransformerInputs(tf.keras.layers.Layer):
    """Prepare the dictionary of inputs expected by the transformer layer"""

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        if isinstance(inputs, tf.RaggedTensor):
            # convert to a dense tensor as HF transformers do not support ragged tensors
            inputs = inputs.to_tensor()
        return {"inputs_embeds": inputs}


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceSummary(TFSequenceSummary):
    def __init__(self, summary: str, initializer_range: float = 0.02, **kwargs):
        config = SimpleNamespace(summary_type=summary)
        super().__init__(config, initializer_range=initializer_range, **kwargs)


@Block.registry.register("sequence_last")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceLast(SequenceSummary):
    def __init__(self, initializer_range: float = 0.02, **kwargs):
        super().__init__("last", initializer_range=initializer_range, **kwargs)


@Block.registry.register("sequence_first")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceFirst(SequenceSummary):
    def __init__(self, initializer_range: float = 0.02, **kwargs):
        super().__init__("first", initializer_range=initializer_range, **kwargs)


@Block.registry.register("sequence_mean")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceMean(SequenceSummary):
    def __init__(self, initializer_range: float = 0.02, **kwargs):
        super().__init__("mean", initializer_range=initializer_range, **kwargs)


@Block.registry.register("sequence_cls_index")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceClsIndex(SequenceSummary):
    def __init__(self, initializer_range: float = 0.02, **kwargs):
        super().__init__("cls_index", initializer_range=initializer_range, **kwargs)
