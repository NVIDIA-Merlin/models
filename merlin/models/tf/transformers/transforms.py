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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: TFBaseModelOutputWithPoolingAndCrossAttentions):
        return inputs.last_hidden_state


@Block.registry.register("inference_hidden_state")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TransformerInferenceHiddenState(Layer):
    """A post-processing layer to select the hidden state
    of the next-item position, during inference.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        testing: bool = False,
    ):
        """Select the hidden state of the target (last) position, during inference.
        During training or testing, the inputs are returned
        without any processing.

        Parameters:
        ----------
        inputs: tf.Tensor
            The 3-D output tensor returned by the transformer block
        training : bool, optional
            Flag that indicates whether in training mode, by default True
        testing : bool, optional
            Flag that indicates whether in evaluation mode, by default True

        Returns
        -------
        tf.Tensor
            If inference, returns a 2-D tensor with the hidden states of
            the target position
        """
        if isinstance(inputs, tf.RaggedTensor):
            batch_size = tf.shape(inputs.row_lengths())[0]
        else:
            batch_size = tf.shape(inputs)[0]
        if not training and not testing:
            if isinstance(inputs, tf.RaggedTensor):
                inputs = inputs[:, -1:, :]
                inputs = tf.squeeze(tf.sparse.to_dense(inputs.to_sparse()), axis=1)

            elif getattr(inputs, "_keras_mask", None) is not None:
                inputs = tf.reshape(
                    tf.boolean_mask(inputs, inputs._keras_mask), (-1, inputs.shape[-1])
                )
            tf.debugging.assert_equal(
                tf.shape(inputs)[0],
                batch_size,
                f"The resulting tensor has {tf.shape(inputs)[0]} rows, which does not match"
                f" the inputs batch-size {batch_size}. During inference only one position "
                "candidate (the last one) should be masked per example",
            )
        return inputs


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        mask = None
        if getattr(inputs, "_keras_mask", None) is not None and isinstance(
            inputs._keras_mask, tf.RaggedTensor
        ):
            mask = inputs._keras_mask.to_tensor()
        if isinstance(inputs, tf.RaggedTensor):
            # convert to a dense tensor as HF transformers do not support ragged tensors
            inputs = inputs.to_tensor()
        if mask is not None:
            inputs._keras_mask = mask
        return {"inputs_embeds": inputs}


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceSummary(TFSequenceSummary):
    def __init__(self, summary: str, initializer_range: float = 0.02, **kwargs):
        self.summary = summary
        config = SimpleNamespace(summary_type=summary)
        super().__init__(config, initializer_range=initializer_range, **kwargs)

    def get_config(self):
        config = super().get_config()
        config["summary"] = self.summary
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        output = SequenceSummary(**config)
        output.__class__ = cls
        return output


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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TransformerOutputToRagged(Block):
    """Converts the dense outputs returned by the transformer layer to
    a ragged tensor using masking information.

    This layer takes dense inputs from the transformer layer and
    applies the masking information (in the `_keras_mask` attribute)
    to produce a ragged tensor output. The resulting tensor contains predictions
    only at masked positions (targets).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        if isinstance(inputs, tf.RaggedTensor):
            return input

        if getattr(inputs, "_keras_mask", None) is not None:
            mask = inputs._keras_mask
            if isinstance(mask, tf.RaggedTensor):
                mask = mask.to_tensor()
            inputs = tf.ragged.boolean_mask(inputs, mask)
        return inputs
