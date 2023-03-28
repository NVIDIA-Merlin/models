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
from typing import Optional, Union

import tensorflow as tf
import transformers
from transformers import (
    AlbertConfig,
    BertConfig,
    GPT2Config,
    PretrainedConfig,
    RobertaConfig,
    TFPreTrainedModel,
    XLNetConfig,
)

from merlin.models.tf.core import combinators
from merlin.models.tf.core.base import Block, block_registry
from merlin.models.tf.transformers.transforms import PrepareTransformerInputs
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)

TransformerBody = Union[TFPreTrainedModel, PretrainedConfig, tf.keras.layers.Layer]


def get_tf_main_layer(hf_model):
    """
    Extract serializable custom keras layer `TF*MainLayer` from the HuggingFace model
    """
    main_layer = [v for _, v in hf_model.__dict__.items() if isinstance(v, tf.keras.layers.Layer)][
        0
    ]
    return main_layer


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TransformerBlock(Block):
    """
    Base class to support HuggingFace Transformers for session-based and
    sequential-based recommendation models.

    Parameters
    ----------
    transformer: TransformerBody
        One of the following HuggingFace classes:
        Pretrained Config or model, or the custom keras layer `TF*MainLayer`,
        related to a specific transformer architecture.
    transformer_pre: tf.keras.layers.Layer
        Prepare the dictionary of inputs expected by the transformer layer
        by default PrepareTransformerInputs()
    transformer_post: [Union[str, tf.keras.layers.Layer]]
        Layer to extract the desired tensors from the HuggingFace dataclass
        `TFBaseModelOutputWithPoolingAndCrossAttentions`
        by default `last_hidden_state`
    pre: Optional[Union[str, tf.keras.layers.Layer]]
        A block to use before the main transformer block, by default None
    post: Optional[Union[str, tf.keras.layers.Layer]]
        A block to use after the main transformer block, by default None
    masking_post:
        A block to use to postprocess the output of the transformer block based on
        keras mask information, by default None
    masking_pre:
        A block to use to prepare the inputs to the transformer block based on
        keras mask information, by default None
    """

    def __init__(
        self,
        transformer: TransformerBody,
        pre: Optional[Union[str, tf.keras.layers.Layer]] = None,
        post: Optional[Union[str, tf.keras.layers.Layer]] = None,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        masking_post: Optional[tf.keras.layers.Layer] = None,
        masking_pre: Optional[tf.keras.layers.Layer] = None,
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
        self.transformer.supports_masking = True
        if "transformer" in inspect.signature(transformer_pre.__init__).parameters:
            transformer_pre = transformer_pre(transformer=self.transformer)
        self.transformer_pre = transformer_pre
        if isinstance(transformer_post, str):
            transformer_post = block_registry.parse(transformer_post)
        self.transformer_post = transformer_post

        if isinstance(pre, str):
            pre = block_registry.parse(pre)

        if isinstance(post, str):
            post = block_registry.parse(post)

        self.post = post
        self.pre = pre
        self._masking_post = masking_post
        self._masking_pre = masking_pre

    @property
    def masking_post(self):
        return self._masking_post

    @masking_post.setter
    def masking_post(self, block):
        self._masking_post = block

    @property
    def masking_pre(self):
        return self._masking_pre

    @masking_pre.setter
    def masking_pre(self, block):
        self._masking_pre = block

    def build(self, input_shape=None):
        """Builds the sequential block

        Parameters
        ----------
        input_shape : tf.TensorShape, optional
            The input shape, by default None
        """
        combinators.build_sequentially(
            self, [*list(self.to_call_pre), self.transformer, *list(self.to_call_post)], input_shape
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        inputs: tf.Tensor
            The 3D tensor of the sequence of interactions embeddings.
        """
        pre = combinators.call_sequentially(list(self.to_call_pre), inputs, **kwargs)
        transformer = self.transformer(pre)
        out = combinators.call_sequentially(list(self.to_call_post), transformer, **kwargs)

        return out

    @property
    def to_call_pre(self):
        if self.masking_pre:
            yield self.masking_pre

        if self.pre:
            yield self.pre

        yield self.transformer_pre

    @property
    def to_call_post(self):
        yield self.transformer_post

        if self.masking_post:
            yield self.masking_post

        if self.post:
            yield self.post

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self,
            config,
            [
                "transformer",
                "pre",
                "post",
                "transformer_pre",
                "transformer_post",
                "masking_pre",
                "masking_post",
            ],
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = maybe_deserialize_keras_objects(
            config,
            [
                "transformer",
                "pre",
                "post",
                "transformer_pre",
                "transformer_post",
                "masking_post",
                "masking_pre",
            ],
        )

        output = TransformerBlock(**config)
        output.__class__ = cls

        return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BertBlock(TransformerBlock):
    """
    Class to prepare the configuration of a `Bert` model
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_layer: int,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        pre=None,
        post=None,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        **kwargs,
    ):
        config = self.create_config(
            d_model,
            n_head,
            n_layer,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token=pad_token,
            log_attention_weights=log_attention_weights,
            **kwargs,
        )

        super().__init__(
            transformer=config,
            pre=pre,
            post=post,
            transformer_pre=transformer_pre,
            transformer_post=transformer_post,
        )

    def create_config(
        self,
        d_model,
        n_head,
        n_layer,
        max_position_embeddings=512,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs,
    ) -> BertConfig:
        return BertConfig(
            hidden_size=d_model,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AlbertBlock(TransformerBlock):
    """
    Class to prepare the configuration of an `Albert` model
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_layer: int,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        pre=None,
        post=None,
        **kwargs,
    ):
        config = self.create_config(
            d_model,
            n_head,
            n_layer,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token=pad_token,
            log_attention_weights=log_attention_weights,
            **kwargs,
        )

        super().__init__(
            config,
            pre=pre,
            post=post,
            transformer_pre=transformer_pre,
            transformer_post=transformer_post,
        )

    @classmethod
    def create_config(
        cls,
        d_model,
        n_head,
        n_layer,
        max_position_embeddings=512,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs,
    ) -> AlbertConfig:
        return AlbertConfig(
            hidden_size=d_model,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            hidden_act=hidden_act,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RobertaBlock(TransformerBlock):
    """
    Class to prepare the configuration of a `RoBerta` model
    """

    def __init__(
        self,
        d_model,
        n_head,
        n_layer,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        pre=None,
        post=None,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        **kwargs,
    ):
        config = self.create_config(
            d_model,
            n_head,
            n_layer,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token=pad_token,
            log_attention_weights=log_attention_weights,
            **kwargs,
        )

        super().__init__(
            config,
            pre=pre,
            post=post,
            transformer_pre=transformer_pre,
            transformer_post=transformer_post,
        )

    @classmethod
    def create_config(
        cls,
        d_model,
        n_head,
        n_layer,
        max_position_embeddings=512,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs,
    ) -> RobertaConfig:
        return RobertaConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            max_position_embeddings=max_position_embeddings + 8,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class XLNetBlock(TransformerBlock):
    """
    Class to prepare the configuration of a `XLNet` model
    """

    def __init__(
        self,
        d_model,
        n_head,
        n_layer,
        attn_type="bi",
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        mem_len=1,
        pre=None,
        post=None,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        **kwargs,
    ):
        config = self.create_config(
            d_model,
            n_head,
            n_layer,
            attn_type=attn_type,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token=pad_token,
            log_attention_weights=log_attention_weights,
            mem_len=mem_len,
            **kwargs,
        )

        super().__init__(
            config,
            pre=pre,
            post=post,
            transformer_pre=transformer_pre,
            transformer_post=transformer_post,
        )

    @classmethod
    def create_config(
        cls,
        d_model,
        n_head,
        n_layer,
        attn_type="bi",
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        mem_len=1,
        **kwargs,
    ) -> XLNetConfig:
        return XLNetConfig(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            attn_type=attn_type,
            ff_activation=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            mem_len=mem_len,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class GPT2Block(TransformerBlock):
    """
    Class to prepare the configuration of a `GPT2` model
    """

    def __init__(
        self,
        d_model,
        n_head,
        n_layer,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        pre=None,
        post=None,
        transformer_pre=PrepareTransformerInputs(),
        transformer_post: Optional[Union[str, tf.keras.layers.Layer]] = "last_hidden_state",
        **kwargs,
    ):
        config = self.create_config(
            d_model,
            n_head,
            n_layer,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token=pad_token,
            log_attention_weights=log_attention_weights,
            **kwargs,
        )

        super().__init__(
            config,
            pre=pre,
            post=post,
            transformer_pre=transformer_pre,
            transformer_post=transformer_post,
        )

    @classmethod
    def create_config(
        cls,
        d_model,
        n_head,
        n_layer,
        max_position_embeddings=512,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs,
    ) -> GPT2Config:
        return GPT2Config(
            n_embd=d_model,
            n_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=max_position_embeddings,
            n_ctx=max_position_embeddings,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )
