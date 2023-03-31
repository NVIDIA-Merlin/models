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
from typing import Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import embedding_ops

import merlin.io
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.metrics.topk import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.models.tf.outputs.base import MetricsFn, ModelOutput
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    tensor_to_df,
)
from merlin.schema import ColumnSchema, Schema

LOG = logging.getLogger("merlin_models")


def default_binary_metrics():
    return (
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    )


def default_categorical_prediction_metrics(k=10):
    return (
        RecallAt(k),
        MRRAt(k),
        NDCGAt(k),
        AvgPrecisionAt(k),
        PrecisionAt(k),
    )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BinaryOutput(ModelOutput):
    """
    Binary-classification prediction block.

    Parameters
    ----------
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for binary-classification
        by 'binary_crossentropy'
    default_metrics_fn: Callable
        A function returning the list of default metrics
        to use for binary-classification
    """

    def __init__(
        self,
        target: Optional[Union[str, ColumnSchema]] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy",
        default_metrics_fn: MetricsFn = default_binary_metrics,
        **kwargs,
    ):
        if isinstance(target, ColumnSchema):
            target = target.name
        to_call = kwargs.pop("to_call", None)
        super().__init__(
            to_call=to_call or tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalOutput(ModelOutput):
    """Categorical output

    Parameters
    ----------
    prediction: Union[Schema, ColumnSchema,
                EmbeddingTable, 'CategoricalTarget',
                'EmbeddingTablePrediction']
        The target feature to predict. To perform weight-tying [1] technique, you should provide
        the `EmbeddingTable` or `EmbeddingTablePrediction` related to the
        target feature.
    negative_samplers: ItemSamplersType, optional
        List of samplers for negative sampling,
        by default None
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1
    name: str, optional
        The name of the task, by default None
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for categorical-classification
        by default 'categorical_crossentropy'
    get_default_metrics: Callable, optional
        A function returning the list of default metrics
        to use for categorical-classification

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).

    """

    def __init__(
        self,
        to_call: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ],
        target_name: str = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics_fn: MetricsFn = default_categorical_prediction_metrics,
        **kwargs,
    ):
        self.max_num_samples = kwargs.pop("max_num_samples", None)
        _to_call = kwargs.pop("to_call", None)

        if to_call is not None:
            if isinstance(to_call, (Schema, ColumnSchema)):
                _to_call = CategoricalTarget(to_call)
                if isinstance(to_call, Schema):
                    to_call = to_call.first
                target_name = target_name or to_call.name
            elif isinstance(to_call, EmbeddingTable):
                _to_call = EmbeddingTablePrediction(to_call)
                target_name = _to_call.table.col_schema.name
            else:
                _to_call = to_call

        self.target_name = kwargs.pop("target", target_name)
        super().__init__(
            to_call=_to_call,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=self.target_name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def to_dataset(self, gpu=True) -> merlin.io.Dataset:
        return merlin.io.Dataset(tensor_to_df(self.to_call.embeddings, gpu=gpu))

    def get_config(self):
        config = super().get_config()
        config["max_num_samples"] = self.max_num_samples
        config["target_name"] = self.target_name
        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalTarget(tf.keras.layers.Dense):
    """Prediction of a categorical feature.

    Parameters
    ----------
    feature : Union[Schema, ColumnSchema]
        The schema description of the categorical feature to predict.
    activation : optional
        Activation function to use, by default None
    use_bias : bool, optional
        Whether the layer uses a bias vector, by default True
    kernel_initializer : str, optional
        Initializer for the kernel weights matrix, by default "glorot_uniform"
    bias_initializer : str, optional
        Initializer for the bias vector., by default "zeros"
    kernel_regularizer : optional
        Regularizer function applied to the kernel weights matrix, by default None
    bias_regularizer : optional
        Regularizer function applied to the bias vector, by default None
    activity_regularizer : optional
        Regularizer function applied to the output of the layer (its "activation"),
        by default None
    kernel_constraint : optional
        Constraint function applied to the kernel weights matrix, by default None
    bias_constraint : optional
        Constraint function applied to the bias vector, by default None
    """

    def __init__(
        self,
        feature: Union[Schema, ColumnSchema] = None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        if feature is not None:
            if isinstance(feature, Schema):
                assert len(feature) == 1, "Schema can have max 1 feature"
                col_schema = feature.first
            else:
                col_schema = feature

            self.num_classes = col_schema.int_domain.max + 1
            units = self.num_classes
        else:
            units = kwargs.pop("units")
            self.num_classes = units

        super().__init__(
            units,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )

    def embedding_lookup(self, inputs: tf.Tensor, **kwargs):
        """Method to retrieve hidden representation vectors from the kernel weight matrix
        based on a given "input" positions

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of indices to retrieve from the weight matrix.

        Returns
        -------
        tf.Tensor
            Tensor of hidden representation vectors.
        """
        return embedding_ops.embedding_lookup(self.embeddings, inputs, **kwargs)

    @property
    def embeddings(self):
        return tf.transpose(self.kernel)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTablePrediction(Layer):
    """Prediction of a categorical feature using weight-sharing [1] with an embedding table

    Parameters
    ----------
    table : EmbeddingTable
        The embedding table to use as the weight matrix
    bias_initializer : str, optional
        Initializer for the bias vector, by default "zeros"

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).
    """

    def __init__(self, table: EmbeddingTable, bias_initializer="zeros", **kwargs):
        self.table = table
        self.num_classes = table.input_dim
        self.bias_initializer = bias_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )
        self.table.build(input_shape)
        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            original_inputs = inputs
            inputs = inputs.flat_values
        logits = tf.matmul(inputs, self.table.table.embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)
        if is_ragged:
            logits = original_inputs.with_flat_values(logits)
        return logits

    @property
    def embeddings(self):
        return self.table.table.embeddings

    def embedding_lookup(self, inputs, **kwargs):
        return self.table.table(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def get_config(self):
        config = maybe_serialize_keras_objects(
            self,
            {
                **super().get_config(),
            },
            ["table"],
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["table"])
        return super().from_config(config)
