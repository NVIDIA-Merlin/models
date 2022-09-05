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
from typing import List, Optional, Protocol, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.outputs.base import MetricsFn, ModelOutput
from merlin.models.tf.outputs.classification import (
    CategoricalTarget,
    EmbeddingTablePrediction,
    default_categorical_prediction_metrics,
)
from merlin.models.tf.outputs.sampling.base import Items, ItemSamplersType, parse_negative_samplers
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    call_layer,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import ColumnSchema, Schema

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContrastiveOutput(ModelOutput):
    """Categorical output

    Parameters
    ----------
    prediction: Union[Schema, ColumnSchema,
                EmbeddingTable, 'CategoricalTarget',
                'EmbeddingTablePrediction']
        The target feature to predict. To perform weight-tying [1] technique, you should provide
        the `EmbeddingTable` or `EmbeddingTablePrediction` related to the
        target feature.
    negative_samplers: ItemSamplersType
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
        negative_samplers: ItemSamplersType = None,
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

        to_call_train_test = kwargs.pop(
            "to_call_train_test",
            SampledLookUps(
                prediction=_to_call,
                negative_samplers=negative_samplers,
                feature_name=target_name,
                max_num_samples=self.max_num_samples,
            ),
        )

        self.target_name = kwargs.pop("target", target_name)
        super().__init__(
            to_call=_to_call,
            to_call_train_test=to_call_train_test,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=self.target_name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def compile(self, negative_sampling=None, downscore_false_negatives=False):
        if negative_sampling is not None:
            negative_sampling = parse_negative_samplers(negative_sampling)
        self.to_call_train_test.negative_samplers = negative_sampling
        self.to_call_train_test.downscore_false_negatives = downscore_false_negatives

    def get_config(self):
        config = super().get_config()
        config["max_num_samples"] = self.max_num_samples
        config["target_name"] = self.target_name
        return config


@runtime_checkable
class LookUpProtocol(Protocol):
    def embedding_lookup(self, inputs, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SampledLookUps(Layer):
    """Contrastive layer for sampled logits.

    This layer can be used to compute the output scores of a multi-classification
    task only on a subset of sampled classes.

    For example, we could use this class to define the sampled softmax task [1] where
    negatives are defined using a popularity-based sampler.

    Parameters
    ----------
    prediction : LookUpProtocol
        The prediction layer used for computing the logits scores. It should be an
        instance of `LookUpProtocol`, i.e. it includes the method `embedding_lookup`
        that indexes the output weights.
    negative_samplers : ItemSamplersType
        List of samplers for negative sampling,
    feature_name : str, optional
        The name of the target feature, by default None
    downscore_false_negatives : bool, optional
        Identify false negatives (sampled item ids equal to the positive item and downscore them
        to the `sampling_downscore_false_negatives_value`),
        by default False
    false_negative_score : float, optional
        Value to be used to downscore false negatives when
        `sampling_downscore_false_negatives=True`,
        by default `np.finfo(np.float32).min / 100.0`

    References:
    -----------
    [1] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312
    """

    def __init__(
        self,
        prediction: LookUpProtocol,
        negative_samplers: ItemSamplersType,
        feature_name: str = None,
        downscore_false_negatives=True,
        false_negative_score: float = MIN_FLOAT,
        **kwargs,
    ):
        self.prediction = prediction
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.feature_name = feature_name

        if negative_samplers is not None:
            self.negative_samplers = parse_negative_samplers(negative_samplers)
        else:
            self.negative_samplers = negative_samplers

        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, features, targets, training=False, testing=False):
        if isinstance(targets, dict):
            if self.feature_name is None:
                raise ValueError(
                    "When training with multi-task, you should specify the "
                    "`target_name` for the sampled softmax task"
                )
            targets = targets[self.feature_name]
        # Get positive weights
        pos_item_id = tf.squeeze(targets)
        positive_weights = self.prediction.embedding_lookup(pos_item_id)

        # Sample negative items
        neg_items = self.sample_negatives(
            Items(pos_item_id, {}), features, training=training, testing=testing
        )
        negative_weights = self.prediction.embedding_lookup(neg_items.id)

        # Apply dot-product
        negative_scores = tf.linalg.matmul(inputs, negative_weights, transpose_b=True)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs, positive_weights), keepdims=True, axis=-1
        )

        if self.downscore_false_negatives:
            negative_scores, _ = rescore_false_negatives(
                pos_item_id, neg_items.id, negative_scores, self.false_negative_score
            )

        outputs = tf.concat([positive_scores, negative_scores], axis=-1)

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        outputs = tf.cast(outputs, tf.float32)
        outputs = tf.squeeze(outputs)

        targets = tf.concat(
            [
                tf.ones([tf.shape(outputs)[0], 1], dtype=outputs.dtype),
                tf.zeros(
                    [tf.shape(outputs)[0], tf.shape(outputs)[1] - 1],
                    dtype=outputs.dtype,
                ),
            ],
            axis=1,
        )

        return Prediction(outputs, targets)

    def sample_negatives(
        self,
        positive_items: Items,
        features: TabularData,
        training=False,
        testing=False,
    ) -> Items:
        """Method to sample negatives from `self.negative_samplers`

        Parameters
        ----------
        positive_items : Items
            Positive items (ids and metadata)
        features : TabularData
            Dictionary of input raw tensors
        training : bool, optional
            Flag for train mode, by default False
        testing : bool, optional
            Flag for test mode, by default False

        Returns
        -------
        Items
            Class containing the sampled negative ids
        """
        negative_items: List[Items] = []
        sampling_kwargs = {"training": training, "testing": testing, "features": features}

        # sample a number of negative ids from self.negative_samplers
        for sampler in self.negative_samplers:
            sampler_items: Items = call_layer(sampler, positive_items, **sampling_kwargs)

            if tf.shape(sampler_items.id)[0] > 0:
                negative_items.append(sampler_items)
            else:
                LOG.warn(
                    f"The sampler {type(sampler).__name__} returned no samples for this batch."
                )

        if len(negative_items) == 0:
            raise Exception(f"No negative items where sampled from samplers {self.samplers}")

        negatives = negative_items[0]
        if len(negative_items) > 1:
            for neg in negative_items[1:]:
                negatives += neg

        return negatives

    @property
    def has_negative_samplers(self) -> bool:
        return self.negative_samplers is not None and len(self.negative_samplers) > 0

    def get_config(self):
        config = maybe_serialize_keras_objects(
            self,
            {
                **super().get_config(),
                "downscore_false_negatives": self.downscore_false_negatives,
                "false_negative_score": self.false_negative_score,
            },
            ["negative_samplers", "prediction"],
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["negative_samplers", "prediction"])
        return super().from_config(config)
