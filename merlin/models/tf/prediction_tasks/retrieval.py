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
from typing import List, Optional, Sequence

import tensorflow as tf
from tensorflow.python.layers.base import Layer

from merlin.schema import Schema, Tags

from ..blocks.core.transformations import L2Norm, PredictionsScaler
from ..blocks.retrieval.base import ItemRetrievalScorer
from ..blocks.sampling.base import ItemSampler
from ..blocks.sampling.in_batch import InBatchSampler
from ..core import Block, MetricOrMetricClass
from ..losses import LossType, loss_registry
from ..metrics.ranking import ranking_metrics
from .classification import MultiClassClassificationTask


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemRetrievalTask(MultiClassClassificationTask):
    """Prediction-task for item-retrieval.

    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        loss: Optional[LossType]
            Loss function.
            Defaults to `categorical_crossentropy`.
        metrics: Sequence[MetricOrMetricClass]
            List of top-k ranking metrics.
            Defaults to MultiClassClassificationTask.DEFAULT_METRICS["ranking"].
        samplers: List[ItemSampler]
            List of samplers for negative sampling, by default `[InBatchSampler()]`
        extra_pre_call: Optional[PredictionBlock]
            Optional extra pre-call block. Defaults to None.
        target_name: Optional[str]
            If specified, name of the target tensor to retrieve from dataloader.
            Defaults to None.
        task_name: Optional[str]
            name of the task.
            Defaults to None.
        task_block: Block
            The `Block` that applies additional layers op to inputs.
            Defaults to None.
        softmax_temperature: float
            Parameter used to reduce model overconfidence, so that softmax(logits / T).
            Defaults to 1.
        normalize: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to True.

    Returns
    -------
        PredictionTask
            The item retrieval prediction task
    """

    DEFAULT_LOSS = "categorical_crossentropy"
    DEFAULT_METRICS = ranking_metrics(top_ks=[10])

    def __init__(
        self,
        schema: Schema,
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS,
        samplers: List[ItemSampler] = (),
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        extra_pre_call: Optional[Block] = None,
        softmax_temperature: float = 1.0,
        normalize: bool = True,
        cache_query: bool = False,
        **kwargs,
    ):
        self.item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.cache_query = cache_query
        pre = self._build_prediction_call(samplers, normalize, softmax_temperature, extra_pre_call)
        self.loss = loss_registry.parse(loss)

        super().__init__(
            loss=self.loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            **kwargs,
        )

    def _build_prediction_call(
        self,
        samplers: List[ItemSampler],
        normalize: bool,
        softmax_temperature: float,
        extra_pre_call: Optional[Block] = None,
    ):
        if samplers is None or len(samplers) == 0:
            samplers = (InBatchSampler(),)

        prediction_call = ItemRetrievalScorer(
            samplers=samplers,
            item_id_feature_name=self.item_id_feature_name,
            cache_query=self.cache_query,
        )

        if normalize:
            prediction_call = L2Norm().connect(prediction_call)

        if softmax_temperature != 1:
            prediction_call = prediction_call.connect(PredictionsScaler(1.0 / softmax_temperature))

        if extra_pre_call is not None:
            prediction_call = prediction_call.connect(extra_pre_call)

        return prediction_call

    @property
    def retrieval_scorer(self):
        return self.pre[0][1]

    def set_retrieval_cache_query(self, value: bool):
        self.retrieval_scorer.cache_query = value
