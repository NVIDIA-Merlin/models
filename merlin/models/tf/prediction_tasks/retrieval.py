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
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.python.layers.base import Layer

from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.transformations import LogitsTemperatureScaler
from merlin.models.tf.blocks.retrieval.base import ItemRetrievalScorer
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.blocks.sampling.in_batch import InBatchSampler
from merlin.models.tf.metrics.ranking import ranking_metrics
from merlin.models.tf.prediction_tasks.classification import MultiClassClassificationTask
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemRetrievalTask(MultiClassClassificationTask):
    """Prediction-task for item-retrieval.

    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        samplers: List[ItemSampler]
            List of samplers for negative sampling, by default `[InBatchSampler()]`
        post_logits: Optional[PredictionBlock]
            Optional extra pre-call block for post-processing the logits, by default None.
            You can for example use `post_logits = mm.PopularitySamplingBlock(item_fequency)`
            for populariy sampling correction.
        target_name: Optional[str]
            If specified, name of the target tensor to retrieve from dataloader.
            Defaults to None.
        task_name: Optional[str]
            name of the task.
            Defaults to None.
        task_block: Block
            The `Block` that applies additional layers op to inputs.
            Defaults to None.
        logits_temperature: float
            Parameter used to reduce the model overconfidence, so that logits / T.
            Defaults to 1.
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
        samplers: Sequence[ItemSampler] = (),
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        post_logits: Optional[Block] = None,
        logits_temperature: float = 1.0,
        cache_query: bool = False,
        store_negative_ids: bool = False,
        **kwargs,
    ):
        self.item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.item_domain = schema_utils.categorical_domains(schema)[self.item_id_feature_name]
        self.cache_query = cache_query
        pre = self._build_prediction_call(
            samplers,
            logits_temperature,
            post_logits,
            store_negative_ids,
        )

        super().__init__(
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            **kwargs,
        )

    def call(self, inputs, training=False, eval_sampling=False, **kwargs):
        return inputs

    def _build_prediction_call(
        self,
        samplers: Sequence[ItemSampler],
        logits_temperature: float,
        post_logits: Optional[Block] = None,
        store_negative_ids: bool = False,
        **kwargs,
    ):
        if samplers is None or len(samplers) == 0:
            samplers = (InBatchSampler(),)

        prediction_call = ItemRetrievalScorer(
            samplers=samplers,
            item_id_feature_name=self.item_id_feature_name,
            item_domain=self.item_domain,
            cache_query=self.cache_query,
            store_negative_ids=store_negative_ids,
        )

        if post_logits is not None:
            prediction_call = prediction_call.connect(post_logits)

        if logits_temperature != 1:
            prediction_call = prediction_call.connect(LogitsTemperatureScaler(logits_temperature))

        return prediction_call

    @property
    def retrieval_scorer(self):
        def find_retrieval_scorer_block(block):
            if isinstance(block, ItemRetrievalScorer):
                return block

            if getattr(block, "layers", None):
                for subblock in block.layers:
                    result = find_retrieval_scorer_block(subblock)
                    if result:
                        return result

            return None

        result = find_retrieval_scorer_block(self.pre)

        if result is None:
            raise Exception("An ItemRetrievalScorer layer was not found in the model.")

        return result

    def set_retrieval_cache_query(self, value: bool):
        self.retrieval_scorer.cache_query = value
