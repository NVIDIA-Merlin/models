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
from typing import Optional

import tensorflow as tf
from tensorflow.python.layers.base import Layer

from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.masking import MaskingHead
from merlin.models.tf.blocks.core.transformations import (
    ItemsPredictionWeightTying,
    L2Norm,
    LabelToOneHot,
    LogitsTemperatureScaler,
    RemovePad3D,
)
from merlin.models.tf.blocks.retrieval.base import ItemRetrievalScorer
from merlin.models.tf.blocks.sampling.cross_batch import PopularityBasedSampler
from merlin.models.tf.losses.base import LossType
from merlin.models.tf.metrics.ranking import ranking_metrics
from merlin.models.tf.prediction_tasks.classification import (
    CategFeaturePrediction,
    MultiClassClassificationTask,
)
from merlin.models.utils.schema_utils import categorical_cardinalities
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin.models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemsPrediction(CategFeaturePrediction):
    def __init__(
        self,
        schema: Schema,
        **kwargs,
    ):
        super(ItemsPrediction, self).__init__(schema, **kwargs)


def ItemsPredictionSampled(
    schema: Schema,
    num_sampled: int,
    min_id: int = 0,
    ignore_false_negatives: bool = True,
):
    """
    Compute the items logits on a subset of sampled candidates to optimize
    training. During inference, the scores are computed over the whole
    catalog of items.
    Reference of the method can be found at [Jean et al., 2014](http://arxiv.org/abs/1412.2007)

    Parameters:
    -----------
        schema: Schema
            The schema object including features to use and their properties.
        num_sampled: int
            The number of candidates to sample during training
        min_id: int
            The minimum id value to be sampled as negative. Useful to ignore the first categorical
            encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
            Defaults to 0.
        ignore_false_negatives: bool
            Ignore sampled items that are equal to the target classes
            Defaults to True

    Returns:
    -------
        targets, logits: tf.Tensor, tf.Tensor
            During training, return the concatenated tensor of true class
            and sampled negatives of shape (bs, num_sampled+1), as well as the related logits.
            During evaluation, returns the input tensor of true class, and the related logits.
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    num_classes = categorical_cardinalities(schema)[item_id_feature_name]
    samplers = PopularityBasedSampler(
        max_num_samples=num_sampled,
        max_id=num_classes,
        min_id=min_id,
        item_id_feature_name=item_id_feature_name,
    )

    logits = ItemRetrievalScorer(
        samplers=[samplers],
        sampling_downscore_false_negatives=ignore_false_negatives,
        item_id_feature_name=item_id_feature_name,
        sampled_softmax_mode=True,
    )

    return logits


def NextItemPredictionTask(
    schema: Schema,
    loss: Optional[LossType] = "categorical_crossentropy",
    metrics=ranking_metrics(top_ks=[10]),
    weight_tying: bool = True,
    masking: bool = True,
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    logits_temperature: float = 1.0,
    normalize: bool = True,
    sampled_softmax: bool = False,
    num_sampled: int = 100,
    min_sampled_id: int = 0,
) -> MultiClassClassificationTask:
    """
    Function to create the NextItemPrediction task with the right parameters.
    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        loss: Optional[LossType]
            Loss function.
            Defaults to `sparse_categorical_crossentropy`.
        metrics: MetricOrMetrics
            List of top-k ranking metrics.
            Defaults to ranking_metrics(top_ks=[10]).
        weight_tying: bool
            The item_id embedding weights are shared with the prediction network layer.
            Defaults to True
        masking: bool
            Whether masking is used to transform inputs and targets or not
            Defaults to True
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
        logits_temperature: float
            Parameter used to reduce the model overconfidence, so that logits / T.
            Defaults to 1.
        normalize: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to True.
        sampled_softmax: bool
            Compute the logits scores over all items of the catalog or
            generate a subset of candidates
            Defaults to False
        num_sampled: int
            When sampled_softmax is enabled, specify the number of
            negative candidates to generate for each batch
            Defaults to 100
        min_sampled_id: int
            The minimum id value to be sampled. Useful to ignore the first categorical
            encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
            Defaults to 0.
    Returns
    -------
        PredictionTask
            The next item prediction task
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    if sampled_softmax:
        prediction_call = ItemsPredictionSampled(
            schema, num_sampled=num_sampled, min_id=min_sampled_id
        )

    else:
        if weight_tying:
            prediction_call = ItemsPredictionWeightTying(schema)

        else:
            prediction_call = ItemsPrediction(schema)

        prediction_call = prediction_call.connect(LabelToOneHot())

    if logits_temperature != 1:
        prediction_call = prediction_call.connect(LogitsTemperatureScaler(logits_temperature))

    if masking:
        prediction_call = MaskingHead(item_id_feature_name=item_id_feature_name).connect(
            RemovePad3D(), prediction_call
        )

    if normalize:
        prediction_call = L2Norm().connect(prediction_call)

    if extra_pre_call is not None:
        prediction_call = prediction_call.connect(extra_pre_call)

    return MultiClassClassificationTask(
        target_name,
        task_name,
        task_block,
        loss=loss,
        metrics=metrics,
        pre=prediction_call,
    )
