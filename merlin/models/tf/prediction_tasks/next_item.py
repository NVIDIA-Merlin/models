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
    PopularityLogitsCorrection,
    RemovePad3D,
)
from merlin.models.tf.blocks.retrieval.base import ItemRetrievalScorer
from merlin.models.tf.blocks.sampling.cross_batch import PopularityBasedSampler
from merlin.models.tf.prediction_tasks.classification import (
    CategFeaturePrediction,
    MultiClassClassificationTask,
)
from merlin.models.utils.schema_utils import categorical_cardinalities, categorical_domains
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


def ItemsPredictionPopSampled(
    schema: Schema,
    num_sampled: int,
    min_id: int = 0,
    ignore_false_negatives: bool = True,
):
    """
    Compute the items logits on a subset of sampled candidates to optimize
    training. During inference, the scores are computed over the whole
    catalog of items.
    The PopularityBasedSampler is used for sampled softmax [1]_ [2]_ [3]_.
    That implementation does not require the actual item frequencies/probabilities
    if the item ids are sorted by frequency. The PopularityBasedSampler
    approximates the item probabilities using the log_uniform (zipfian) distribution.

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
        A SequenceBlock that performs popularity-based sampling of negatives, scores
        the items and applies the logQ correction for sampled softmax

    References
    ----------
    .. [1] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic
       Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial
       Intelligence and Statistics (AISTATS).

    .. [2 Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312

    .. [3] Jean, Sébastien, et al. "On using very large target vocabulary for neural
        machine translation." arXiv preprint arXiv:1412.2007 (2014).
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    item_domain = categorical_domains(schema)[item_id_feature_name]
    num_classes = categorical_cardinalities(schema)[item_id_feature_name]
    sampler = PopularityBasedSampler(
        max_num_samples=num_sampled,
        max_id=num_classes - 1,
        min_id=min_id,
        item_id_feature_name=item_id_feature_name,
    )

    retrieval_scorer = ItemRetrievalScorer(
        samplers=[sampler],
        sampling_downscore_false_negatives=ignore_false_negatives,
        item_id_feature_name=item_id_feature_name,
        item_domain=item_domain,
        sampled_softmax_mode=True,
    )

    expected_items_distribution = sampler.get_distribution_probs()
    logq_correction = PopularityLogitsCorrection(expected_items_distribution, schema=schema)

    return retrieval_scorer.connect(logq_correction)


def NextItemPredictionTask(
    schema: Schema,
    weight_tying: bool = True,
    masking: bool = True,
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    logits_temperature: float = 1.0,
    l2_normalization: bool = False,
    sampled_softmax: bool = False,
    num_sampled: int = 100,
    min_sampled_id: int = 0,
    post_logits: Optional[Block] = None,
) -> MultiClassClassificationTask:
    """
    Function to create the NextItemPrediction task with the right parameters.
    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
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
        l2_normalization: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to False.
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
        post_logits: Optional[PredictionBlock]
            Optional extra pre-call block for post-processing the logits, by default None.
            You can for example use `post_logits = mm.PopularitySamplingBlock(item_fequency)`
            for populariy sampling correction.
    Returns
    -------
        PredictionTask
            The next item prediction task
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    if sampled_softmax:
        prediction_call = ItemsPredictionPopSampled(
            schema, num_sampled=num_sampled, min_id=min_sampled_id
        )

    else:
        if weight_tying:
            prediction_call = ItemsPredictionWeightTying(schema)

        else:
            prediction_call = ItemsPrediction(schema)

        prediction_call = prediction_call.connect(LabelToOneHot())

    if post_logits is not None:
        prediction_call = prediction_call.connect(post_logits)

    if logits_temperature != 1:
        prediction_call = prediction_call.connect(LogitsTemperatureScaler(logits_temperature))

    if masking:
        prediction_call = MaskingHead(item_id_feature_name=item_id_feature_name).connect(
            RemovePad3D(), prediction_call
        )

    if l2_normalization:
        prediction_call = L2Norm().connect(prediction_call)

    if extra_pre_call is not None:
        prediction_call = prediction_call.connect(extra_pre_call)

    return MultiClassClassificationTask(
        target_name,
        task_name,
        task_block,
        pre=prediction_call,
    )
