from typing import Optional

from merlin.models.tf import Model
from merlin.models.tf.blocks.core.base import PredictionOutput
from merlin.models.tf.prediction_blocks.base import PredictionBlock


class NegativeSampling(PredictionBlock):
    def __init__(
        self,
        samplers: Sequence[ItemSampler] = (),
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value: float = MIN_FLOAT,
        item_id_feature_name: str = "item_id",
        query_name: str = "query",
        item_name: str = "item",
        cache_query: bool = False,
        sampled_softmax_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value
        self.item_id_feature_name = item_id_feature_name
        self.query_name = query_name
        self.item_name = item_name
        self.cache_query = cache_query

        if not isinstance(samplers, (list, tuple)):
            samplers = (samplers,)  # type: ignore
        self.samplers = samplers
        self.sampled_softmax_mode = sampled_softmax_mode

        self.set_required_features()

    def call(self, inputs: PredictionOutput, training=False, **kwargs) -> PredictionOutput:
        return super().call(inputs, training, **kwargs)
