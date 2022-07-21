from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.predictions.base import PredictionBlock


class BinaryPrediction(PredictionBlock):
    """Binary-classification prediction block"""

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss="binary_crossentropy",
        default_metrics=(
            tf.keras.metrics.Precision,
            tf.keras.metrics.Recall,
            tf.keras.metrics.BinaryAccuracy,
            tf.keras.metrics.AUC,
        ),
        **kwargs,
    ):
        super().__init__(
            prediction=tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )
