from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.predictions.base import PredictionBlock


class RegressionPrediction(PredictionBlock):
    """Regression prediction block"""

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss="mse",
        default_metrics=(tf.keras.metrics.RootMeanSquaredError(),),
        **kwargs,
    ):
        super().__init__(
            prediction=tf.keras.layers.Dense(1, activation="linear"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )
