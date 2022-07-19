import tensorflow as tf

from merlin.models.tf.predictions.base import PredictionBlock


class RegressionPrediction(PredictionBlock):
    """Regression prediction block"""

    def __init__(
        self,
        target=None,
        pre=None,
        post=None,
        logits_temperature=1.0,
        default_loss="mse",
        default_metrics=(tf.keras.metrics.RootMeanSquaredError,),
    ):
        super().__init__(
            prediction=tf.keras.layers.Dense(1, activation="linear"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
        )
