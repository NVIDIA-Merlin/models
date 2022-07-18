import tensorflow as tf

from merlin.models.tf.predictions.base import PredictionBlock


class BinaryPrediction(PredictionBlock):
    def __init__(
            self,
            default_loss="binary_crossentropy",
            default_metrics=(
                    tf.keras.metrics.Precision,
                    tf.keras.metrics.Recall,
                    tf.keras.metrics.BinaryAccuracy,
                    tf.keras.metrics.AUC,
            ),
            target=None,
            pre=None,
            post=None,
            logits_temperature=1.0
    ):
        super().__init__(
            prediction=tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature
        )
