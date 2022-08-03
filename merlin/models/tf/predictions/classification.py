from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.predictions.base import PredictionBlock


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BinaryPrediction(PredictionBlock):
    """
    Binary-classification prediction block.

    Parameters
    ----------
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for binary-classification
        by 'binary_crossentropy'
    default_metrics: Sequence[tf.keras.metrics.Metric], optional
        Default metrics to use for binary-classification
    """

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss="binary_crossentropy",
        default_metrics=(
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ),
        **kwargs,
    ):
        prediction = kwargs.pop("prediction", None)
        super().__init__(
            prediction=prediction or tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )
