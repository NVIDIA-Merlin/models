from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.predictions.base import PredictionBlock


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RegressionPrediction(PredictionBlock):
    """Regression prediction block

    Parameters
    ----------
    target: str, optional
        The name of the target.
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the regression scores,
        by default None
    post: Optional[Block], optional
        Optional block to transform the regression scores,
        by default None
    name: str, optional
        The name of the task.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for regression
        by 'mse'
    default_metrics: Sequence[tf.keras.metrics.Metric], optional
        Default metrics to use for regression
    """

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        name: Optional[str] = None,
        default_loss="mse",
        default_metrics=(tf.keras.metrics.RootMeanSquaredError(name="rmse"),),
        **kwargs,
    ):
        prediction = kwargs.pop("prediction", None)
        super().__init__(
            prediction=prediction or tf.keras.layers.Dense(1, activation="linear"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            name=name,
            **kwargs,
        )
