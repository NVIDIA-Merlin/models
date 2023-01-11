from typing import Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.outputs.base import MetricsFn, ModelOutput
from merlin.schema import ColumnSchema


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RegressionOutput(ModelOutput):
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
    get_default_metrics: Callable, optional
        A function returning the list of default metrics to set
        if the user does not specify any
        Default metrics to use for regression
    """

    def __init__(
        self,
        target: Optional[Union[str, ColumnSchema]] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        name: Optional[str] = None,
        default_loss="mse",
        default_metrics_fn: MetricsFn = lambda: (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
        **kwargs,
    ):
        if isinstance(target, ColumnSchema):
            target = target.name

        to_call = kwargs.pop("to_call", None)
        super().__init__(
            to_call=to_call or tf.keras.layers.Dense(1, activation="linear"),
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            target=target,
            pre=pre,
            post=post,
            name=name,
            **kwargs,
        )
