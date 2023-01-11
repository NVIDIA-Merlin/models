from functools import partial

import tensorflow as tf

from merlin.models.utils.registry import Registry

metrics_registry: Registry = Registry.class_registry("tf.metrics")


metrics_registry.register_with_multiple_names("auc")(partial(tf.keras.metrics.AUC, name="auc"))
metrics_registry.register_with_multiple_names("precision")(
    partial(tf.keras.metrics.Precision, name="precision")
)
metrics_registry.register_with_multiple_names("recall")(
    partial(tf.keras.metrics.Recall, name="recall")
)
metrics_registry.register_with_multiple_names("binary_accuracy")(
    partial(tf.keras.metrics.BinaryAccuracy, name="binary_accuracy")
)
metrics_registry.register_with_multiple_names("mean_squared_error", "mse")(
    partial(tf.keras.metrics.MeanSquaredError, name="mean_squared_error")
)
metrics_registry.register_with_multiple_names("root_mean_squared_error", "rmse")(
    partial(tf.keras.metrics.RootMeanSquaredError, name="root_mean_squared_error")
)

metrics_registry.register_with_multiple_names("categorical_accuracy")(
    partial(tf.keras.metrics.CategoricalAccuracy, name="categorical_accuracy")
)
metrics_registry.register_with_multiple_names("accuracy")(
    partial(tf.keras.metrics.Accuracy, name="accuracy")
)
