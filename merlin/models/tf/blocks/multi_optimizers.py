import collections
from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.utils import tf_utils

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """An optimizer that composes multiple individual optimizers.

    It allows different optimizers to be applied to different subsets of the
    model's variables. For example, it makes it possible to apply one
    optimizer to the blocks which contains the model's embeddings (sparse variables)
    and another optimizer to the rest of its variables (other blocks).

    To specify which optimizer should apply to each block, pass a list of
    pairs of (optimizer instance, blocks the optimizer should apply to).

    For example:
    ```python
      user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER), ml.MLPBlock([512, 256]))
      item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM), ml.MLPBlock([512, 256]))
      two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower})
      model = ml.Model(two_tower, ml.ItemRetrievalTask())
      optimizer = CompositeOptimizer([
          (tf.keras.optimizers.SGD(), user_tower),
          (tf.keras.optimizers.Adam(), item_tower),
      ])
    ```
    """

    def __init__(
        self,
        default_optimizer: tf.keras.optimizers.Optimizer,
        optimizers_and_blocks: Sequence[Tuple[tf.keras.optimizers.Optimizer, Sequence[Block]]],
        name: str = "MultiOptimizer",
    ) -> None:
        """Initializes an CompositeOptimizer instance.

        Args:
          default: default optimizer for all blocks not specified in optimizers_and_blocks
          optimizers_and_blocks:  List of tuples of (optimizer instance, function
            returning variables that the optimizer should apply to).
          name: The optimizer name.
        """
        super().__init__(name=name)
        if not default_optimizer:
            raise ValueError("`default` can't be empty")
        if not optimizers_and_blocks:
            raise ValueError("`optimizers_and_blocks` can't be empty")
        self.optimizers_and_blocks = optimizers_and_blocks
        self.default_optimizer = default_optimizer
        for i, optimizer_and_blocks in enumerate(optimizers_and_blocks):
            optimizer = optimizer_and_blocks[0]
            self._track_trackable(optimizer, name=f"Optimizer{i}")

    def apply_gradients(
        self,
        grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
        name: Optional[str] = None,
        experimental_aggregate_gradients: bool = True,
    ) -> None:
        """See base class."""
        var_optimizer_dict = {}

        for optimizer, block in self.optimizers_and_blocks:
            # TODO: iterate child block
            for v in block.trainable_variables():
                if v.ref() in var_optimizer_dict:
                    raise ValueError(
                        f"The set of variables handled by each optimizer should be "
                        f"disjoint, but variable {v} of block {block} is handled both "
                        f"by {var_optimizer_dict[v.ref()]} and {optimizer}."
                    )
                var_optimizer_dict[v.ref()] = optimizer

        optimizer_grads_and_vars = collections.defaultdict(list)
        for g, v in grads_and_vars:
            if v.ref() in var_optimizer_dict:
                optimizer = var_optimizer_dict[v.ref()]
                optimizer_grads_and_vars[optimizer].append((g, v))
            else:
                optimizer_grads_and_vars[self.default_optimizer].append((g, v))
        for optimizer, opt_grads_and_vars in optimizer_grads_and_vars.items():
            optimizer.apply_gradients(
                opt_grads_and_vars,
                name=name,
                experimental_aggregate_gradients=experimental_aggregate_gradients,
            )

    def get_config(self):
        config = dict()
        config = tf_utils.maybe_serialize_keras_objects(
            self,
            config,
            ["default_optimizer", "optimizers_and_blocks"],
        )
        config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config, ["default_optimizer", "optimizers_and_blocks"]
        )
        return cls(**config)

    @property
    def iterations(self):
        """See base class."""
        # Returning iterations from the first optimizer.
        return self.optimizers_and_blocks[0][0].iterations

    @iterations.setter
    def iterations(self, variable):
        """See base class."""
        for optimizer, _ in self.optimizers_and_blocks:
            optimizer.iterations = variable

    def variables(self):
        """Returns the optimizer's variables."""
        # OptimizerV2.variables() returns self._weights, so override that method.
        return self.weights

    @property
    def weights(self) -> List[tf.Variable]:
        """Returns the optimizer's variables."""
        weights = []
        for optimizer, _ in self.optimizers_and_blocks:
            weights += optimizer.weights
        return weights

    @property
    def optimizers(self) -> List[tf.keras.optimizers.Optimizer]:
        """Returns the optimizers in composite optimizer (in the original order)."""
        return [optimizer for optimizer, _ in self._optimizers_and_blocks]
