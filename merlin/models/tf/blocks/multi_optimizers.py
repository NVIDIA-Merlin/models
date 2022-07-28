#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import collections
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf

import merlin.models.tf as ml
from merlin.models.tf.core.base import Block
from merlin.models.tf.utils import tf_utils

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


@dataclass
class OptimizerBlocks:
    """dataclass for a pair of optimizer and blocks that the optimizer should apply to.
    Example:
        ml.OptimizerBlocks("sgd", [user_tower, third_tower])
        ml.OptimizerBlocks("adam", item_tower)
    """

    optimizer: Union[str, tf.keras.optimizers.Optimizer]
    blocks: Sequence[Block]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """An optimizer that composes multiple individual optimizers.

    It allows different optimizers to be applied to different subsets of the model's variables. For
    example, it is possible to apply one optimizer to the blocks which contains the model's
    embeddings (sparse variables) and another optimizer to the rest of its variables (other blocks).

    To specify which optimizer should apply to each block, pass a list of pairs of (optimizer
    instance, blocks the optimizer should apply to).

    For example:
    ```python
      import merlin.models.tf as ml
      user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([512, 256]))
      item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([512, 256]))
      third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([64]))
      three_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower, "third": third_tower})
      model = ml.Model(three_tower,  ml.BinaryClassificationTask("click"))

      # The third_tower would be assigned the default_optimizer ("adagrad" in this example)
      optimizer = ml.MultiOptimizer(default_optimizer="adagrad",
        optimizers_and_blocks=[
          ml.OptimizerBlocks(tf.keras.optimizers.SGD(), user_tower),
          ml.OptimizerBlocks(tf.keras.optimizers.Adam(), item_tower),
        ])

      # The string identification of optimizer is also acceptable, here "sgd" for the third_tower
      # The variables of BinaryClassificationTask("click") would still use the default_optimizer
      optimizer = ml.MultiOptimizer(default_optimizer="adam",
        optimizers_and_blocks=[
          ml.OptimizerBlocks("sgd", [user_tower, third_tower]),
          ml.OptimizerBlocks("adam", item_tower),
        ])
    ```

    """

    def __init__(
        self,
        optimizers_and_blocks: Sequence[OptimizerBlocks],
        default_optimizer: Union[str, tf.keras.optimizers.Optimizer] = "rmsprop",
        name: str = "MultiOptimizer",
    ):
        """Initializes an MultiOptimizer instance.

        Parameters
        ----------
        optimizers_and_blocks: Sequence[OptimizerBlocks]
            List of OptimizerBlocks(dataclass), the OptimizerBlocks contains two items, one is
            optimizer, another one is a list of blocks or a block that the optimizer should apply
            to. See 'class OptimizerBlocks'
        default_optimizer: Union[str, tf.keras.optimizers.Optimizer]
            Default optimizer for the rest variables not specified in optimizers_and_blocks, by
            default "rmsprop".
        name:str
            The name of MultiOptimizer.
        """
        super().__init__(name=name)
        self.name = name
        if not optimizers_and_blocks:
            raise ValueError("`optimizers_and_blocks` can't be empty")
        self.default_optimizer = tf.keras.optimizers.get(default_optimizer)
        self.optimizers_and_blocks = []
        for i, pair in enumerate(optimizers_and_blocks):
            pair.optimizer = tf.keras.optimizers.get(pair.optimizer)
            self._track_trackable(pair.optimizer, name=f"Optimizer{i}")
            self.optimizers_and_blocks.append(pair)

    def get_trainable_variables_optimizer_dict(self):
        var_optimizer_dict = {}
        attribute = "_trainable_weights"
        for pair in self.optimizers_and_blocks:
            optimizer = pair.optimizer
            blocks = pair.blocks
            for block in blocks:
                # Iterate all submodule (BFS) except ModelContext
                # Note: block.trainable_variables is not used because modelcontext contain all
                # variables, you may iterate the same variable twice in different block, causing
                # disjoint error. Consider replace this iteration method to simply call
                # block.trainable_variables in the future when ModelContext is deleted
                deque = collections.deque()
                deque.append(block)
                while deque:
                    current_module = deque.popleft()
                    if hasattr(current_module, attribute):
                        for v in current_module._trainable_weights:
                            if v.ref() in var_optimizer_dict:
                                raise ValueError(
                                    f"The set of variables handled by each optimizer should be "
                                    f"disjoint, but variable {v} of {current_module} in block "
                                    f"{block} is handled both by {var_optimizer_dict[v.ref()]} and "
                                    f"{optimizer}."
                                )
                            var_optimizer_dict[v.ref()] = optimizer

                    for sub_module in current_module._flatten_modules(
                        include_self=False, recursive=False
                    ):
                        # filter out modelcontext to avoiding assign two optimizers to one variable
                        if type(sub_module) != ml.ModelContext:
                            deque.append(sub_module)
        return var_optimizer_dict

    def apply_gradients(
        self,
        grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
        name: Optional[str] = None,
        experimental_aggregate_gradients: bool = True,
    ) -> None:
        # Can be replaced by block.trainable_variables if ModelContext is removed
        var_optimizer_dict = self.get_trainable_variables_optimizer_dict()
        optimizer_grads_and_vars = collections.defaultdict(list)
        # Category variables by the optimizer
        for g, v in grads_and_vars:
            if v.ref() in var_optimizer_dict:
                optimizer = var_optimizer_dict[v.ref()]
                optimizer_grads_and_vars[optimizer].append((g, v))
            # for variables not in optimizers_and_blocks, assign default optimizer
            else:
                optimizer_grads_and_vars[self.default_optimizer].append((g, v))
        # Apply gradient for each optimizer
        for optimizer, opt_grads_and_vars in optimizer_grads_and_vars.items():
            optimizer.apply_gradients(
                opt_grads_and_vars,
                name=name,
                experimental_aggregate_gradients=experimental_aggregate_gradients,
            )

    def add(
        self,
        optimizer_blocks: OptimizerBlocks,
    ):
        """add another optimzier and specify which block to apply this optimizer to"""
        len_exist_optimizers = len(self.optimizers_and_blocks)
        optimizer_blocks.optimizer = tf.keras.optimizers.get(optimizer_blocks.optimizer)
        optimizer = optimizer_blocks.optimizer
        # Check if already track the optimizer
        optimizer_not_exists = True
        for opt_blocks in self.optimizers_and_blocks:
            if optimizer == opt_blocks.optimizer:
                optimizer_not_exists = False
        if optimizer_not_exists:
            self._track_trackable(optimizer, name=f"Optimizer{1+len_exist_optimizers}")

        self.optimizers_and_blocks.append(optimizer_blocks)
        return

    def get_config(self):
        config = dict()
        config = tf_utils.maybe_serialize_keras_objects(self, config, ["default_optimizer"])
        config["name"] = self.name
        config["optimizers_and_blocks"] = []
        for optimizer_blocks in self.optimizers_and_blocks:
            optimizer = optimizer_blocks.optimizer
            blocks = optimizer_blocks.blocks
            config["optimizers_and_blocks"].append(
                (
                    tf.keras.utils.serialize_keras_object(optimizer),
                    [tf.keras.utils.serialize_keras_object(block) for block in blocks],
                )
            )
        return config

    @classmethod
    def from_config(cls, config):
        config["default_optimizer"] = tf.keras.optimizers.deserialize(config["default_optimizer"])
        optimizers_and_blocks = []
        for optimizer, blocks in config["optimizers_and_blocks"]:
            optimizers_and_blocks.append(
                OptimizerBlocks(
                    tf.keras.optimizers.deserialize(optimizer),
                    [tf.keras.layers.deserialize(block) for block in blocks],
                )
            )
        config.update({"optimizers_and_blocks": optimizers_and_blocks})
        return cls(**config)

    @property
    def iterations(self):
        """See base class."""
        # Returning iterations from the first optimizer.
        return self.optimizers_and_blocks[0].optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        """See base class."""
        for optimizer_blocks in self.optimizers_and_blocks:
            optimizer_blocks.optimizer.iterations = variable

    def variables(self):
        """Returns the optimizer's variables."""
        # OptimizerV2.variables() returns self._weights, so override that method.
        return self.weights

    @property
    def weights(self) -> List[tf.Variable]:
        """Returns the optimizer's variables."""
        weights = []
        for optimizer_blocks in self.optimizers_and_blocks:
            weights += optimizer_blocks.optimizer.weights
        return weights

    @property
    def optimizers(self) -> List[tf.keras.optimizers.Optimizer]:
        """Returns the optimizers in MultiOptimizer (in the original order). Note: default_optimizer
        is included here"""
        return [pair.optimizer for pair in self.optimizers_and_blocks] + [self.default_optimizer]
