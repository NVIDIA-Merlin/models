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
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import merlin.models.tf as ml
from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ParallelBlock
from merlin.models.tf.utils import tf_utils

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


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
      user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER), ml.MLPBlock([512, 256]))
      item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM), ml.MLPBlock([512, 256]))
      third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM), ml.MLPBlock([64]))
      two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower})
      model = ml.Model(two_tower, ml.ItemRetrievalTask())
      optimizer = MultiOptimizer(default_optimizer="adam",
        optimizers_and_blocks=[
          (tf.keras.optimizers.SGD(), [user_tower, third_tower]),
          (tf.keras.optimizers.Adam(), item_tower),
        ])

      # The string identification of optimizer is also acceptable:
      optimizer = MultiOptimizer(default_optimizer="adam",
        optimizers_and_blocks=[
          ("sgd", [user_tower, third_tower]),
          ("adam", item_tower),
        ])
    ```

    """

    def __init__(
        self,
        optimizers_and_blocks: Sequence[
            Tuple[Union[str, tf.keras.optimizers.Optimizer], Sequence[Block]]
        ],
        default_optimizer: Union[str, tf.keras.optimizers.Optimizer] = "rmsprop",
        name: str = "MultiOptimizer",
    ):
        """Initializes an CompositeOptimizer instance.

        Parameters
        ----------
        optimizers_and_blocks:  List of tuples of (optimizer instance, blocks that the optimizer
            should apply to, which could be a list of blocks or only one block).
        default: default optimizer for all blocks not specified in optimizers_and_blocks, by
            default "rmsprop".
        name: The optimizer name.
        """
        super().__init__(name=name)
        self.name = name
        if not optimizers_and_blocks:
            raise ValueError("`optimizers_and_blocks` can't be empty")
        self.default_optimizer = tf.keras.optimizers.get(default_optimizer)
        self.optimizers_and_blocks = []
        for i, optimizer_and_blocks in enumerate(optimizers_and_blocks):
            optimizer = tf.keras.optimizers.get(optimizer_and_blocks[0])
            self._track_trackable(optimizer, name=f"Optimizer{i}")
            self.optimizers_and_blocks.append((optimizer, optimizer_and_blocks[1]))
        self.update_optimzier_and_block = []

    def _get_trainable_variables_optimizer_dict(self, optimizers_and_blocks, require_disjoint=True):

        attribute = "_trainable_weights"
        for optimizer, blocks in optimizers_and_blocks:
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
                            if require_disjoint and v.ref() in self.var_optimizer_dict:
                                raise ValueError(
                                    f"The set of variables handled by each optimizer should be "
                                    f"disjoint, but variable {v} of {current_module} in block "
                                    f"{block} is handled both by {self.var_optimizer_dict[v.ref()]}"
                                    f" and {optimizer}."
                                )
                            self.var_optimizer_dict[v.ref()] = optimizer

                    for sub_module in current_module._flatten_modules(
                        include_self=False, recursive=False
                    ):
                        # filter out modelcontext to avoiding assign two optimizers to one variable
                        if type(sub_module) != ml.ModelContext:
                            deque.append(sub_module)
        return

    def apply_gradients(
        self,
        grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
        name: Optional[str] = None,
        experimental_aggregate_gradients: bool = True,
    ) -> None:
        # Can be replaced by block.trainable_variables if ModelContext is removed
        self.var_optimizer_dict = {}
        self._get_trainable_variables_optimizer_dict(
            self.optimizers_and_blocks, require_disjoint=True
        )
        if len(self.update_optimzier_and_block) > 0:
            self._get_trainable_variables_optimizer_dict(
                self.update_optimizer_and_blocks, require_disjoint=False
            )
        optimizer_grads_and_vars = collections.defaultdict(list)
        # Category variables by the optimizer
        for g, v in grads_and_vars:
            if v.ref() in self.var_optimizer_dict:
                optimizer = self.var_optimizer_dict[v.ref()]
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
        return

    def add(
        self,
        optimizer: Union[str, tf.keras.optimizers.Optimizer],
        blocks: Sequence[Block],
    ):
        """add another optimzier and specify which block to apply this optimizer to. If the block is
        specified with an optimizer before, it may raise conflict error, please consider calling
        self.update() method"""
        len_exist_optimizers = len(self.optimizers_and_blocks)
        optimizer = tf.keras.optimizers.get(optimizer)
        # Check if already track the optimizer
        optimizer_not_exists = True
        for opt, blocks in self.optimizers_and_blocks:
            if optimizer == opt:
                optimizer_not_exists = False
        if optimizer_not_exists:
            self._track_trackable(optimizer, name=f"Optimizer{1+len_exist_optimizers}")

        self.optimizers_and_blocks.append((optimizer, blocks))
        return

    def update(
        self,
        optimizer: Union[str, tf.keras.optimizers.Optimizer],
        blocks: Sequence[Block],
    ):
        """update the optimzier of a block, which would update the block's optimizer no matter
        what optimizer it used to utilize. If the block is not specified with an optimizer before,
        this functions would have the same functionality as self.add()"""
        len_exist_optimizers = len(self.optimizers_and_blocks)
        optimizer = tf.keras.optimizers.get(optimizer)
        # Check if already track the optimizer
        optimizer_not_exists = True
        for opt, pre_blocks in enumerate(self.optimizers_and_blocks):
            if optimizer == opt:
                optimizer_not_exists = False
            for pre_block in pre_blocks:
                if pre_block in blocks:
                    pre_blocks.remove(pre_block)
        if optimizer_not_exists:
            self._track_trackable(optimizer, name=f"Optimizer{1+len_exist_optimizers}")
        self.update_optimzier_and_blocks.append((optimizer, blocks))
        return

    def get_config(self):
        config = dict()
        config = tf_utils.maybe_serialize_keras_objects(self, config, ["default_optimizer"])
        config["name"] = self.name
        config["optimizers_and_blocks"] = []
        for optimizer, blocks in self.optimizers_and_blocks:
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
                (
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
        return [optimizer for optimizer, _ in self.optimizers_and_blocks]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class LazyAdam(tf.keras.optimizers.Adam):
    """Variant of the Adam optimizer that handles sparse updates more efficiently.

    The original Adam algorithm maintains two moving-average accumulators for each trainable
    variable; the accumulators are updated at every step. This class provides lazier handling of
    gradient updates for sparse variables.  It only updates moving-average accumulators for sparse
    variable indices that appear in the current batch, rather than updating the accumulators for all
    indices. Compared with the original Adam optimizer, it can provide large improvements in model
    training throughput for some applications. However, it provides slightly different semantics
    than the original Adam algorithm, and may lead to different empirical results.

    Note, amsgrad is currently not supported and the argument can only be False.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        amsgrad: bool = False,
        name: str = "LazyAdam",
        **kwargs,
    ):
        """Constructs a new LazyAdam optimizer.

        Parameters
        ----------
        learning_rate: Union[FloatTensorLike, Callable]
            A `Tensor` or a floating point value. or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule` The learning rate.
        beta_1: FloatTensorLike
            A `float` value or a constant `float` tensor. The exponential decay rate for the 1st
            moment estimates.
        beta_2: FloatTensorLike
            A `float` value or a constant `float` tensor.   The exponential decay rate for the 2nd
            moment estimates.
        epsilon: FloatTensorLike
            A small constant for numerical stability. This epsilon is "epsilon hat" in [Adam: A
            Method for Stochastic Optimization. Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
            (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: bool
            Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of
            Adam and beyond". Note that this argument is currently not supported and the argument
            can only be `False`.
        name: str
            Optional name for the operations created when applying gradients. Defaults to
            "LazyAdam".
        **kwargs:
            keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm`
            is clip gradients by norm; `clipvalue` is clip gradients by value, `decay` is included
            for backward compatibility to allow time inverse decay of learning rate. `lr` is
            included for backward compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr = lr_t * tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # \\(m := beta1 * m + (1 - beta1) * g_t\\)
        m = self.get_slot(var, "m")
        m_t_slice = beta_1_t * tf.gather(m, indices) + (1 - beta_1_t) * grad
        m_update_op = self._resource_scatter_update(m, indices, m_t_slice)

        # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
        v = self.get_slot(var, "v")
        v_t_slice = beta_2_t * tf.gather(v, indices) + (1 - beta_2_t) * tf.math.square(grad)
        v_update_op = self._resource_scatter_update(v, indices, v_t_slice)

        # \\(variable += -learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
        var_slice = lr * m_t_slice / (tf.math.sqrt(v_t_slice) + epsilon_t)
        var_update_op = self._resource_scatter_sub(var, indices, var_slice)

        return tf.group(*[var_update_op, m_update_op, v_update_op])

    def _resource_scatter_update(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterUpdate
        )

    def _resource_scatter_sub(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterSub
        )

    def _resource_scatter_operate(self, resource, indices, update, resource_scatter_op):
        resource_update_kwargs = {
            "resource": resource.handle,
            "indices": indices,
            "updates": update,
        }
        return resource_scatter_op(**resource_update_kwargs)


def split_embeddings_on_size(
    embeddings: ParallelBlock, threshold: int
) -> Tuple[List[Block], List[Block]]:
    """split embedding tables in ParallelBlock based on size threshold (first dimension of embedding
    tables), return a tuple of two lists, which contain large embeddings and small embeddings"""
    large_embeddings, small_embeddings = [], []
    for key, block in embeddings.parallel_dict.items():
        if block.input_dim >= threshold:
            large_embeddings.append(block)
        else:
            small_embeddings.append(block)
    if len(large_embeddings) < 1:
        warnings.warn(
            f"All embedding tables in given ParallelBlock {embeddings.name} have smaller "
            f"input dim than threshold {threshold}, thus return empty list."
        )
    return (large_embeddings, small_embeddings)
