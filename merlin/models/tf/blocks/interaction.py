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

import tensorflow as tf

from merlin.models.tf.blocks.core.base import Block

_INTERACTION_TYPES = (None, "field_all", "field_each", "field_interaction")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DotProductInteraction(tf.keras.layers.Layer):
    """
    Layer implementing the factorization machine style feature
    interaction layer suggested by the DLRM and DeepFM architectures,
    generalized to include a dot-product version of the parameterized
    interaction suggested by the FiBiNet architecture (which normally
    uses element-wise multiplication instead of dot product). Maps from
    tensors of shape `(batch_size, num_features, embedding_dim)` to
    tensors of shape `(batch_size, (num_features - 1)*num_features // 2)`
    if `self_interaction` is `False`, otherwise `(batch_size, num_features**2)`.

    Parameters
    ------------------------
    interaction_type: {}
        The type of feature interaction to use. `None` defaults to the
        standard factorization machine style interaction, and the
        alternatives use the implementation defined in the FiBiNet
        architecture (with the element-wise multiplication replaced
        with a dot product).
    self_interaction: bool
        Whether to calculate the interaction of a feature with itself.
    """.format(
        _INTERACTION_TYPES
    )

    def __init__(self, interaction_type=None, self_interaction=False, name=None, **kwargs):
        if interaction_type not in _INTERACTION_TYPES:
            raise ValueError("Unknown interaction type {}".format(interaction_type))
        self.interaction_type = interaction_type
        self.self_interaction = self_interaction
        super(DotProductInteraction, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.interaction_type is None:
            self.built = True
            return

        kernel_shape = [input_shape[2], input_shape[2]]
        if self.interaction_type in _INTERACTION_TYPES[2:]:
            idx = _INTERACTION_TYPES.index(self.interaction_type)
            for _ in range(idx - 1):
                kernel_shape.insert(0, input_shape[1])

        self.kernel = self.add_weight(
            name="bilinear_interaction_kernel",
            shape=kernel_shape,
            initializer="glorot_normal",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        right = inputs

        # first transform v_i depending on the interaction type
        if self.interaction_type is None:
            left = inputs
        elif self.interaction_type == "field_all":
            left = tf.matmul(inputs, self.kernel)
        elif self.interaction_type == "field_each":
            left = tf.einsum("b...k,...jk->b...j", inputs, self.kernel)
        else:
            left = tf.einsum("b...k,f...jk->bf...j", inputs, self.kernel)

        # do the interaction between v_i and v_j
        # output shape will be (batch_size, num_features, num_features)
        if self.interaction_type != "field_interaction":
            interactions = tf.matmul(left, right, transpose_b=True)
        else:
            interactions = tf.einsum("b...jk,b...k->b...j", left, right)

        # mask out the appropriate area
        ones = tf.reduce_sum(tf.zeros_like(interactions), axis=0) + 1
        mask = tf.linalg.band_part(ones, 0, -1)  # set lower diagonal to zero
        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)  # get rid of diagonal
        mask = tf.cast(mask, tf.bool)
        x = tf.boolean_mask(interactions, mask, axis=1)

        # masking destroys shape information, set explicitly
        x.set_shape(self.compute_output_shape(inputs.shape))
        return x

    def compute_output_shape(self, input_shape):
        if self.self_interaction:
            output_dim = input_shape[1] ** 2
        else:
            output_dim = input_shape[1] * (input_shape[1] - 1) // 2

        return input_shape[0], output_dim

    def get_config(self):
        return {
            "interaction_type": self.interaction_type,
            "self_interaction": self.self_interaction,
        }


class XDeepFmOuterProduct(tf.keras.layers.Layer):
    """
    Layer implementing the outer product transformation used in
    the Compressed Interaction Network (CIN) proposed in
    in https://arxiv.org/abs/1803.05170. Treats the feature dimension
    H_k of a B x H_k x D feature embedding tensor as a feature map
    of the D embedding elements, and computes element-wise multiplication
    interaction between these maps and those from an initial input tensor
    x_0 before taking the inner product with a parameter matrix.

    Parameters
    ------------
    dim : int
      Feature dimension of the layer. Output will be of shape
      (batch_size, dim, embedding_dim)
    """

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(**kwargs)

    def build(self, input_shapes):
        if not isinstance(input_shapes[0], (tuple, tf.TensorShape)):
            raise ValueError("Should be called on a list of inputs.")
        if len(input_shapes) != 2:
            raise ValueError("Should only have two inputs, found {}".format(len(input_shapes)))
        for shape in input_shapes:
            if len(shape) != 3:
                raise ValueError("Found shape {} without 3 dimensions".format(shape))
        if input_shapes[0][-1] != input_shapes[1][-1]:
            raise ValueError(
                "Last dimension should match, found dimensions {} and {}".format(
                    input_shapes[0][-1], input_shapes[1][-1]
                )
            )

        # H_k x H_{k-1} x m
        shape = (self.dim, input_shapes[0][1], input_shapes[1][1])
        self.kernel = self.add_weight(
            name="kernel", initializer="glorot_uniform", trainable=True, shape=shape
        )
        self.built = True

    def call(self, inputs):
        """
        Parameters
        ------------
        inputs : array-like(tf.Tensor)
          The two input tensors, the first of which should be the
          output of the previous layer, and the second of which
          should be the input to the CIN.
        """
        x_k_minus_1, x_0 = inputs

        # need to do shape manipulations so that we
        # can do element-wise multiply
        x_k_minus_1 = tf.expand_dims(x_k_minus_1, axis=2)  # B x H_{k-1} x 1 x D
        x_k_minus_1 = tf.tile(x_k_minus_1, [1, 1, x_0.shape[1], 1])  # B x H_{k-1} x m x D
        x_k_minus_1 = tf.transpose(x_k_minus_1, (1, 0, 2, 3))  # H_{k-1} x B x m x D
        z_k = x_k_minus_1 * x_0  # H_{k-1} x B x m x D
        z_k = tf.transpose(z_k, (1, 0, 2, 3))  # B x H_{k-1} x m x D

        # now we need to map to B x H_k x D
        x_k = tf.tensordot(self.kernel, z_k, axes=[[1, 2], [1, 2]])
        x_k = tf.transpose(x_k, (1, 0, 2))
        return x_k

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], self.dim, input_shapes[0][2])


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class FMPairwiseInteraction(Block):
    """Compute pairwise (2nd-order) feature interactions like defined in
    Factorized Machine [1].

    References
    ----------
    [1] Steffen, Rendle, "Factorization Machines" IEEE International
    Conference on Data Mining, 2010. https://ieeexplore.ieee.org/document/5694074
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        if len(input_shapes) != 3:
            raise ValueError("Found shape {} without 3 dimensions".format(input_shapes))
        super(FMPairwiseInteraction, self).build(input_shapes)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : array-like(tf.Tensor)
          A 3-D tensor of shape (bs, n_features, embedding_dim)
          containing the stacked embeddings of input features.

        Returns
        -------
        A 2-D tensor of shape (bs, K) containing pairwise interactions
        """
        assert len(inputs.shape) == 3, "inputs should be a 3-D tensor"

        # sum_square part
        summed_square = tf.square(tf.reduce_sum(inputs, 1))

        # square_sum part
        squared_sum = tf.reduce_sum(tf.square(inputs), 1)

        # second order
        return 0.5 * tf.subtract(summed_square, squared_sum)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[2])
