from typing import Dict, List, Optional, Union

import tensorflow as tf

from merlin.models.tf.core.tabular import TabularBlock
from merlin.models.tf.distributed.backend import dmp, dmp_installed, hvd_installed
from merlin.models.utils.schema_utils import infer_embedding_dim
from merlin.schema import Schema


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DistributedEmbeddings(TabularBlock):
    """Large embedding table that automatically distributes embedding tables
    to multiple GPUs.

    Parameters
    ----------
    schema: Schema
        Schema containing the columns used in embedding tables.
    dim: Optional[Union[Dict[str, int], int]], optional
        If int, the embedding size to use for all features, or a
        dictionary-like {"feature_name": embedding size, ...}.
        By default, None.
    strategy:
    column_slice_threshold:
    dp_input:
    input_table_map:
    """

    def __init__(
        self,
        schema: Schema,
        dim: Optional[Union[Dict[str, int], int]] = None,
        strategy: str = "basic",
        column_slice_threshold: Optional[int] = None,
        dp_input=True,
        input_table_map=None,
        **kwargs,
    ):
        if not hvd_installed or not dmp_installed:
            raise ImportError(
                "'horovod' and 'distributed-embeddings' are required to use "
                f"{self.__class__.__name__}."
            )

        super(DistributedEmbeddings, self).__init__(schema=schema, **kwargs)

        self.dim = dim
        self.table_names = []
        self.embedding_layers = []

        for col in self.schema:
            table_name = col.int_domain.name or col.name
            self.table_names.append(table_name)
            self.embedding_layers.append(
                tf.keras.layers.Embedding(
                    input_dim=self._infer_input_dim(col),
                    output_dim=self._infer_output_dim(col, dim),
                    name=table_name,
                )
            )

        self.embedding_layers = dmp.DistributedEmbedding(
            self.embedding_layers,
            strategy=strategy,
            column_slice_threshold=column_slice_threshold,
            dp_input=dp_input,
            input_table_map=input_table_map,
        )

    def _infer_input_dim(self, col_schema):
        return col_schema.int_domain.max + 1

    def _infer_output_dim(self, col_schema, embedding_dims):
        if isinstance(embedding_dims, dict):
            dim = embedding_dims.get(col_schema.name)
        elif isinstance(embedding_dims, int):
            dim = embedding_dims
        else:
            dim = None

        if dim is None:
            dim = infer_embedding_dim(col_schema)

        return dim

    def build(self, input_shapes):
        super().build(input_shapes)

        if self.embedding_layers.built is True:
            return

        if isinstance(input_shapes, dict):
            ordered_input_shapes = []
            for feature_name in self.table_names:
                ordered_input_shapes.append(input_shapes[feature_name])
        elif isinstance(input_shapes, list):
            ordered_input_shapes = input_shapes
        else:
            raise ValueError(f"Unexpected input type encountered: {input_shapes}")
        self.embedding_layers.build(ordered_input_shapes)

    @tf.function
    def call(
        self, inputs: Union[Dict[str, tf.Tensor], List[tf.Tensor]]
    ) -> Union[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        Parameters
        ----------
        inputs : Union[Dict[str, tf.Tensor], List[tf.Tensor]]
            Tensors or dictionary of tensors representing the input batch.

        Returns
        -------
        A tensor or dict of tensors corresponding to the embeddings for inputs
        """

        def _validate_inputs(tensor):
            depth = 100
            if isinstance(tensor, tf.SparseTensor):
                max_value = tf.reduce_max(tensor.values)
                min_value = tf.reduce_min(tensor.values)
            else:
                max_value = tf.reduce_max(tensor)
                min_value = tf.reduce_min(tensor)
            condition = tf.logical_and(
                tf.greater(tf.cast(depth, max_value.dtype), max_value),
                tf.greater_equal(min_value, tf.cast(0, min_value.dtype)),
            )
            return condition

        if isinstance(inputs, dict):
            ordered_inputs = []
            outputs = {}
            for feature_name in self.table_names:
                with tf.control_dependencies([_validate_inputs(inputs[feature_name])]):
                    ordered_inputs.append(inputs[feature_name])
            ordered_outputs = self.embedding_layers(ordered_inputs)
            for feature_name, output in zip(self.schema.column_names, ordered_outputs):
                outputs[feature_name] = output
        elif isinstance(inputs, list):
            with tf.control_dependencies([_validate_inputs(inputs)]):
                outputs = self.embedding_layers(inputs)
        else:
            raise ValueError(f"Unexpected input type encountered: {inputs}")

        return outputs

    @tf.function
    def compute_call_output_shape(self, input_shapes):
        def _get_output_shape(input_shape):
            batch_size = input_shape[0]
            output_shape = tf.TensorShape([batch_size, self.dim])
            return output_shape

        if isinstance(input_shapes, dict):
            output_shapes = {k: _get_output_shape(v) for k, v in input_shapes.items()}
        elif isinstance(input_shapes, list):
            output_shapes = [_get_output_shape(x) for x in input_shapes]
        else:
            raise ValueError(f"Unexpected input type encountered: {input_shapes}")

        return output_shapes
