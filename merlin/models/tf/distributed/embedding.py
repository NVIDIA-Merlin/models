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
        if isinstance(inputs, dict):
            ordered_inputs = []
            for feature_name in self.table_names:
                ordered_inputs.append(inputs[feature_name])

            ordered_outputs = self.embedding_layers(ordered_inputs)
            outputs = {}
            for feature_name, output in zip(self.schema.column_names, ordered_outputs):
                outputs[feature_name] = output
        else:
            outputs = self.embedding_layers(inputs)
        return outputs
