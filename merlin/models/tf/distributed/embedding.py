from typing import List, Union

import tensorflow as tf

from merlin.models.tf.inputs.embedding import EmbeddingTableBase
from merlin.models.tf.distributed.backend import dmp
from merlin.schema import ColumnSchema
from merlin.models.tf.typing import TabularData


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DistributedEmbeddingTable(EmbeddingTableBase):
    """Large embedding table that automatically distributes embedding tables to multiple GPUs."""

    def __init__(
        self,
        dim: int,
        *col_schemas: ColumnSchema,
        embeddings_initializer="uniform",
        # sequence_combiner: Optional[CombinerType] = None,
        trainable: bool = True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super(DistributedEmbeddingTable, self).__init__(
            dim,
            *col_schemas,
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs,
        )

        self.embedding_layers = []
        self.embeddings_initializer = embeddings_initializer

        self._create_embeddings()

    def _create_embeddings(self):
        for table_size in self.table_sizes:
            # if model_flags.test_combiner:
            #  self.embedding_layers.append(
            #      embedding.Embedding(input_dim=table_size,
            #                          output_dim=self.embedding_dim,
            #                          embeddings_initializer=DLRMInitializer(),
            #                          combiner='sum'))
            # else:
            self.embedding_layers.append(
                tf.keras.layers.Embedding(
                    input_dim=self.col_schema.int_domain.max,
                    output_dim=self.dim,
                    embeddings_initializer=self.embeddings_initializer,
                )
            )
        self.embedding_layers = dmp.DistributedEmbedding(self.embedding_layers)

    def call(self, inputs: Union[tf.Tensor, TabularData]) -> Union[tf.Tensor, TabularData]:
        """
        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
            Tensors or dictionary of tensors representing the input batch.
        Returns
        -------
        A tensor or dict of tensors corresponding to the embeddings for inputs
        """
        if isinstance(inputs, dict):
            outputs = {}
            for feature_name in self.schema.column_names:
                if feature_name in inputs:
                    embedding_outputs = self.embedding_layers(inputs[feature_name])
                    #outputs[feature_name] = tf.concat(embedding_outputs, 1)
        else:
            embedding_outputs = self.embedding_layers(inputs)
            #outputs = tf.concat(embedding_outputs, 1)

        #return outputs
        return embedding_outputs
