from typing import Union

import tensorflow as tf

from merlin.models.tf.distributed.backend import hvd_installed, sok, sok_installed
from merlin.models.tf.inputs.embedding import EmbeddingTableBase
from merlin.models.utils.schema_utils import (
    create_categorical_column,
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import ColumnSchema


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SOKEmbedding(EmbeddingTableBase):
    """
    Wrap GPU accelerated opererations dedicated for sparse training / inference case.
    dim: int The last dimension of the variable
    vocab_sizes: list, rows of the variable list
    initializer: string, list = "uniform"
            When it's string, it specifies the initializer used to generate initial values.
            For sok.DynamicVariable, currently, only support "random" or string of a float
            value(meaning const initializer).
            For sok.Variable, it is compatible with tf.Variable.
            Default value is "uniform".
            When it's list, it specifies the values in the embedding table.
            For sok.DynamicVariable, initializer[i] must be list of [index, value],
            and will be used as the initial indices and value for i-th sok.DynamicVariable.
            For sok.Variable, initializer[i] must be a numpy with shape
            [vocab_size[i], embedding_vec_size],
            and will be used as the initial value for i-th sok.Variable.
    use_dynamic_variable: bool = "False" use sok.DynamicVariable or sok.Variable. DynamicVariable
            can allocates memory dynamically. Variable is a model-parallel distributed variable
    localized: When utilizing sok.Variable, we change choose two mode: distributed(Distributed Va
            riable) and localized(Localized Variable). If set to None, use Distributed Variable,
            otherwise Localized Variable. where the list indicates which GPU you want to put this
            variable on.
            Default is None.
    Examples
    --------
    .. code-block:: python
    Notes
    -----
    """

    def __init__(
        self,
        dim: int,
        *col_schemas: ColumnSchema,
        vocab_sizes: list,
        initializer: Union[str, tf.Tensor, list] = "uniform",
        use_dynamic_variable=False,
        localized=None,
        trainable=True,
        name=None,
        dtype=None,
        **kwargs,
    ):
        if not hvd_installed or not sok_installed:
            raise ImportError(
                "'horovod' and 'sparse_operation_kit' are required to use "
                f"{self.__class__.__name__}."
            )

        super(SOKEmbedding, self).__init__(
            dim, *col_schemas, trainable=trainable, name=name, dtype=dtype, **kwargs
        )
        self._embedding_vec_size = dim
        self._vocab_sizes = vocab_sizes
        self._use_dynamic_variable = use_dynamic_variable
        self._localized = localized
        self._initializer = initializer
        self._vars = []
        if self._localized is None and self._use_dynamic_variable is False:
            for i in range(len(vocab_sizes)):
                if isinstance(initializer, str):
                    v = sok.Variable(
                        shape=[self._vocab_sizes[i], self._embedding_vec_size],
                        initializer=tf.keras.initializers.get(initializer),
                        dtype=tf.float32,
                    )
                else:
                    v = sok.Variable(initializer[i])
        else:
            for i in range(len(vocab_sizes)):
                if self._use_dynamic_variable:
                    if isinstance(initializer, str):
                        v = sok.DynamicVariable(
                            dimension=self._embedding_vec_size, initializer=initializer
                        )
                    else:
                        v = sok.DynamicVariable(
                            dimension=self._embedding_vec_size, initializer="random"
                        )
                        indices = tf.convert_to_tensor(initializer[i][0], dtype=tf.int64)
                        values = tf.convert_to_tensor(initializer[i][1], dtype=tf.float32)
                        sok.assign(v, indices, values)
                elif self._localized is not None:
                    if isinstance(initializer, str):
                        v = sok.Variable(
                            shape=[self._vocab_sizes[i], self._embedding_vec_size],
                            initializer=tf.keras.initializers.get(initializer),
                            dtype=tf.float32,
                            mode="localized:%d" % self._localized[i],
                        )
                    else:
                        v = sok.Variable(
                            initializer[i],
                            mode="localized:%d" % self._localized[i],
                        )
                else:
                    raise ValueError("Wrong Configuration!!!")
        self._trainable_weights.append(v)
        self._vars.append(v)

    def call(self, inputs, combiners, training=True):
        """
        inputs: list, tuple
            a list or tuple of tf.SparseTensor or tf.RaggedTensor.
        combiners: list, tuple
            a list or tuple of string to specify the combiner of each lookup.
        """
        is_list = isinstance(inputs, list) or isinstance(inputs, tuple)
        if is_list:
            for cur_input in inputs:
                if not isinstance(cur_input, tf.SparseTensor):
                    if not isinstance(cur_input, tf.RaggedTensor):
                        raise ValueError(
                            "The input must be a list of tf.SparseTensor or tf.RaggedTensor"
                        )
                    else:
                        if not len(cur_input.shape) == 2:
                            raise ValueError("The rank of input RaggedTensor must be 2")
        else:
            if not isinstance(cur_input, tf.SparseTensor):
                if not isinstance(cur_input, tf.RaggedTensor):
                    raise ValueError(
                        "The input must be a list of tf.SparseTensor or tf.RaggedTensor"
                    )
                else:
                    if not len(cur_input.shape) == 2:
                        raise ValueError("The rank of input RaggedTensor must be 2")
        emb_vectors = sok.lookup_sparse(
            params=self._vars,
            sp_ids=inputs,
            combiners=combiners,
        )
        return emb_vectors

    @classmethod
    def from_pretrained(
        cls,
        dim: int,
        vocab_sizes: list,
        data: list,
        trainable=True,
        name=None,
        col_schema=None,
        use_dynamic_variable=True,
        localized=None,
        **kwargs,
    ) -> "SOKEmbedding":
        """Create From pre-trained embeddings from a Dataset.
        Parameters
        ----------
        data :
            A list of numpy.array or A list of dict {"indice": numpy.array, "values": numpy.array}
        trainable : bool
            Whether the layer should be trained or not.
        name : str
            The name of the layer.
        """

        if not col_schema:
            if not name:
                raise ValueError("`name` is required when not using a ColumnSchema")
            col_schema = create_categorical_column(name, sum(vocab_sizes) - 1)

        weights = []
        for i, item in enumerate(data):
            if use_dynamic_variable:
                if isinstance(item, dict) and "indice" in item and "values" in item:
                    weights.append([item["indice"], item["values"]])
                else:
                    raise ValueError("DynamicVariable should be initialized with indice and values")
            else:
                weights.append(item)

        return cls(
            dim,
            col_schema,
            vocab_sizes=vocab_sizes,
            name=name,
            initializer=weights,
            use_dynamic_variable=use_dynamic_variable,
            localized=localized,
            trainable=trainable,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim

        schema = schema_to_tensorflow_metadata_json(self.schema)
        config["schema"] = schema
        config["vocab_sizes"] = self._vocab_sizes
        config["initializer"] = self._initializer
        config["use_dynamic_variable"] = self._use_dynamic_variable
        config["localized"] = self._localized

        return config

    @classmethod
    def from_config(cls, config):
        dim = config.pop("dim")
        schema = tensorflow_metadata_json_to_schema(config.pop("schema"))
        vocab_size = config.pop("vocab_sizes")
        initializer = config.pop("initializer")
        use_dynamic_variable = config.pop("use_dynamic_variable")
        localized = config.pop("localized")

        return cls(
            dim,
            *schema,
            vocab_sizes=vocab_size,
            initializer=initializer,
            use_dynamic_variable=use_dynamic_variable,
            localized=localized,
            **config,
        )
