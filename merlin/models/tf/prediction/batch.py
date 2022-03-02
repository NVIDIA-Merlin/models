import tempfile
import typing as tp

import numpy as np
import tensorflow as tf

from merlin.core.dispatch import DataFrameType, concat_columns, get_lib
from merlin.schema import Schema, Tags

from ...utils.schema import select_targets
from ..core import Block, Model, RetrievalModel
from ..dataset import Dataset


class ModelEncode:
    def __init__(
        self,
        model,
        output_names,
        data_iterator_func=None,
        model_load_func=None,
        model_encode_func=None,
        output_concat_func=None,
    ):
        super().__init__()
        self._model = model
        self.output_names = [output_names] if isinstance(output_names, str) else output_names
        self.data_iterator_func = data_iterator_func
        self.model_load_func = model_load_func
        self.model_encode_func = model_encode_func
        self.output_concat_func = output_concat_func

    @property
    def model(self):
        if isinstance(self._model, str):
            self._model = self.model_load_func(self._model)
        return self._model

    def __call__(self, df: DataFrameType) -> DataFrameType:
        # Set defaults
        iterator_func = self.data_iterator_func or (lambda x: [x])
        encode_func = self.model_encode_func or (lambda x, y: x(y))
        concat_func = self.output_concat_func or np.concatenate

        # Iterate over batches of df and collect predictions
        new_df = concat_columns(
            [
                df,
                type(df)(
                    concat_func([encode_func(self.model, batch) for batch in iterator_func(df)]),
                    columns=self.output_names,
                    # index=_df.index,
                ),
            ]
        )

        # Return result
        return new_df

    def transform(self, col_selector, df: DataFrameType) -> DataFrameType:
        return self(df[col_selector])


class TFModelEncode(ModelEncode):
    def __init__(
        self,
        model: tp.Union[Model, tf.keras.Model],
        output_names: tp.List[str] = None,
        batch_size: int = 512,
        save_path: tp.Optional[str] = None,
        block_load_func: tp.Optional[tp.Callable[[str], Block]] = None,
        schema: tp.Optional[Schema] = None,
        output_concat_func=None,
    ):
        save_path = save_path or tempfile.mkdtemp()
        model.save(save_path)

        model_load_func = block_load_func if block_load_func else tf.keras.models.load_model
        output_names = output_names or model.block.last.task_names
        if not output_concat_func:
            output_concat_func = np.concatenate if len(output_names) == 1 else get_lib().concat

        self.schema = schema or model.schema

        super().__init__(
            save_path,
            output_names,
            data_iterator_func=data_iterator_func(self.schema, batch_size=batch_size),
            model_load_func=model_load_func,
            model_encode_func=model_encode,
            output_concat_func=output_concat_func,
        )

    # def fit_transform(self, data) -> nvt.Dataset:
    #     features = self.schema.column_names >> self
    #
    #     # Fit and transform
    #     processor = nvt.Workflow(features)
    #     output = processor.fit_transform(data)
    #
    #     return output


class ItemEmbeddings(TFModelEncode):
    def __init__(
        self, model: Model, dim: int, batch_size: int = 512, save_path: tp.Optional[str] = None
    ):
        item_block = model.block.first.item_block()
        schema = item_block.schema
        output_names = [str(i) for i in range(dim)]

        super().__init__(
            item_block,
            output_names,
            save_path=save_path,
            batch_size=batch_size,
            schema=schema,
            output_concat_func=np.concatenate,
        )


class QueryEmbeddings(TFModelEncode):
    def __init__(
        self,
        model: RetrievalModel,
        dim: int,
        batch_size: int = 512,
        save_path: tp.Optional[str] = None,
    ):
        query_block = model.block.first.query_block()
        schema = query_block.schema
        output_names = [str(i) for i in range(dim)]

        super().__init__(
            query_block,
            output_names,
            save_path=save_path,
            batch_size=batch_size,
            schema=schema,
            output_concat_func=np.concatenate,
        )


def model_encode(model, batch):
    # TODO: How to handle list outputs?

    model_outputs = model(batch[0])

    if isinstance(model_outputs, dict):
        return get_lib().DataFrame({key: encode_output(val) for key, val in model_outputs.items()})

    return encode_output(model_outputs)


def encode_output(output: tf.Tensor):
    if len(output.shape) == 2 and output.shape[1] == 1:
        output = tf.squeeze(output)

    return output.numpy()


def data_iterator_func(schema, batch_size: int = 512):
    import merlin.io.dataset

    cat_cols = schema.select_by_tag(Tags.CATEGORICAL).column_names
    cont_cols = schema.select_by_tag(Tags.CONTINUOUS).column_names
    targets = select_targets(schema).column_names

    def data_iterator(dataset):
        return Dataset(
            merlin.io.dataset.Dataset(dataset),
            batch_size=batch_size,
            cat_names=cat_cols,
            cont_names=cont_cols,
            label_names=targets,
        )

    return data_iterator
