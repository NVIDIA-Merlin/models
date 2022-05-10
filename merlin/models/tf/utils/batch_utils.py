import tempfile
import typing as tp

import numpy as np
import tensorflow as tf

from merlin.core.dispatch import DataFrameType, concat_columns, get_lib
from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.models.base import Model, RetrievalModel
from merlin.models.utils.schema_utils import select_targets
from merlin.schema import Schema, Tags


class ModelEncode:
    def __init__(
        self,
        model,
        output_names=None,
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

    def __call__(
        self,
        df: DataFrameType,
        filter_input_columns: tp.Optional[tp.List[str]] = None,
        filter_output_columns: tp.Optional[tp.List[str]] = None,
    ) -> DataFrameType:
        # Set defaults
        iterator_func = self.data_iterator_func or (lambda x: [x])
        encode_func = self.model_encode_func or (lambda x, y: x(y))
        concat_func = self.output_concat_func or np.concatenate

        # Iterate over batches of df and collect predictions
        outputs = concat_func([encode_func(self.model, batch) for batch in iterator_func(df)])
        output_names = self.output_names or [str(i) for i in range(outputs.shape[1])]
        model_output_df = type(df)(outputs, columns=output_names)
        if filter_output_columns:
            model_output_df = model_output_df[filter_output_columns]
        input_df = df if not filter_input_columns else df[filter_input_columns]

        output_df = concat_columns([input_df, model_output_df])

        return output_df

    def transform(self, col_selector, df: DataFrameType, **kwargs) -> DataFrameType:
        return self(df[col_selector], **kwargs)


class TFModelEncode(ModelEncode):
    def __init__(
        self,
        model: tp.Union[Model, tf.keras.Model],
        output_names: tp.Optional[tp.List[str]] = None,
        batch_size: int = 512,
        save_path: tp.Optional[str] = None,
        block_load_func: tp.Optional[tp.Callable[[str], Block]] = None,
        schema: tp.Optional[Schema] = None,
        output_concat_func=None,
    ):
        save_path = save_path or tempfile.mkdtemp()
        model.save(save_path)

        model_load_func = block_load_func if block_load_func else tf.keras.models.load_model
        if not output_names:
            try:
                output_names = model.block.last.task_names
            except AttributeError:
                pass
        if not output_concat_func:
            if len(output_names) == 1:  # type: ignore
                output_concat_func = np.concatenate
            else:
                output_concat_func = get_lib().concat  # type: ignore

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
    def __init__(self, model: Model, batch_size: int = 512, save_path: tp.Optional[str] = None):
        item_block = model.block.first.item_block()
        schema = item_block.schema

        super().__init__(
            item_block,
            save_path=save_path,
            batch_size=batch_size,
            schema=schema,
            output_concat_func=np.concatenate,
        )


class QueryEmbeddings(TFModelEncode):
    def __init__(
        self,
        model: RetrievalModel,
        batch_size: int = 512,
        save_path: tp.Optional[str] = None,
    ):
        query_block = model.block.first.query_block()
        schema = query_block.schema

        super().__init__(
            query_block,
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

    cat_cols = schema.select_by_tag(Tags.CATEGORICAL).excluding_by_tag(Tags.TARGET).column_names
    cont_cols = schema.select_by_tag(Tags.CONTINUOUS).excluding_by_tag(Tags.TARGET).column_names
    targets = select_targets(schema).column_names

    def data_iterator(dataset):
        return BatchedDataset(
            merlin.io.dataset.Dataset(dataset),
            batch_size=batch_size,
            cat_names=cat_cols,
            cont_names=cont_cols,
            label_names=targets,
            shuffle=False,
        )

    return data_iterator
