import tempfile
import typing as tp

import numpy as np
import nvtabular as nvt
import tensorflow as tf
from nvtabular.dispatch import get_lib
from nvtabular.ops import ModelEncode as _ModelEncode

from merlin_standard_lib import Schema, Tag

from .core import Block, Model
from .dataset import Dataset


class TFModelEncode(_ModelEncode):
    def __init__(
        self,
        model: Model,
        output_names: tp.List[str] = None,
        batch_size: int = 512,
        save_path: tp.Optional[str] = None,
        block_load_func: tp.Optional[tp.Callable[[str], Block]] = None,
        schema: tp.Optional[Schema] = None,
    ):
        save_path = save_path or tempfile.mkdtemp()
        model.save(save_path)

        model_load_func = block_load_func if block_load_func else tf.keras.models.load_model
        output_names = output_names or model.block.last.task_names
        output_concat_func = np.concat if len(output_names) == 1 else get_lib().concat

        self.schema = schema or model.schema

        super().__init__(
            save_path,
            output_names,
            data_iterator_func=data_iterator_func(self.schema, batch_size=batch_size),
            model_load_func=model_load_func,
            model_encode_func=model_encode,
            output_concat_func=output_concat_func,
        )

    def fit_transform(self, data) -> nvt.Dataset:
        features = self.schema.column_names >> self

        # Fit and transform
        processor = nvt.Workflow(features)
        output = processor.fit_transform(data)

        return output


class ItemEmbeddings(TFModelEncode):
    def __init__(
        self, model: Model, dim: int, batch_size: int = 512, save_path: tp.Optional[str] = None
    ):
        block_load_func = model.block.first.load_item_block
        item_block = model.block.first.item_block()
        schema = item_block.schema
        output_names = [str(i) for i in range(dim)]

        super().__init__(
            model,
            output_names,
            save_path=save_path,
            batch_size=batch_size,
            block_load_func=block_load_func,
            schema=schema,
        )


class UserEmbeddings(TFModelEncode):
    def __init__(
        self, model: Model, dim: int, batch_size: int = 512, save_path: tp.Optional[str] = None
    ):
        block_load_func = model.block.first.load_user_block
        user_block = model.block.first.user_block()
        schema = user_block.schema
        output_names = [str(i) for i in range(dim)]

        super().__init__(
            model,
            output_names,
            save_path=save_path,
            batch_size=batch_size,
            block_load_func=block_load_func,
            schema=schema,
        )


def model_encode(model, batch):
    # TODO: How to handle list outputs?

    model_outputs = model(batch[0])

    if isinstance(model_outputs, dict):
        return get_lib().DataFrame({key: encode_output(val) for key, val in model_outputs.items()})

    return encode_output(model_outputs)


def encode_output(output: tf.Tensor):
    if len(output.shape) == 2:
        output = tf.squeeze(output)

    return output.numpy()


def data_iterator_func(schema, batch_size: int = 512):
    cat_cols = schema.select_by_tag(Tag.CATEGORICAL).column_names
    cont_cols = schema.select_by_tag(Tag.CONTINUOUS).column_names
    targets = schema.select_targets().column_names

    def data_iterator(dataset):
        return Dataset(
            nvt.Dataset(dataset),
            batch_size=batch_size,
            cat_names=cat_cols,
            cont_names=cont_cols,
            label_names=targets,
        )

    return data_iterator
