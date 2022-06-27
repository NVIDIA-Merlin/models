from typing import Dict, NamedTuple

import tensorflow as tf
from merlin.schema import Schema, Tags
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.composite_tensor import CompositeTensor


class ItemsShape(NamedTuple):
    ids: tf.TensorShape
    metadata: Dict[str, tf.TensorShape]


class Items(CompositeTensor):
    def __init__(self, ids: tf.Tensor, **metadata: tf.Tensor):
        self._ids = ids
        self._metadata = metadata

    @classmethod
    def from_schema(cls, schema: Schema, batch: Dict[str, tf.Tensor]):
        id_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

        metadata_names = schema.select_by_tag(Tags.ITEM)
        metadata = {}
        for col in metadata_names.column_names:
            metadata[col] = batch[col]

        ids = metadata.pop(id_name)

        return cls(ids, **metadata)

    @property
    def ids(self) -> tf.Tensor:
        return self._ids

    @property
    def metadata(self) -> Dict[str, tf.Tensor]:
        return self._metadata

    @property
    def shape(self) -> ItemsShape:
        return ItemsShape(
            self.ids.shape,
            {key: val.shape for key, val in self.metadata.items()}
        )

    def _type_spec(self):
        return tf.TensorSpec((
            self.ids.spec,
            {key: val.shape for key, val in self.metadata.items()}
        ))

    def _shape_invariant_to_type_spec(self, shape):
        pass

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.components == other.components and
                self.metadata == other.metadata)