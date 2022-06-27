from copy import copy
from typing import Dict, Optional

import tensorflow as tf

from merlin.models.tf.inputs.collection import FeatureCollection
from merlin.schema import ColumnSchema, Schema, Tags


class ItemCollection(FeatureCollection):
    EMBEDDING_KEY = "__embedding__"

    def __init__(self, features: FeatureCollection, embeddings: Optional[tf.Tensor] = None):
        schema, values = copy(features.schema), features.values
        if embeddings:
            schema.column_schemas[self.EMBEDDING_KEY] = ColumnSchema(
                self.EMBEDDING_KEY, tags=["EMBEDDING"]
            )
            values[self.EMBEDDING_KEY] = embeddings

        super(ItemCollection, self).__init__(schema, values)

    @classmethod
    def from_ids(
        cls,
        ids: tf.Tensor,
        features: Optional[FeatureCollection] = None,
        embeddings: Optional[tf.Tensor] = None,
    ):
        pass

    @classmethod
    def from_features(
        cls, schema: Schema, features: Dict[str, tf.Tensor], embeddings: Optional[tf.Tensor] = None
    ):
        values = {}
        for col in schema.column_names:
            if col in features:
                values[col] = features[col]

        return cls(FeatureCollection(schema, values), embeddings=embeddings)

    @property
    def ids(self) -> tf.Tensor:
        return self.select_value_by_tag(Tags.ITEM_ID)

    @property
    def embeddings(self) -> Optional[tf.Tensor]:
        return self.values.get(self.EMBEDDING_KEY, None)

    @property
    def features(self) -> Dict[str, tf.Tensor]:
        outputs = {**self.values}
        id_col = self.schema.select_by_tag(Tags.ITEM_ID).first.name

        outputs.pop(id_col)

        return outputs
