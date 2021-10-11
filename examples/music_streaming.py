import merlin_standard_lib as msl
import tensorflow as tf
from merlin_standard_lib import Schema, Tag

import merlin_models.tf as ml
from merlin_models.data.synthetic import generate_recsys_data

synthetic_music_recsys_data_schema = Schema(
    [
        # Item
        msl.ColumnSchema.create_categorical(
            "item_id",
            num_items=10000,
            tags=[Tag.ITEM_ID],
        ),
        msl.ColumnSchema.create_categorical(
            "item_category",
            num_items=100,
            tags=[Tag.ITEM],
        ),
        msl.ColumnSchema.create_continuous(
            "item_recency",
            min_value=0,
            max_value=1,
            tags=[Tag.ITEM],
        ),
        msl.ColumnSchema.create_categorical(
            "item_genres",
            num_items=100,
            value_count=msl.schema.ValueCount(1, 20),
            tags=[Tag.ITEM],
        ),
        # User
        msl.ColumnSchema.create_categorical(
            "country",
            num_items=100,
            tags=[Tag.USER],
        ),
        msl.ColumnSchema.create_continuous(
            "user_age",
            is_float=False,
            min_value=18,
            max_value=50,
            tags=[Tag.USER],
        ),
        msl.ColumnSchema.create_categorical(
            "user_genres",
            num_items=100,
            value_count=msl.schema.ValueCount(1, 20),
        ),
        # Targets
        msl.ColumnSchema("click").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
        msl.ColumnSchema("play").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
        msl.ColumnSchema("like").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
    ]
)


def build_two_tower(schema: Schema, target="play") -> ml.Model:
    user_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.USER), [512, 256])
    item_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.ITEM), [512, 256])
    body = ml.Retrieval(user_tower, item_tower)

    return ml.Head.from_schema(schema.select_by_name(target), body).to_model()


def build_dnn(schema: Schema) -> ml.Model:
    body = ml.MLPBlock.from_schema(schema, [512, 256])

    return ml.Head.from_schema(synthetic_music_recsys_data_schema, body).to_model()


def build_dlrm(schema: Schema) -> ml.Model:
    body = ml.DLRMBlock.from_schema(
        schema, bottom_mlp=ml.MLPBlock([512, 128]), top_mlp=ml.MLPBlock([512, 128])
    )

    return ml.Head.from_schema(synthetic_music_recsys_data_schema, body).to_model()


def data_from_schema(schema, num_items=1000) -> tf.data.Dataset:
    data_df = generate_recsys_data(num_items, schema)

    targets = {}
    for target in synthetic_music_recsys_data_schema.select_by_tag(Tag.BINARY_CLASSIFICATION):
        targets[target.name] = data_df.pop(target.name)

    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), targets))

    return dataset


if __name__ == "__main__":
    dataset = data_from_schema(synthetic_music_recsys_data_schema).batch(100)
    model = build_dnn(synthetic_music_recsys_data_schema)
    # model = build_dlrm(synthetic_music_recsys_data_schema)
    # model = build_two_tower(synthetic_music_recsys_data_schema, target="play")

    model.compile(optimizer="adam", run_eagerly=True)

    batch = [i for i in dataset.as_numpy_iterator()][0][0]

    # TODO: remove this after fix in T4Rec
    out = model(batch)

    a = 5
