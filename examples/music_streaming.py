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
        # Bias
        msl.ColumnSchema.create_continuous(
            "position",
            is_float=False,
            min_value=1,
            max_value=100,
            tags=["bias"],
        ),
        # Targets
        msl.ColumnSchema("click").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
        msl.ColumnSchema("play").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
        msl.ColumnSchema("like").with_tags(tags=[Tag.BINARY_CLASSIFICATION]),
    ]
)


def build_two_tower(schema: Schema, target="play") -> ml.Model:
    # user_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.USER), [512, 256])
    # item_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.ITEM), [512, 256])
    # body = ml.Retrieval(user_tower, item_tower)
    model = ml.Retrieval.from_schema(schema, [512, 256]).to_model(schema.select_by_name(target))

    inputs: ml.TabularBlock = ml.TabularFeatures.from_schema(schema)
    inputs.route_by_tag(Tag.ITEM, ml.MLPBlock([512, 256]), output_name="query")
    inputs.route_by_tag(Tag.USER, ml.MLPBlock([512, 256]), output_name="user")

    inputs.route_by_tag("bias", ml.MLPBlock([512, 256]), output_name="bias")

    return model


def build_dnn(schema: Schema) -> ml.Model:
    schema = schema.remove_by_tag("bias")

    return ml.MLPBlock.from_schema(schema, [512, 256]).to_model(schema)


def build_dcn(schema: Schema) -> ml.Model:
    schema = schema.remove_by_tag("bias")

    cross = ml.CrossBlock.from_schema(schema, depth=3)
    deep_cross = cross.add_in_parallel(ml.MLPBlock([512, 256]), aggregation="concat")

    return deep_cross.to_model(schema)


def build_advanced_ranking_model(schema: Schema) -> ml.Model:
    # TODO: Change msl to be able to make this a single function call.
    bias_schema = schema.select_by_tag("bias")
    schema = schema.remove_by_tag("bias")

    body = ml.DLRMBlock.from_schema(
        schema, bottom_mlp=ml.MLPBlock([512, 128]), top_mlp=ml.MLPBlock([512, 128])
    )
    # bias_block = ml.MLPBlock.from_schema(bias_schema, [64])

    return ml.MMOEHead.from_schema(
        schema,
        body,
        task_blocks=ml.MLPBlock([64, 32]),
        expert_block=ml.MLPBlock([64, 32]),
        num_experts=3,
        # bias_block=bias_block,
    ).to_model()

    # return ml.PLEHead.from_schema(
    #     schema,
    #     body,
    #     task_blocks=ml.MLPBlock([64, 32]),
    #     expert_block=ml.MLPBlock([64, 32]),
    #     num_shared_experts=2,
    #     num_task_experts=2,
    #     depth=2,
    #     bias_block=bias_block,
    # ).to_model()


def build_dlrm(schema: Schema) -> ml.Model:
    model: ml.Model = ml.DLRMBlock.from_schema(
        schema, bottom_mlp=ml.MLPBlock([512, 128]), top_mlp=ml.MLPBlock([512, 128])
    ).to_model(schema)

    return model


def data_from_schema(schema, num_items=1000) -> tf.data.Dataset:
    data_df = generate_recsys_data(num_items, schema)

    targets = {}
    for target in synthetic_music_recsys_data_schema.select_by_tag(Tag.BINARY_CLASSIFICATION):
        targets[target.name] = data_df.pop(target.name)

    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), targets))

    return dataset


if __name__ == "__main__":
    dataset = data_from_schema(synthetic_music_recsys_data_schema).batch(100)
    # model = build_dnn(synthetic_music_recsys_data_schema)
    model = build_advanced_ranking_model(synthetic_music_recsys_data_schema)
    # model = build_dcn(synthetic_music_recsys_data_schema)
    # model = build_dlrm(synthetic_music_recsys_data_schema)
    # model = build_two_tower(synthetic_music_recsys_data_schema, target="play")

    model.compile(optimizer="adam", run_eagerly=True)

    batch = [i for i in dataset.as_numpy_iterator()][0][0]

    # TODO: remove this after fix in T4Rec
    out = model(batch)

    a = 5
