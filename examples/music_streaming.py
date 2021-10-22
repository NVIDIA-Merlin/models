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


def build_two_tower(schema: Schema, target="play", dims=(512, 256)) -> ml.Model:
    def method_1() -> ml.Model:
        return ml.Retrieval.from_schema(schema, dims).to_model(schema.select_by_name(target))

    def method_2() -> ml.Model:
        user_schema, item_schema = schema.select_by_tag(Tag.USER), schema.select_by_tag(Tag.ITEM)
        user_tower = ml.MLPBlock([512, 256]).with_inputs(user_schema, aggregation="concat")
        item_tower = ml.MLPBlock([512, 256]).with_inputs(item_schema, aggregation="concat")

        return ml.Retrieval(user_tower, item_tower).to_model(schema)

    def method_3() -> ml.Model:
        def routes_verbose(inputs, schema: Schema):
            user_features = schema.select_by_tag(Tag.USER).filter_columns_from_dict(inputs)
            item_features = schema.select_by_tag(Tag.ITEM).filter_columns_from_dict(inputs)

            user_tower = ml.MLPBlock(dims)(user_features)
            item_tower = ml.MLPBlock(dims)(item_features)

            return ml.ParallelBlock(dict(user=user_tower, item=item_tower), aggregation="cosine")

        # routes = {
        #     Tag.USER: block.as_tabular("user"),
        #     Tag.ITEM: block.copy().as_tabular("item"),
        # }
        # two_tower = inputs.routes(routes, aggregation="cosine")

        # two_tower = inputs.match_keys(
        #     ml.Match(Tag.USER, block.as_tabular("user")),
        #     ml.Match(Tag.ITEM, block.copy().as_tabular("user")),
        #     aggregation="cosine"
        # )
        user_tower = Tag.USER >> ml.MLPBlock(dims, pre_aggregation="concat").as_tabular("user")
        item_tower = Tag.ITEM >> ml.MLPBlock(dims, pre_aggregation="concat").as_tabular("item")

        inputs: ml.TabularBlock = ml.TabularFeatures.from_schema(schema)
        two_tower = inputs.branch(user_tower, item_tower, aggregation="cosine").to_model(schema)

        return two_tower

    return method_3()


def build_dnn(schema: Schema, residual=False) -> ml.Model:
    schema = schema.remove_by_tag("bias")

    if residual:
        block = ml.DenseResidualBlock().repeat(2).with_inputs(schema, aggregation="concat")
    else:
        block = ml.MLPBlock([512, 256]).with_inputs(schema, aggregation="concat")

    return block.to_model(schema)


def build_dcn(schema: Schema) -> ml.Model:
    schema = schema.remove_by_tag("bias")

    cross = ml.CrossBlock.from_schema(schema, depth=3)
    deep_cross = cross.add_in_parallel(ml.MLPBlock([512, 256]), aggregation="concat")

    return deep_cross.to_model(schema)


def build_advanced_ranking_model(schema: Schema, head="ple") -> ml.Model:
    # TODO: Change msl to be able to make this a single function call.
    bias_schema = schema.select_by_tag("bias")
    schema = schema.remove_by_tag("bias")

    body = ml.DLRMBlock(
        schema, bottom_block=ml.MLPBlock([512, 128]), top_block=ml.MLPBlock([128, 64])
    )
    bias_block = ml.MLPBlock([512, 256]).with_inputs(bias_schema, aggregation="concat")

    # expert_block, output_names = ml.MLPBlock([64, 32]), ml.Head.task_names_from_schema(schema)
    # mmoe = ml.MMOE(expert_block, num_experts=3, output_names=output_names)
    # model = body.add(mmoe).to_model(schema)

    if head == "mmoe":
        return ml.MMOEHead.from_schema(
            schema,
            body,
            task_blocks=ml.MLPBlock([64, 32]),
            expert_block=ml.MLPBlock([64, 32]),
            bias_block=bias_block,
            num_experts=3,
        ).to_model()
    elif head == "ple":
        return ml.PLEHead.from_schema(
            schema,
            body,
            task_blocks=ml.MLPBlock([64, 32]),
            expert_block=ml.MLPBlock([64, 32]),
            num_shared_experts=2,
            num_task_experts=2,
            depth=2,
            bias_block=bias_block,
        ).to_model()

    return body.to_model(schema)


def build_dlrm(schema: Schema) -> ml.Model:
    model: ml.Model = ml.DLRMBlock(
        schema, bottom_block=ml.MLPBlock([512, 128]), top_block=ml.MLPBlock([512, 128])
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
    # model = build_dnn(synthetic_music_recsys_data_schema, residual=True)
    # model = build_advanced_ranking_model(synthetic_music_recsys_data_schema)
    # model = build_dcn(synthetic_music_recsys_data_schema)
    # model = build_dlrm(synthetic_music_recsys_data_schema)
    model = build_two_tower(synthetic_music_recsys_data_schema, target="play")

    model.compile(optimizer="adam", run_eagerly=True)

    batch = [i for i in dataset.as_numpy_iterator()][0][0]

    # TODO: remove this after fix in T4Rec
    out = model(batch)

    a = 5
