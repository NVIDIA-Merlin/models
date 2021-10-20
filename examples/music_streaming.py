from typing import Optional

import merlin_standard_lib as msl
import tensorflow as tf
from merlin_standard_lib import Schema, Tag

import merlin_models.tf as ml
from merlin_models.data.synthetic import generate_recsys_data
from merlin_models.tf.layers import DotProductInteraction

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
    # user_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.USER), [512, 256])
    # item_tower = ml.MLPBlock.from_schema(schema.select_by_tag(Tag.ITEM), [512, 256])
    # body = ml.Retrieval(user_tower, item_tower)

    # model = ml.Retrieval.from_schema(schema, dims).to_model(schema.select_by_name(target))

    inputs: ml.TabularBlock = ml.TabularFeatures.from_schema(schema)
    routes = {Tag.USER: ml.MLPBlock(dims).as_tabular("user"),
              Tag.ITEM: ml.MLPBlock(dims).as_tabular("item")}
    model = inputs.add_routes(routes, aggregation="cosine").to_model(schema)

    return model


def build_dnn(schema: Schema, residual=False) -> ml.Model:
    schema = schema.remove_by_tag("bias")

    if residual:
        block = ml.MLPResidualBlock.from_schema(schema, depth=2)

    else:
        block = ml.MLPBlock.from_schema(schema, [512, 256])

    return block.to_model(schema)


def build_dcn(schema: Schema) -> ml.Model:
    # bias_schema = schema.select_by_tag("bias")
    schema = schema.remove_by_tag("bias")

    cross = ml.CrossBlock.from_schema(schema, depth=3)
    deep_cross = cross.add_in_parallel(ml.MLPBlock([512, 256]), aggregation="concat")

    # inputs = ml.TabularFeatures.from_schema(schema)
    # inputs.add_route(schema.select_by_tag("bias"), ml.MLPBlock([512, 256]))
    # deep_cross = ml.CrossBlock(3).add_in_parallel(ml.MLPBlock([512, 256]), aggregation="concat")
    # inputs.add_route(schema.remove_by_tag("bias"), deep_cross)

    return deep_cross.to_model(schema)


def MMOE(expert_block: ml.Block, num_experts: int, output_names, gate_dim: int = 32):
    agg = ml.StackFeatures(axis=1)
    experts = expert_block.repeat_in_parallel(num_experts, prefix="expert_", aggregation=agg)
    gates = ml.MMOEGate(num_experts, dim=gate_dim).repeat_in_parallel(names=output_names)
    mmoe = expert_block.add_with_shortcut(experts, block_outputs_name="experts")
    mmoe = mmoe.add(gates, block_name="MMOE")

    return mmoe


def build_advanced_ranking_model(schema: Schema) -> ml.Model:
    # TODO: Change msl to be able to make this a single function call.
    # bias_schema = schema.select_by_tag("bias")
    schema = schema.remove_by_tag("bias")

    # body = ml.DLRMBlock.from_schema(
    #     schema, bottom_mlp=ml.MLPBlock([512, 128]), top_mlp=ml.MLPBlock([512, 128])
    # )
    body = DLRMBlock(schema, bottom_block=ml.MLPBlock([512, 128]), top_block=ml.MLPBlock([128, 64]))
    expert_block, output_names = ml.MLPBlock([64, 32]), ml.Head.task_names_from_schema(schema)
    model = body.add(MMOE(expert_block, num_experts=3, output_names=output_names)).to_model(schema)

    a = 5

    return model

    # head = ml.Head.from_schema(
    #     schema,
    #     body,
    #     task_blocks=ml.MLPBlock([64, 32]),
    #     expert_block=ml.MLPBlock([64, 32]),
    #     num_experts=3,
    #     # bias_block=bias_block,
    # )
    # # head.add_in_parallel()
    #
    # return head.to_model()

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


def DLRMBlock(schema, bottom_block: ml.Block, top_block: Optional[ml.Block] = None):
    con, cat = schema.select_by_tag(Tag.CONTINUOUS), schema.select_by_tag(Tag.CATEGORICAL)
    emb = ml.EmbeddingFeatures.from_schema(cat, embedding_dim_default=bottom_block.layers[-1].units)
    continuous = ml.ContinuousFeatures.from_schema(con, aggregation="concat").add(bottom_block)
    inputs = ml.ParallelBlock(dict(embeddings=emb, continuous=continuous), aggregation="stack")
    dlrm = inputs.add(DotProductInteraction())

    if top_block:
        dlrm = dlrm.add(top_block)

    return dlrm



    # routes = {schema.select_by_tag(Tag.CONTINUOUS): bottom_block.as_tabular("continuous")}
    # Same as:
    inp: ml.TabularBlock = ml.TabularFeatures.from_schema(schema)
    routes = {Tag.CONTINUOUS: bottom_block.as_tabular("continuous")}
    dlrm = inp.add_routes(routes, add_rest=True, aggregation="stack").add(DotProductInteraction())

    if top_block:
        dlrm = dlrm.add(top_block)


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
