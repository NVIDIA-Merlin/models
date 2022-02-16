# import tensorflow as tf
#
# import merlin_models.tf as ml
# import merlin_standard_lib as msl
# from merlin_models.data import SyntheticDataset
# from merlin.schema import Schema
# from merlin.schema import Tags

#
# # RETRIEVAL
#
#
# def build_matrix_factorization(schema: Schema, dim=128):
#     model = ml.MatrixFactorizationBlock(schema, dim).to_model(schema)
#
#     return model
#
#
# def build_youtube_dnn(schema: Schema, dims=(512, 256), num_sampled=50) -> ml.Model:
#     user_schema = schema.select_by_tag(Tags.USER)
#     dnn = ml.inputs(user_schema, post="continuous-powers").apply(ml.MLPBlock(dims))
#     prediction_task = ml.SampledItemPredictionTask(schema, dim=dims[-1], num_sampled=num_sampled)
#
#     model = dnn.to_model(prediction_task)
#
#     return model
#
#
# def build_two_tower(schema: Schema, target="play", dims=(512, 256)) -> ml.Model:
#     def method_1() -> ml.Model:
#         return ml.TwoTowerBlock(schema, ml.MLPBlock(dims)).to_model(schema.select_by_name(target))
#
#     def method_2() -> ml.Model:
#         user_tower = ml.inputs(schema.select_by_tag(Tags.USER), ml.MLPBlock([512, 256]))
#         item_tower = ml.inputs(schema.select_by_tag(Tags.ITEM), ml.MLPBlock([512, 256]))
#         two_tower = ml.merge({"user": user_tower, "item": item_tower}, aggregation="cosine")
#         model = two_tower.to_model(schema.select_by_name(target))
#
#         return model
#
#     def method_3() -> ml.Model:
#         def routes_verbose(inputs, schema: Schema):
#             user_features = schema.select_by_tag(Tags.USER).filter_columns_from_dict(inputs)
#             item_features = schema.select_by_tag(Tags.ITEM).filter_columns_from_dict(inputs)
#
#             user_tower = ml.MLPBlock(dims)(user_features)
#             item_tower = ml.MLPBlock(dims)(item_features)
#
#             return ml.ParallelBlock(dict(user=user_tower, item=item_tower), aggregation="cosine")
#
#         user_tower = ml.MLPBlock(dims, filter=Tags.USER).as_tabular("user")
#         item_tower = ml.MLPBlock(dims, filter=Tags.ITEM).as_tabular("item")
#
#         two_tower = ml.inputs(schema).branch(user_tower, item_tower, aggregation="cosine")
#         model = two_tower.to_model(schema.select_by_name(target))
#
#         return model
#
#     return method_2()
#
#
# # RANKING
#
#
# def build_dnn(schema: Schema, residual=False) -> ml.Model:
#     bias_block = ml.MLPBlock([256, 128]).from_inputs(schema.select_by_tag("bias"))
#     schema = schema.remove_by_tag("bias")
#
#     if residual:
#         block = ml.inputs(schema, ml.DenseResidualBlock(depth=2))
#     else:
#         block = ml.inputs(schema, ml.MLPBlock([512, 256]))
#
#     return block.to_model(schema, bias_block=bias_block)
#
#
# def build_dcn(schema: Schema) -> ml.Model:
#     schema = schema.remove_by_tag("bias")
#
#     # deep_cross = ml.inputs(schema, ml.CrossBlock(3)).apply(ml.MLPBlock([512, 256]))
#
#     deep_cross = ml.inputs(schema).branch(
#         ml.CrossBlock(3), ml.MLPBlock([512, 256]), aggregation="concat"
#     )
#
#     # deep_cross = ml.inputs(schema, ml.CrossBlock(3))
#     # deep_cross = deep_cross.apply_with_shortcut(ml.MLPBlock([512, 256]), aggregation="concat")
#
#     return deep_cross.to_model(schema)
#
#
# def build_advanced_ranking_model(schema: Schema, head="ple") -> ml.Model:
#     # TODO: Change msl to be able to make this a single function call.
#     bias_block = ml.MLPBlock([512, 256]).from_inputs(schema.select_by_tag("bias"))
#     body = ml.DLRMBlock(
#         schema.remove_by_tag("bias"),
#         bottom_block=ml.MLPBlock([512, 128]),
#         top_block=ml.MLPBlock([128, 64]),
#     )
#
#     # expert_block, output_names = ml.MLPBlock([64, 32]), ml.Head.task_names_from_schema(schema)
#     # mmoe = ml.MMOE(expert_block, num_experts=3, output_names=output_names)
#     # prediction = body.add(mmoe).to_model(schema)
#
#     if head == "mmoe":
#         return ml.MMOEHead.from_schema(
#             schema,
#             body,
#             task_blocks=ml.MLPBlock([64, 32]),
#             expert_block=ml.MLPBlock([64, 32]),
#             bias_block=bias_block,
#             num_experts=3,
#         ).to_model()
#     elif head == "ple":
#         return ml.PLEHead.from_schema(
#             schema,
#             body,
#             task_blocks=ml.MLPBlock([64, 32]),
#             expert_block=ml.MLPBlock([64, 32]),
#             num_shared_experts=2,
#             num_task_experts=2,
#             depth=2,
#             bias_block=bias_block,
#         ).to_model()
#
#     return body.to_model(schema)
#
#
# def build_dlrm(schema: Schema) -> ml.Model:
#     model: ml.Model = ml.DLRMBlock(
#         schema, bottom_block=ml.MLPBlock([512, 128]), top_block=ml.MLPBlock([512, 128])
#     ).to_model(schema)
#
#     return model
#
#
# def data_from_schema(schema, num_items=1000, next_item_prediction=False) -> tf.data.Dataset:
#     data_df = generate_recsys_data(num_items, schema)
#
#     if next_item_prediction:
#         targets = {"item_id": data_df.pop("item_id")}
#     else:
#         targets = {}
#         for target in schema.select_by_tag(Tags.BINARY_CLASSIFICATION):
#             targets[target.name] = data_df.pop(target.name)
#
#     dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), targets))
#
#     return dataset
#
#
# if __name__ == "__main__":
#     music_streaming_data = SyntheticDataset.create_music_streaming_data()
#     dataset = music_streaming_data.get_tf_dataloader(100)
#
#     # model = build_dnn(music_streaming_data.schema, residual=True)
#     # model = build_advanced_ranking_model(music_streaming_data.schema)
#     # model = build_dcn(music_streaming_data.schema)
#     model = build_dlrm(music_streaming_data.schema)
#     # model = build_two_tower(music_streaming_data.schema, target="play")
#
#     # dataset = data_from_schema(music_streaming_data.schema,
#     #                            next_item_prediction=True).batch(100)
#     # prediction = build_youtube_dnn(music_streaming_data.schema)
#
#     model.compile(optimizer="adam", run_eagerly=True)
#
#     inputs, targets = [i for i in dataset.as_numpy_iterator()][0]
#
#     # TODO: remove this after fix in T4Rec
#     predictions = model(inputs)
#     # loss = prediction.compute_loss(predictions, targets)
#
#     model.fit(dataset)
#
#     a = 5
