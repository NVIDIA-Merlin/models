# import pytest
# import tensorflow as tf
# from merlin_standard_lib import Schema, Tag
#
# import merlin_models.tf as ml
# from merlin_models.tf.block.sampling import MemoryBankBlock
#
# # from merlin_models.tf.head.retrieval import RetrievalPredictionTask
#
#
# def _create_vectors(batch_size=100, dim=64):
#     return {
#         str(Tag.ITEM): tf.random.uniform((batch_size, dim)),
#         str(Tag.USER): tf.random.uniform((batch_size, dim)),
#     }
#
#
# # def test_negative_sampling():
# #     queue = ItemQueue(num_batches=3)
# #
# #     for _ in range(5):
# #         queue(_create_vectors())
# #
# #         negative_samples = queue.fetch()
# #         a = 5
# #     a = 5
#
#
# @pytest.mark.parametrize("add_targets", [True, False])
# @pytest.mark.parametrize("in_batch_negatives", [True, False])
# def test_retrieval_task(add_targets, in_batch_negatives):
#     vectors = _create_vectors()
#     if add_targets:
#         targets = tf.cast(tf.random.uniform((100, 1), maxval=2, dtype=tf.int32), tf.float32)
#     else:
#         targets = None
#
#     task = RetrievalPredictionTask(in_batch_negatives=in_batch_negatives)
#
#     if not add_targets and not in_batch_negatives:
#         with pytest.raises(ValueError) as excinfo:
#             loss = task.compute_loss(vectors, targets)
#         err_message = "Targets are required when in-batch negative sampling is disabled"
#         assert err_message in str(excinfo.value)
#     else:
#         loss = task.compute_loss(vectors, targets)
#
#         assert loss is not None
#
#
# schema: Schema = Schema()
#
# # Variant (b)
# # two_tower = ml.TwoTowerBlock(schema, ml.MLPBlock([512, 256]))
# # negatives = MemoryBankBlock(num_batches=10, post=two_tower["item"], no_outputs=True)
# # two_tower = two_tower.add_branch(
# #     "negatives",
# #     ml.Filter(schema.select_by_tag(Tag.ITEM)).apply(negatives)
# # )
# # two_tower.to_model(RetrievalPredictionTask(extra_negatives=negatives))
# #
# # # Variant (c)
# # two_tower = ml.TwoTowerBlock(schema, ml.MLPBlock([512, 256]))
# # negatives = MemoryBankBlock(num_batches=10)
# # two_tower.apply_to_branch("item", negatives)
# # two_tower.to_model(RetrievalPredictionTask(extra_negatives=negatives))
# #
# #
# youtube_dnn = ml.TwoTowerBlock(
#     schema,
#     ml.MLPBlock([512, 256]),
#     item_tower=ml.EmbeddingFeatures.from_schema(schema.select_by_tag(Tag.ITEM_ID)),
# )
#
#
# ml.inputs(schema, add_to_context=[Tag.ITEM_ID, Tag.USER_ID])
# # weight_tying = ml.inputs(schema).apply_with_shortcut(
# #     ml.MLPBlock([512, 256]),
# #     shortcut_filter=ml.Filter(Tag.ITEM_ID)
# # )
