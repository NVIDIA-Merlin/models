import pytest
import torch

import merlin.models.torch as mm
from merlin.models.torch.outputs.classification import CategoricalTarget
from merlin.models.torch.outputs.contrastive import (
    ContrastiveOutput,
    DotProduct,
    rescore_false_negatives,
)
from merlin.models.torch.utils.module_utils import module_test
from merlin.schema import Schema


class TestContrastiveOutput:
    def test_initialize_from_schema(self, item_id_col_schema, user_id_col_schema):
        contrastive = ContrastiveOutput()

        dot = ContrastiveOutput(schema=Schema([item_id_col_schema, user_id_col_schema]))
        assert isinstance(dot.to_call, DotProduct)

        target = ContrastiveOutput(schema=Schema([item_id_col_schema]))
        assert isinstance(target.to_call, CategoricalTarget)

        with pytest.raises(ValueError):
            contrastive.initialize_from_schema(1)

        with pytest.raises(ValueError):
            contrastive.initialize_from_schema(Schema(["a", "b", "c"]))

    def test_outputs_without_downscore(self, item_id_col_schema):
        contrastive = ContrastiveOutput(item_id_col_schema, downscore_false_negatives=False)

        query = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        positive = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        negative = torch.tensor([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]])

        id_tensor = torch.tensor(1)
        out = contrastive.contrastive_outputs(query, positive, negative, id_tensor, id_tensor)

        expected_out = torch.tensor(
            [[0.1700, 0.2900, 0.3500, 0.4100], [0.5300, 0.6700, 0.8100, 0.9500]]
        )
        expected_target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        assert torch.allclose(out, expected_out, atol=1e-4)
        assert torch.equal(contrastive.target, expected_target)

    def test_outputs_with_rescore_false_negatives(self, item_id_col_schema):
        contrastive = ContrastiveOutput(item_id_col_schema, false_negative_score=-100.0)

        query = torch.tensor([[0.1, 0.2]])
        positive = torch.tensor([[0.5, 0.6]])
        negative = torch.tensor([[0.5, 0.6], [0.9, 1.0]])
        positive_id = torch.tensor([[0]])
        negative_id = torch.tensor([[0, 1]])

        out = contrastive.contrastive_outputs(query, positive, negative, positive_id, negative_id)

        # Explanation:
        # 1. positive_scores = dot(query, positive) = dot([0.1, 0.2], [0.5, 0.6]) = 0.17
        # 2. negative_scores = matmul(query, negative.T)
        #   = matmul([[0.1, 0.2]], [[0.5, 0.9], [0.6, 1.0]]) = [[0.17, 0.29]]
        # 3. Since the first negative sample is a false negative (its id is in positive_id),
        #   we downscore it to -100.0
        # 4. The final output is a concatenation of the positive_scores and the
        #   rescored negative_scores: [[0.17, -100.0, 0.29]]

        expected_out = torch.tensor([[0.17, -100.0, 0.29]])
        expected_target = torch.tensor([[1.0, 0.0, 0.0]])

        assert torch.allclose(out, expected_out, atol=1e-4)
        assert torch.equal(contrastive.target, expected_target)

    def test_outputs_raises_runtime_error(self, item_id_col_schema):
        contrastive = ContrastiveOutput(item_id_col_schema)

        query = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        positive = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        negative = torch.tensor([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]])
        positive_id = torch.tensor([0, 1])
        negative_id = torch.tensor([1, 2, 3])

        with pytest.raises(RuntimeError):
            contrastive.contrastive_outputs(query, positive, negative, positive_id)

        with pytest.raises(RuntimeError):
            contrastive.contrastive_outputs(query, positive, negative, negative_id=negative_id)

    def test_call_categorical_target(self, item_id_col_schema):
        model = mm.Block(
            mm.EmbeddingTable(10, Schema([item_id_col_schema])),
            mm.Concat(),
            ContrastiveOutput(item_id_col_schema),
        )

        item_id = torch.tensor([1, 2, 3])
        features = {"item_id": item_id}
        batch = mm.Batch(features)

        outputs = module_test(model, features, batch=batch)
        contrastive_outputs = model(features, batch=batch.replace(targets=item_id + 1))

        assert contrastive_outputs.shape == (3, 4)
        assert outputs.shape == (3, 11)

        self.assert_contrastive_outputs(contrastive_outputs, model[-1].target, outputs)

    def test_call_dot_product(self, user_id_col_schema, item_id_col_schema):
        schema = Schema([user_id_col_schema, item_id_col_schema])
        model = mm.Block(
            mm.EmbeddingTables(10, schema),
            ContrastiveOutput(item_id_col_schema),
        )

        user_id = torch.tensor([1, 2, 3])
        item_id = torch.tensor([1, 2, 3])

        data = {"user_id": user_id, "item_id": item_id}
        contrastive_outputs = model(data, batch=mm.Batch(data))

        model.eval()
        outputs = module_test(model, data, batch=mm.Batch(data))

        self.assert_contrastive_outputs(contrastive_outputs, model[-1].target, outputs)

    @pytest.mark.parametrize("negative_sampling", ["popularity", "in-batch"])
    def test_call_weight_tying(self, user_id_col_schema, item_id_col_schema, negative_sampling):
        schema = Schema([user_id_col_schema, item_id_col_schema])
        embeddings = mm.EmbeddingTables(10, schema)
        model = mm.Block(
            embeddings,
            mm.MLPBlock([10]),
            ContrastiveOutput.with_weight_tying(
                embeddings, item_id_col_schema, negative_sampling=negative_sampling
            ),
        )

        user_id = torch.tensor([1, 2, 3])
        item_id = torch.tensor([1, 2, 3])

        data = {"user_id": user_id, "item_id": item_id}
        contrastive_outputs = model(data, batch=mm.Batch(data, targets=item_id + 1))

        model.eval()
        outputs = module_test(model, data, batch=mm.Batch(data))

        assert outputs.shape == (3, 11)

        if negative_sampling == "popularity":
            assert contrastive_outputs.shape[0] == 3
            assert contrastive_outputs.shape[1] >= 4
        else:
            assert contrastive_outputs.shape == (3, 4)
            self.assert_contrastive_outputs(contrastive_outputs, model[-1].target, outputs)

    def assert_contrastive_outputs(self, contrastive_outputs, targets, outputs):
        assert contrastive_outputs.shape == (3, 4)
        assert targets.shape == (3, 4)
        assert not torch.equal(outputs, contrastive_outputs)
        assert targets[:, 0].all().item()
        assert torch.equal(targets[:, 1:], torch.zeros_like(targets[:, 1:]))


class Test_rescore_false_negatives:
    def test_no_false_negatives(self):
        positive_item_ids = torch.tensor([1, 3, 5])
        neg_samples_item_ids = torch.tensor([2, 4, 6])
        negative_scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        false_negatives_score = -100.0

        rescored_neg_scores, valid_negatives_mask = rescore_false_negatives(
            positive_item_ids, neg_samples_item_ids, negative_scores, false_negatives_score
        )

        assert torch.equal(rescored_neg_scores, negative_scores)
        assert torch.equal(
            valid_negatives_mask, torch.ones_like(valid_negatives_mask, dtype=torch.bool)
        )

    def test_with_false_negatives(self):
        positive_item_ids = torch.tensor([1, 3, 5])
        neg_samples_item_ids = torch.tensor([1, 4, 5])
        negative_scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        false_negatives_score = -100.0

        rescored_neg_scores, valid_negatives_mask = rescore_false_negatives(
            positive_item_ids, neg_samples_item_ids, negative_scores, false_negatives_score
        )

        expected_rescored_neg_scores = torch.tensor(
            [[-100.0, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, -100.0]]
        )
        expected_valid_negatives_mask = torch.tensor(
            [[False, True, True], [True, True, True], [True, True, False]], dtype=torch.bool
        )

        assert torch.equal(rescored_neg_scores, expected_rescored_neg_scores)
        assert torch.equal(valid_negatives_mask, expected_valid_negatives_mask)

    def test_all_false_negatives(self):
        positive_item_ids = torch.tensor([1, 3, 5])
        neg_samples_item_ids = torch.tensor([1, 3, 5])
        negative_scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        false_negatives_score = -100.0

        rescored_neg_scores, valid_negatives_mask = rescore_false_negatives(
            positive_item_ids, neg_samples_item_ids, negative_scores, false_negatives_score
        )

        expected_rescored_neg_scores = torch.tensor(
            [[-100.0, 0.2, 0.3], [0.4, -100.0, 0.6], [0.7, 0.8, -100.0]]
        )
        expected_valid_negatives_mask = torch.tensor(
            [[False, True, True], [True, False, True], [True, True, False]], dtype=torch.bool
        )

        assert torch.equal(rescored_neg_scores, expected_rescored_neg_scores)
        assert torch.equal(valid_negatives_mask, expected_valid_negatives_mask)


class TestDotProduct:
    def test_less_than_two_inputs(self):
        dp = DotProduct()
        with pytest.raises(RuntimeError, match=r"DotProduct requires at least two inputs"):
            dp({"candidate": torch.tensor([1, 2, 3])})

    def test_no_query_name_more_than_two_inputs(self):
        dp = DotProduct()
        with pytest.raises(
            RuntimeError, match=r"DotProduct requires query_name to be set when more than"
        ):
            dp(
                {
                    "candidate": torch.tensor([1, 2, 3]),
                    "extra": torch.tensor([4, 5, 6]),
                    "another_extra": torch.tensor([7, 8, 9]),
                }
            )

    def test_valid_dot_product_operation(self):
        dp = DotProduct(query_name="query")
        result = dp.forward(
            {"query": torch.tensor([1.0, 2.0, 3.0]), "candidate": torch.tensor([4.0, 5.0, 6.0])}
        )
        assert torch.allclose(result, torch.tensor([32.0]), atol=1e-4)

    def test_no_query_name_3_inputs(self):
        dp = DotProduct(query_name=None)
        with pytest.raises(RuntimeError):
            dp(
                {
                    "query": torch.tensor([1.0, 2.0, 3.0]),
                    "candidate": torch.tensor([1.0, 2.0, 3.0]),
                    "extra": torch.tensor([4.0, 5.0, 6.0]),
                }
            )

    def test_should_apply_contrastive(self):
        dp = DotProduct()
        assert dp.should_apply_contrastive(None) == dp.training
