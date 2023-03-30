import pytest
import torch

from merlin.models.torch.blocks.interaction import DotProduct
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.inputs.embedding import Embeddings
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.contrastive import ContrastiveOutput, rescore_false_negatives
from merlin.schema import Schema, Tags


class TestContrastiveOutput:
    def test_outputs_without_downscore(self, item_id_col_schema):
        contrastive = ContrastiveOutput(item_id_col_schema, downscore_false_negatives=False)

        query = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        positive = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        negative = torch.tensor([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]])

        out, targets = contrastive.contrastive_outputs(query, positive, negative)

        expected_out = torch.tensor(
            [[0.1700, 0.2900, 0.3500, 0.4100], [0.5300, 0.6700, 0.8100, 0.9500]]
        )
        expected_targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        assert torch.allclose(out, expected_out, atol=1e-4)
        assert torch.equal(targets, expected_targets)

    def test_outputs_with_rescore_false_negatives(self, item_id_col_schema):
        contrastive = ContrastiveOutput(item_id_col_schema, false_negative_score=-100.0)

        query = torch.tensor([[0.1, 0.2]])
        positive = torch.tensor([[0.5, 0.6]])
        negative = torch.tensor([[0.5, 0.6], [0.9, 1.0]])
        positive_id = torch.tensor([[0]])
        negative_id = torch.tensor([[0, 1]])

        out, targets = contrastive.contrastive_outputs(
            query, positive, negative, positive_id, negative_id
        )

        # Explanation:
        # 1. positive_scores = dot(query, positive) = dot([0.1, 0.2], [0.5, 0.6]) = 0.17
        # 2. negative_scores = matmul(query, negative.T)
        #   = matmul([[0.1, 0.2]], [[0.5, 0.9], [0.6, 1.0]]) = [[0.17, 0.29]]
        # 3. Since the first negative sample is a false negative (its id is in positive_id),
        #   we downscore it to -100.0
        # 4. The final output is a concatenation of the positive_scores and the
        #   rescored negative_scores: [[0.17, -100.0, 0.29]]

        expected_out = torch.tensor([[0.17, -100.0, 0.29]])
        expected_targets = torch.tensor([[1.0, 0.0, 0.0]])

        assert torch.allclose(out, expected_out, atol=1e-4)
        assert torch.equal(targets, expected_targets)

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
        model = Model(
            Embeddings(Schema([item_id_col_schema]), 10, aggregation="concat"),
            ContrastiveOutput(item_id_col_schema),
        )

        item_id = torch.tensor([1, 2, 3])

        outputs = model(item_id)
        contrastive_outputs, targets = model(item_id, targets=item_id + 1)

        self.assert_contrastive_outputs(contrastive_outputs, targets, outputs)

    def test_call_dot_product(self, user_id_col_schema, item_id_col_schema):
        schema = Schema([user_id_col_schema, item_id_col_schema])
        model = Model(
            Embeddings(schema),
            ContrastiveOutput(item_id_col_schema, dot_product=DotProduct(*schema.column_names)),
        )

        user_id = torch.tensor([1, 2, 3])
        item_id = torch.tensor([1, 2, 3])

        data = {"user_id": user_id, "item_id": item_id}
        contrastive_outputs, targets = model(data)

        model.eval()
        outputs = model(data)

        self.assert_contrastive_outputs(contrastive_outputs, targets, outputs)

    def test_call_dot_product_table(self, user_id_col_schema, item_id_col_schema):
        schema = Schema([user_id_col_schema, item_id_col_schema])
        embeddings = Embeddings(schema, 10, aggregation="concat")
        model = Model(
            embeddings,
            MLPBlock([10]),
            ContrastiveOutput(
                embeddings.select_by_tag(Tags.ITEM_ID).first,
            ),
        )

        user_id = torch.tensor([1, 2, 3])
        item_id = torch.tensor([1, 2, 3])

        data = {"user_id": user_id, "item_id": item_id}
        contrastive_outputs, targets = model(data, targets=item_id + 1)

        model.eval()
        outputs = model(data)

        self.assert_contrastive_outputs(contrastive_outputs, targets, outputs)

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
