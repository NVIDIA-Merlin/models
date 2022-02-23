#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# import pytest
# import merlin.models.torch as ml

# examples_utils = pytest.importorskip("merlin.models.torch.losses")


# @pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.6])
# def test_item_prediction_with_label_smoothing_ce_loss(
#     torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, label_smoothing
# ):
#     np = pytest.importorskip("numpy")
#     custom_loss = examples_utils.LabelSmoothCrossEntropyLoss(
#         reduction="mean", smoothing=label_smoothing
#     )
#     input_module = torch_yoochoose_tabular_transformer_features
#     body = ml.SequentialBlock(input_module, ml.MLPBlock([64]))
#     head = ml.Head(
#         body, ml.NextItemPredictionTask(weight_tying=True, loss=custom_loss), inputs=input_module
#     )
#
#     body_outputs = body(torch_yoochoose_like)
#
#     trg_flat = input_module.masking.masked_targets.flatten()
#     non_pad_mask = trg_flat != input_module.masking.padding_idx
#     labels_all = torch.masked_select(trg_flat, non_pad_mask)
#     predictions = head(body_outputs)
#
#     loss = head.prediction_task_dict["next-item"].compute_loss(
#         inputs=body_outputs,
#         targets=labels_all,
#     )
#
#     n_classes = 51997
#     manuall_loss = torch.nn.NLLLoss(reduction="mean")
#     target_with_smoothing = labels_all * (1 - label_smoothing) + label_smoothing / n_classes
#     manual_output_loss = manuall_loss(predictions, target_with_smoothing.to(torch.long))
#
#     assert np.allclose(manual_output_loss.detach().numpy(), loss.detach().numpy(), rtol=1e-3)
