import pytest
import tensorflow as tf
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPoolingAndCrossAttentions

from merlin.models.tf.core.base import block_registry
from merlin.models.tf.transformers import transforms

TRANSFORMER_IN = tf.random.uniform((1, 1, 8))
TRANSFORMER_OUT = TFBaseModelOutputWithPoolingAndCrossAttentions(
    last_hidden_state=tf.random.uniform((1, 1, 8)),
    pooler_output=tf.random.uniform((1, 8)),
    past_key_values=list(tf.random.uniform((1, 8))),
    hidden_states=tuple(tf.random.uniform((1, 8))),
    attentions=tuple(tf.random.uniform((1, 8))),
    cross_attentions=tuple(tf.random.uniform((1, 8))),
)


@pytest.mark.parametrize(
    "in_out",
    [
        (transforms.PrepareTransformerInputs, {"inputs_embeds": TRANSFORMER_IN}),
    ],
)
def test_transformer_pre(in_out):
    transform_layer, expected_output = in_out
    out = transform_layer()(TRANSFORMER_IN)
    assert type(out) == type(expected_output)
    for out_name, out_val in out.items():
        tf.assert_equal(out_val, expected_output[out_name])


@pytest.mark.parametrize(
    "in_out",
    [
        (transforms.LastHiddenState, TRANSFORMER_OUT.last_hidden_state),
        (transforms.PoolerOutput, TRANSFORMER_OUT.pooler_output),
        (transforms.HiddenStates, TRANSFORMER_OUT.hidden_states),
        (transforms.AttentionWeights, TRANSFORMER_OUT.attentions),
        (
            transforms.LastHiddenStateAndAttention,
            (TRANSFORMER_OUT.last_hidden_state, TRANSFORMER_OUT.attentions[-1]),
        ),
    ],
)
def test_transformer_post(in_out):
    transform_layer, expected_output = in_out
    out = transform_layer()(TRANSFORMER_OUT)
    assert type(out) == type(expected_output)
    if isinstance(expected_output, (list, tuple)):
        for out_element, expected_output_element in zip(out, expected_output):
            tf.assert_equal(out_element, expected_output_element)
    else:
        tf.assert_equal(out, expected_output)


@pytest.mark.parametrize("post", ["first", "last", "mean", "cls_index"])
def test_post(post):
    transformer_post = block_registry.parse("sequence_" + post)

    assert transformer_post(TRANSFORMER_IN) is not None
