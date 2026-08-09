import numpy as np
import pytest

from conftest import numerical_gradient
from model import (
    build_causal_mask,
    choose_attention_head_config,
    compute_key,
    compute_query,
    create_multihead_output_projection,
    create_multihead_qkv_projections,
    merge_heads_to_d_model,
    multihead_attention_backward,
    multihead_attention_forward,
    multihead_masked_softmax_scores,
    split_into_heads,
    transpose_heads_to_back,
)

TOL = dict(rtol=1e-5, atol=1e-6)


def test_causal_mask_is_lower_triangular():
    mask = build_causal_mask(5)
    row, col = np.indices((5, 5))
    assert np.array_equal(mask, col <= row)


def test_choose_attention_head_config_splits_evenly_and_validates():
    assert choose_attention_head_config(8, 2) == {'n_heads': 2, 'd_head': 4, 'd_model': 8}
    with pytest.raises(ValueError):
        choose_attention_head_config(8, 3)


def test_reshape_and_transpose_heads_round_trip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 8))
    heads = split_into_heads(x, n_heads=2)
    back = merge_heads_to_d_model(transpose_heads_to_back(heads))
    assert np.allclose(x, back)


def test_multihead_masked_softmax_scores_rows_sum_to_one():
    np.random.seed(0)
    proj = create_multihead_qkv_projections(8, 2)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5, 8))
    q = split_into_heads(compute_query(x, proj['W_q']), 2)
    k = split_into_heads(compute_key(x, proj['W_k']), 2)
    weights = multihead_masked_softmax_scores(q, k)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_changing_last_token_does_not_change_earlier_outputs():
    np.random.seed(0)
    attn = {'n_heads': 2, **create_multihead_qkv_projections(8, 2), 'W_o': create_multihead_output_projection(8)}
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 5, 8))
    y1 = multihead_attention_forward(x, attn)['y']

    x2 = x.copy()
    x2[:, -1, :] = rng.normal(size=(2, 8))
    y2 = multihead_attention_forward(x2, attn)['y']

    assert np.allclose(y1[:, :-1, :], y2[:, :-1, :])
    assert not np.allclose(y1[:, -1, :], y2[:, -1, :])


def test_multihead_attention_backward_matches_numerical_gradient():
    np.random.seed(0)
    attn = {'n_heads': 2, **create_multihead_qkv_projections(8, 2), 'W_o': create_multihead_output_projection(8)}
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 5, 8)) * 0.5

    out = multihead_attention_forward(x, attn)
    dy = rng.normal(size=out['y'].shape)
    analytic = multihead_attention_backward(dy, out['cache'])

    def loss(_):
        return np.sum(multihead_attention_forward(x, attn)['y'] * dy)

    numeric_dx = numerical_gradient(loss, x)
    np.testing.assert_allclose(analytic['dx'], numeric_dx, **TOL)

    for key, grad_key in [('W_q', 'dW_q'), ('W_k', 'dW_k'), ('W_v', 'dW_v'), ('W_o', 'dW_o')]:
        numeric = numerical_gradient(loss, attn[key])
        np.testing.assert_allclose(analytic[grad_key], numeric, **TOL)
