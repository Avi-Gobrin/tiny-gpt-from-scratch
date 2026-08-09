import numpy as np

from conftest import TINY_BLOCK, TINY_N_LAYERS, TINY_VOCAB, make_tiny_params, numerical_gradient
from model import batch_cross_entropy, full_model_backward, full_model_forward

TOL = dict(rtol=1e-5, atol=1e-6)


def test_full_model_forward_logits_shape():
    params = make_tiny_params()
    rng = np.random.default_rng(0)
    ids = rng.integers(0, TINY_VOCAB, size=(3, TINY_BLOCK))
    logits = full_model_forward(params, ids)['logits']
    assert logits.shape == (3, TINY_BLOCK, TINY_VOCAB)


def test_full_model_causality_end_to_end():
    params = make_tiny_params()
    rng = np.random.default_rng(1)
    ids = rng.integers(0, TINY_VOCAB, size=(2, TINY_BLOCK))
    logits1 = full_model_forward(params, ids)['logits']

    ids2 = ids.copy()
    ids2[:, -1] = (ids2[:, -1] + 1) % TINY_VOCAB
    logits2 = full_model_forward(params, ids2)['logits']

    assert np.allclose(logits1[:, :-1, :], logits2[:, :-1, :])
    assert not np.allclose(logits1[:, -1, :], logits2[:, -1, :])


def test_full_model_backward_matches_numerical_gradient_for_every_parameter():
    params = make_tiny_params()
    rng = np.random.default_rng(2)
    ids = rng.integers(0, TINY_VOCAB, size=(2, TINY_BLOCK))
    targets = rng.integers(0, TINY_VOCAB, size=(2, TINY_BLOCK))

    out = full_model_forward(params, ids)
    grads = full_model_backward(params, out['cache'], targets)

    def loss(_):
        return batch_cross_entropy(full_model_forward(params, ids)['logits'], targets)

    np.testing.assert_allclose(grads['tok_emb'], numerical_gradient(loss, params['tok_emb']), **TOL)
    np.testing.assert_allclose(grads['pos_emb'], numerical_gradient(loss, params['pos_emb']), **TOL)
    np.testing.assert_allclose(
        grads['ln_f']['gamma'], numerical_gradient(loss, params['ln_f']['gamma']), **TOL)
    np.testing.assert_allclose(
        grads['ln_f']['beta'], numerical_gradient(loss, params['ln_f']['beta']), **TOL)
    np.testing.assert_allclose(
        grads['lm_head']['w_lm'], numerical_gradient(loss, params['lm_head']['w_lm']), **TOL)
    np.testing.assert_allclose(
        grads['lm_head']['b_lm'], numerical_gradient(loss, params['lm_head']['b_lm']), **TOL)

    for i in range(TINY_N_LAYERS):
        block, block_grads = params['blocks'][i], grads['blocks'][i]
        for ln in ('ln1', 'ln2'):
            for key in ('gamma', 'beta'):
                np.testing.assert_allclose(
                    block_grads[ln][key], numerical_gradient(loss, block[ln][key]), **TOL)
        for key in ('W_q', 'W_k', 'W_v', 'W_o'):
            np.testing.assert_allclose(
                block_grads['attn'][key], numerical_gradient(loss, block['attn'][key]), **TOL)
        for key in ('w1', 'b1', 'w2', 'b2'):
            np.testing.assert_allclose(
                block_grads['ffn'][key], numerical_gradient(loss, block['ffn'][key]), **TOL)
