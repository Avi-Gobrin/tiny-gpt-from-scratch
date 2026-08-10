import numpy as np

from conftest import numerical_gradient
from model import (
    bias_add_backward_db,
    bias_add_forward,
    cross_entropy_loss,
    derive_dx_on_paper,
    derive_linear_dw_on_paper,
    layernorm_backward_full,
    layernorm_backward_implementation,
    layernorm_forward,
    linear_backward_dx,
    linear_backward_dw,
    linear_forward,
    logits_to_probs_rowwise,
    relu_backward,
    relu_forward,
    softmax_cross_entropy_backward,
)

TOL = dict(rtol=1e-5, atol=1e-6)


def test_derive_dx_on_paper_states_the_gradient():
    text = derive_dx_on_paper()
    assert text.count('\n') > 1
    assert 'dL/dX = dY @ W.T' in text


def test_derive_linear_dw_on_paper_states_the_gradient():
    text = derive_linear_dw_on_paper()
    assert text.count('\n') > 1
    assert 'dL/dW = X.T @ dY' in text


def test_layernorm_backward_full_states_the_gradient():
    text = layernorm_backward_full()
    assert text.count('\n') > 1
    assert 'dx = (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat)) / std' in text


def test_linear_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    w = rng.normal(size=(3, 5))
    dy = rng.normal(size=(4, 5))
    cache = linear_forward(x, w)['cache']

    analytic_dx = linear_backward_dx(dy, cache)
    analytic_dw = linear_backward_dw(dy, cache)
    numeric_dx = numerical_gradient(lambda x_: np.sum(linear_forward(x_, w)['y'] * dy), x)
    numeric_dw = numerical_gradient(lambda w_: np.sum(linear_forward(x, w_)['y'] * dy), w)

    np.testing.assert_allclose(analytic_dx, numeric_dx, **TOL)
    np.testing.assert_allclose(analytic_dw, numeric_dw, **TOL)


def test_bias_add_backward_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 5))
    b = rng.normal(size=(5,))
    dy = rng.normal(size=(4, 5))

    analytic_db = bias_add_backward_db(dy)
    numeric_db = numerical_gradient(lambda b_: np.sum(bias_add_forward(x, b_)['y'] * dy), b)

    np.testing.assert_allclose(analytic_db, numeric_db, **TOL)


def test_relu_backward_matches_numerical_gradient():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 1.5, size=(4, 5)) * rng.choice([-1.0, 1.0], size=(4, 5))
    dy = rng.normal(size=(4, 5))

    analytic = relu_backward(dy, {'x': x})
    numeric = numerical_gradient(lambda x_: np.sum(relu_forward(x_)['y'] * dy), x)

    np.testing.assert_allclose(analytic, numeric, **TOL)


def test_softmax_cross_entropy_backward_matches_numerical_gradient():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, 6))
    targets = rng.integers(0, 6, size=4)

    probs = logits_to_probs_rowwise(logits)
    analytic = softmax_cross_entropy_backward(probs, targets)
    numeric = numerical_gradient(
        lambda l_: cross_entropy_loss(logits_to_probs_rowwise(l_), targets), logits)

    np.testing.assert_allclose(analytic, numeric, **TOL)


def test_layernorm_backward_matches_numerical_gradient():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 6))
    gamma = rng.normal(size=(6,)) + 1.0
    beta = rng.normal(size=(6,))
    dy = rng.normal(size=(4, 6))

    cache = layernorm_forward(x, gamma, beta)['cache']
    analytic = layernorm_backward_implementation(dy, cache)
    numeric_dx = numerical_gradient(lambda x_: np.sum(layernorm_forward(x_, gamma, beta)['y'] * dy), x)
    numeric_dgamma = numerical_gradient(lambda g_: np.sum(layernorm_forward(x, g_, beta)['y'] * dy), gamma)
    numeric_dbeta = numerical_gradient(lambda b_: np.sum(layernorm_forward(x, gamma, b_)['y'] * dy), beta)

    np.testing.assert_allclose(analytic['dx'], numeric_dx, **TOL)
    np.testing.assert_allclose(analytic['dgamma'], numeric_dgamma, **TOL)
    np.testing.assert_allclose(analytic['dbeta'], numeric_dbeta, **TOL)
