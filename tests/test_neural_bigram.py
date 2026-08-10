import numpy as np

from model import (
    compute_dlogits,
    cross_entropy_loss,
    derive_dlogits_on_paper,
    derive_dw_on_paper,
    forward_logits_lookup,
    initialize_w_random,
    logits_to_probs_rowwise,
    observe_lookup_equivalence,
    run_one_training_step,
    scale_w_small,
    softmax_cross_entropy_backward,
    train_neural_bigram_loop,
)


def test_derive_dlogits_on_paper_states_the_gradient():
    text = derive_dlogits_on_paper()
    assert text.count('\n') > 1
    assert '(probs - onehot(targets)) / B' in text


def test_derive_dw_on_paper_states_the_gradient():
    text = derive_dw_on_paper()
    assert text.count('\n') > 1
    assert 'dL/dW = O.T @ dlogits' in text


def test_observe_lookup_equivalence_matches_onehot_and_index():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 6))
    ids = np.array([0, 3, 5, 2])
    result = observe_lookup_equivalence(w, ids)
    assert np.allclose(result['onehot_result'], result['index_result'])


def test_cross_entropy_loss_perfect_and_uniform():
    targets = np.array([0, 1, 2])
    perfect = np.eye(3)[targets]
    assert cross_entropy_loss(perfect, targets) < 1e-8

    uniform = np.full((3, 4), 0.25)
    assert np.isclose(cross_entropy_loss(uniform, targets), np.log(4))


def test_compute_dlogits_matches_softmax_cross_entropy_backward():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 4))
    targets = rng.integers(0, 4, size=5)
    probs = logits_to_probs_rowwise(logits)
    assert np.allclose(compute_dlogits(probs, targets), softmax_cross_entropy_backward(probs, targets))


def test_one_training_step_lowers_loss_on_repeated_batch():
    rng = np.random.default_rng(0)
    w = scale_w_small(initialize_w_random(5, rng), 0.1)
    ids = np.tile(np.array([0, 1, 2, 3, 4]), 4)
    targets = np.tile(np.array([1, 2, 3, 4, 0]), 4)
    result = run_one_training_step(w, ids, targets, learning_rate=0.5)
    new_probs = logits_to_probs_rowwise(forward_logits_lookup(result['w'], ids))
    assert cross_entropy_loss(new_probs, targets) < result['loss']


def test_train_neural_bigram_loop_loss_decreases_overall():
    w = scale_w_small(initialize_w_random(6, np.random.default_rng(0)), 0.1)
    data = np.tile(np.arange(6), 40)
    result = train_neural_bigram_loop(
        w, data, block_size=4, batch_size=16, learning_rate=0.5, num_steps=100, log_every=10)
    assert result['loss_history'][-1] < result['loss_history'][0]
