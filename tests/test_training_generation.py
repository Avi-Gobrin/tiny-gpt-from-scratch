import numpy as np

from conftest import TINY_BLOCK, TINY_VOCAB, make_tiny_params
from model import (
    adam_bias_correction,
    adam_increment_step,
    adam_parameter_update,
    adam_update_first_moment,
    adam_update_second_moment,
    apply_temperature,
    build_itos,
    build_stoi,
    build_vocab,
    crop_context_to_block_size,
    decode_final_sequence,
    encode_prompt,
    generation_loop_for_n_steps,
    stable_softmax_1d,
    top_k_filter,
    wire_full_training_loop,
)


def test_adam_bias_correction_matches_known_values():
    hats = adam_bias_correction(m=np.array([1.0]), v=np.array([4.0]), beta1=0.9, beta2=0.999, t=1)
    assert np.isclose(hats['m_hat'][0], 10.0)
    assert np.isclose(hats['v_hat'][0], 4000.0)


def test_adam_minimizes_a_simple_quadratic():
    x = np.array([5.0])
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    for _ in range(200):
        grad = 2 * x
        t = adam_increment_step(t)
        m = adam_update_first_moment(m, grad)
        v = adam_update_second_moment(v, grad)
        hats = adam_bias_correction(m, v, 0.9, 0.999, t)
        x = adam_parameter_update(x, hats['m_hat'], hats['v_hat'], learning_rate=0.1)
    assert abs(x[0]) < 0.1


def test_wire_full_training_loop_lowers_loss():
    params = make_tiny_params()
    data = np.tile(np.arange(TINY_VOCAB), 30)
    result = wire_full_training_loop(
        params, data, block_size=TINY_BLOCK, batch_size=8,
        learning_rate=0.02, num_steps=100, log_every=10)
    assert result['loss_history'][-1] < result['loss_history'][0]


def test_apply_temperature_changes_distribution_sharpness():
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    cold = stable_softmax_1d(apply_temperature(logits, 0.5))
    hot = stable_softmax_1d(apply_temperature(logits, 2.0))
    assert cold.max() > hot.max()


def test_top_k_filter_keeps_exactly_k_largest():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=20)
    filtered = top_k_filter(logits, k=5)
    finite = filtered[np.isfinite(filtered)]
    assert finite.size == 5
    assert np.array_equal(np.sort(finite), np.sort(logits)[-5:])


def test_crop_context_never_exceeds_block_size():
    ids = np.arange(20)
    cropped = crop_context_to_block_size(ids, block_size=5)
    assert len(cropped) == 5
    assert np.array_equal(cropped, ids[-5:])

    short_ids = np.arange(3)
    cropped_short = crop_context_to_block_size(short_ids, block_size=5)
    assert len(cropped_short) <= 5
    assert np.array_equal(cropped_short, short_ids)


def test_generation_returns_prompt_plus_n_new_tokens_and_decodes():
    params = make_tiny_params()
    vocab = build_vocab("abcdefghijk")
    stoi, itos = build_stoi(vocab), build_itos(vocab)
    prompt_ids = encode_prompt("abc", stoi)

    rng = np.random.default_rng(0)
    out_ids = generation_loop_for_n_steps(
        params, prompt_ids, n_new_tokens=7, block_size=TINY_BLOCK, top_k=5, rng=rng)

    assert len(out_ids) == len(prompt_ids) + 7
    assert np.array_equal(out_ids[:len(prompt_ids)], prompt_ids)
    text = decode_final_sequence(out_ids, itos)
    assert isinstance(text, str)
    assert len(text) == len(out_ids)
