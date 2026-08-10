import numpy as np
import pytest

from model import (
    add_one_smoothing,
    allocate_count_matrix,
    average_nll,
    decode_generated_sequence,
    generate_sequence,
    get_batch,
    loop_fill_counts,
    normalize_counts_to_probs,
    pick_block_size,
    pick_split_point,
    read_text_file,
    sample_random_batch_offsets,
    slice_train_and_val,
    vectorize_counts_add_at,
)


def test_read_text_file_validates_input():
    assert read_text_file("hello") == "hello"
    with pytest.raises(TypeError):
        read_text_file(123)
    with pytest.raises(ValueError):
        read_text_file("")


def test_split_sizes_cover_the_data_without_overlap():
    data = np.arange(100)
    idx = pick_split_point(len(data), 0.8)
    train, val = slice_train_and_val(data, idx)
    assert len(train) == idx
    assert len(train) + len(val) == len(data)
    assert np.array_equal(np.concatenate([train, val]), data)


def test_pick_block_size_is_at_least_one():
    assert pick_block_size(8) == 8
    assert pick_block_size(0) == 1


def test_sample_random_batch_offsets_stay_in_range():
    rng = np.random.default_rng(1)
    data_len, block_size = 50, 6
    offsets = sample_random_batch_offsets(data_len, block_size, batch_size=500, rng=rng)
    assert offsets.min() >= 0
    assert offsets.max() < data_len - block_size


def test_get_batch_shape_and_y_is_x_shifted_by_one():
    data = np.arange(200)
    rng = np.random.default_rng(0)
    xb, yb = get_batch(data, block_size=10, batch_size=8, rng=rng)
    assert xb.shape == (8, 10)
    assert yb.shape == (8, 10)
    assert np.array_equal(yb, xb + 1)


def test_vectorized_scatter_add_matches_looped_counts():
    rng = np.random.default_rng(3)
    data = rng.integers(0, 5, size=200)
    looped = loop_fill_counts(allocate_count_matrix(5), data)
    vectorized = vectorize_counts_add_at(5, data)
    assert np.array_equal(looped, vectorized)


def test_normalize_counts_to_probs_rows_sum_to_one():
    counts = add_one_smoothing(np.array([[0, 3, 0], [1, 1, 1], [0, 0, 5]]))
    probs = normalize_counts_to_probs(counts)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_add_one_smoothing_removes_zeros():
    counts = np.array([[0, 3, 0], [1, 0, 1], [0, 0, 5]])
    assert np.any(counts == 0)
    smoothed = add_one_smoothing(counts)
    assert np.all(smoothed > 0)


def test_average_nll_is_nonnegative():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 4, size=50)
    counts = add_one_smoothing(vectorize_counts_add_at(4, data))
    probs = normalize_counts_to_probs(counts)
    assert average_nll(probs, data) >= 0.0


def test_generate_sequence_length_and_start():
    rng = np.random.default_rng(4)
    probs = normalize_counts_to_probs(add_one_smoothing(np.zeros((3, 3), dtype=int)))
    seq = generate_sequence(probs, start_id=1, length=6, rng=rng)
    assert len(seq) == 6
    assert seq[0] == 1
    text = decode_generated_sequence(seq, {0: 'a', 1: 'b', 2: 'c'})
    assert len(text) == 6
