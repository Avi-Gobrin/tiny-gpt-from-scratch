import numpy as np

from model import (
    array_exp,
    array_log,
    make_2d_random,
    make_2d_zeros,
    matmul,
    max_along_axis,
    naive_softmax_1d,
    softmax_overflow_demo,
    stable_softmax_1d,
    stable_softmax_2d_rowwise,
    sum_all,
    sum_axis0,
    sum_axis1,
    sum_keepdims,
    transpose_matrix,
)


def test_make_2d_zeros_shape_and_fill():
    arr = make_2d_zeros(3, 4)
    assert arr.shape == (3, 4)
    assert np.all(arr == 0)


def test_make_2d_random_is_seeded_and_bounded():
    a = make_2d_random(3, 4, seed=7)
    b = make_2d_random(3, 4, seed=7)
    assert np.array_equal(a, b)
    assert np.all(a >= 0) and np.all(a < 1)


def test_array_exp_and_log_are_inverses():
    x = np.array([0.1, 1.0, 2.5, 4.0])
    assert np.allclose(array_log(array_exp(x)), x)


def test_sum_reductions_agree_with_each_other():
    arr = np.arange(12.0).reshape(3, 4)
    assert np.isclose(sum_axis0(arr).sum(), sum_all(arr))
    assert np.isclose(sum_axis1(arr).sum(), sum_all(arr))
    assert sum_keepdims(arr, 1).shape == (3, 1)
    assert np.isclose(sum_keepdims(arr, 1).sum(), sum_all(arr))


def test_max_along_axis_bounds_every_row():
    arr = np.array([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])
    row_max = max_along_axis(arr, 1)
    assert np.all(arr <= row_max[:, None])


def test_transpose_matrix_shape_and_involution():
    arr = np.arange(15.0).reshape(3, 5)
    t = transpose_matrix(arr)
    assert t.shape == (5, 3)
    assert np.array_equal(transpose_matrix(t), arr)


def test_matmul_matches_known_product():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.array_equal(matmul(a, b), np.array([[19.0, 22.0], [43.0, 50.0]]))


def test_naive_and_stable_softmax_agree_on_small_inputs():
    logits = np.array([1.0, 2.0, -1.0, 0.5])
    assert np.allclose(naive_softmax_1d(logits), stable_softmax_1d(logits))


def test_naive_softmax_overflows_but_stable_does_not():
    demo = softmax_overflow_demo(1000.0)
    assert demo['overflowed']
    assert np.isinf(demo['naive_exp'])
    stable = stable_softmax_1d(np.array([1000.0, 0.0, 0.0]))
    assert np.all(np.isfinite(stable))


def test_stable_softmax_1d_sums_to_one():
    probs = stable_softmax_1d(np.array([3.0, -2.0, 0.5, 10.0]))
    assert np.isclose(probs.sum(), 1.0)


def test_stable_softmax_2d_rowwise_matches_1d_and_sums_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [10.0, -5.0, 0.0], [0.0, 0.0, 0.0]])
    rowwise = stable_softmax_2d_rowwise(logits)
    assert np.allclose(rowwise.sum(axis=1), 1.0)
    for i in range(logits.shape[0]):
        assert np.allclose(rowwise[i], stable_softmax_1d(logits[i]))
