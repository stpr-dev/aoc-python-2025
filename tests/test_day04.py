import pytest
from hypothesis import given
import hypothesis.strategies as st

import aoc2025.day04 as d04


# ---------------------------------------------------------------------------
# Parametrized deterministic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "data", "expected"),
    [
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[5, 12], [21, 32]],
        ),
        (
            [[-1, 2], [3, -4]],
            [[5, -6], [-7, 8]],
            [[-5, -12], [-21, -32]],
        ),
        (
            [[1.0, 0.5], [0.25, 0.1]],
            [[2.0, 4.0], [8.0, 10.0]],
            [[2.0, 2.0], [2.0, 1.0]],
        ),
    ],
)
def test_hadamard_product(kernel, data, expected) -> None:
    assert d04.Conv2d.hadamard_product(kernel, data) == expected


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([[1, 2], [3, 4]], 10),
        ([[-1, -2], [3, 0]], 0),
        ([[1.5, 2.5], [3.0, 4.0]], pytest.approx(11.0)),
    ],
)
def test_reduce_matrix(matrix, expected) -> None:
    assert d04.Conv2d.reduce_matrix(matrix) == expected


def test_pad_matrix_basic() -> None:
    matrix = [
        [1, 2],
        [3, 4],
    ]
    padded = d04.Conv2d.pad_matrix(matrix, padding=1)

    expected = [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]

    assert padded == expected


def test_pad_matrix_padding_2() -> None:
    matrix = [
        [5, 6],
    ]
    padded = d04.Conv2d.pad_matrix(matrix, padding=2)

    expected = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 5, 6, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert padded == expected


def test_pad_matrix_float_zero() -> None:
    matrix = [
        [1.5, -2.0],
    ]
    padded = d04.Conv2d.pad_matrix(matrix, padding=1)

    expected_zero = 1.5 - 1.5  # float zero
    assert padded[0][0] == expected_zero
    assert padded[-1][-1] == expected_zero
    assert all(len(row) == 4 for row in padded)  # width check
    assert len(padded) == 3  # height check


def test_pad_matrix_preserves_values() -> None:
    matrix = [
        [7],
        [8],
    ]
    padded = d04.Conv2d.pad_matrix(matrix, padding=1)

    assert padded[1][1] == 7
    assert padded[2][1] == 8


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


# Rectangular non-empty matrices of floats
@st.composite
def matrix_strategy(draw):
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))

    row_strategy = st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=cols,
        max_size=cols,
    )
    return draw(st.lists(row_strategy, min_size=rows, max_size=rows))


@given(
    a=matrix_strategy(),
    b=matrix_strategy(),
)
def test_hadamard_product_elementwise(a, b) -> None:
    """Checks: result[i][j] == a[i][j] * b[i][j] for all i,j."""
    assume_same_shape = len(a) == len(b) and all(
        len(a[i]) == len(b[i]) for i, _ in enumerate(a)
    )
    if not assume_same_shape:
        return  # skip cases where shapes differ

    result = d04.Conv2d.hadamard_product(a, b)
    for i, row in enumerate(a):
        for j, _ in enumerate(row):
            assert result[i][j] == a[i][j] * b[i][j]


@given(matrix=matrix_strategy())
def test_reduce_matrix_matches_manual_sum(matrix) -> None:
    """Checks: reduce_matrix(matrix) == sum(flat(matrix))."""
    expected = sum(sum(row) for row in matrix)
    assert d04.Conv2d.reduce_matrix(matrix) == expected


@given(
    matrix=matrix_strategy(),
    pad=st.integers(min_value=1, max_value=3),
)
def test_pad_matrix_property(matrix, pad):
    padded = d04.Conv2d.pad_matrix(matrix, padding=pad)

    rows, cols = len(matrix), len(matrix[0])
    expected_rows = rows + 2 * pad
    expected_cols = cols + 2 * pad

    # shape is correct
    assert len(padded) == expected_rows
    assert all(len(row) == expected_cols for row in padded)

    # inner region is preserved
    for i, row in enumerate(matrix):
        for j, _ in enumerate(row):
            assert padded[i + pad][j + pad] == matrix[i][j]

    # border region is zero of correct type
    zero = matrix[0][0] - matrix[0][0]
    for i, row in enumerate(padded):
        for j, _ in enumerate(row):
            if i < pad or i >= pad + rows or j < pad or j >= pad + cols:
                assert row[j] == zero
