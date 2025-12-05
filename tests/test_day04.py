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
