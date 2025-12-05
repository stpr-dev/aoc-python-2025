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


def test_extract_window_basic() -> None:
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    w = d04.Conv2d.extract_window(matrix, top=1, left=1, height=2, width=2)
    assert w == [
        [5, 6],
        [8, 9],
    ]


def test_extract_window_full() -> None:
    matrix = [
        [1, 2],
        [3, 4],
    ]
    w = d04.Conv2d.extract_window(matrix, 0, 0, 2, 2)
    assert w == [[1, 2], [3, 4]]


def test_extract_window_raises() -> None:
    matrix = [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        d04.Conv2d.extract_window(matrix, 1, 0, 3, 2)  # too


def test_convolve_identity_kernel() -> None:
    kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    conv = d04.Conv2d(kernel)

    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    out = conv.convolve(data, padding=1)
    # identity kernel → same as input
    expected = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert out == expected


def test_convolve_summing_kernel() -> None:
    kernel = [[1, 1], [1, 1]]
    conv = d04.Conv2d(kernel)

    data = [
        [1, 2],
        [3, 4],
    ]

    out = conv.convolve(data, padding=0)
    # sum of all elements = 10
    assert out == [[10]]


def test_convolve_padding() -> None:
    kernel = [[1]]
    conv = d04.Conv2d(kernel)
    data = [[5]]
    out = conv.convolve(data, padding=1)
    # kernel = [[1]], so just copies padded matrix
    assert out == [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]


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


@given(
    matrix=matrix_strategy(),
    data=st.data(),
)
def test_extract_window_property(matrix, data):
    rows = len(matrix)
    cols = len(matrix[0])

    # draw valid window sizes
    height = data.draw(st.integers(min_value=1, max_value=rows))
    width = data.draw(st.integers(min_value=1, max_value=cols))

    # draw valid top-left corner
    top = data.draw(st.integers(min_value=0, max_value=rows - height))
    left = data.draw(st.integers(min_value=0, max_value=cols - width))

    w = d04.Conv2d.extract_window(matrix, top, left, height, width)

    # shape correct
    assert len(w) == height
    assert all(len(row) == width for row in w)

    # values preserved
    for i in range(height):
        for j in range(width):
            assert w[i][j] == matrix[top + i][left + j]


@st.composite
def conv_input_strategy(draw):
    rows = draw(st.integers(min_value=2, max_value=6))
    cols = draw(st.integers(min_value=2, max_value=6))
    data = draw(
        st.lists(
            st.lists(st.integers(-10, 10), min_size=cols, max_size=cols),
            min_size=rows,
            max_size=rows,
        )
    )
    krows = draw(st.integers(min_value=1, max_value=rows))
    kcols = draw(st.integers(min_value=1, max_value=cols))
    kernel = draw(
        st.lists(
            st.lists(st.integers(-2, 2), min_size=kcols, max_size=kcols),
            min_size=krows,
            max_size=krows,
        )
    )
    padding = draw(st.integers(0, 2))
    return data, kernel, padding


@given(data=conv_input_strategy())
def test_convolve_shape(data):
    data, kernel, padding = data
    conv = d04.Conv2d(kernel)

    out = conv.convolve(data, padding=padding)
    padded_h = len(d04.Conv2d.pad_matrix(data, padding))
    padded_w = len(d04.Conv2d.pad_matrix(data, padding)[0])

    expected_h = padded_h - len(kernel) + 1
    expected_w = padded_w - len(kernel[0]) + 1

    assert len(out) == expected_h
    assert all(len(row) == expected_w for row in out)
