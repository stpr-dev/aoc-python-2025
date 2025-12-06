# tests/test_day06.py

import pytest
from hypothesis import given
from hypothesis import strategies as st

import aoc2025.day06 as d06


# ---------------------------------------------------------------------------
# Parametrized deterministic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix, ops, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [sum, sum],
            [6, 15],
        ),
        (
            [[2, 3], [5, 7], [11, 13]],
            [sum, sum, sum],
            [5, 12, 24],
        ),
        (
            [[2, 3], [4, 5]],
            [sum, d06.math.prod],
            [5, 20],
        ),
    ],
)
def test_row_reduce(matrix, ops, expected):
    out = d06.MatrixReduce.row_reduce(matrix, ops)
    assert out == expected


@pytest.mark.parametrize(
    "matrix, ops, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [sum, sum, sum],
            [5, 7, 9],
        ),
        (
            [[2, 3], [5, 7], [11, 13]],
            [sum, sum],
            [18, 23],
        ),
        (
            [[2, 3], [4, 5]],
            [sum, d06.math.prod],
            [6, 15],
        ),
    ],
)
def test_column_reduce(matrix, ops, expected):
    out = d06.MatrixReduce.column_reduce(matrix, ops)
    assert out == expected


@pytest.mark.parametrize(
    "matrix, op, expected",
    [
        (
            [[1, 2], [3, 4]],
            sum,
            sum([sum([1, 2]), sum([3, 4])]),
        ),
        (
            [[2, 3], [4, 5]],
            d06.math.prod,
            d06.math.prod([d06.math.prod([2, 3]), d06.math.prod([4, 5])]),
        ),
    ],
)
def test_all_reduce(matrix, op, expected):
    out = d06.MatrixReduce.all_reduce(matrix, op)
    assert out == expected


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [[1, 2], [3]],
        [[1], [2, 3]],
    ],
)
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        d06.MatrixReduce.row_reduce(matrix, [sum])
    with pytest.raises(ValueError):
        d06.MatrixReduce.column_reduce(matrix, [sum])
    with pytest.raises(ValueError):
        d06.MatrixReduce.all_reduce(matrix, sum)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


# Strategy for non-empty rectangular integer matrices
@st.composite
def matrix_strategy(draw):
    rows = draw(st.integers(min_value=1, max_value=10))
    cols = draw(st.integers(min_value=1, max_value=10))
    mat = draw(
        st.lists(
            st.lists(
                st.integers(min_value=-10, max_value=10), min_size=cols, max_size=cols
            ),
            min_size=rows,
            max_size=rows,
        )
    )
    return mat


# Strategy for selecting reduce functions from {sum, prod}
def reduce_fn_strategy():
    return st.sampled_from([sum, d06.math.prod])


# Combined strategy: matrix + per-column or per-row ops
@st.composite
def conv_input_strategy(draw):
    matrix = draw(matrix_strategy())
    rows = len(matrix)
    cols = len(matrix[0])
    row_ops = draw(st.lists(reduce_fn_strategy(), min_size=rows, max_size=rows))
    col_ops = draw(st.lists(reduce_fn_strategy(), min_size=cols, max_size=cols))
    return matrix, row_ops, col_ops


@given(data=conv_input_strategy())
def test_row_and_column_transpose_consistency(data):
    matrix, row_ops, col_ops = data

    # Row reduce on M should match column reduce on M^T with corresponding ops
    row_result = d06.MatrixReduce.row_reduce(matrix, row_ops)

    # Transpose matrix
    tmat = list(map(list, zip(*matrix, strict=True)))

    # column_reduce(matrix, ops) reduces columns => apply row_ops to rows of transpose
    col_on_t = d06.MatrixReduce.column_reduce(tmat, row_ops)

    assert row_result == col_on_t


@given(data=conv_input_strategy())
def test_double_transpose_idempotent(data):
    matrix, row_ops, col_ops = data

    # T(T(M)) == M structurally
    t1 = list(map(list, zip(*matrix, strict=True)))
    t2 = list(map(list, zip(*t1, strict=True)))
    assert t2 == matrix


@given(data=conv_input_strategy())
def test_reduce_output_types(data):
    matrix, row_ops, col_ops = data

    # Check row_reduce returns list[T]
    rr = d06.MatrixReduce.row_reduce(matrix, row_ops)
    assert isinstance(rr, list)
    assert len(rr) == len(matrix)

    # Check column_reduce returns list[T]
    cr = d06.MatrixReduce.column_reduce(matrix, col_ops)
    assert isinstance(cr, list)
    assert len(cr) == len(matrix[0])


@given(data=conv_input_strategy())
def test_all_reduce_matches_row_of_column(data):
    matrix, row_ops, col_ops = data

    # Use sum or prod uniformly for all_reduce
    # pick one op arbitrarily
    op = sum

    ar = d06.MatrixReduce.all_reduce(matrix, op)

    # Equivalent: first row-reduce with op everywhere, then op again
    row_vals = [op(row) for row in matrix]
    expected = op(row_vals)

    assert ar == expected
