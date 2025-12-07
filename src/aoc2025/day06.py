import math
from collections.abc import Callable, Iterable
from pathlib import Path
from pprint import pprint
from typing import TypeVar

T = TypeVar("T")
ReduceFn = Callable[[Iterable[T]], T]


class MatrixReduce:
    """A class for performing matrix reduction operations."""

    @staticmethod
    def is_valid_matrix(matrix: list[list[T]]) -> bool:
        """Check if a matrix is valid."""
        if not matrix or not all(len(row) == len(matrix[0]) for row in matrix):
            return False
        return True

    @staticmethod
    def all_reduce(
        matrix: list[list[T]], reduce_fn: ReduceFn = sum, ragged: bool = False
    ) -> T:
        """Reduce a matrix to a single value by applying a reduction function."""

        if ragged:
            return reduce_fn(
                MatrixReduce.row_reduce(matrix, [reduce_fn] * len(matrix[0]), ragged)
            )

        if not MatrixReduce.is_valid_matrix(matrix):
            raise ValueError("Matrix must be valid")
        return reduce_fn((reduce_fn(row) for row in matrix))

    @staticmethod
    def row_reduce(
        matrix: list[list[T]], reduce_fns: list[ReduceFn], ragged: bool = False
    ) -> list[T]:
        """Reduce each row of a matrix to a single value."""

        if ragged and len(reduce_fns) != len(matrix):
            raise ValueError("Number of reduce functions must match number of rows")

        if not ragged and not MatrixReduce.is_valid_matrix(matrix):
            raise ValueError("Matrix must be valid")

        return [
            reduce_fn(row) for row, reduce_fn in zip(matrix, reduce_fns, strict=True)
        ]

    @staticmethod
    def column_reduce(matrix: list[list[T]], reduce_fns: list[ReduceFn]) -> list[T]:
        """Reduce each column of a matrix to a single value."""
        if not MatrixReduce.is_valid_matrix(matrix):
            raise ValueError("Matrix must be valid")
        return [
            reduce_fn(col)
            for col, reduce_fn in zip(
                zip(*matrix, strict=True), reduce_fns, strict=True
            )
        ]


def transposed_parsing(data: list[str]) -> list[list[int]]:
    """Parse data constructing numbers column-wise instead of row-wise. Produces
    ragged matrices."""

    transposed_data: list[str] = ["".join(d) for d in zip(*data, strict=True)]

    matrix: list[list[int]] = []
    row: list[int] = []

    for symbol in transposed_data:
        sym = symbol.strip()
        if sym == "":
            matrix.append(row)
            row = []
        else:
            row.append(int(sym))  # Will raise ValueError if symbol is not an int

    if row:
        matrix.append(row)

    return matrix


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day06.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = data_path.read_text().splitlines()

    if not data:
        raise ValueError("Input data is empty")

    pprint(data)

    # The last line is the operators, the remaining form the matrix.

    operators: list[str] = data[-1].split()

    matrix: list[list[int]] = [list(map(int, row.split())) for row in data[:-1]]

    # Convert operators from str to callable. There will be only two: "*" or "+".

    reduce_fns: list[ReduceFn] = [math.prod if op == "*" else sum for op in operators]

    col_reduced: list[int] = MatrixReduce.column_reduce(matrix, reduce_fns)

    print(f"Solution to part 1: {sum(col_reduced)}")

    matrix_transposed: list[list[int]] = transposed_parsing(data[:-1])

    row_reduced: list[int] = MatrixReduce.row_reduce(
        matrix_transposed, reduce_fns, ragged=True
    )

    print(f"Solution to part 2: {sum(row_reduced)}")


if __name__ == "__main__":
    main()
