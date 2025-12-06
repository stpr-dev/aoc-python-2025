import math
from collections.abc import Callable, Iterable
from pathlib import Path
from pprint import pprint
from typing import TypeVar

from aoc2025.utils.io import read_input_lines

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
    def all_reduce(matrix: list[list[T]], reduce_fn: ReduceFn = sum) -> T:
        """Reduce a matrix to a single value by applying a reduction function."""
        if not MatrixReduce.is_valid_matrix(matrix):
            raise ValueError("Matrix must be valid")
        return reduce_fn((reduce_fn(row) for row in matrix))

    @staticmethod
    def row_reduce(matrix: list[list[T]], reduce_fns: list[ReduceFn]) -> list[T]:
        """Reduce each row of a matrix to a single value."""
        if not MatrixReduce.is_valid_matrix(matrix):
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


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day06.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

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


if __name__ == "__main__":
    main()
