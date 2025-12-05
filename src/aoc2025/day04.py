from pathlib import Path
from pprint import pprint
from typing import TypeVar, Protocol

from aoc2025.utils.io import read_input_lines
# from aoc2025.utils.benchmark import time_callable


class Numeric(Protocol):
    def __add__(self, other: "Numeric", /) -> "Numeric": ...
    def __sub__(self, other: "Numeric", /) -> "Numeric": ...
    def __mul__(self, other: "Numeric", /) -> "Numeric": ...
    def __truediv__(self, other: "Numeric", /) -> "Numeric": ...
    def __neg__(self) -> "Numeric": ...


T = TypeVar("T", bound=Numeric)


class Conv2d[T]:
    """A simple conv2d implementation in Python."""

    def __init__(self, kernel: list[list[T]]) -> None:
        if not Conv2d.is_valid_matrix(kernel):
            raise ValueError("Kernel must be a valid matrix")
        self.kernel: list[list[T]] = kernel

    @staticmethod
    def is_valid_matrix(matrix: list[list[T]]) -> bool:
        """Check if a matrix is valid."""
        if not matrix or not all(len(row) == len(matrix[0]) for row in matrix):
            return False
        return True

    @staticmethod
    def generate_constant_kernel(
        size: tuple[int, int], constant: T = 1
    ) -> list[list[T]]:
        """Generate a kernel of constant value of the given size."""
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Size must be positive")
        return [[constant for _ in range(size[1])] for _ in range(size[0])]

    @staticmethod
    def hadamard_product(
        kernel: list[list[T]],
        data: list[list[T]],
    ) -> list[list[T]]:
        """Compute the Hadamard product of a kernel and a data matrix."""
        if not Conv2d.is_valid_matrix(kernel) or not Conv2d.is_valid_matrix(data):
            raise ValueError("Kernel and data must be valid matrices")
        return [
            [kernel[i][j] * data[i][j] for j, _ in enumerate(row)]
            for i, row in enumerate(kernel)
        ]

    @staticmethod
    def reduce_matrix(matrix: list[list[T]]) -> T:
        """Reduce a matrix to a single value by summing all elements."""
        if not Conv2d.is_valid_matrix(matrix):
            raise ValueError("Matrix must be valid")
        first: T = matrix[0][0]
        zero: T = first - first
        return sum((sum(row, start=zero) for row in matrix), start=zero)

    @staticmethod
    def pad_matrix(matrix: list[list[T]], padding: int = 1) -> list[list[T]]:
        """Pad a matrix with zeros on all sides."""
        zero = matrix[0][0] - matrix[0][0]

        zero_row = [zero] * (2 * padding + len(matrix[0]))

        out = [zero_row.copy()] * padding

        for row in matrix:
            out.append([zero] * padding + row + [zero] * padding)

        out += [zero_row.copy()] * padding

        return out


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day04example.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    # Numbers should be a string of digits.
    pprint(data)


if __name__ == "__main__":
    main()
