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


def hadamard_product(
    kernel: list[list[T]],
    data: list[list[T]],
) -> list[list[T]]:
    """Compute the Hadamard product of a kernel and a data matrix."""
    return [
        [kernel[i][j] * data[i][j] for j, _ in enumerate(row)]
        for i, row in enumerate(kernel)
    ]


def reduce_matrix(matrix: list[list[T]]) -> T:
    """Reduce a matrix to a single value by summing all elements."""
    first: T = matrix[0][0]
    zero: T = first - first
    return sum((sum(row, start=zero) for row in matrix), start=zero)


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
