from pathlib import Path
from pprint import pprint
from collections.abc import Sequence
from typing import TypeVar

from aoc2025.utils.io import read_input_lines


T = TypeVar("T")


def find_occurrences(seq: Sequence[T], symbols: Sequence[T]) -> list[int]:
    return [idx for idx, item in enumerate(seq) if item in symbols]


def validate_indices(indices: Sequence[int], length: int) -> bool:
    if not all(0 < idx < length for idx in indices):
        return False
    return True


def process_layer(inputs: list[int], layer: list[int]) -> tuple[list[int], int]:
    """Process input to a layer and return the output as well as the number of
    splits."""
    # The rule is if a number in inputs is present in layer, the output will contain
    # n-1 and n+1. Otherwise the number n is added directly to the output. Each time
    # a hit occurs, increment a counter. There are possibilites of duplicates,
    # but we don't consider them separate.

    splits: int = 0
    outputs: set[int] = set()

    for ip in inputs:
        if ip in layer:
            outputs.update([ip - 1, ip + 1])
            splits += 1
        else:
            outputs.add(ip)

    return sorted(outputs), splits


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day07.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    pprint(data)

    symbols: str = "S^"

    indices: list[list[int]] = []

    for row in data:
        idx = find_occurrences(row, symbols)
        if not validate_indices(idx, len(row)):
            raise ValueError(f"Invalid indices {idx}")
        indices.append(idx)

    # pprint(indices)

    inputs: list[int] = sorted(set(indices[0]))
    layers: list[list[int]] = indices[1:]
    splits: list[int] = []

    for layer in layers:
        output, sp = process_layer(inputs, layer)
        splits.append(sp)
        inputs = output

    print(f"Number of splits: {sum(splits)}")


if __name__ == "__main__":
    main()
