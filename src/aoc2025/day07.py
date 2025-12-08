from pathlib import Path
from pprint import pprint
from collections.abc import Sequence, Iterable, Mapping
from collections import Counter
from typing import TypeVar

from aoc2025.utils.io import read_input_lines


T = TypeVar("T")


def find_occurrences(seq: Sequence[T], symbols: Sequence[T]) -> list[int]:
    return [idx for idx, item in enumerate(seq) if item in symbols]


def validate_indices(indices: Sequence[int], length: int) -> bool:
    if not all(0 < idx < length for idx in indices):
        return False
    return True


def process_layer(inputs: Iterable[int], layer: Iterable[int]) -> tuple[set[int], int]:
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

    return set(outputs), splits


def process_layer_counts(
    counts: Mapping[int, int],
    layer: set[int],
) -> Counter[int]:
    """Propagate timeline counts through one layer."""
    next_counts: Counter[int] = Counter()
    for pos, c in counts.items():
        if pos in layer:
            next_counts[pos - 1] += c
            next_counts[pos + 1] += c
        else:
            next_counts[pos] += c
    return next_counts


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day07.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    pprint(data)

    symbols: str = "S^"

    indices: list[set[int]] = []

    for row in data:
        idx = find_occurrences(row, symbols)
        if not validate_indices(idx, len(row)):
            raise ValueError(f"Invalid indices {idx}")
        indices.append(set(idx))

    # pprint(indices)

    inputs: set[int] = set(indices[0])
    layers: list[set[int]] = indices[1:]
    splits: list[int] = []

    for layer in layers:
        output, sp = process_layer(inputs, layer)
        splits.append(sp)
        inputs = output

    # print(f"Splits: {splits}")
    print(f"Number of splits: {sum(splits)}")

    # initial timelines: 1 timeline at each starting S (should normally be exactly one)
    counts = Counter()
    for start_pos in set(indices[0]):  # in case there are multiple S
        counts[start_pos] += 1

    for layer in layers:
        counts = process_layer_counts(counts, layer)

    total_timelines = sum(counts.values())
    print(f"Number of timelines: {total_timelines}")


if __name__ == "__main__":
    main()
