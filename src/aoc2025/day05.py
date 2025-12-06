from pathlib import Path
from pprint import pprint

from aoc2025.utils.io import read_input_lines


from collections.abc import Iterator


class MultiRange:
    """A class representing a set of ranges."""

    def __init__(self, ranges: list[list[int]]) -> None:
        self.ranges: list[list[int]] = ranges
        self.range_objects: list[range] = [range(*args) for args in ranges]

    def __contains__(self, value: int) -> bool:
        return any(value in r for r in self.range_objects)

    def __iter__(self) -> Iterator[int]:
        for r in self.range_objects:
            yield from r

    def __len__(self) -> int:
        return sum(len(r) for r in self.range_objects)

    def __repr__(self) -> str:
        return f"MultiRange({self.ranges!r})"

    @staticmethod
    def optimize_ranges(ranges: list[list[int]]) -> list[list[int]]:
        """
        Merge overlapping or adjacent half-open intervals [start, stop).

        Input:  [[s1, e1], [s2, e2], ...]
        Output: merged, sorted list of intervals.
        """
        if not ranges:
            return []

        # Sort by starting coordinate
        ranges_sorted = sorted(ranges, key=lambda r: r[0])

        merged: list[list[int]] = []
        cur_start, cur_end = ranges_sorted[0]

        for start, end in ranges_sorted[1:]:
            # Overlap or adjacency: start <= cur_end
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append([cur_start, cur_end])
                cur_start, cur_end = start, end

        merged.append([cur_start, cur_end])
        return merged


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day05.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    pprint(data)

    # The data file contains both ranges and product ids, separate them.
    sep_index: int = data.index("")

    ranges: list[str] = data[:sep_index]
    product_ids: list[str] = data[sep_index + 1 :]

    # ranges is a string encoding of the form f"{lower}-{upper}" for each range.
    ranges: list[list[int]] = [[int(x) for x in r.split("-")] for r in ranges]
    product_ids: set[int] = set(map(int, product_ids))

    pprint(ranges)
    pprint(product_ids)

    # Adjust the ranges to be inclusive of both ends.
    ranges: list[list[int]] = [[r[0], r[1] + 1] for r in ranges]

    ranges: list[list[int]] = MultiRange.optimize_ranges(ranges)

    multi_range: MultiRange = MultiRange(ranges)

    num_invalid: int = sum(1 for pid in product_ids if pid in multi_range)

    print(f"Solution to part 1: {num_invalid}")

    print(f"Solution to part 2: {len(multi_range)}")


if __name__ == "__main__":
    main()
