import math
from collections.abc import Sequence
from pathlib import Path
from pprint import pprint
from typing import TypeVar

from aoc2025.utils.io import read_input_lines

T = TypeVar("T")


class Grouper[T]:
    def __init__(self, points: Sequence[Sequence[T]]):
        self.groups: list[set[int]] = []
        self.points: Sequence[Sequence[T]] = points
        self.pairwise_distances: list[list[float]] = []
        self.nearest_neighbours: list[int] = []

    def _calculate_pairwise_euclidean_distance(self) -> None:
        """Compute the pairwise Euclidean distance between two sequences."""
        self.pairwise_distances = [
            [math.dist(x, y) for y in self.points] for x in self.points
        ]

    def _make_diagonal_infinity(self) -> None:
        """Adjust the diagonal entries of a matrix to infinity."""
        if not self.pairwise_distances:
            raise ValueError("No pairwise distances to adjust")
        self.pairwise_distances = [
            [float("inf") if i == j else x for j, x in enumerate(row)]
            for i, row in enumerate(self.pairwise_distances)
        ]

    def _find_nearest_neighbour(self) -> None:
        """Find the index of the closest neighbour in each row of data."""
        if not self.pairwise_distances:
            raise ValueError("No pairwise distances to compute nearest neighbours")
        self.nearest_neighbours = [
            min(range(len(row)), key=lambda i: row[i])
            for row in self.pairwise_distances
        ]

    def _get_group_index(self, point_index: int) -> int | None:
        for idx, group in enumerate(self.groups):
            if point_index in group:
                return idx
        return None

    def _point_index_in_group(self, point_index: int) -> bool:
        return self._get_group_index(point_index) is not None

    def _make_groups(self):
        if not self.nearest_neighbours:
            raise ValueError("No nearest neighbours to form groups")

        # The logic is as follows:
        # For each point in the points sequence, see if the point index is in the
        # groups. If it is, then nothing needs to be done as it's already grouped.from
        # If the above is false, check to see if the nearest neighbour is in the groups.
        # If it is, then add the point to the group. If it is not, then create a new
        # group.
        for point_idx, nearest_neighbour_idx in enumerate(self.nearest_neighbours):
            if not self._point_index_in_group(point_idx):
                neighbour_group_idx = self._get_group_index(nearest_neighbour_idx)
                if neighbour_group_idx is not None:
                    self.groups[neighbour_group_idx].add(point_idx)
                else:
                    self.groups.append({point_idx, nearest_neighbour_idx})

    def group(self) -> list[set[int]]:
        self._calculate_pairwise_euclidean_distance()
        self._make_diagonal_infinity()
        self._find_nearest_neighbour()
        self._make_groups()
        return self.groups


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day08example.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    # Each line is comma separated list of numbers.
    data_int: list[list[int]] = [[int(x) for x in line.split(",")] for line in data]

    grouper = Grouper[int](data_int)
    groups = grouper.group()
    pprint(groups)

    lengths: list[int] = sorted((len(group) for group in groups), reverse=True)
    pprint(lengths)

    print(f"Solution to part 1: {math.prod(lengths[:3])}")


if __name__ == "__main__":
    main()
