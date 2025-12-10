import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import TypeVar

from aoc2025.utils.io import read_input_lines

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Point3D:
    x: int
    y: int
    z: int

    def distance(self, other: "Point3D") -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))

    def __repr__(self) -> str:
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"


@dataclass(frozen=True, slots=True)
class Point3DPair:
    point1: Point3D
    point2: Point3D
    distance: float
    indices: tuple[int, int]

    def __repr__(self) -> str:
        return f"Point3DPair(point1={self.point1}, point2={self.point2}, distance={self.distance}, indices={self.indices})"


class Grouper:
    def __init__(self, points: Sequence[Point3D]):
        self.groups: list[set[int]] = []
        self.points: list[Point3D] = list(points)
        self.pairwise_distances: list[list[float]] = []
        self.flattened_distances: list[Point3DPair] = []

    def _calculate_pairwise_euclidean_distance(self) -> None:
        """Compute the pairwise Euclidean distance between two sequences."""
        self.pairwise_distances = [
            [point.distance(other) for other in self.points] for point in self.points
        ]

    def _make_lower_diagonal_infinity(self):
        for i, row in enumerate(self.pairwise_distances):
            for j in range(i, len(row)):
                row[j] = float("inf")

    def _flatten_and_sort(self):
        self.flattened_distances = [
            Point3DPair(point1, point2, distance, (i, j))
            for i, row in enumerate(self.pairwise_distances)
            for j, distance in enumerate(row)
            for point1, point2 in [(self.points[i], self.points[j])]
        ]

        self.flattened_distances.sort(key=lambda x: x.distance)

    def _get_group_index(self, point_index: int) -> int | None:
        for idx, group in enumerate(self.groups):
            if point_index in group:
                return idx
        return None

    def _point_index_in_group(self, point_index: int) -> bool:
        return self._get_group_index(point_index) is not None

    def _make_groups(self, max_num: int = -1):
        if not self.flattened_distances:
            raise ValueError("No flattened distances to form groups")

        max_iterations = len(self.flattened_distances) if max_num < 0 else max_num
        # The logic is as follows:
        # For each point pair in the sorted list, if the point pair is not in a group yet,
        # add the point pair to the group of the nearest neighbour.
        # If the point pair is in a group, add the point pair to the group of the point itself.
        for point_pair in self.flattened_distances[:max_iterations]:
            point_idx, nearest_neighbour_idx = point_pair.indices
            neighbour_group_idx = self._get_group_index(nearest_neighbour_idx)
            point_group_idx = self._get_group_index(point_idx)
            if neighbour_group_idx is not None:
                self.groups[neighbour_group_idx].add(point_idx)
            elif point_group_idx is not None:
                self.groups[point_group_idx].add(nearest_neighbour_idx)
            else:
                self.groups.append({point_idx, nearest_neighbour_idx})

            # Now there are cases where adding a new connecting would merge two
            # groups. If so, update one of them and remove the other. We can
            # determine that by looking to see if BOTH point_group_idx and
            # neighbour_group_idx are not None.
            if (
                neighbour_group_idx is not None
                and point_group_idx is not None
                and neighbour_group_idx != point_group_idx
            ):
                self.groups[neighbour_group_idx] |= self.groups[point_group_idx]
                self.groups.pop(point_group_idx)

        return self.groups

    def group(self, max_num: int = -1) -> list[set[int]]:
        self._calculate_pairwise_euclidean_distance()
        self._make_lower_diagonal_infinity()
        self._flatten_and_sort()
        return self._make_groups(max_num)


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day08.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    # Each line is comma separated list of numbers.
    data_int: list[list[int]] = [[int(x) for x in line.split(",")] for line in data]

    # Convert to point objects.
    points: list[Point3D] = [Point3D(x, y, z) for x, y, z in data_int]

    grouper = Grouper(points)
    groups = grouper.group(max_num=1000)
    pprint(groups)

    lengths: list[int] = sorted((len(group) for group in groups), reverse=True)
    pprint(lengths)

    print(f"Solution to part 1: {math.prod(lengths[:3])}")


if __name__ == "__main__":
    main()
