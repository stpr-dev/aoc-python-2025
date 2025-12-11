from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations


# -------------------------
# Data Structures
# -------------------------


@dataclass(frozen=True, slots=True)
class Point2D:
    x: int
    y: int

    def sq_distance(self, other: "Point2D") -> int:
        dx: int = self.x - other.x
        dy: int = self.y - other.y
        return dx * dx + dy * dy

    def manhattan_distance(self, other: "Point2D") -> int:
        dx: int = self.x - other.x
        dy: int = self.y - other.y
        return abs(dx) + abs(dy)

    def __repr__(self) -> str:
        return f"Point2D(x={self.x}, y={self.y})"


def solve(points: Sequence[Point2D]) -> tuple[Point2D, Point2D, int]:
    if len(points) < 2:
        raise ValueError("At least two points are required")

    max_area: int = -1
    best_pair: tuple[Point2D, Point2D] = (points[0], points[1])

    for p1, p2 in combinations(points, 2):
        dx: int = abs(p1.x - p2.x) + 1
        dy: int = abs(p1.y - p2.y) + 1
        area: int = dx * dy
        if area > max_area:
            max_area = area
            best_pair = (p1, p2)

    return best_pair[0], best_pair[1], max_area


# -------------------------
# I/O + Main
# -------------------------


def main() -> None:
    data_path = Path(__file__).parent.parent.parent / "data" / "2025" / "day09.txt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = data_path.read_text().strip().splitlines()
    coords = [[int(x) for x in line.split(",")] for line in raw]
    points = [Point2D(x, y) for x, y in coords]

    p1, p2, area = solve(points)
    print(f"Best pair of points: {p1}, {p2}")
    print(f"Area of the best pair: {area}")


if __name__ == "__main__":
    main()
