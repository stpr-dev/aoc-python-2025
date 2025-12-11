import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


# -------------------------
# Data Structures
# -------------------------


@dataclass(frozen=True, slots=True)
class Point3D:
    x: int
    y: int
    z: int

    def sq_distance(self, other: "Point3D") -> int:
        dx: int = self.x - other.x
        dy: int = self.y - other.y
        dz: int = self.z - other.z
        return dx * dx + dy * dy + dz * dz


class DSU:
    def __init__(self, n: int) -> None:
        self.parent: list[int] = list(range(n))
        self.size: list[int] = [1] * n

    def find(self, x: int) -> int:
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra: int = self.find(a)
        rb: int = self.find(b)
        if ra == rb:
            return self.size[ra]
        # Union by size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

        return self.size[ra]


# -------------------------
# Solver
# -------------------------


def solve(points: Sequence[Point3D], k: int) -> int:
    n: int = len(points)

    # Generate all pairwise edges as (sq_distance, i, j)
    edges: list[tuple[int, int, int]] = []
    for i in range(n):
        pi = points[i]
        for j in range(i):
            pj = points[j]
            d2 = pi.sq_distance(pj)
            edges.append((d2, i, j))

    m = len(edges)

    # Get k smallest edges using heap if k << m
    if k < m:
        edges_k = heapq.nsmallest(k, edges, key=lambda e: e[0])
    else:
        edges.sort(key=lambda e: e[0])
        edges_k = edges[:k]

    # Initialize DSU and add edges
    dsu = DSU(n)
    for _, i, j in edges_k:
        _ = dsu.union(i, j)

    # Compute sizes of all connected components
    comp: dict[int, int] = {}
    for v in range(n):
        r = dsu.find(v)
        comp[r] = comp.get(r, 0) + 1

    # Sort sizes descending
    sizes = sorted(comp.values(), reverse=True)

    # Multiply top 3 component sizes
    return math.prod(sizes[:3])


def solve_part2(points: Sequence[Point3D]) -> int:
    n: int = len(points)

    # Generate all pairwise edges as (sq_distance, i, j)
    edges: list[tuple[int, int, int]] = []
    for i in range(n):
        pi = points[i]
        for j in range(i):
            pj = points[j]
            d2 = pi.sq_distance(pj)
            edges.append((d2, i, j))

    # Get k smallest edges using heap if k << m
    edges.sort(key=lambda e: e[0])

    # Initialize DSU and add edges
    dsu = DSU(n)
    for _, i, j in edges:
        new_size = dsu.union(i, j)

        if new_size == n:
            return points[i].x * points[j].x

    raise ValueError("No solution found!")


# -------------------------
# I/O + Main
# -------------------------


def main() -> None:
    data_path = Path(__file__).parent.parent.parent / "data" / "2025" / "day08.txt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = data_path.read_text().strip().splitlines()
    coords = [[int(x) for x in line.split(",")] for line in raw]
    points = [Point3D(x, y, z) for x, y, z in coords]

    result = solve(points, k=1000)
    print(f"Solution to part 1: {result}")

    result = solve_part2(points)
    print(f"Solution to part 2: {result}")


if __name__ == "__main__":
    main()
