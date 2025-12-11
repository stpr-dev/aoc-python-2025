# tests/test_day08.py

import pytest
from hypothesis import given
from hypothesis import strategies as st

import aoc2025.day08 as d08


# ---------------------------------------------------------------------------
# Deterministic Tests: Point3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, q, expected",
    [
        (d08.Point3D(0, 0, 0), d08.Point3D(0, 0, 0), 0),
        (d08.Point3D(1, 2, 3), d08.Point3D(1, 2, 3), 0),
        (d08.Point3D(1, 2, 3), d08.Point3D(2, 2, 3), 1),
        (d08.Point3D(1, 2, 3), d08.Point3D(1, 5, 3), 9),
        (d08.Point3D(1, 2, 3), d08.Point3D(1, 2, 7), 16),
        (d08.Point3D(0, 0, 0), d08.Point3D(3, 4, 0), 25),
    ],
)
def test_sq_distance(p, q, expected):
    assert p.sq_distance(q) == expected


# ---------------------------------------------------------------------------
# Deterministic Tests: DSU
# ---------------------------------------------------------------------------


def test_dsu_singletons():
    n = 5
    dsu = d08.DSU(n)
    for i in range(n):
        assert dsu.find(i) == i
        assert dsu.size[i] == 1


def test_dsu_union_simple():
    dsu = d08.DSU(4)
    dsu.union(0, 1)
    assert dsu.find(0) == dsu.find(1)

    dsu.union(2, 3)
    assert dsu.find(2) == dsu.find(3)

    dsu.union(1, 2)
    r = dsu.find(0)
    assert dsu.find(3) == r
    assert dsu.size[r] == 4


# ---------------------------------------------------------------------------
# Deterministic Tests: Small Graphs for solve() and solve_part2()
# ---------------------------------------------------------------------------


def test_solve_small_manual():
    pts = [
        d08.Point3D(0, 0, 0),
        d08.Point3D(1, 0, 0),
        d08.Point3D(2, 0, 0),
    ]
    val = d08.solve(pts, k=2)
    assert val == 3


def test_solve_part2_small_manual():
    pts = [
        d08.Point3D(0, 0, 0),
        d08.Point3D(1, 0, 0),
        d08.Point3D(3, 0, 0),
    ]
    result = d08.solve_part2(pts)
    assert result == 3


# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------

coord = st.integers(min_value=-1000, max_value=1000)


@given(x=coord, y=coord, z=coord, x2=coord, y2=coord, z2=coord)
def test_sq_distance_symmetric(x, y, z, x2, y2, z2):
    p = d08.Point3D(x, y, z)
    q = d08.Point3D(x2, y2, z2)
    assert p.sq_distance(q) == q.sq_distance(p)


@given(n=st.integers(min_value=1, max_value=20), data=st.data())
def test_dsu_properties(n, data):
    """
    Random sequences of union operations must produce consistent find() behavior.
    """
    dsu = d08.DSU(n)

    ops = data.draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
            ),
            min_size=1,
            max_size=50,
        )
    )

    for a, b in ops:
        dsu.union(a, b)

    # After unions, all nodes in a connected component should share the same root
    comps = {}
    for i in range(n):
        r = dsu.find(i)
        comps.setdefault(r, []).append(i)

    for group in comps.values():
        roots = {dsu.find(x) for x in group}
        assert len(roots) == 1


@given(st.lists(st.tuples(coord, coord, coord), min_size=2, max_size=7))
def test_partial_kruskal_components_monotonic(raw_points):
    """
    After each union during Kruskal's edge-adding,
    the number of components should never increase.
    """
    pts = [d08.Point3D(x, y, z) for x, y, z in raw_points]
    n = len(pts)

    dsu = d08.DSU(n)

    edges = []
    for i in range(n):
        for j in range(i):
            edges.append((pts[i].sq_distance(pts[j]), i, j))

    edges.sort(key=lambda e: e[0])
    prefix = edges[: min(len(edges), 20)]

    comp_counts = []
    for _, i, j in prefix:
        dsu.union(i, j)
        roots = {dsu.find(x) for x in range(len(pts))}
        comp_counts.append(len(roots))

    for a, b in zip(comp_counts, comp_counts[1:], strict=False):
        assert b <= a


@given(st.lists(st.tuples(coord, coord, coord), min_size=2, max_size=7))
def test_solve_part2_correctness_small_scale(raw_points):
    """
    For small N, solve_part2 must match brute-force Kruskal.
    """
    pts = [d08.Point3D(x, y, z) for x, y, z in raw_points]
    n = len(pts)

    # Brute-force Kruskal
    edges = []
    for i in range(n):
        for j in range(i):
            edges.append((pts[i].sq_distance(pts[j]), i, j))
    edges.sort(key=lambda e: e[0])

    dsu = d08.DSU(n)
    expected = None
    for _, i, j in edges:
        new_size = dsu.union(i, j)
        if new_size == n:
            expected = pts[i].x * pts[j].x
            break

    assert expected is not None

    result = d08.solve_part2(pts)
    assert result == expected
