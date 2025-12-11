# tests/test_day09.py

import itertools
import pytest
from hypothesis import given, strategies as st

import aoc2025.day09 as d09


# ---------------------------------------------------------------------------
# Parametrized deterministic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "points, expected_area",
    [
        # Two points — simplest non-trivial case
        ([d09.Point2D(0, 0), d09.Point2D(3, 4)], (3 + 1) * (4 + 1)),
        # Square shape
        ([d09.Point2D(1, 1), d09.Point2D(4, 4)], (4 - 1 + 1) * (4 - 1 + 1)),
        # All points collinear horizontally -> area = 0
        ([d09.Point2D(x, 5) for x in range(3)], (2 + 1) * (0 + 1)),
        # All points collinear vertically -> area = 0
        ([d09.Point2D(7, y) for y in range(3)], (2 + 1) * (0 + 1)),
        # Given example from prompt (expected 50 area between (2,5) and (11,1))
        (
            [
                d09.Point2D(7, 1),
                d09.Point2D(11, 1),
                d09.Point2D(11, 7),
                d09.Point2D(9, 7),
                d09.Point2D(9, 5),
                d09.Point2D(2, 5),
                d09.Point2D(2, 3),
                d09.Point2D(7, 3),
            ],
            50,
        ),
    ],
)
def test_solve_area(points, expected_area):
    """
    Deterministic verification of correct area computation for known configurations.
    """
    p1, p2, area = d09.solve(points)
    assert area == expected_area
    assert isinstance(p1, d09.Point2D)
    assert isinstance(p2, d09.Point2D)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


@st.composite
def points_strategy(draw, min_size=2, max_size=15):
    """
    Strategy producing a list of distinct Point2D objects within bounded coordinates.
    """
    xs = st.integers(min_value=-1000, max_value=1000)
    ys = st.integers(min_value=-1000, max_value=1000)
    coords = draw(
        st.lists(st.tuples(xs, ys), min_size=min_size, max_size=max_size, unique=True)
    )
    return [d09.Point2D(x, y) for x, y in coords]


@given(points_strategy())
def test_solve_returns_valid_pair(points):
    """
    Property: The returned pair must be members of the input and area non-negative.
    """
    p1, p2, area = d09.solve(points)
    assert p1 in points
    assert p2 in points
    assert area >= 0


@given(points_strategy())
def test_solve_matches_reference(points):
    """
    Property: The computed area must equal the maximum area among all point pairs.
    """
    p1, p2, area = d09.solve(points)
    ref_area = max(
        (abs(a.x - b.x) + 1) * (abs(a.y - b.y) + 1)
        for a, b in itertools.combinations(points, 2)
    )
    assert area == ref_area


@given(points_strategy())
def test_symmetry_of_area(points):
    """
    Property: Swapping the two points does not affect area value.
    """
    for a, b in itertools.combinations(points, 2):
        area1 = (abs(a.x - b.x) + 1) * (abs(a.y - b.y) + 1)
        area2 = (abs(b.x - a.x) + 1) * (abs(b.y - a.y) + 1)
        assert area1 == area2
