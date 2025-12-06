# tests/test_day05.py

import pytest
from hypothesis import given
import hypothesis.strategies as st

import aoc2025.day05 as d05


# ---------------------------------------------------------------------------
# Parametrized deterministic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ranges, value, expected",
    [
        ([[3, 6]], 3, True),  # inclusive lower
        ([[3, 6]], 5, True),  # interior
        ([[3, 6]], 6, False),  # upper not included (Python range semantics)
        ([[3, 6], [10, 15]], 12, True),
        ([[3, 6], [10, 15]], 9, False),
        ([[0, 1]], 0, True),
        ([[0, 1]], 1, False),
        ([[10, 11], [11, 12]], 11, True),  # two independent ranges touching
    ],
)
def test_contains(ranges, value, expected):
    mr = d05.MultiRange(ranges)
    assert (value in mr) is expected


@pytest.mark.parametrize(
    "ranges, expected_values",
    [
        ([[0, 3]], [0, 1, 2]),
        ([[3, 6], [10, 13]], [3, 4, 5, 10, 11, 12]),
        ([[5, 6], [6, 7]], [5, 6]),  # two singletons
    ],
)
def test_iteration(ranges, expected_values):
    mr = d05.MultiRange(ranges)
    assert list(mr) == expected_values


@pytest.mark.parametrize(
    "ranges, expected_len",
    [
        ([[0, 3]], 3),
        ([[3, 6], [10, 13]], 3 + 3),
        ([[5, 6], [6, 10]], 1 + 4),
    ],
)
def test_len(ranges, expected_len):
    mr = d05.MultiRange(ranges)
    assert len(mr) == expected_len


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ([], []),
        # Single range
        ([[3, 6]], [[3, 6]]),
        # Already sorted, non-overlapping
        ([[0, 3], [5, 10]], [[0, 3], [5, 10]]),
        # Overlapping
        ([[3, 6], [5, 12]], [[3, 12]]),
        # Contained intervals
        ([[3, 10], [4, 5], [6, 9]], [[3, 10]]),
        # Adjacent intervals: [3,6) and [6,10) → merge
        ([[3, 6], [6, 10]], [[3, 10]]),
        # Unsorted input
        ([[10, 20], [0, 5], [4, 12]], [[0, 20]]),
        # Multiple disjoint + merge cluster
        ([[1, 3], [10, 15], [2, 4], [20, 22]], [[1, 4], [10, 15], [20, 22]]),
    ],
)
def test_optimize_ranges_deterministic(ranges, expected):
    result = d05.MultiRange.optimize_ranges(ranges)
    assert result == expected


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


def range_strategy():
    """
    Strategy producing valid Python-range-style interval specs [start, stop)
    where stop > start.
    """
    start = st.integers(min_value=-10_000, max_value=10_000)
    length = st.integers(min_value=1, max_value=5_000)
    return st.builds(lambda strt, leng: [strt, strt + leng], start, length)


def multi_range_strategy():
    """
    Strategy producing a list of ranges.
    """
    return st.lists(range_strategy(), min_size=0, max_size=20)


@st.composite
def membership_test_strategy(draw):
    ranges = draw(multi_range_strategy())
    value = draw(st.integers(min_value=-20_000, max_value=20_000))
    return ranges, value


@given(data=membership_test_strategy())
def test_contains_matches_reference(data):
    """
    Property: MultiRange containment must match union-of-ranges semantics.
    """
    ranges, value = data

    mr = d05.MultiRange(ranges)

    # Reference implementation using explicit check.
    # Because ranges are [start, stop) intervals:
    ref = any(start <= value < stop for start, stop in ranges)

    assert (value in mr) == ref


@given(ranges=multi_range_strategy())
def test_iteration_matches_reference(ranges):
    """
    Property: Iteration should match flattening the underlying ranges.
    """
    mr = d05.MultiRange(ranges)
    ref = [x for start, stop in ranges for x in range(start, stop)]
    assert list(mr) == ref


@given(ranges=multi_range_strategy())
def test_len_matches_reference(ranges):
    """
    Property: len(multi_range) should equal the total span of all ranges.
    """
    mr = d05.MultiRange(ranges)
    ref = sum(stop - start for start, stop in ranges)
    assert len(mr) == ref


def interval_strategy():
    """
    Half-open intervals [start, stop) with stop > start.
    """
    start = st.integers(min_value=-10_000, max_value=10_000)
    length = st.integers(min_value=1, max_value=5_000)
    return st.builds(lambda strt, leng: [strt, strt + leng], start, length)


def intervals_strategy():
    return st.lists(interval_strategy(), min_size=0, max_size=50)


@given(ranges=intervals_strategy())
def test_optimize_ranges_sorted(ranges):
    merged = d05.MultiRange.optimize_ranges(ranges)
    starts = [s for s, _ in merged]
    assert starts == sorted(starts)


@given(ranges=intervals_strategy())
def test_optimize_ranges_no_overlaps_or_adjacency(ranges):
    merged = d05.MultiRange.optimize_ranges(ranges)
    for (_, e1), (s2, _) in zip(merged, merged[1:], strict=False):
        assert s2 > e1  # strict: no touching or overlapping intervals


def in_any(value: int, intervals: list[list[int]]) -> bool:
    return any(start <= value < stop for start, stop in intervals)


@given(ranges=intervals_strategy(), point=st.integers(-20_000, 20_000))
def test_optimize_ranges_preserves_membership(ranges, point):
    merged = d05.MultiRange.optimize_ranges(ranges)
    ref_orig = in_any(point, ranges)
    ref_merged = in_any(point, merged)
    assert ref_orig == ref_merged
