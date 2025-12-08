from collections import Counter

from hypothesis import given, strategies as st

from aoc2025.day07 import (
    find_occurrences,
    process_layer,
    validate_indices,
    process_layer_counts,
)


def test_find_occurrences_basic():
    assert find_occurrences("abcabc", "b") == [1, 4]
    assert find_occurrences([1, 2, 3, 2], [2]) == [1, 3]


def test_validate_indices_valid():
    assert validate_indices([1, 2, 3], 5) is True


def test_validate_indices_invalid_negative():
    assert validate_indices([-1, 2], 5) is False


def test_validate_indices_invalid_oob():
    assert validate_indices([1, 5], 5) is False  # 5 is out of range: 0..4


def test_process_layer_no_splits():
    inputs = [3, 5]
    layer = [10]
    output, splits = process_layer(inputs, layer)
    assert output == {3, 5}
    assert splits == 0


def test_process_layer_with_split():
    inputs = [3, 5]
    layer = [5]
    output, splits = process_layer(inputs, layer)
    assert splits == 1
    assert output == {3, 4, 6}


def test_process_layer_multiple_splits():
    inputs = [2, 4, 6]
    layer = [2, 6]
    output, splits = process_layer(inputs, layer)
    # 2 hits → outputs (1,3)
    # 6 hits → outputs (5,7)
    # 4 unaffected
    assert splits == 2
    assert output == {1, 3, 4, 5, 7}


def test_process_layer_counts_no_split():
    counts = Counter({4: 1})
    layer = {10}  # no hit
    next_counts = process_layer_counts(counts, layer)

    assert next_counts == Counter({4: 1})


def test_process_layer_counts_single_split():
    counts = Counter({4: 1})
    layer = {4}
    next_counts = process_layer_counts(counts, layer)

    # 4 splits → 3 and 5
    assert next_counts == Counter({3: 1, 5: 1})


def test_process_layer_counts_multiple_positions():
    counts = Counter({2: 1, 4: 2})
    layer = {4}

    next_counts = process_layer_counts(counts, layer)

    # 2 (count=1) does not split
    # 4 (count=2) splits into 3 and 5 twice
    expected = Counter(
        {
            2: 1,  # unchanged
            3: 2,  # two splits from 4
            5: 2,
        }
    )

    assert next_counts == expected


def test_process_layer_counts_merge_paths():
    counts = Counter({5: 3})
    layer = {5}

    next_counts = process_layer_counts(counts, layer)

    # three timelines at 5 split into three at 4 and three at 6
    assert next_counts == Counter({4: 3, 6: 3})


@given(
    seq=st.lists(st.integers(min_value=0, max_value=10), min_size=0, max_size=50),
    symbol=st.integers(min_value=0, max_value=10),
)
def test_find_occurrences_hypothesis(seq, symbol):
    result = find_occurrences(seq, [symbol])
    # All indices returned must satisfy: seq[i] == symbol
    assert all(seq[i] == symbol for i in result)
    # Order must be increasing
    assert result == sorted(result)


@given(
    inputs=st.lists(
        st.integers(min_value=1, max_value=50), min_size=1, max_size=50, unique=True
    ),
    layer=st.lists(
        st.integers(min_value=1, max_value=50), min_size=0, max_size=50, unique=True
    ),
)
def test_process_layer_properties(inputs, layer):
    output, splits = process_layer(inputs, layer)

    # Property 1: output has no duplicates
    assert len(output) == len(set(output))

    # Property 2: number of splits = number of unique hits in inputs
    layer_set = set(layer)
    expected_splits = sum(1 for x in set(inputs) if x in layer_set)
    assert splits == expected_splits

    # Property 3: each input either stays or expands to ±1
    out_set = set(output)
    for ip in set(inputs):
        if ip in layer_set:
            assert {ip - 1, ip + 1}.issubset(out_set)
        else:
            assert ip in out_set


@given(
    # positions are ≥1 as in your invariant
    counts=st.dictionaries(
        keys=st.integers(min_value=1, max_value=50),
        values=st.integers(min_value=1, max_value=10),  # nonzero counts
        min_size=1,
        max_size=20,
    ),
    layer=st.sets(
        st.integers(min_value=1, max_value=50),
        min_size=0,
        max_size=20,
    ),
)
def test_process_layer_counts_properties(counts, layer):
    # convert dict to Counter
    counts = Counter(counts)
    next_counts = process_layer_counts(counts, layer)

    # Property 1: all resulting counts are positive integers
    assert all(v > 0 for v in next_counts.values())

    # Property 2: for each position p:
    #   If p ∉ layer, its weight moves to same pos.
    #   If p ∈ layer, its weight moves to p-1 and p+1.
    expected = Counter()
    for pos, c in counts.items():
        if pos in layer:
            expected[pos - 1] += c
            expected[pos + 1] += c
        else:
            expected[pos] += c

    assert next_counts == expected

    # Property 3: total timelines conserved except for splitting logic
    # i.e., total output = sum of:
    #        (for each p ∉ layer: c)
    #        (for each p ∈ layer: 2*c)

    total_out = sum(next_counts.values())

    expected_out = 0
    for pos, c in counts.items():
        if pos in layer:
            expected_out += 2 * c
        else:
            expected_out += c

    assert total_out == expected_out
