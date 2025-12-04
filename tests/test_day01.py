import aoc2025.day01 as d01


def test_data_to_signed_int():
    data = "L68 L30 R48 L5 R60 L55 L1 L99 R14 L82".split()

    assert d01.data_to_signed_int(data) == [
        -68,
        -30,
        +48,
        -5,
        60,
        -55,
        -1,
        -99,
        +14,
        -82,
    ]


def test_signed_int_to_cumulative_index():
    data = [-68, -30, +48, -5, 60, -55, -1, -99, +14, -82]
    out = d01.signed_int_to_cumulative_index(data, start=50)
    assert out == [50, -18, -48, 0, -5, 55, 0, -1, -100, -86, -168]
    assert out[-1] % 100 == 32


def test_count_multiples_in_range():
    cumulative_indices = [
        -10,
        10,  # (0,)
        210,  # (100, 200)
        220,  # (,)
        -110,  # (200, 100, 0, -100)
        -220,  # (-200,)
    ]

    assert d01.count_multiples_in_range(cumulative_indices, value=100) == (
        1 + 2 + 0 + 4 + 1
    )


def test_count_multiples_in_range_optimised():
    cumulative_indices = [
        -10,
        10,  # (0,)
        210,  # (100, 200)
        220,  # (,)
        -110,  # (200, 100, 0, -100)
        -220,  # (-200,)
    ]

    assert d01.count_multiples_in_range_optimised(cumulative_indices, value=100) == (
        1 + 2 + 0 + 4 + 1
    )


def test_count_multiples_in_range_optimised_v2():
    cumulative_indices = [
        -10,
        10,  # (0,)
        210,  # (100, 200)
        220,  # (,)
        -110,  # (200, 100, 0, -100)
        -220,  # (-200,)
    ]

    assert d01.count_multiples_in_range_optimised_v2(cumulative_indices, value=100) == (
        1 + 2 + 0 + 4 + 1
    )
