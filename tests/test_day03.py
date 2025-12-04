import aoc2025.day03 as d03


def test_part1_examples():
    data = [
        "987654321111111",
        "811111111111119",
        "234234234234278",
        "818181911112111",
    ]

    out = d03.part1_get_largest_2_digit(data)

    assert out == [98, 89, 78, 92]
