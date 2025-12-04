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


def test_get_largest_sub_list_basic():
    assert d03.get_largest_sub_list(list("987654")) == list("98765")
    assert d03.get_largest_sub_list(list("123456")) == list("23456")
    assert d03.get_largest_sub_list(list("132456")) == list("32456")


def test_part2_examples():
    data = [
        "987654321111111",
        "811111111111119",
        "234234234234278",
        "818181911112111",
    ]

    out = d03.part2_get_largest_k_digit(data, k=12)

    assert out == [
        987654321111,
        811111111119,
        434234234278,
        888911112111,
    ]
