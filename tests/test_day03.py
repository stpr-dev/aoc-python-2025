import aoc2025.day03 as d03
import pytest


@pytest.mark.parametrize(
    "num,expected",
    [
        (["987654321111111"], [98]),
        (["811111111111119"], [89]),
        (["234234234234278"], [78]),
        (["818181911112111"], [92]),
    ],
)
def test_part1_examples(num, expected):
    out = d03.part1_get_largest_2_digit(num)

    assert out == expected


@pytest.mark.parametrize(
    "input_list,expected",
    [
        (list("987654"), list("98765")),
        (list("123456"), list("23456")),
        (list("132456"), list("32456")),
    ],
)
def test_get_largest_sub_list_basic(input_list, expected):
    assert d03.get_largest_sub_list(input_list) == expected


@pytest.mark.parametrize(
    "num,expected",
    [
        (["987654321111111"], [987654321111]),
        (["811111111111119"], [811111111119]),
        (["234234234234278"], [434234234278]),
        (["818181911112111"], [888911112111]),
    ],
)
def test_part2_get_largest_k_digit(num, expected):
    out = d03.part2_get_largest_k_digit(num, k=12)

    assert out == expected


@pytest.mark.parametrize(
    "num,expected",
    [
        (["987654321111111"], [987654321111]),
        (["811111111111119"], [811111111119]),
        (["234234234234278"], [434234234278]),
        (["818181911112111"], [888911112111]),
    ],
)
def test_part2_get_largest_k_digit_optimised(num, expected):
    out = d03.part2_get_largest_k_digit_optimised(num, k=12)

    assert out == expected
