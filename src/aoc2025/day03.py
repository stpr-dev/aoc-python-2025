from pathlib import Path
from pprint import pprint
from typing import Protocol, TypeVar

from utils.io import read_input_lines


def part1_get_largest_2_digit(data: list[str]) -> list[int]:
    """Return a list of the largest two digit numbers in each string in the input data."""
    largest_2_digit_numbers: list[int] = []

    # The way the logic works, we are looking for picking two digits such that they
    # form the largest number. The twist being that when you pick a digit, you can
    # pick only subsequent digits for making the second digit. The logic here is
    # pretty straightforward:
    # - Find the largest digit in str[:-1] and it's index.
    # - Find the largest digit in str[index+1:].
    # - Concatenate them and add to the list.

    for line in data:
        digits: list[str] = [char for char in line if char.isdigit()]
        first_digit_index: int = digits.index(max(digits[:-1]))
        second_digit_index: int = digits.index(max(digits[first_digit_index + 1 :]))
        largest_2_digit_numbers.append(
            int(digits[first_digit_index] + digits[second_digit_index])
        )

    return largest_2_digit_numbers


T = TypeVar("T", bound="Comparable")


class Comparable(Protocol):
    def __lt__(self: T, other: T) -> bool: ...


def get_largest_sub_list(lst: list[T]) -> list[T]:
    """Return the largest sublist in the input list formed by removing exactly one
    element."""

    if len(lst) < 2:
        raise ValueError(
            f"Got {len(lst)=} elements in the input list, expected at least 2."
        )

    for idx in range(len(lst) - 1):
        if lst[idx] < lst[idx + 1]:
            return lst[:idx] + lst[idx + 1 :]

    return lst[:-1]


def part2_get_largest_k_digit(data: list[str], k: int = 12) -> list[int]:
    """Return a list of the largest k digit numbers in each string in the input data."""
    largest_k_digit_numbers: list[int] = []

    # This is an extension of part 1, but now we need to pick the largest k digits.
    # One optimal way to do this is to find the argsort of the list, pop the first
    # n-k elements, then use the rest of the list to find the largest k digits.

    for line in data:
        digits: list[str] = [char for char in line if char.isdigit()]

        if k > len(digits):
            raise ValueError(f"Got {k=} but only {len(digits)=} digits in the input.")

        while len(digits) > k:
            digits = get_largest_sub_list(digits)

        largest_k_digit_numbers.append(int("".join(digits)))

    return largest_k_digit_numbers


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day03.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    # Numbers should be a string of digits.
    pprint(data)

    largest_2_digit_numbers: list[int] = part1_get_largest_2_digit(data)
    pprint(largest_2_digit_numbers)

    print(f"Solution to part 1 is: {sum(largest_2_digit_numbers)}.")

    largest_k_digit_numbers: list[int] = part2_get_largest_k_digit(data, k=12)
    pprint(largest_k_digit_numbers)

    print(f"Solution to part 2 is: {sum(largest_k_digit_numbers)}.")


if __name__ == "__main__":
    main()
