from pathlib import Path
from pprint import pprint
from typing import Protocol, TypeVar

from utils.io import read_input_lines
from utils.benchmark import time_callable


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


def part1_get_largest_2_digit_optimised(data: list[str]) -> list[int]:
    """Return a list of the largest two digit numbers in each string in the input data."""
    largest_2_digit_numbers: list[int] = []

    def largest_2_digit(str_digits: str) -> int:
        best_first: str = str_digits[0]
        best_second: str = "0"
        for idx, digit in enumerate(str_digits):
            if idx == 0:
                continue
            if idx < len(str_digits) - 1:
                if digit > best_first:
                    best_first = digit
                    best_second = str_digits[idx + 1]
                elif digit > best_second:
                    best_second = digit
            else:
                best_second = max(best_second, str_digits[-1])

        return int(best_first + best_second)

    for line in data:
        largest_2_digit_numbers.append(largest_2_digit(line.strip()))

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


def max_k_digits(digits: list[T], k: int) -> list[T]:
    drop = len(digits) - k
    stack: list[T] = []

    for d in digits:
        while drop > 0 and stack and stack[-1] < d:
            stack.pop()
            drop -= 1
        stack.append(d)

    return stack[:k]


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


def part2_get_largest_k_digit_optimised(data: list[str], k: int = 12) -> list[int]:
    """Return a list of the largest k digit numbers in each string in the input data."""
    largest_k_digit_numbers: list[int] = []

    # This is an extension of part 1, but now we need to pick the largest k digits.
    # One optimal way to do this is to find the argsort of the list, pop the first
    # n-k elements, then use the rest of the list to find the largest k digits.

    for line in data:
        digits: list[str] = [char for char in line if char.isdigit()]

        if k > len(digits):
            raise ValueError(f"Got {k=} but only {len(digits)=} digits in the input.")

        digits = max_k_digits(digits, k=k)

        largest_k_digit_numbers.append(int("".join(digits)))

    return largest_k_digit_numbers


def benchmark(
    data: list[str],
    *,
    k: int = 12,
    number: int = 10,
) -> dict[str, float]:
    """
    Benchmark the three target functions.
    Assumes the functions are already imported and available in the namespace.
    """

    timings: dict[str, float] = {
        "part1": time_callable(
            part1_get_largest_2_digit,
            data,
            number=number,
        ),
        "part1_optimised": time_callable(
            part1_get_largest_2_digit_optimised,
            data,
            number=number,
        ),
        "part2": time_callable(part2_get_largest_k_digit, data, k, number=number),
        "part2_optimised": time_callable(
            part2_get_largest_k_digit_optimised, data, k, number=number
        ),
    }

    return timings


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

    largest_2_digit_numbers_opt = part1_get_largest_2_digit_optimised(data)

    print(f"Solution to part 1 (optimised) is: {sum(largest_2_digit_numbers_opt)}.")
    if largest_2_digit_numbers != largest_2_digit_numbers_opt:
        raise ValueError("Optimised solution is incorrect!")

    largest_k_digit_numbers: list[int] = part2_get_largest_k_digit(data, k=12)
    pprint(largest_k_digit_numbers)

    print(f"Solution to part 2 is: {sum(largest_k_digit_numbers)}.")

    largest_k_digit_numbers_opt = part2_get_largest_k_digit_optimised(data, k=12)
    print(f"Solution to part 2 (optimised) is: {sum(largest_k_digit_numbers_opt)}.")

    if largest_k_digit_numbers != largest_k_digit_numbers_opt:
        raise ValueError("Optimised solution is incorrect!")

    timings = benchmark(data, number=100)
    pprint(timings)


if __name__ == "__main__":
    main()
