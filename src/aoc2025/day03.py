from pathlib import Path
from pprint import pprint

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


if __name__ == "__main__":
    main()
