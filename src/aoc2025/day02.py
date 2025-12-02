from pathlib import Path
from pprint import pprint

from utils.io import read_input


def get_invalid_ids_brute_force(ranges: list[list[int]]) -> list[int]:
    """Get invalid IDs as defined by Part 1 of the AoC challenge.

    Args:
        ranges: List of integer ranges, where each range is a list of two integers.

    Returns:
        List of invalid IDs.

    Notes:
        This approach uses brute force and is extremely slow. Only use for
            validation purposes.
    """

    # So essentially, invalid ids are numbers in given range that have "half symmetry"
    # i.e. if we look at the first half of the digits, it is exactly equal to the
    # second half. E.g.: 11, 1212, 123123, ...

    # The easiest way is to do the following:
    # For each given range, go through the range and generate numbers (upper is
    # inclusive). For each number and convert it to a string.
    # Assume that there are an even number of digits in the string. If the first half
    # of the string is equal to the second half, then the number is "invalid". Do
    # that for each range.

    invalid_ids: list[int] = []

    for range_ in ranges:
        lower_number: int = range_[0]
        lower_number_str: str = str(lower_number)

        higher_number: int = range_[1]
        higher_number_str: str = str(higher_number)

        # Trivial check is if both lower AND higher digits have odd number of digits,
        # then don't bother checking as there will be no invalid ids.
        if len(lower_number_str) % 2 == 1 and len(higher_number_str) % 2 == 1:
            continue

        # Now loop through the range.
        for num in range(lower_number, higher_number + 1):
            num_str: str = str(num)

            if len(num_str) % 2 == 1:
                continue

            lower_half: str = num_str[: len(num_str) // 2]
            higher_half: str = num_str[len(num_str) // 2 :]

            if lower_half == higher_half:
                invalid_ids.append(num)

    return invalid_ids


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day02.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: str = read_input(data_path)

    # The data should be a single line of text wilth comma separated ranges. E.g.:
    # 1-5,10-30,...

    ranges: list[list[int]] = [
        list(map(int, pair.split("-"))) for pair in data.split(",")
    ]

    # Validate that there are exactly two numbers in each range and that the
    # first is <= the second.
    for range_ in ranges:
        if len(range_) != 2 or range_[0] > range_[1]:
            raise ValueError(f"Invalid range: {range_}")

    pprint(ranges)

    invalid_ids: list[int] = get_invalid_ids_brute_force(ranges)
    pprint(f"Invalid IDs: {invalid_ids}")

    pprint(f"Part 1 solution is: {sum(invalid_ids)}.")


if __name__ == "__main__":
    main()
