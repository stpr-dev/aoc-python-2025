from pathlib import Path
import math
from utils.benchmark import time_callable

from utils.io import read_input_lines


def data_to_signed_int(data: list[str]) -> list[int]:
    """Convert data to signed integers."""
    rotation_to_signed_int: list[int] = [
        int(rotation[1:]) * (-1 if rotation[0] == "L" else 1) for rotation in data
    ]
    return rotation_to_signed_int


def signed_int_to_cumulative_index(data: list[int], start: int = 0) -> list[int]:
    """Convert signed integers to cumulative indices."""
    cumulative_data: list[int] = []
    cumulative_sum: int = start
    cumulative_data.append(cumulative_sum)
    for rotation in data:
        cumulative_sum += rotation
        cumulative_data.append(cumulative_sum)

    return cumulative_data


def part1_wrapper(
    data: list[str],
    start: int = 0,
    total: int = 100,
) -> tuple[list[int], list[int]]:
    """Part 1 of aoc-python-2025."""
    signed_int_data: list[int] = data_to_signed_int(data)
    cumulative_indices: list[int] = signed_int_to_cumulative_index(
        signed_int_data, start=start
    )

    indices: list[int] = [i % total for i in cumulative_indices[1:]]

    return indices, cumulative_indices


def count_multiples_in_range(cumulative_indices: list[int], value: int = 100) -> int:
    """Count multiples of a value in a list of cumulative indices.

    This function treats consecutive indices as ranges and checks to see how many
    values in those ranges are multiples of a value.

    Args:
        cumulative_indices: List of cumulative indices.
        value: The value to check for multiples. Defaults to 100.

    Returns:
        The count of multiples of the given value within the ranges defined by the
            cumulative indices.

    Notes:
        This function is EXTREMELY unoptimized. It is route 101 solution to the
        problem. It should be optimized for better performance.
    """

    if len(cumulative_indices) < 2:
        raise ValueError(
            "List of cumulative indices must contain at least two elements."
        )

    multiples: int = 0

    for idx in range(len(cumulative_indices) - 1):
        start: int = cumulative_indices[idx]
        end: int = cumulative_indices[idx + 1]

        # Since it is possible to turn backwards, we need to check for both cases.

        if start <= end:
            multiples += sum(1 for i in range(start, end, +1) if i % value == 0)
        else:
            multiples += sum(1 for i in range(start, end, -1) if i % value == 0)

    return multiples


def count_multiples_in_range_optimised(
    cumulative_indices: list[int], value: int = 100
) -> int:
    """Count multiples of a value in a list of cumulative indices.

    This function treats consecutive indices as ranges and checks to see how many
    values in those ranges are multiples of a value.

    Args:
        cumulative_indices: List of cumulative indices.
        value: The value to check for multiples. Defaults to 100.

    Returns:
        The count of multiples of the given value within the ranges defined by the
            cumulative indices.
    """

    if len(cumulative_indices) < 2:
        raise ValueError(
            "List of cumulative indices must contain at least two elements."
        )

    multiples: int = 0

    for idx in range(len(cumulative_indices) - 1):
        start: int = cumulative_indices[idx]
        end: int = cumulative_indices[idx + 1]

        delta: int = end - start

        if delta >= 0:
            multiples += end // value - start // value
        else:
            multiples += (start - 1) // value - (end - 1) // value

    return multiples


def count_multiples_in_range_optimised_v2(
    cumulative_indices: list[int], value: int = 100
) -> int:
    """Count multiples of a value in a list of cumulative indices.

    This function treats consecutive indices as ranges and checks to see how many
    values in those ranges are multiples of a value.

    Args:
        cumulative_indices: List of cumulative indices.
        value: The value to check for multiples. Defaults to 100.

    Returns:
        The count of multiples of the given value within the ranges defined by the
            cumulative indices.
    """

    if len(cumulative_indices) < 2:
        raise ValueError(
            "List of cumulative indices must contain at least two elements."
        )

    multiples: int = 0

    for idx in range(len(cumulative_indices) - 1):
        start: int = cumulative_indices[idx]
        end: int = cumulative_indices[idx + 1]

        if start == end:
            continue

        # So this problem is asking:
        # Assume (a, b]. Find c such that a < c <= b. Let c = k * value. How many k's
        # satisfy the equation?
        # We can rewrite equation as
        # a/value < k <= b/value.
        # So now simply, we are asking the span of "normalized" start and end.
        # since we are couting up from a to b, we simply take the ceiling and floor
        # operations to clamp results to the nearest integer. Then it is a simple
        # subtraction to find the number of multiples.
        # If however, a > b, then we need to reverse the operations to find the min
        # and max multiples.

        # So essentially we are asking how many integers are between normalized start
        # and normalized end. The +/-1 is important to make sure we don't include the
        # start but we do need to include the end.

        # If start is less than ened, this is canonical count up so things are easy.
        # If start is greater than end, this is canonical count down. In this case
        # the rounding operations flip around. Essentially we are rounding "towards
        # the other end". If the other end is greater we ceil the start and floor the
        # end. If the other end is less we floor the start and ceil the end.
        if start < end:
            min_factor: int = math.ceil((start + 1) / value)
            max_factor: int = math.floor(end / value)
        else:
            max_factor: int = math.floor((start - 1) / value)
            min_factor: int = math.ceil(end / value)

        num_multiples: int = max_factor - min_factor + 1

        multiples += num_multiples

    return multiples


def benchmark(
    data: list[str],
    *,
    start: int = 0,
    total: int = 100,
    number: int = 10,
) -> dict[str, float]:
    """
    Benchmark the three target functions.
    Assumes the functions are already imported and available in the namespace.
    """

    # Run part1_wrapper once to get cumulative data needed for part2.
    cumulative_indices, _ = part1_wrapper(data, start=start, total=total)

    timings: dict[str, float] = {
        "part1_wrapper": time_callable(
            part1_wrapper,
            data,
            start,
            total,
            number=number,
        ),
        "count_multiples_in_range": time_callable(
            count_multiples_in_range,
            cumulative_indices,
            100,
            number=number,
        ),
        "count_multiples_in_range_optimised": time_callable(
            count_multiples_in_range_optimised,
            cumulative_indices,
            100,
            number=number,
        ),
        "count_multiples_in_range_optimised_v2": time_callable(
            count_multiples_in_range_optimised_v2,
            cumulative_indices,
            100,
            number=number,
        ),
    }

    return timings


def main() -> None:
    data_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "2025" / "day01.txt"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data: list[str] = read_input_lines(data_path)

    # Puzzle dictates we are starting at 50 and we have 100 positions.
    start: int = 50
    total_positions: int = 100

    indices, cumulative_indices = part1_wrapper(
        data, start=start, total=total_positions
    )

    # Print zipped indices and data for verification.
    print(list(zip(indices, data, strict=True)))

    # The solution to the puzzle is essentially calculating the number of times we
    # hit 0.
    element_of_interest: int = 0
    count: int = indices.count(element_of_interest)
    print(f"Number of times we hit {element_of_interest}: {count} (Part 1 solution).")

    # Now part 2 dictates that in addition to finding zero, we need to count how many
    # times we “run past zero” as well.

    # First, naive brute force approach.
    wrap_counts: int = count_multiples_in_range(
        cumulative_indices, value=total_positions
    )
    print(f"Number of times we wrapped around: {wrap_counts} (Part 2 solution).")

    # Second approach, partially using ChatGPT.
    multiple_of_interest: int = count_multiples_in_range_optimised(
        cumulative_indices, value=total_positions
    )

    print(f"Number of multiples of {total_positions}: {multiple_of_interest}.")

    if multiple_of_interest == wrap_counts:
        print("Part 2 solution is correct!")
    else:
        print("Part 2 solution is incorrect!")

    # Third approach, which is mine.

    multiple_of_interest_v2: int = count_multiples_in_range_optimised_v2(
        cumulative_indices, value=total_positions
    )

    print(f"Number of multiples of {total_positions}: {multiple_of_interest_v2}.")

    if multiple_of_interest_v2 == wrap_counts:
        print("Part 2 solution is correct!")
    else:
        print("Part 2 solution is incorrect!")

    from pprint import pprint

    result = benchmark(data, start=start, total=total_positions, number=100)
    pprint(result)


if __name__ == "__main__":
    main()
