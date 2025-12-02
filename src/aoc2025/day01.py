from pathlib import Path

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
    signed_int_data: list[int] = data_to_signed_int(data)
    cumulative_indices: list[int] = signed_int_to_cumulative_index(
        signed_int_data, start=start
    )

    indices: list[int] = [i % total_positions for i in cumulative_indices]

    # Print zipped indices and data for verification.
    print(list(zip(indices, data, strict=True)))

    # The solution to the puzzle is essentially calculating the number of times we
    # hit 0.
    element_of_interest: int = 0
    count: int = indices.count(element_of_interest)
    print(f"Number of times we hit {element_of_interest}: {count} (Part 1 solution).")

    # Now part 2 dictates that in addition to finding zero, we need to count how many
    # times we “run past zero” as well.
    wrap_counts: int = count_multiples_in_range(
        cumulative_indices, value=total_positions
    )
    print(f"Number of times we wrapped around: {wrap_counts} (Part 2 solution).")


if __name__ == "__main__":
    main()
