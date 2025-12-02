from pathlib import Path
import math

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


def count_hits(data: list[int], target: int, total: int = 100) -> int:
    """Count the number of times the target is reached in modulo space.

    Args:
        data (list[int]): Signed indices.
        target (int): Target integer to count hits for.
        total (int, optional): Total number of positions. Defaults to 100.

    Returns:
        int: Number of times the target is reached.
    """
    # The goal is to count in a rotation, how many times we "hit" a target.
    # This could happen in 4 ways:
    # - Go from > target to < target
    # - Go from < target to > target
    # - Go from < target to >> target (wrapping around)
    # - Go from > target to << target (wrapping around)

    # Trivially, we can subtract the target from the cumulative indices to zero
    # centre it.
    zero_centred_data: list[int] = [i - target for i in data]

    # Next for counting wrap arounds, we can work with divisors.
    divisors: list[int] = [math.trunc(zdata / total) for zdata in zero_centred_data]
    # Now put together all cases.
    counts: list[int] = []

    for idx, zdata in enumerate(zero_centred_data):
        total_hits: int = 0
        if idx == 0:
            # We need at least two zdata points to see if we ran past zero.
            continue

        # First calculate if sign change occurred.
        sign_change: bool = zero_centred_data[idx - 1] * zdata <= 0

        if sign_change:
            total_hits += 1

        # Sign change only gives us one hit but that isn't the full picture.
        # For instance, consider the case where the total is 10 and we are at -5. If
        # the transition was -5 -> 21, we also need to count 10->21 as a hit since it
        # wrapped around to zero and started again. Same logic applies to the other
        # direction.
        # A good way to find the correct to apply is to diff the divisors.
        total_hits += abs(divisors[idx] - divisors[idx - 1])
        
        counts.append(total_hits)

    return sum(counts)


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
    print(list(zip(indices, data)))

    # The solution to the puzzle is essentially calculating the number of times we
    # hit 0.
    element_of_interest: int = 0
    count: int = indices.count(element_of_interest)
    print(
        f"Number of times we hit {element_of_interest}:"
        f" {count} (Part 1 solution)."
    )

    # Now part 2 dictates that instead of finding zero, we need to count how many
    # times we “run past zero”.
    num_hits: int = count_hits(
        cumulative_indices, target=element_of_interest, total=total_positions
    )
    print(f"Number of times we run past zero: {num_hits} (Part 2 solution).")

if __name__ == "__main__":
    main()