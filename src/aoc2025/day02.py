from pathlib import Path
from pprint import pprint
from collections.abc import Iterable
import math

from aoc2025.utils.io import read_input
from aoc2025.utils.benchmark import time_callable


def part1_get_invalid_ids_brute_force(ranges: list[list[int]]) -> list[int]:
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

    return sorted(set(invalid_ids))


def generate_invalid_ids_in_range(low: int, high: int) -> list[int]:
    """Generate all invalid IDs within [low, high].

    Invalid IDs are numbers with even digit count 2k formed as xx where x has k digits.
    """
    result: list[int] = []

    # Maximum number of digits we need to consider
    max_digits: int = math.ceil(math.log10(high))

    # Iterate over even digit counts: 2, 4, 6, ...
    for digits in range(2, max_digits + 1, 2):
        k = digits // 2

        base = 10**k  # 10^k
        x_min = base // 10  # smallest k-digit number
        x_max = base - 1  # largest  k-digit number

        # full = x * (10^k + 1)
        multiplier = base + 1

        # Solve L <= full = x * multiplier <= R
        # → x >= ceil(L / multiplier)
        # → x <= floor(R / multiplier)

        x_low = (low + multiplier - 1) // multiplier
        x_high = high // multiplier

        # Clamp to valid k-digit range
        x_low = max(x_low, x_min)
        x_high = min(x_high, x_max)

        if x_low <= x_high:
            for x in range(x_low, x_high + 1):
                full = x * multiplier
                result.append(full)

    return result


def collect_invalid_ids(ranges: Iterable[list[int]]) -> list[int]:
    """Collect invalid IDs across all ranges."""
    all_invalid: list[int] = []
    for low, high in ranges:
        all_invalid.extend(generate_invalid_ids_in_range(low, high))
    return sorted(set(all_invalid))


def divisors(n: int) -> list[int]:
    """Return all divisors of n except n itself."""
    result: list[int] = []
    for d in range(1, int(math.sqrt(n)) + 1):
        if n % d == 0:
            if d < n:
                result.append(d)
            q = n // d
            if q != d and q < n:
                result.append(q)
    return sorted(result)


def generate_periodic_ids_in_range(low: int, high: int) -> list[int]:
    """Generate all periodic (invalid) IDs within [low, high].

    A number is invalid if its digits form S repeated k>=2 times.
    """
    result: list[int] = []

    max_digits = math.ceil(math.log10(high))

    # Loop over digit lengths
    for n in range(2, max_digits + 1):
        n_min = 10 ** (n - 1)
        n_max = 10**n - 1

        # Skip digit lengths that do not overlap range
        if n_max < low or n_min > high:
            continue

        # Get all period lengths p dividing n
        for p in divisors(n):
            r = n // p

            # multiplier = 111...111 (r copies) in base 10^p
            # multiplier = (10^(p*r) - 1) // (10^p - 1)
            pow_p = 10**p
            multiplier = (pow_p**r - 1) // (pow_p - 1)

            # Solve bounds on base
            base_low = (low + multiplier - 1) // multiplier
            base_high = high // multiplier

            # Clamp to valid p-digit bases
            p_min = pow_p // 10
            p_max = pow_p - 1

            base_low = max(base_low, p_min)
            base_high = min(base_high, p_max)

            if base_low <= base_high:
                for base in range(base_low, base_high + 1):
                    result.append(base * multiplier)

    return result


def collect_invalid_ids_part2(ranges: Iterable[list[int]]) -> list[int]:
    all_invalid: list[int] = []
    for low, high in ranges:
        all_invalid.extend(generate_periodic_ids_in_range(low, high))
    return sorted(set(all_invalid))


def factors(n: int) -> list[int]:
    return [factor for factor in range(1, n + 1) if n % factor == 0]


def part2_get_invalid_ids_brute_force(ranges: list[list[int]]) -> list[int]:
    """Get invalid IDs as defined by Part 2 of the AoC challenge.

    Args:
        ranges: List of integer ranges, where each range is a list of two integers.

    Returns:
        List of invalid IDs.

    Notes:
        This approach uses brute force and is extremely slow. Only use for
            validation purposes.
    """

    # Unlike part 1, this time it's not just HALF of the string, but any n-substring
    # of the number that is repeated. So 123123123 is valid now, for instance.
    # So looking at it from another lens, the question is essentially asking: if we
    # have a string of length n, can we decompose it into smaller substrings and test
    # for equality? From a mathematical perspective, it is essentially finding
    # factors of the length of the original string and finding if repeating THAT
    # substring gives us the original string.

    invalid_ids: list[int] = []

    for range_ in ranges:
        lower_number: int = range_[0]

        higher_number: int = range_[1]

        # Now loop through the range.
        for num in range(lower_number, higher_number + 1):
            num_str: str = str(num)

            fac: list[int] = factors(len(num_str))
            _ = fac.pop()  # Remove the last element, which is the number itself

            for factor in fac:
                if num_str[:factor] * (len(num_str) // factor) == num_str:
                    invalid_ids.append(num)
                    break

    return sorted(set(invalid_ids))


def benchmark(
    data: Iterable[list[int]],
    *,
    number: int = 10,
) -> dict[str, float]:
    """
    Benchmark the three target functions.
    Assumes the functions are already imported and available in the namespace.
    """

    timings: dict[str, float] = {
        "part1_bruteforce": time_callable(
            part1_get_invalid_ids_brute_force,
            data,
            number=number,
        ),
        "part1_optimised": time_callable(
            collect_invalid_ids,
            data,
            number=number,
        ),
        "part2_bruteforce": time_callable(
            collect_invalid_ids_part2,
            data,
            number=number,
        ),
        "part2_optimised": time_callable(
            collect_invalid_ids_part2,
            data,
            number=number,
        ),
    }

    return timings


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

    invalid_ids_periodic: list[int] = part1_get_invalid_ids_brute_force(ranges)
    pprint(f"Invalid IDs: {invalid_ids_periodic}")

    pprint(f"Part 1 solution is: {sum(invalid_ids_periodic)}.")

    invalid_ids_v2: list[int] = collect_invalid_ids(ranges)

    if not invalid_ids_periodic == invalid_ids_v2:
        raise ValueError(
            f"Invalid IDs mismatch: {invalid_ids_periodic} != {invalid_ids_v2}"
        )
    else:
        print("Part 1 optimised version is correct!")

    invalid_ids_periodic = part2_get_invalid_ids_brute_force(ranges)
    pprint(f"Invalid IDs: {invalid_ids_periodic}")
    pprint(f"Part 2 solution is: {sum(invalid_ids_periodic)}.")

    invalid_ids_periodic_v2: list[int] = collect_invalid_ids_part2(ranges)

    if not invalid_ids_periodic_v2 == invalid_ids_periodic:
        raise ValueError(
            f"Invalid IDs mismatch: {invalid_ids_periodic_v2} != {invalid_ids_periodic}"
        )
    else:
        print("Part 2 optimised version is correct!")

    result = benchmark(ranges, number=100)
    pprint(result)


if __name__ == "__main__":
    main()
