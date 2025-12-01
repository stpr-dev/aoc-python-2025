from pathlib import Path

from utils.io import read_input_lines


def rotation_to_index(data: list[str], start: int = 0, total: int = 100) -> list[int]:
    """Convert L/R rotation to index.
    
    Args:
        data (list[str]): List of L/R rotations.
        start (int, optional): Starting index. Defaults to 0.
        total (int, optional): Total number of positions. Defaults to 100.
    
    Returns:
        list[int]: Indices after each rotation.
    """
    
    # The logic here is simple: the data is expected to be of the format "L/R{x}"
    # where x is the number of rotations. From the given start, we essentially are
    # calculating where we end up after the given rotation. The effects are cumulative.
    # This is essentially glorified modulo arithmetic but with cumulative history.
    
    rotation_to_signed_int: list[int] = [int(rotation[1:]) * (-1 if rotation[0] ==
                                                                    "L" else 1) for rotation in data]
    # Next, calculate the new position cumulatively.
    cumulative_data: list[int] = []
    cumulative_sum: int = start
    for rotation in rotation_to_signed_int:
        cumulative_sum += rotation
        cumulative_data.append(cumulative_sum % total)
    
    return cumulative_data
    
def main() -> None:
    data_path: Path = Path(__file__).parent.parent.parent / "data" / "2025" / "day01.txt"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data: list[str] = read_input_lines(data_path)
    
    # Puzzle dictates we are starting at 50 and we have 100 positions.
    indices: list[int] = rotation_to_index(data, start=50, total=100)
    
    # Print zipped indices and data for verification.
    print(list(zip(indices, data)))
    
    # Next part of the puzzle is essentially calculating the number of times we hit 0.
    element_of_interest: int = 0
    print(f"Number of times we hit {element_of_interest}:"
          f" {indices.count(element_of_interest)}")

if __name__ == "__main__":
    main()