from pathlib import Path


def read_input(path: str | Path) -> str:
    """Read input from a file path and return its content as a string.

    Args:
        path (str | Path): The file path to read from.

    Returns:
        str: The content of the file as a string.
    """
    p: Path = Path(path)
    return p.read_text(encoding="utf-8").strip()


def read_input_lines(path: str | Path) -> list[str]:
    """Read input from a file path and return its content as a list of lines.
    Args:
        path (str | Path): The file path to read from.

    Returns:
        list[str]: The content of the file as a list of lines.
    """
    p: Path = Path(path)
    return p.read_text(encoding="utf-8").strip().splitlines()
