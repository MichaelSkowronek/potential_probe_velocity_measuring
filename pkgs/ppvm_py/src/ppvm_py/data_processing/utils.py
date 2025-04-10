# -*- coding: UTF-8 -*-


from pathlib import Path


def read_first_n_lines(
    file_path,
    n
):
    """
    Reads the first n lines from a file without loading the entire file into memory.

    Parameters:
        file_path (str): The path to the file.
        n (int): The number of lines to read from the beginning of the file.

    Returns:
        list: A list containing the first n lines of the file.
    """
    file_path = Path(file_path)
    lines = []
    
    with open(file_path, 'r') as file:
        for _ in range(n):
            line = file.readline()
            if not line:  # Stop if we reach EOF
                break
            lines.append(line.strip())
    
    return lines


