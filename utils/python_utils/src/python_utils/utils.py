# -*- coding: UTF-8 -*-
"""Python Utils.
"""


import pickle
from pathlib import Path


def save_object(obj, file_path):
    """
    Save a Python object to a file, creating parent directories if necessary.
    
    :param obj: The Python object to save
    :param file_path: The path of the file where the object will be stored (can be a string or Path)
    """
    file_path = Path(file_path)

    # Ensure parent directories exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the object using pickle
    with file_path.open('wb') as file:
        pickle.dump(obj, file)
    print(f"Object successfully saved to {file_path.resolve(strict=True)}")
        

def load_object(file_path):
    """
    Load a Python object from a file.
    
    :param file_path: The path of the file where the object is stored
    :return: The loaded Python object
    """
    file_path = Path(file_path)
    try:
        with file_path.open('rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {file_path.resolve(strict=True)}")
        return obj
    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        return None


def truncate_file_at_line(filepath, line_number):
    """
    Truncates a file at the specified line number.

    Args:
        filepath (str): The path to the file.
        line_number (int): The line number at which to truncate the file.
    """
    filepath = Path(filepath)
    try:
        with filepath.open('r+') as f:
            lines = f.readlines()
            if line_number > 0 and line_number <= len(lines):
                f.seek(0) # go to the beginning of the file
                f.writelines(lines[:line_number]) # write the first line_number lines
                f.truncate() # remove the rest of the file.
            else:
                print(f"Error: Line number {line_number} is out of range.")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath.resolve(strict=True)}")
    except Exception as e:
        print(f"An error occurred: {e}")