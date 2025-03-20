# -*- coding: UTF-8 -*-
"""Python Utils.
"""


import os
import pickle
from pathlib import Path

def save_object(obj, file_path):
    """
    Save a Python object to a file, creating parent directories if necessary.
    
    :param obj: The Python object to save
    :param file_path: The path of the file where the object will be stored (can be a string or Path)
    """
    # Convert file_path to a Path object if it's not already one
    file_path = Path(file_path)

    # Ensure parent directories exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the object using pickle
    with file_path.open('wb') as file:
        pickle.dump(obj, file)

    # Print the absolute path of the saved object
    print(f"Object successfully saved to {file_path.resolve()}")
        

def load_object(file_path):
    """
    Load a Python object from a file.
    
    :param file_path: The path of the file where the object is stored
    :return: The loaded Python object
    """
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        # Get the absolute path and print it
        complete_file_path = os.path.abspath(file_path)
        print(f"Object successfully loaded from {complete_file_path}")
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
    try:
        with open(filepath, 'r+') as f:  # Open for reading and writing
            lines = f.readlines()
            if line_number > 0 and line_number <= len(lines):
                f.seek(0) # go to the beginning of the file
                f.writelines(lines[:line_number]) # write the first line_number lines
                f.truncate() # remove the rest of the file.
            else:
                print(f"Error: Line number {line_number} is out of range.")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")