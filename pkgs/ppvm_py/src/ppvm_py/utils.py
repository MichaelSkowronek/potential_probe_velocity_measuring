# -*- coding: UTF-8 -*-
"""Python Utils.
"""


import pickle
from pathlib import Path
import numpy as np


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


def zero_smallest_absolute_to_meet_ratio(parameters, ratio):
    """
    Zero out the smallest elements (based on absolute value) in 'parameters'
    while ensuring that at most '(1 - ratio) * original_sum' is removed.
    Works for arrays of any shape.

    Args:
        parameters (ndarray): Array of numerical values (any shape).
        ratio (float): Ratio of total parameter amount to retain (0 < ratio <= 1).

    Returns:
        tuple: 
            - Modified ndarray with some values set to zero (same shape as input).
            - Threshold (absolute value) at which parameters are not removed anymore.
    """
    # Ensure ratio is valid
    if not (0 < ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    # Flatten the parameters for processing
    flat_parameters = parameters.flatten()
    
    # Compute original total sum
    original_sum = np.sum(np.abs(flat_parameters))  # Use absolute sum
    
    # Determine maximum removable sum based on (1 - ratio)
    max_removal_sum = (1 - ratio) * original_sum
    
    # Sort indices by absolute value in ascending order
    sorted_indices = np.argsort(np.abs(flat_parameters))
    
    # Initialize variables for cumulative removal sum and modified array
    removal_sum = 0
    modified_flat_parameters = np.copy(flat_parameters)
    
    threshold = None  # Initialize threshold variable
    
    # Iterate over sorted indices and zero out elements until max_removal is met
    for index in sorted_indices:
        if removal_sum + abs(modified_flat_parameters[index]) > max_removal_sum:
            threshold = abs(modified_flat_parameters[index])  # Set threshold when stopping
            break  # Stop when exceeding allowed removal
        
        # Zero out this element and update removal sum
        removal_sum += abs(modified_flat_parameters[index])
        modified_flat_parameters[index] = 0
    
    if threshold is None:  
        # If all elements were removed, set threshold to infinity or a meaningful upper bound
        threshold = float('inf')
    
    # Reshape the modified flat array back to original shape
    modified_parameters = modified_flat_parameters.reshape(parameters.shape)
    
    return modified_parameters, threshold
