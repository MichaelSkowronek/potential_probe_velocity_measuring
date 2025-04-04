# -*- coding: UTF-8 -*-
"""Data Processing.
"""


import re
import numpy as np
from pathlib import Path


def parse_tecplot_slice_to_numpy_timeseries(
        file_path,
        *,
        dtype=float,
        ):
    """
    Parses a Tecplot slice file and returns a 3D NumPy array,
    where the first dimension represents time steps in the timeseries.
    """
    file_path = Path(file_path)

    # Precompile regex patterns
    re_title = re.compile(r'title')
    re_variables = re.compile(r'variables')
    re_zone = re.compile(r'zone')

    data_arrays = []
    current_snapshot = []
    headers = []
    found_headers = False

    with file_path.open('r') as file:
        for line in file:
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            if re_title.match(line):  # New snapshot starts here
                if current_snapshot:  # Save previous snapshot if it exists
                    data_arrays.append(np.array(current_snapshot, dtype=dtype))
                    current_snapshot = []  # Reset for next snapshot

            elif re_variables.match(line):  # Extract headers only once
                if found_headers is not True:
                    found_headers = True
                    headers = re.findall(r'"(.*?)"', line)

            elif not re_zone.match(line):  # Process data lines
                try:
                    values = [dtype(v) for v in line.split()]  # Faster parsing than np.fromstring
                    current_snapshot.append(values)
                except ValueError:
                    print(f"Warning: Could not convert line to float values: {line}")

        # Append the last snapshot after reading all lines
        if current_snapshot:
            data_arrays.append(np.array(current_snapshot, dtype=dtype))

    # Ensure all snapshots have consistent shapes before stacking them into a single array.
    try:
        stacked_array = np.stack(data_arrays)
        print(f"Data parsed from path: {file_path.resolve(strict=True)}")
        return stacked_array, headers
    except ValueError as e:
        raise ValueError("Snapshots have inconsistent shapes and cannot be stacked into a single 3D array.") from e


def print_loaded_data_example(data):
    """
    Prints information about the loaded timeseries data.

    Parameters:
    data (dict): A dictionary containing 'timeseries' and 'labels'.
                 'timeseries' should be a 3D NumPy array of snapshots,
                 and 'labels' should be a list of corresponding labels.

    The function prints:
    - The number of timeseries snapshots (time steps).
    - The dimensions of each snapshot in terms of n_y, n_z, and n_v.
    - The labels associated with the timeseries.
    - A preview of the first 5 rows and first 3 columns of the first snapshot.
    """
    print("The data looks as follows:")
    
    # Extract timeseries and labels from the input dictionary
    timeseries = data['timeseries']
    labels = data['labels']
    
    # Number of time steps (snapshots)
    num_snapshots = timeseries.shape[0]
    
    # Dimensions per snapshot: n_y * n_z grid points and n_v variables
    n_x_y = timeseries.shape[1]  # Number of spatial grid points (n_y * n_z)
    n_v = timeseries.shape[2]   # Number of variables per grid point
    
    # Assuming a square grid for simplicity; calculate n_y and n_z
    n_y = int(np.sqrt(n_x_y))
    if n_y * n_y != n_x_y:
        raise ValueError("The number of spatial grid points is not a perfect square. Cannot infer square grid dimensions.")
    
    print(f"N (number of snapshots) = {num_snapshots}")
    
    print(f"n_y (grid height) = {n_y}, "
          f"n_z (grid width) = {n_y}, "  # Assuming square grid so height equals width
          f"n_v (number of variables) = {n_v}")
    
    print("Labels:")
    print(labels)
    
    # Preview: First snapshot, first 5 rows, first 3 columns
    print("Preview of the first snapshot:")
    
    first_snapshot_preview = timeseries[0, :5, :3]
    
    print(first_snapshot_preview)
