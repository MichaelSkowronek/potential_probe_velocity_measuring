# -*- coding: UTF-8 -*-
"""Data Processing.
"""


import re
import numpy as np
from pathlib import Path


def parse_tecplot_slice_to_numpy_timeseries(file_path, *, dtype=float):
    """
    Parses a Tecplot slice file and returns a list of NumPy arrays,
    where each array represents a snapshot in the timeseries.
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
    print(f"Data parsed from path: {file_path.resolve(strict=True)}")
    return data_arrays, headers


def print_loaded_data_example(data):
    """
    Prints information about the loaded timeseries data.

    Parameters:
    data (dict): A dictionary containing 'timeseries' and 'labels'.
                 'timeseries' should be a list or array of snapshots,
                 and 'labels' should be a list of corresponding labels.

    The function prints:
    - The number of timeseries snapshots.
    - The dimensions of the first snapshot in terms of n_y, n_z, and n_v.
    - The labels associated with the timeseries.
    - A preview of the first 5 rows and first 3 columns of the first snapshot.
    """
    print(f"The data looks as follows:")
    timeseries = data['timeseries']
    labels = data['labels']
    
    print("N =", len(timeseries))
    
    first_snapshot = timeseries[0]
    n_x_y, n_v = first_snapshot.shape
    n_y = n_z = np.sqrt(n_x_y)
    
    print(f"n_y = {n_y}, n_z = {n_z}, n_v = {n_v}")
    print("Labels:")
    print(labels)
    print(first_snapshot[0:5, 0:3])
