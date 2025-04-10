# -*- coding: UTF-8 -*-
"""Data Processing.
"""


import re
import numpy as np
from pathlib import Path


from pathlib import Path
import re
import numpy as np


def _process_snapshot_data_with_fixed_coords(
        snapshot_data,
        headers,
        y_index,
        z_index,
        dtype,
        y_coords,
        z_coords,
        coord_to_index_map,
):
    """
    Processes the data of a single Tecplot snapshot using pre-defined coordinates.

    Args:
        snapshot_data (list): A list of data rows for the current time step.
        headers (list): The list of variable headers.
        y_index (int): The index of the 'y' coordinate column.
        z_index (int): The index of the 'z' coordinate column.
        dtype (type): The NumPy data type for the output array.
        y_coords (list): The sorted list of unique 'y' coordinates from the first snapshot.
        z_coords (list): The sorted list of unique 'z' coordinates from the first snapshot.
        coord_to_index_map (dict): A dictionary mapping (y_coord, z_coord)
                                   tuples to their (y_index, z_index) in the output array.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the data for the snapshot,
                       indexed by y, z, and variables. Missing coordinates are filled with NaN.
    """
    num_y = len(y_coords)
    num_z = len(z_coords)
    num_vars = len(headers) - 2 if headers else 0
    snapshot_array = np.empty((num_y, num_z, num_vars), dtype=dtype)
    snapshot_array[:] = np.nan  # Initialize with NaN

    for row in snapshot_data:
        y_coord = row[y_index]
        z_coord = row[z_index]
        if (y_coord, z_coord) in coord_to_index_map:
            y_idx, z_idx = coord_to_index_map[(y_coord, z_coord)]
            data_values = np.array([row[i] for i in range(len(row)) if i not in [y_index, z_index]], dtype=dtype)
            snapshot_array[y_idx, z_idx, :] = data_values
        else:
            print(f"Warning: Coordinate (y={y_coord}, z={z_coord}) not found in the initial coordinate map.")
    return snapshot_array


def parse_tecplot_yz_slice(
        file_path,
        *,
        dtype=float,
):
    """
    Parses a Tecplot yz-slice file and returns a 4D NumPy array,
    where the dimensions represent:
    1. Time steps
    2. Y-coordinates (indexed)
    3. Z-coordinates (indexed)
    4. Variables

    Also returns a dictionary mapping (y_coord, z_coord) tuples to their
    corresponding index in the 4D array. This version assumes that the
    y and z coordinates and their corresponding indices remain the same
    across all snapshots, extracting them only from the first snapshot.

    Args:
        file_path (pathlib.Path or str): The path to the Tecplot slice file.
        dtype (type, optional): The NumPy data type for the output array. Defaults to float.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A 4D NumPy array where the first dimension is time,
                             the second is the y-coordinate index, the third is the
                             z-coordinate index, and the fourth contains the variable values.
            - dict: A dictionary mapping (y_coord, z_coord) tuples to their
                    (y_index, z_index) in the 4D array.
            - list: A list of variable headers (excluding 'y' and 'z').
    """
    file_path = Path(file_path)

    # Precompile regex patterns
    re_title = re.compile(r'title')
    re_variables = re.compile(r'variables')
    re_zone = re.compile(r'zone')

    time_series_data = []
    headers = []
    found_headers = False
    y_index = -1
    z_index = -1
    initial_snapshot_data = []
    y_coords = []
    z_coords = []
    coord_to_index_map = {}
    first_snapshot_processed = False

    with file_path.open('r') as file:
        current_snapshot_data = []
        for line in file:
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            if re_title.match(line):  # New snapshot starts here
                if current_snapshot_data and headers and y_index != -1 and z_index != -1:
                    if not first_snapshot_processed:
                        unique_y_set = set(row[y_index] for row in initial_snapshot_data)
                        unique_z_set = set(row[z_index] for row in initial_snapshot_data)
                        y_coords = sorted(list(unique_y_set))
                        z_coords = sorted(list(unique_z_set))
                        coord_to_index_map = {(y, z): (y_coords.index(y), z_coords.index(z)) for y in y_coords for z in z_coords}
                        first_snapshot_processed = True

                    if first_snapshot_processed:
                        snapshot_array = _process_snapshot_data_with_fixed_coords(
                            current_snapshot_data,
                            headers,
                            y_index,
                            z_index,
                            dtype,
                            y_coords,
                            z_coords,
                            coord_to_index_map,
                        )
                        time_series_data.append(snapshot_array)

                current_snapshot_data = []  # Reset for next snapshot

            elif re_variables.match(line):  # Extract headers
                if not found_headers:
                    found_headers = True
                    headers = [h.strip('"') for h in re.findall(r'"(.*?)"', line)]
                    try:
                        y_index = headers.index("y")
                        z_index = headers.index("z")
                    except ValueError:
                        print("Warning: 'y' and 'z' variables not found in headers.")
                        y_index = -1
                        z_index = -1

            elif not re_zone.match(line):  # Process data lines
                try:
                    values = [dtype(v) for v in line.split()]
                    current_snapshot_data.append(values)
                    if not first_snapshot_processed and found_headers:
                        initial_snapshot_data.append(values)
                except ValueError:
                    print(f"Warning: Could not convert line to float values: {line}")

        # Process the last snapshot
        if current_snapshot_data and headers and y_index != -1 and z_index != -1:
            if not first_snapshot_processed:
                unique_y_set = set(row[y_index] for row in initial_snapshot_data)
                unique_z_set = set(row[z_index] for row in initial_snapshot_data)
                y_coords = sorted(list(unique_y_set))
                z_coords = sorted(list(unique_z_set))
                coord_to_index_map = {(y, z): (y_coords.index(y), z_coords.index(z)) for y in y_coords for z in z_coords}
                first_snapshot_processed = True

            if first_snapshot_processed:
                snapshot_array = _process_snapshot_data_with_fixed_coords(
                    current_snapshot_data,
                    headers,
                    y_index,
                    z_index,
                    dtype,
                    y_coords,
                    z_coords,
                    coord_to_index_map,
                )
                time_series_data.append(snapshot_array)

    if not time_series_data:
        return np.empty((0, 0, 0, 0), dtype=dtype), {}, headers

    first_shape = time_series_data[0].shape
    for i, snapshot in enumerate(time_series_data):
        if snapshot.shape != first_shape:
            raise ValueError(f"Snapshot at time index {i} has inconsistent shape {snapshot.shape} compared to the first snapshot {first_shape}.")

    stacked_array = np.stack(time_series_data)
    print(f"Data parsed from path: {file_path.resolve(strict=True)}")

    return stacked_array, coord_to_index_map, [h for h in headers if h not in ["y", "z"]]


def print_yz_slice_example_from_parsed_data(
        data,
):
    """
    Prints information about the loaded timeseries data (for 4D array).

    Args:
        data (dict): A dictionary containing 'timeseries', 'coord_to_index_map', and 'labels'.
                     'timeseries' should be a 4D NumPy array of snapshots,
                     'coord_to_index_map' should be a dictionary mapping (y, z) to indices,
                     and 'labels' should be a list of corresponding variable labels.

    The function prints:
    - The number of timeseries snapshots (time steps).
    - The dimensions of each snapshot in terms of n_y, n_z, and n_v.
    - The labels associated with the timeseries variables.
    - A preview of the first time step, first 2 y-indices, first 2 z-indices,
      and all variable values, along with their corresponding (y, z) coordinates.
    """
    print("The data looks as follows:")

    # Extract timeseries, coordinate map, and labels from the input dictionary
    timeseries = data['timeseries']
    coord_to_index_map = data['coord_to_index_map']
    labels = data['labels']

    # Number of time steps (snapshots)
    num_snapshots = timeseries.shape[0]

    # Dimensions per snapshot: n_y, n_z grid points and n_v variables
    n_y = timeseries.shape[1]
    n_z = timeseries.shape[2]
    n_v = timeseries.shape[3]

    print(f"N (number of snapshots) = {num_snapshots}")
    print(f"n_y (grid height) = {n_y}, "
          f"n_z (grid width) = {n_z}, "
          f"n_v (number of variables) = {n_v}")

    print("Labels:")
    print(labels)

    # Preview: First snapshot, first 2 y-indices, first 2 z-indices, all variables
    print("Preview of the first snapshot (first 2 y, first 2 z) with coordinates:")

    if timeseries.ndim == 4 and timeseries.shape[1] > 0 and timeseries.shape[2] > 0:
        preview_y_count = min(2, timeseries.shape[1])
        preview_z_count = min(2, timeseries.shape[2])
        first_snapshot = timeseries[0]

        # Create a reverse mapping from index to coordinate
        index_to_coord = {(y_idx, z_idx): coord for coord, (y_idx, z_idx) in coord_to_index_map.items()}

        for y_idx in range(preview_y_count):
            for z_idx in range(preview_z_count):
                if (y_idx, z_idx) in index_to_coord:
                    y_coord, z_coord = index_to_coord[(y_idx, z_idx)]
                    data_values = first_snapshot[y_idx, z_idx, :]
                    print(f"  (y={y_coord:.2f}, z={z_coord:.2f}): {data_values}")
                else:
                    print(f"  Index ({y_idx}, {z_idx}) not found in coordinate map.")
    else:
        print("Timeseries data or coordinate map is empty or not in the expected format.")

