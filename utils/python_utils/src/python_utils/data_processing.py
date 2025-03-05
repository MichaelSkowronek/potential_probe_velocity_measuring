# -*- coding: UTF-8 -*-
"""Data Processing.
"""


import re
import numpy as np

import re
import numpy as np

def parse_tecplot_slice_to_numpy_timeseries(file_path, *, dtype=float):
    """
    Parses a Tecplot slice file and returns a list of NumPy arrays,
    where each array represents a snapshot in the timeseries.
    """
    # Precompile regex patterns
    re_title = re.compile(r'title')
    re_variables = re.compile(r'variables')
    re_zone = re.compile(r'zone')

    data_arrays = []
    current_snapshot = []
    headers = []
    found_headers = False

    with open(file_path, "r") as file:
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

    return data_arrays, headers
