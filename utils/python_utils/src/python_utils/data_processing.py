# -*- coding: UTF-8 -*-
"""Data Processing.
"""


import re
import pandas as pd


def parse_tecplot_slice_to_dataframes_timeseries(file_path):
    # Initialize variables to store parsed data
    data_frames = []
    current_snapshot = []
    headers = []

    # Regular expressions to detect headers
    re_title = re.compile(r'title')
    re_variables = re.compile(r'variables')
    re_zone = re.compile(r'zone')

    # Read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            if not line:  # Skip empty lines
                continue

            if re_title.match(line):
                if current_snapshot:  # If there's existing data, store it
                    df = pd.DataFrame(current_snapshot, columns=headers)
                    data_frames.append(df)
                    current_snapshot.clear()  # Clear for the next snapshot
            
            elif re_variables.match(line):
                headers = re.findall(r'"(.*?)"', line)
            
            elif not re_zone.match(line):  # Ignore 'zone' lines
                try:
                    values = list(map(float, line.split()))
                    current_snapshot.append(values)  # Append float values to the snapshot
                except ValueError:
                    print(f"Warning: Could not convert line to float values: {line}")

    # Append the last snapshot if there is one
    if current_snapshot:
        df = pd.DataFrame(current_snapshot, columns=headers)
        data_frames.append(df)

    return data_frames
