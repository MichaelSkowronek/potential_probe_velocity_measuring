# -*- coding: UTF-8 -*-
"""Time Series Extraction.
"""


import numpy as np
from matplotlib import pyplot as plt


class ClosestMatchFound(Exception):
    """Custom exception to indicate a closest match was found."""

    def __init__(
            self,
            index,
            closest_y,
            closest_z,
            provided_y,
            provided_z,
            ):
        self.index = index
        self.closest_y = closest_y
        self.closest_z = closest_z
        self.provided_y = provided_y
        self.provided_z = provided_z
        
        super().__init__(f"Closest match found at index {index}, "
                         f"closest coordinates (y={closest_y}, z={closest_z}), "
                         f"provided coordinates (y={provided_y}, z={provided_z})")


def find_closest_coordinate_index(
        coord_to_index_map,
        y_coord,
        z_coord,
        *,
        return_closest=False,
    ):
    """
    Finds the indices (y_index, z_index) in a coordinate-to-index map where the y and z coordinates match.
    If no exact match is found, it either returns the closest indices or raises an exception, based on the 'return_closest' parameter.

    Args:
        coord_to_index_map (dict): A dictionary mapping (y_coord, z_coord) tuples to (y_index, z_index) tuples.
        y_coord (float or int): The y-coordinate to filter by.
        z_coord (float or int): The z-coordinate to filter by.
        return_closest (bool, optional): If True, returns the indices of the closest match if no exact match is found. 
                                         If False (default), raises ClosestMatchFound.

    Returns:
        tuple or None: The indices (y_index, z_index) of the matching row in the map, or None if the map is empty.
                      If return_closest is True and no exact match is found, returns indices of closest match.

    Raises:
        ClosestMatchFound: If no exact match is found and return_closest is False.
    """

    if not coord_to_index_map:  # Check if the map is empty
        return None

    # Check for exact match
    key = (y_coord, z_coord)
    if key in coord_to_index_map:
        return coord_to_index_map[key]

    # Find closest matching row using Euclidean distance
    all_coords = np.array(list(coord_to_index_map.keys()))  # Extract all keys as an array
    y_values = all_coords[:, 0]
    z_values = all_coords[:, 1]

    euclidean_distance = np.sqrt((y_values - y_coord)**2 + (z_values - z_coord)**2)
    
    closest_key_idx_in_array = np.argmin(euclidean_distance)  # Get index within array

    closest_y, closest_z = all_coords[closest_key_idx_in_array]
    
    closest_key = (closest_y, closest_z)
    
    if return_closest:
        return coord_to_index_map[closest_key]  # Return corresponding indices from map
    else:
        raise ClosestMatchFound(
            coord_to_index_map[closest_key], 
            closest_y,
            closest_z,
            y_coord,
            z_coord
        )


def _extract_timeseries_of_one_column_and_row_index(
        time_series_arrays: list[np.ndarray],
        column_index: int,
        row_index: int,
        *,
        dtype=None,
        ) -> np.ndarray:
    """
    Extracts values from a specific location across multiple time series arrays.

    This function iterates through a list of NumPy arrays, each representing a time series,
    and retrieves the value located at the specified row and column index.

    Args:
        time_series_arrays (list of np.ndarray):
            A list of NumPy arrays, where each array represents a time series.
        column_index (int):
            The column index from which to extract values.
        row_index (int):
            The row index from which to extract values.
        dtype (data-type, optional):
            Desired data type for the output array. If None, infers dtype from 
            the input arrays.

    Returns:
        np.ndarray:
            A NumPy array containing the extracted values from each time series.

    Raises:
        IndexError:
            If the provided row or column index is out of bounds for any of 
            the arrays.
    """
    
    # Return an empty array if no input is provided
    if not time_series_arrays:
        return np.array([])

    # Infer data type if not explicitly provided
    inferred_dtype = (
        dtype 
        if dtype is not None 
        else time_series_arrays[0][row_index, column_index].dtype
    )

    num_time_series = len(time_series_arrays)
    
    # Initialize an empty array with specified/inferred dtype
    extracted_values = np.empty(num_time_series, dtype=inferred_dtype)

    for i, arr in enumerate(time_series_arrays):
        try:
            extracted_values[i] = arr[row_index, column_index]
        except IndexError:
            raise IndexError(f"IndexError: Row or column index out of bounds for array at index {i}.")

    return extracted_values


def extract_timeseries_by_column_and_row_index(
        time_series_arrays: list[np.ndarray],
        column_indices: int | list[int] | np.ndarray,
        row_index: int,
        *,
        dtype=None,
        ) -> np.ndarray:
    """
    Extracts values from specific locations across multiple time series arrays.

    This function iterates through a list of NumPy arrays, each representing a time series,
    and retrieves the values located at the specified row index and multiple column indices.
    If only one column index is provided (either as an int or a single-element list),
    it delegates to `_extract_timeseries_of_one_column_and_row_index`.

    Args:
        time_series_arrays (list of np.ndarray): A list of NumPy arrays, where each array
                                                   represents a time series.
        column_indices (int or list/np.ndarray): A single column index or a list/array of column indices
                                                 from which to extract values.
        row_index (int): The row index from which to extract values.
        dtype (data-type, optional): Desired data type for the output array. If None,
                                     infers dtype from the input arrays.

    Returns:
        np.ndarray: A 1D NumPy array if one column index is provided as an integer. Otherwise,
                    for multiple columns or if one column is provided as a list,
                    returns a 2D NumPy array. Each row corresponds to one time series,
                    and each column corresponds to a value from `column_indices`.

    Raises:
        IndexError: If any provided row or column index is out of bounds for any of the arrays.
    """
    
    # Handle single-column case when passed as an integer
    if isinstance(column_indices, int):
        return _extract_timeseries_of_one_column_and_row_index(
            time_series_arrays,
            column_indices,
            row_index,
            dtype=dtype
        )

    # Handle single-column case when passed as a single-element list
    if isinstance(column_indices, (list, np.ndarray)) and len(column_indices) == 1:
        extracted_values = _extract_timeseries_of_one_column_and_row_index(
            time_series_arrays,
            column_indices[0],  # Extract the first (and only) element in the list
            row_index,
            dtype=dtype
        )
        # Convert result into 2D array with shape (num_time_series x 1)
        return extracted_values[:, np.newaxis]

    # Handle multi-column case
    if not time_series_arrays:
        return np.array([])  # Return an empty array if the input list is empty

    num_time_series = len(time_series_arrays)
    num_columns = len(column_indices)
    
    # Infer dtype if not explicitly provided
    inferred_dtype = (
        dtype 
        if dtype is not None 
        else time_series_arrays[0][row_index, column_indices[0]].dtype
    )
    
    # Initialize an empty 2D array for extracted values with specified/inferred dtype
    extracted_values = np.empty((num_time_series, num_columns), dtype=inferred_dtype)

    for i, arr in enumerate(time_series_arrays):
        try:
            extracted_values[i] = arr[row_index, column_indices]
        except IndexError:
            raise IndexError(f"IndexError: Row or column index out of bounds for array at index {i}.")

    return extracted_values


def extract_timeseries_by_row_index(
        data: dict,
        column_labels: str | list[str],
        row_index: int,
        *,
        dtype=None,
    ) -> np.ndarray:
    """
    Wrapper for 'extract_timeseries_by_column_and_row_index' that uses column labels instead of indices.

    This function accepts a dictionary containing time series arrays and their corresponding
    column labels, allowing extraction of values using one or more column labels.

    Args:
        data (dict): Dictionary with two keys:
            - "timeseries" (list[np.ndarray]): List of NumPy arrays representing time series data.
            - "labels" (list[str]): List of column labels corresponding to the columns in the arrays.
        column_labels (str or list[str]): The label(s) of the columns to extract values from. Can be a single label or a list of labels.
        row_index (int): The row index from which to extract values.
        dtype (data-type, optional): Desired data type for the output array. If None,
                                     infers dtype from the input arrays.

    Returns:
        np.ndarray: A 1D NumPy array if one label is provided as a string. Otherwise,
                    for multiple labels, returns a 2D NumPy array. Each row corresponds
                    to one time series, and each column corresponds to a value from `column_labels`.

    Raises:
        KeyError: If any label in 'column_labels' is not found in 'data["labels"]'.
        IndexError: If the provided row index is out of bounds for any of the arrays.
    """
    
    # Ensure both keys are present in the dictionary
    if "timeseries" not in data or "labels" not in data:
        raise ValueError("The 'data' dictionary must contain 'timeseries' and 'labels' keys.")
    
    # Normalize `column_labels` to always be a list
    if isinstance(column_labels, str):
        column_labels = [column_labels]
    
    # Find indices corresponding to all provided column labels
    try:
        column_indices = [data["labels"].index(label) for label in column_labels]
    except ValueError as e:
        missing_label = next(label for label in column_labels if label not in data["labels"])
        raise KeyError(f"The label '{missing_label}' was not found in 'data['labels']'.") from e
    
    # Call original function using extracted indices and timeseries
    return extract_timeseries_by_column_and_row_index(
                time_series_arrays=data["timeseries"],
                column_indices=column_indices,
                row_index=row_index,
                dtype=dtype
            )


def extract_timeseries_by_column_index(
        arrays,
        column_indices,
        y_coord,
        z_coord,
        *,
        return_closest=False,
        dtype=None,
        ):
    """
    Extracts timeseries data from a list of numpy arrays, where the first two
    columns of each array represent y and z coordinates. The function retrieves
    the data from the specified column index/indices at the row corresponding to
    the given y and z coordinates. It supports finding either an exact match or 
    the closest match for y and z coordinates based on 'return_closest'.

    Args:
        arrays (list of np.ndarray): List of numpy arrays (timeseries).
        column_indices (int or list[int] or np.ndarray): The column index/indices to extract data from.
        y_coord (float or int): The y-coordinate to filter by.
        z_coord (float or int): The z-coordinate to filter by.
        return_closest (bool, optional): If True, extracts data from the row with the closest matching 
                                         coordinates if no exact match is found. Default is False.
        dtype (data-type, optional): Desired data type for the output array. If None,
                                     infers dtype from the input arrays.

    Returns:
        np.ndarray or None: A NumPy array containing the extracted data, or None
                            if no matching row is found in any of the timeseries.
                            Returns a 1D array if one column index is provided as an integer;
                            otherwise returns a 2D array for multiple columns.

    Raises:
        ClosestMatchFound: If no exact match is found and 'return_closest' is False.
                           This exception provides details about the closest match.
    """

    # Find matching row index based on y and z coordinates
    try:
        row_index = find_matching_row_index_from_timeseries(
            arrays=arrays,
            y_coord=y_coord,
            z_coord=z_coord,
            return_closest=return_closest
        )
        
    except ClosestMatchFound as e:
        # Log details about why no exact match was found and re-raise exception
        print(f"No exact match found for coordinates ({e.provided_y}, {e.provided_z}). "
              f"Closest match at index {e.index} with values "
              f"(y={e.closest_y}, z={e.closest_z}).")
        
        raise

    # If no matching row was found in any scenario, return None
    if row_index is None:
        return None

    # Use updated extract_timeseries_by_column_and_row_index function to handle single/multiple columns flexibly
    return extract_timeseries_by_column_and_row_index(
        time_series_arrays=arrays,
        column_indices=column_indices,
        row_index=row_index,
        dtype=dtype,
    )


def extract_timeseries(
        data,
        column_labels,
        y_coord,
        z_coord,
        *,
        return_closest=False,
        dtype=None,
    ):
    """
    Extended wrapper for 'extract_timeseries_by_column_index' function.
    This function accepts a dictionary containing timeseries data and labels, 
    and uses one or more column labels instead of column indices.

    Args:
        data (dict): Dictionary with two keys:
            - "timeseries" (list of np.ndarray): List of numpy arrays (timeseries).
            - "labels" (list[str]): List of column labels corresponding to the columns in the arrays.
        column_labe (str or list[str]): The label(s) of the columns to extract data from. Can be a single label or a list of labels.
        y_coord (float or int): The y-coordinate to filter by.
        z_coord (float or int): The z-coordinate to filter by.
        return_closest (bool, optional): If True, extracts data from the row with the closest matching 
                                         coordinates if no exact match is found. Default is False.
        dtype (data-type, optional): Desired data type for the output array. If None,
                                     infers dtype from the input arrays.

    Returns:
        np.ndarray or None: A NumPy array containing the extracted data, or None
                            if no matching row is found in any of the timeseries.

    Raises:
        KeyError: If any label in 'column_labels' is not found in 'data["labels"]'.
    """
    
    # Ensure both keys are present in the dictionary
    if "timeseries" not in data or "labels" not in data:
        raise ValueError("The 'data' dictionary must contain 'timeseries' and 'labels' keys.")
    
    # Normalize column_labels to always be a list
    if isinstance(column_labels, str):
        column_labels = [column_labels]
    
    # Find indices corresponding to all provided column labels
    try:
        column_indices = [data["labels"].index(label) for label in column_labels]
    except ValueError as e:
        missing_label = next(label for label in column_labels if label not in data["labels"])
        raise KeyError(f"The label '{missing_label}' was not found in 'data['labels']'.") from e
    
    # Call original function using extracted indices and timeseries
    return extract_timeseries_by_column_index(
                arrays=data["timeseries"],
                column_indices=column_indices,
                y_coord=y_coord,
                z_coord=z_coord,
                return_closest=return_closest,
                dtype=dtype
            )


def plot_timeseries(
        data_list,
        *,
        labels=None,
        xlabel='Time',
        ylabel='Value',
        linewidth=1,
        y_limits=None,
    ):
    # Check if data_list is a single 1D array or list
    if not isinstance(data_list, list):
        data_list = [data_list]  # Wrap in a list if it's not already
    
    # Create an array for the x-axis (time)
    time = np.arange(len(data_list[0]))  # Assuming all data series have the same length
    
    # Create a figure for plotting
    plt.figure(figsize=(10, 5))
    
    # Plot each time series in the provided list
    for i, data in enumerate(data_list):
        label = labels[i] if labels is not None else f"Timeseries {i+1}"
        plt.plot(time, data, linewidth=linewidth, label=label)
    
    plt.title('Multiple Timeseries')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if y_limits is not None:  # Set y-axis limits if provided
        plt.ylim(y_limits)

    plt.grid()
    plt.legend()  # Show legend to identify different timeseries
    plt.show()
