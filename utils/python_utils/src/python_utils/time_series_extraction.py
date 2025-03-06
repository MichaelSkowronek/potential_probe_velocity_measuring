# -*- coding: UTF-8 -*-
"""Time Series Extraction.
"""


import numpy as np


class ClosestMatchFound(Exception):
    """Custom exception to indicate a closest match was found."""
    def __init__(self, index, closest_y, closest_z, provided_y, provided_z):
        self.index = index
        self.closest_y = closest_y
        self.closest_z = closest_z
        self.provided_y = provided_y
        self.provided_z = provided_z
        super().__init__(f"Closest match found at index {index}, closest coordinates (y={closest_y}, z={closest_z}), provided coordinates (y={provided_y}, z={provided_z})")


def find_matching_row_index_from_timeseries(arrays, y_coord, z_coord, *, return_closest=False):
    """
    Finds the row index in the first numpy array of a timeseries where the y and z coordinates match.
    If no exact match is found, it either returns the index of the closest match or raises an exception, based on the 'return_closest' parameter.

    Args:
        arrays (list of np.ndarray): A list of numpy arrays (timeseries). Each array is expected to have at least two columns: y and z.
        y_coord (float or int): The y-coordinate to filter by.
        z_coord (float or int): The z-coordinate to filter by.
        return_closest (bool, optional): If True, returns the index of the closest match if no exact match is found. If False (default), raises ClosestMatchFound.

    Returns:
        int or None: The row index of the matching row in the first array, or None if the array list or first array is empty.
        int: if return_closest is True, and no exact match is found, the index of the closest match is returned.

    Raises:
        ClosestMatchFound: If no exact match is found and return_closest is False.
    """

    if not arrays:  # Check if the list is empty
        return None

    first_array = arrays[0]

    if first_array.size == 0 or first_array.shape[0] == 0:  # Check if the array is empty
        return None

    y_values = first_array[:, 0]
    z_values = first_array[:, 1]

    matching_indices = np.where((y_values == y_coord) & (z_values == z_coord))[0]

    if matching_indices.size > 0:
        return matching_indices[0]  # Return the index of the first matching row
    else:
        # Find closest matching row
        y_diff = np.abs(y_values - y_coord)
        z_diff = np.abs(z_values - z_coord)
        total_diff = y_diff + z_diff
        closest_index = np.argmin(total_diff)
        closest_y = y_values[closest_index]
        closest_z = z_values[closest_index]
        if return_closest:
            return closest_index
        else:
            raise ClosestMatchFound(closest_index, closest_y, closest_z, y_coord, z_coord)


def extract_time_series_values(time_series_arrays, column_index, row_index):
    """
    Extracts values from a specific location across multiple time series arrays.

    This function iterates through a list of NumPy arrays, each representing a time series,
    and retrieves the value located at the specified row and column index.

    Args:
        time_series_arrays (list of np.ndarray): A list of NumPy arrays, where each array
                                                   represents a time series.
        column_index (int): The column index from which to extract values.
        row_index (int): The row index from which to extract values.

    Returns:
        np.ndarray: A NumPy array containing the extracted values from each time series.

    Raises:
        IndexError: If the provided row or column index is out of bounds for any of the arrays.
    """
    if not time_series_arrays:
        return np.array([])  # Return an empty array if the input list is empty

    num_time_series = len(time_series_arrays)
    dtype = time_series_arrays[0][row_index, column_index].dtype
    extracted_values = np.empty(num_time_series, dtype=dtype)

    for i, arr in enumerate(time_series_arrays):
        try:
            extracted_values[i] = arr[row_index, column_index]
        except IndexError:
            raise IndexError(f"IndexError: Row or column index out of bounds for array at index {i}.")

    return extracted_values


def extract_timeseries_data_by_coordinates_and_column(arrays, column_index, y_coord, z_coord):
    """
    Extracts timeseries data from a list of numpy arrays, where the first two
    columns of each array represent y and z coordinates. The function retrieves
    the data from the specified column index, at the row corresponding to the
    given y and z coordinates. It returns a NumPy array containing the extracted
    data, assuming no missing data and that the row and column are present in all
    arrays.

    Args:
        arrays (list of np.ndarray): List of numpy arrays (timeseries).
        column_index (int): The column index to extract data from.
        y_coord (float or int): The y-coordinate to filter by.
        z_coord (float or int): The z-coordinate to filter by.

    Returns:
        numpy.ndarray or None: A NumPy array containing the extracted data, or None
                               if no matching row is found in the first
                               array of the timeseries.
    """

    row_index = find_matching_row_index_from_timeseries(arrays, y_coord, z_coord)
    if row_index is None:
        return None

    return extract_time_series_values(arrays, column_index, row_index)