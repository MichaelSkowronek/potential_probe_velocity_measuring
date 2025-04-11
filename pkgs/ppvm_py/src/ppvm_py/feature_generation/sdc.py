# -*- coding: UTF-8 -*-
"""Feature Generation.
"""

import numpy as np

from ppvm_py.data_processing.utils import generate_coord_nd


def symmetric_difference_coefficient(
        former_point,
        latter_point,
        delta_l,
        ):
    delta_f = latter_point - former_point
    sdc = delta_f / delta_l
    return sdc


def compute_symmetric_difference_coefficients(
    points_3d,
    direction_axis,
    coord_3d,
):
    """
    Computes symmetric difference coefficients for a 3D numpy array of points,
    given the coordinate array.

    Args:
        points_3d (np.ndarray): A 3D numpy array of shape (x, y, z) containing point values.
        direction_axis (int): The axis index (0 for x, 1 for y, 2 for z) along which to compute the coefficients.
        coord_3d (np.ndarray): A 3D numpy array of the same shape as points_3d containing coordinate values.

    Returns:
        np.ndarray: A 3D numpy array containing symmetric difference coefficients.
    """
    # Shift the matrix in positive and negative directions along the given axis
    shifted_positive_points = np.roll(points_3d, -1, axis=direction_axis)
    shifted_negative_points = np.roll(points_3d, 1, axis=direction_axis)

    shifted_positive_coords = np.roll(coord_3d, -1, axis=direction_axis)
    shifted_negative_coords = np.roll(coord_3d, 1, axis=direction_axis)

    # Compute delta_f by subtracting matrices of points
    delta_f = shifted_positive_points - shifted_negative_points

    # Compute delta_l by subtracting matrices of coordinates
    delta_l = np.abs(shifted_positive_coords - shifted_negative_coords)

    # Compute symmetric difference coefficient without masking or error handling
    sdc_array = delta_f / delta_l

    return sdc_array


def compute_symmetric_difference_coefficients_with_index_to_coord_map(
    points_3d,
    direction_axis,
    index_to_coord_map,
):
    """
    Computes symmetric difference coefficients for a 3D numpy array of points.

    Args:
        points_3d (np.ndarray): A 3D numpy array of shape (x, y, z) containing point values.
        direction_axis (int): The axis index (0 for x, 1 for y, 2 for z) along which to compute the coefficients.
        index_to_coord_map (dict): A dictionary mapping indices (x, y, z) to their respective coordinates.

    Returns:
        np.ndarray: A 3D numpy array containing symmetric difference coefficients.
    """

    # Generate the coordinate matrix using the helper function
    coord_3d = generate_coord_nd(
        points_3d.shape,
        index_to_coord_map,
        coord_index=direction_axis,
    )

    # Call the now public function with the generated coordinate array
    sdc_array = compute_symmetric_difference_coefficients(
        points_3d,
        direction_axis,
        coord_3d,
    )

    return sdc_array
