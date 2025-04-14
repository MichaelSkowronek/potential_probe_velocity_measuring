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
    points_nd,
    direction_axis,
    coord_nd,
):
    """
    Computes symmetric difference coefficients for an n-dimensional numpy array of points,
    given the coordinate array. Works for both 3D and 4D input.

    Note: Periodic boundary conditions are assumed to calculate the sdc at the boundaries.

    Args:
        points_nd (np.ndarray): An n-dimensional numpy array containing point values.
                                 Shape can be (x, y, z) for a single sample or (batch, x, y, z) for a batch.
        direction_axis (int): The axis index along which to compute the coefficients.
                               If points_nd is 3D, axis can be 0, 1, or 2.
                               If points_nd is 4D, axis can be 1, 2, or 3.
        coord_nd (np.ndarray): An n-dimensional numpy array of the same shape as points_nd
                                 containing coordinate values.

    Returns:
        np.ndarray: An n-dimensional numpy array containing symmetric difference coefficients.
    """
    ndim = points_nd.ndim
    if ndim not in [3, 4]:
        raise ValueError("Input 'points_nd' must be a 3D or 4D array.")

    if ndim == 3:
        if not 0 <= direction_axis <= 2:
            raise ValueError("For a 3D array, direction_axis must be 0, 1, or 2.")
    else:  # ndim == 4
        if not 1 <= direction_axis <= 3:
            raise ValueError("For a 4D array, direction_axis must be 1, 2, or 3.")

    shifted_positive_points = np.roll(points_nd, -1, axis=direction_axis)
    shifted_negative_points = np.roll(points_nd, 1, axis=direction_axis)

    shifted_positive_coords = np.roll(coord_nd, -1, axis=direction_axis)
    shifted_negative_coords = np.roll(coord_nd, 1, axis=direction_axis)

    delta_f = shifted_positive_points - shifted_negative_points
    delta_l = np.abs(shifted_positive_coords - shifted_negative_coords)

    sdc_array = delta_f / delta_l
    return sdc_array


def compute_symmetric_difference_coefficients_with_index_to_coord_map(
    points_nd,
    direction_axis,
    index_to_coord_map,
):
    """
    Computes symmetric difference coefficients for an n-dimensional numpy array of points,
    using an index-to-coordinate map. Works for both 3D and 4D input.

    Args:
        points_nd (np.ndarray): An n-dimensional numpy array containing point values.
                                 Shape can be (x, y, z) for a single sample or (batch, x, y, z) for a batch.
        direction_axis (int): The axis index along which to compute the coefficients.
                               If points_nd is 3D, axis can be 0, 1, or 2.
                               If points_nd is 4D, axis can be 1, 2, or 3.
        index_to_coord_map (dict): A dictionary mapping indices (as tuples) to their respective
                                     coordinates.

    Returns:
        np.ndarray: An n-dimensional numpy array containing symmetric difference coefficients.
    """
    ndim = points_nd.ndim
    if ndim not in [3, 4]:
        raise ValueError("Input 'points_nd' must be a 3D or 4D array.")

    if ndim == 3:
        if not 0 <= direction_axis <= 2:
            raise ValueError("For a 3D array, direction_axis must be 0, 1, or 2.")
        shape = points_nd.shape
        coord_nd = generate_coord_nd(
            shape,
            index_to_coord_map,
            coord_index=direction_axis,
        )
    else:  # ndim == 4
        if not 1 <= direction_axis <= 3:
            raise ValueError("For a 4D array, direction_axis must be 1, 2, or 3.")
        sample_shape = points_nd.shape[1:]
        coord_3d_sample = generate_coord_nd(
            sample_shape,
            index_to_coord_map,
            coord_index=direction_axis - 1,  # Adjust for 3D map
        )
        coord_nd = np.repeat(coord_3d_sample[np.newaxis, ...], points_nd.shape[0], axis=0)

    sdc_array = compute_symmetric_difference_coefficients(
        points_nd,
        direction_axis,
        coord_nd,
    )
    return sdc_array
