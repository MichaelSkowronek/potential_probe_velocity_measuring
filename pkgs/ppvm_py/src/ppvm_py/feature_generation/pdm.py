# -*- coding: UTF-8 -*-
"""Feature Generation.
"""


from ppvm_py.feature_generation.sdc import symmetric_difference_coefficient
from ppvm_py.feature_generation.sdc import compute_symmetric_difference_coefficients
from ppvm_py.feature_generation.sdc import compute_symmetric_difference_coefficients_with_index_to_coord_map


def pdm_approximated_velocity(
        former_point,
        latter_point,
        delta_l,
        B,
        ):
    """Calculate potential-difference-method approximated velocity.

    Args:
        former_point (_type_): _description_
        latter_point (_type_): _description_
        delta_l (_type_): _description_
        B (_type_): Scale (reciprocal), usually the magnetic field strength.

    Returns:
        _type_: pdm_approximated_velocity
    """
    sdc = symmetric_difference_coefficient(
        former_point=former_point,
        latter_point=latter_point,
        delta_l=delta_l,
        )
    pdm_approximated_velocity = sdc / B
    return pdm_approximated_velocity


def compute_pdm_approximated_velocities(
    points_nd,
    direction_axis,
    coord_nd,
    B,
):
    """
    Computes potential-difference-method approximated velocities for an n-dimensional
    numpy array of points, given the coordinate array and the magnetic field strength.
    Works for both 3D and 4D input.

    Args:
        points_nd (np.ndarray): An n-dimensional numpy array containing point values.
                                    Shape can be (x, y, z) for a single sample or (batch, x, y, z) for a batch.
        direction_axis (int): The axis index along which to compute the velocities.
                                If points_nd is 3D, axis can be 0, 1, or 2.
                                If points_nd is 4D, axis can be 1, 2, or 3.
        coord_nd (np.ndarray): An n-dimensional numpy array of the same shape as points_nd
                                    containing coordinate values.
        B (float or np.ndarray): The magnetic field strength (reciprocal scale).
                                  Can be a scalar applied to all velocities or an array
                                  of the same shape as the output sdc_array.

    Returns:
        np.ndarray: An n-dimensional numpy array containing PDM approximated velocities.
    """
    sdc_array = compute_symmetric_difference_coefficients(
        points_nd,
        direction_axis,
        coord_nd,
    )
    pdm_velocity_array = sdc_array / B
    return pdm_velocity_array


def compute_pdm_approximated_velocities_with_index_to_coord_map(
    points_nd,
    direction_axis,
    index_to_coord_map,
    B,
):
    """
    Computes potential-difference-method approximated velocities for an n-dimensional
    numpy array of points, using an index-to-coordinate map and the magnetic field strength.
    Works for both 3D and 4D input.

    Args:
        points_nd (np.ndarray): An n-dimensional numpy array containing point values.
                                    Shape can be (x, y, z) for a single sample or (batch, x, y, z) for a batch.
        direction_axis (int): The axis index along which to compute the velocities.
                                If points_nd is 3D, axis can be 0, 1, or 2.
                                If points_nd is 4D, axis can be 1, 2, or 3.
        index_to_coord_map (dict): A dictionary mapping indices (as tuples) to their respective
                                    coordinates.
        B (float or np.ndarray): The magnetic field strength (reciprocal scale).
                                  Can be a scalar applied to all velocities or an array
                                  of the same shape as the output velocity array.

    Returns:
        np.ndarray: An n-dimensional numpy array containing PDM approximated velocities.
    """
    sdc_array = compute_symmetric_difference_coefficients_with_index_to_coord_map(
        points_nd,
        direction_axis,
        index_to_coord_map,
    )
    pdm_velocity_array = sdc_array / B
    return pdm_velocity_array
