# -*- coding: UTF-8 -*-
"""Utils for sigma (conductivity) and B (magnetic field strength) estimation.
"""


import numpy as np


def calc_conductivity_with_z_at_site(
        partial_phi_data,
        partial_phi_label,
        current_data,
        current_label,
        y_idx,
        z_idx,
        *,
        print_statistics=False,
    ):
    """
    Calculate the conductivity (Sigma) at a specific site using z-component data.

    This function computes the conductivity by dividing the z-component 
    of the current density (`j_z_center`) by the z-component of the 
    partial derivative of phi (`part_phi_z`) at a given spatial index 
    (`y_idx`, `z_idx`). Optionally, it can print statistics such as mean, 
    standard deviation, and median of the calculated conductivity values.

    Args:
        partial_phi_data (dict): A dictionary containing timeseries data 
            and corresponding labels for partial derivatives of phi. 
            The key 'timeseries' refers to a multidimensional array.
        partial_phi_label (str): The label identifying which component 
            of `partial_phi_data` to use for calculation.
        current_data (dict): A dictionary containing timeseries data and 
            corresponding labels for current density. The key 'timeseries' 
            refers to a multidimensional array.
        current_label (str): The label identifying which component of 
            `current_data` to use for calculation.
        y_idx (int): The index along the y-axis specifying the spatial site.
        z_idx (int): The index along the z-axis specifying the spatial site.
        print_statistics (bool, optional): If True, prints statistical metrics
            (mean, standard deviation, and median) of computed conductivity values. 
            Defaults to False.

    Returns:
        np.ndarray: An array representing the estimated conductivity values
            (`sigma_est`) computed as j_z_center / part_phi_z.

    Prints:
        If `print_statistics` is True:
          - Mean of conductivity estimates.
          - Standard deviation of conductivity estimates.
          - Median of conductivity estimates.
    """
    part_phi_z = partial_phi_data['timeseries'][
        :,
        y_idx,
        z_idx,
        partial_phi_data['labels'].index(partial_phi_label),
    ]
    j_z_center = current_data['timeseries'][
        :,
        y_idx,
        z_idx,
        current_data['labels'].index(current_label),
    ]
    sigma_est = np.divide(
        j_z_center,
        part_phi_z,
    )
    
    mean = np.mean(sigma_est)
    std = np.std(sigma_est)
    median = np.median(sigma_est)

    if print_statistics > 0:
        print(f"mean(conductivity estimate) = {mean}")
        print(f"std(conductivity estimate) = {std}")
        print(f"median(conductivity estimate) = {median}")

    return sigma_est


def calc_conductivity_with_z(
        partial_phi_data,
        partial_phi_label,
        current_data,
        current_label,
    ):
    """
    Calculate the conductivity (Sigma) using z-component data.

    This function computes the conductivity by dividing the z-component 
    of the current density (`j_z_center`) by the z-component of the 
    partial derivative of phi (`part_phi_z`).

    Args:
        partial_phi_data (dict): A dictionary containing timeseries data 
            and corresponding labels for partial derivatives of phi. 
            The key 'timeseries' refers to a multidimensional array.
        partial_phi_label (str): The label identifying which component 
            of `partial_phi_data` to use for calculation.
        current_data (dict): A dictionary containing timeseries data and 
            corresponding labels for current density. The key 'timeseries' 
            refers to a multidimensional array.
        current_label (str): The label identifying which component of 
            `current_data` to use for calculation.

    Returns:
        np.ndarray: An array representing the estimated conductivity values 
            (`sigma_est`) computed as j_z_center / part_phi_z.
    """
    part_phi_z = partial_phi_data['timeseries'][
        :,
        :,
        :,
        partial_phi_data['labels'].index(partial_phi_label),
    ]
    j_z_center = current_data['timeseries'][
        :,
        :,
        :,
        current_data['labels'].index(current_label),
    ]
    sigma_est = np.divide(
        j_z_center,
        part_phi_z,
    )
    return sigma_est
