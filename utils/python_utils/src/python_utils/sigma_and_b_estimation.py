# -*- coding: UTF-8 -*-
"""Utils for sigma (conductivity) and B (magnetic field strength) estimation.
"""


import numpy as np


from .time_series_extraction import extract_timeseries


def calc_conductivity_with_z_at_site(
        partial_phi_data,
        partial_phi_label,
        current_data,
        current_label,
        y_coord,
        z_coord,
        *,
        print_statistics=False,
        ):
    """_summary_

    Sigma = conductivity.

    Args:
        partial_phi_data (_type_): _description_
        partial_phi_label (_type_): _description_
        current_data (_type_): _description_
        current_label (_type_): _description_
        y_coord (_type_): _description_
        z_coord (_type_): _description_
    """
    part_phi_z = extract_timeseries(
        data=partial_phi_data,
        column_labels=partial_phi_label,
        y_coord=y_coord,
        z_coord=z_coord,
        )
    j_z_center = extract_timeseries(
        data=current_data,
        column_labels=current_label,
        y_coord=y_coord,
        z_coord=z_coord,
        )
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



