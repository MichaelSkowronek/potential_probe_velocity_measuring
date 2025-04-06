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
    """_summary_

    Sigma = conductivity.

    Args:
        partial_phi_data (_type_): _description_
        partial_phi_label (_type_): _description_
        current_data (_type_): _description_
        current_label (_type_): _description_
        y_idx (_type_): _description_
        z_idx (_type_): _description_
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
