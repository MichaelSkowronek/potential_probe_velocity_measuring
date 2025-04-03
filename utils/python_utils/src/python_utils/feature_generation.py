# -*- coding: UTF-8 -*-
"""Feature Generation.
"""


def symmetric_difference_coefficient(
        former_point,
        latter_point,
        delta_l,
        ):
    delta_f = latter_point - former_point
    sdc = delta_f / delta_l
    return sdc


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
