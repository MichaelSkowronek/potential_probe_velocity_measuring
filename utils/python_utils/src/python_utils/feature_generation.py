# -*- coding: UTF-8 -*-
"""Feature Generation.
"""


def symmetric_difference_coefficient(
        *,
        former_point,
        latter_point,
        delta_l,
        ):
    delta_f = latter_point - former_point
    sdc = delta_f / delta_l
    return sdc
