# -*- coding: UTF-8 -*-
"""YZSlice data utils.
"""


import numpy as np



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

