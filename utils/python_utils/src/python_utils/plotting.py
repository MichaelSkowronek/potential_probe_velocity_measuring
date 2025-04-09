# -*- coding: UTF-8 -*-
"""Plotting Utils.
"""


import numpy as np
from matplotlib import pyplot as plt


def plot_timeseries(
        data_list,
        *,
        labels=None,
        xlabel='Time',
        ylabel='Value',
        linewidth=1,
        y_limits=None,
    ):
    # Check if data_list is a single 1D array or list
    if not isinstance(data_list, list):
        data_list = [data_list]  # Wrap in a list if it's not already
    
    # Create an array for the x-axis (time)
    time = np.arange(len(data_list[0]))  # Assuming all data series have the same length
    
    # Create a figure for plotting
    plt.figure(figsize=(10, 5))
    
    # Plot each time series in the provided list
    for i, data in enumerate(data_list):
        label = labels[i] if labels is not None else f"Timeseries {i+1}"
        plt.plot(time, data, linewidth=linewidth, label=label)
    
    plt.title('Multiple Timeseries')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if y_limits is not None:  # Set y-axis limits if provided
        plt.ylim(y_limits)

    plt.grid()
    plt.legend()  # Show legend to identify different timeseries
    plt.show()


def plot_heatmap(
    data,
    *,
    title='2D Heat Map',
    x_label='X-axis',
    y_label='Y-axis',
    index_to_space_dict=None,
    round_decimals=None,
    max_ticks=None,
):
    """
    Plots a 2D heatmap from a provided 2D NumPy array.

    Parameters:
        data (np.ndarray): A 2D NumPy array containing the data to be plotted.
        title (str): Title of the heat map.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        index_to_space_dict (dict): A dictionary where keys are tuple indices 
                                    (i, j) in 'data' and values are tuple 
                                    spatial coordinates (x, y).
                                    Example: {(0, 0): (-10.12345, -100.54321), ...}
        round_decimals (int or None): Number of decimal places to round axis tick values.
                                      If None, no rounding is applied.
        max_ticks (int or None): Maximum number of ticks to display on each axis. 
                                 If None, all unique ticks are displayed.
                                 Includes zero if it's within range.
    """
    
    if index_to_space_dict is not None:
        # Extract unique x and y spatial positions from the dictionary
        x_positions = sorted({v[0] for v in index_to_space_dict.values()})  # Unique X-values sorted
        y_positions = sorted({v[1] for v in index_to_space_dict.values()})  # Unique Y-values sorted
        
        # Apply rounding if specified
        if round_decimals is not None:
            x_positions = [round(x, round_decimals) for x in x_positions]
            y_positions = [round(y, round_decimals) for y in y_positions]

        # Ensure zero is included in ticks if it's within range
        if min(x_positions) <= 0 <= max(x_positions):
            x_positions.append(0)
            x_positions = sorted(x_positions)
        
        if min(y_positions) <= 0 <= max(y_positions):
            y_positions.append(0)
            y_positions = sorted(y_positions)

        # Reduce number of ticks if max_ticks is specified
        if max_ticks is not None:
            def reduce_ticks_with_zero(positions):
                indices = np.linspace(0, len(positions) - 1, min(len(positions), max_ticks), dtype=int).tolist()
                zero_index = positions.index(0) if 0 in positions else None

                if zero_index is not None and zero_index not in indices:
                    indices.append(zero_index)
                    indices.sort()

                return [positions[i] for i in indices], indices
            
            x_reduced_ticks, x_indices = reduce_ticks_with_zero(x_positions)
            y_reduced_ticks, y_indices = reduce_ticks_with_zero(y_positions)

            plt.xticks(ticks=x_indices, labels=x_reduced_ticks)
            plt.yticks(ticks=y_indices, labels=y_reduced_ticks)

        else:
            plt.xticks(ticks=np.arange(len(x_positions)), labels=x_positions)
            plt.yticks(ticks=np.arange(len(y_positions)), labels=y_positions)

    # Plot heatmap
    plt.imshow(
        data,
        cmap='hot',
        interpolation='nearest'
    )
    
    plt.colorbar()  
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
