#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:42:14 2024

@author: ozanbaris
"""

import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
def read_results_from_csv(filename):
    """
    Read optimization results from a CSV file, skipping the 'Parameters' column,
    and convert them back into a nested dictionary format.

    Args:
        filename (str): The path to the CSV file to be read.

    Returns:
        dict: A dictionary containing the optimization results structured by sensor count and house ID.
    """
    results_df = pd.read_csv(filename)
    results_dict = {}

    for _, row in results_df.iterrows():
        sensor_count = row['Sensor Count']
        house_id = row['House ID']
        sensor = row['Sensor']
        # Skip the 'Parameters' column
        train_rmse = row['Training RMSE']
        test_rmse = row['Testing RMSE']
        
        # Initialize nested dictionary if not exist
        if sensor_count not in results_dict:
            results_dict[sensor_count] = {}
        if house_id not in results_dict[sensor_count]:
            results_dict[sensor_count][house_id] = {
                'train_errors': {},
                'test_errors': {}
            }
        
        # Populate the dictionary
        results_dict[sensor_count][house_id]['train_errors'][sensor] = train_rmse
        results_dict[sensor_count][house_id]['test_errors'][sensor] = test_rmse
    
    return results_dict


filename = "optimization_results.csv"
results_lm = read_results_from_csv(filename)

filename = "onestep_results.csv"
results_onestep= read_results_from_csv(filename)
#%%
def extract_errors_by_sensor_count(results_dict):
    """
    Extract training and testing errors by sensor count from results dictionary.

    Args:
    - results_dict (dict): Dictionary containing results data for either results_onestep or results_lm.

    Returns:
    - tuple: training_errors_by_sensor_count, testing_errors_by_sensor_count
    """
    training_errors_by_sensor_count = {}
    testing_errors_by_sensor_count = {}

    # Loop through the results dictionary to collect training and testing errors
    for sensor_count, results in results_dict.items():
        training_errors = {}
        testing_errors = {}

        # Loop through each house and its results
        for house_id, house_results in results.items():
            training_errors[house_id] = house_results['train_errors']
            testing_errors[house_id] = house_results['test_errors']

        training_errors_by_sensor_count[sensor_count] = training_errors
        testing_errors_by_sensor_count[sensor_count] = testing_errors

    return training_errors_by_sensor_count, testing_errors_by_sensor_count

training_errors_onestep, testing_errors_onestep = extract_errors_by_sensor_count(results_onestep)
training_errors_lm, testing_errors_lm = extract_errors_by_sensor_count(results_lm)

#%%



def plot_combined_errors(training_errors_onestep, testing_errors_onestep, training_errors_lm, testing_errors_lm, legend_loc=(0.5, 0.99)):
    fig, axs = plt.subplots(2, 1, figsize=(11, 11))
    datasets = [testing_errors_onestep, testing_errors_lm]
    descriptions = ['Testing Errors (One Step)', 'Testing Errors (One Day)']

    # Define hatches with distinctive patterns
    hatches = ['/', '\\', 'o', '*', '+', 'x']  # Six distinct hatches

    # Prepare to synchronize hatches in legend
    sensor_hatches = {sensor: hatches[i % len(hatches)] for i, sensor in enumerate(sensor_columns)}

    for ax, errors, desc in zip(axs, datasets, descriptions):
        all_positions = []
        offset = 0

        sensor_counts = sorted(errors.keys())
        sensor_count_positions = []

        # First plot the boxplots
        for sensor_count in sensor_counts:
            error_data = errors[sensor_count]
            sensor_errors = {sensor: [error_data[house_id][sensor] for house_id in error_data] for sensor in sensor_columns}
            sorted_sensors = sorted(sensor_errors, key=lambda x: np.mean(sensor_errors[x]))

            positions = np.array(range(len(sorted_sensors))) + offset
            all_positions.extend(positions)
            offset += len(sensor_columns) + 1  # Adjust offset as needed

            # Draw boxplots and apply thick hatches
            for i, sensor in enumerate(sorted_sensors):
                box = ax.boxplot(sensor_errors[sensor], positions=[positions[i]], widths=1, notch=True,
                                 patch_artist=True, boxprops=dict(facecolor=color_map[sensor]), showfliers=False)
                for patch in box['boxes']:
                    patch.set_hatch(sensor_hatches[sensor])  # Set hatch using synchronized indexing
                    patch.set_linewidth(2)  # Make lines of the hatch thicker
                for median in box['medians']:
                    median.set(linewidth=2, color='black')

            sensor_count_positions.append((positions[0] + positions[-1]) / 2)

        # Draw the dashed line separators, except for the last sensor_count
        for position in sensor_count_positions[:-1]:
            ax.axvline(x=position + 3.35, linestyle='--', color='grey')

        # Add description inside each plot
        ax.set_title(desc, fontsize=16)
        ax.set_ylabel('RMSE(°F)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks(sensor_count_positions)
        ax.set_xticklabels(sensor_counts)

    # Create legend patches with hatches
    legend_patches = [Patch(facecolor=color_map[sensor], hatch=sensor_hatches[sensor], label=legend_label_map[sensor])
                      for sensor in sensor_columns]

    # Add legend to the figure with a customizable location
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=legend_loc, fontsize=16, ncol=6)
    ax.set_xlabel('Number of additional sensors', fontsize=20)
    fig.tight_layout()
    plt.savefig('combined_errors.pdf', bbox_inches='tight')
    plt.show()

# Define color map and legend labels
color_map = {'random_average': 'red', 'worst_case_average': 'lightblue', 'motion_average':'orange', 'average_all': 'green',
             'Indoor_AverageTemperature': 'yellow', 'Thermostat_Temperature': 'purple'}
legend_label_map = {'random_average': 'Random', 'worst_case_average': 'Worst', 'motion_average': 'Motion', 'average_all': 'Full',
                    'Indoor_AverageTemperature': 'Actual', 'Thermostat_Temperature': 'Thermostat'}

sensor_columns = list(color_map.keys())

# Call the function with your data
plot_combined_errors(training_errors_onestep, testing_errors_onestep, training_errors_lm, testing_errors_lm)




