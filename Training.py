#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:26:01 2023

@author: ozanbaris
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
os.environ['MKL_VERBOSE'] = '0'
from sklearn.linear_model import Ridge

def read_csvs_to_dict(main_output_directory, mode):
    # Ensure mode is either 'cooling' or 'heating'
    if mode not in ['cooling', 'heating']:
        raise ValueError("Mode must be either 'cooling' or 'heating'.")

    # Determine the column name based on the mode
    runtime_column = "CoolingRunTime" if mode == 'cooling' else "HeatingRunTime"
    
    all_houses_reduced = {}
    
    # Iterate through each subdirectory in the main directory
    for house_group_folder in os.listdir(main_output_directory):
        house_group_path = os.path.join(main_output_directory, house_group_folder)
        
        if os.path.isdir(house_group_path):
            house_group = house_group_folder.replace("house_group_", "")
            all_houses_reduced[int(house_group)] = {}

            # Iterate through each CSV file in the subdirectory
            for csv_file in os.listdir(house_group_path):
                if csv_file.endswith('.csv'):
                    house_id = csv_file.replace("house_id_", "").replace(".csv", "")
                    csv_path = os.path.join(house_group_path, csv_file)
                    df = pd.read_csv(csv_path)

                    # Rename the specified runtime column to "RunTime"
                    df.rename(columns={runtime_column: "RunTime"}, inplace=True)
                    
                    sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, int(house_group)+1)]
                    df[sensor_motion_names] = df[sensor_motion_names].fillna(0)
                    
                    # Add the DataFrame to the nested dictionary
                    all_houses_reduced[int(house_group)][house_id] = df

    return all_houses_reduced


main_output_directory = "house_data_csvs1"
mode = "cooling"  # or "heating"
all_houses_reduced = read_csvs_to_dict(main_output_directory, mode)



#%%
# Random average across a randomly selected subset of sensors
def random_average(row):
    # Exclude NaN values to only consider available sensors
    available_sensors = row[sensor_columns].dropna()
    if available_sensors.empty:
        # If no sensors are available, use Thermostat_Temperature as fallback
        return row['Thermostat_Temperature']
    # Randomly choose one or more sensors
    selected_sensors = available_sensors.sample(n=np.random.randint(1, len(available_sensors)+1))
    return selected_sensors.mean()


# Worst-case scenario: alternating between min and max sensor readings
def worst_case_average(idx, row):
    available_sensors = row[sensor_columns].dropna()
    if available_sensors.empty:
        return row['Thermostat_Temperature']
    if idx % 2 == 0:
        return available_sensors.min()
    else:
        return available_sensors.max()


def compute_motion_average(house_dict, num_sensors):
    # Define sensor names
    sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, num_sensors + 1)]
    sensor_temp_names = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, num_sensors + 1)]
    
    for house_id, dataset in house_dict.items():
        # Create a function to compute motion-based average for a row
        def motion_based_avg(row):
            occupied_temp_values = [row[temp] for temp, motion in zip(sensor_temp_names, sensor_motion_names) if row[motion] == 1]
            if occupied_temp_values:
                return sum(occupied_temp_values) / len(occupied_temp_values)
            else:
                return row['Thermostat_Temperature']
        
        # Apply the function to each row to compute the 'motion_average' column
        dataset['motion_average'] = dataset.apply(motion_based_avg, axis=1)
        house_dict[house_id] = dataset

    return house_dict



def compute_averages(dataframe):
    """
    Inner function to compute and add average columns to a single DataFrame.
    """
    # Average across all sensors
    dataframe['average_all'] = dataframe[sensor_columns].mean(axis=1)

    # Random average across a randomly selected subset of sensors
    dataframe['random_average'] = dataframe.apply(random_average, axis=1)

    # Worst-case scenario average
    dataframe['worst_case_average'] = [worst_case_average(idx, row) for idx, row in dataframe.iterrows()]

    return dataframe

# Apply compute_averages to each DataFrame in all_houses_reduced
for sensor_count, houses in all_houses_reduced.items():
    # Define the sensor columns
    sensor_columns = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, sensor_count + 1)]
    sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, sensor_count + 1)]
    for house_id, house_data in houses.items():
        if house_data.empty:
            print(f"Skipping empty DataFrame for {house_id}")
            continue
            
        all_houses_reduced[sensor_count][house_id]=compute_averages(house_data)


for sensor_count, houses in all_houses_reduced.items():
    all_houses_reduced[sensor_count] = compute_motion_average(houses, sensor_count)

#%%

def ridge_regression(X, y, alpha=1.0):
    ridge = Ridge(alpha=alpha, fit_intercept=False)  # Set fit_intercept to False
    ridge.fit(X, y)
    return ridge.coef_


def process_houses(houses_dict, sensor_count):
    models = {}
    train_errors = {}
    test_errors = {}
    singular_houses = []
    results={}
    # Define the sensor columns
    sensor_columns = ['random_average','average_all','worst_case_average','motion_average','Indoor_AverageTemperature','Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, sensor_count + 1)]

    for house_id, data in houses_dict.items():
        print("Processing house:", house_id)
        
        house_models = {}
        house_train_errors = {}
        house_test_errors = {}
        
        for sensor_col in sensor_columns:
            print("Processing house:", house_id, ", sensor:", sensor_col)
            data['RunTime'].fillna(0, inplace=True)
            # Features
            X = data[['RunTime', 'Outdoor_Temperature','GHI']].values[:-1]
            
            sensor_values = data[sensor_col].values[:-1].reshape(-1, 1)
            X = np.hstack([X, sensor_values])
            
            # Target
            y = data[sensor_col].values[1:]

            feature_cols = ['RunTime', 'Outdoor_Temperature','GHI', sensor_col]
            
            # Check for NaN values in feature columns
            for col in feature_cols:
                if data[col].isna().any():
                    print(f"House: {house_id}, Sensor: {sensor_col}, Column '{col}' has NaN values!")
                    print(data[sensor_col])
        
            # Check if there's data left after filtering
            if X.shape[0] == 0:
                print("No valid data for house:", house_id, ", sensor:", sensor_col)
                continue  # Skip the current loop iteration
            
            # Split the data into halves
            half_index = int(0.5 * len(X))
            first_half_X = X[:half_index]
            first_half_y = y[:half_index]

            # Within the first half, split into training and testing
            split_index = int(0.875 * len(first_half_X))
            X_train = first_half_X[:split_index]
            X_test = first_half_X[split_index:]
            y_train = first_half_y[:split_index]
            y_test = first_half_y[split_index:]
            
            theta = ridge_regression(X_train, y_train)
            house_models[sensor_col] = theta
        
            # No need to add bias to X_train and X_test
            # Directly use them for predictions
            train_predictions = X_train.dot(theta)
            test_predictions = X_test.dot(theta)

        
            train_error = np.sqrt(np.mean((y_train - train_predictions) ** 2))
            test_error = np.sqrt(np.mean((y_test - test_predictions) ** 2))
        
            house_train_errors[sensor_col] = train_error
            house_test_errors[sensor_col] = test_error

        models[house_id] = house_models
        train_errors[house_id] = house_train_errors
        test_errors[house_id] = house_test_errors
    
        # Store results
        results[house_id] = {
            'models': house_models,
            'train_errors': house_train_errors,
            'test_errors': house_test_errors
        }

    return results

results_onestep = {}
for sensor_count, house_dict in all_houses_reduced.items():
    
    results_onestep[sensor_count]=process_houses(house_dict, sensor_count)
    
print("Results:", results_onestep)


#%%



def create_regressor(outdoor_temp, sensor_temp, cooling_runtime,solar_data, h, na, nb, nd, k, P):
    """
    Creates a regressor Z for the given inputs. 
    Uses recursively predicted sensor_temp values beyond the kth time step.
    """
    Z = []
    #print("Lengths of inputs at iteration {}:".format(i))
    #print("cooling_runtime:", len(cooling_runtime))
    #print("outdoor_temp:", len(outdoor_temp))
    #print("solar_data:", len(solar_data))
    #print("k:", k)
    #print("i:", i)
   # print("k+i:", k+i)

    # Start with the actual sensor_temp at time k
    predicted_values = [sensor_temp[k]]
    #print('predcited vals', predicted_values)
    for i in range(P):
        z = [[cooling_runtime[k+i]]+[outdoor_temp[k+i]]+[solar_data[k+i]]+[predicted_values[-1]]]

        #print('z',z)
        print(f"Iteration {i}, Z length: {len(Z)}, predicted_values length: {len(predicted_values)}")

        Z.append(z)
        z = np.array(z).flatten()
        # Compute the next predicted value using the current regressor
        y_pred = np.dot(z, h)
        #if i<20:
         #   print('in time k:',k,'i',i,'for Z:', z,'y_pred:',y_pred)
        predicted_values.append(y_pred)
    print("Shape of Z:", np.array(Z).shape)
    return np.array(Z)


def objective(params, outdoor_temp, sensor_temp,cooling_runtime, solar_data,y_true, P, N):
    h = np.array([params['h0'], params['h1'], params['h2'],params['h3']])
    residuals = []
    
    for k in range(N - P):
        # Get the regressor Z for the current k value
        Z = create_regressor(outdoor_temp, sensor_temp, cooling_runtime, solar_data, h, 1, 1, 1, k, P)
        for i in range(1,P):
            y_pred = np.dot(Z[i], h)
            residual = y_pred - y_true[k+i]
            residuals.append(residual)
            
    return residuals

def identify_parameters(outdoor_temp, sensor_temp, cooling_runtime, solar_data, y_true, P, init_guess, N):
    params = Parameters()
    params.add('h0', value=init_guess[0])
    params.add('h1', value=init_guess[1])
    params.add('h2', value=init_guess[2])
    params.add('h3', value=init_guess[3])
    # Call the minimize function. Note that we don't need the intermediate Z anymore.
    result = minimize(objective, params, args=(outdoor_temp, sensor_temp, cooling_runtime,solar_data, y_true, P, N))

    h = np.array([result.params['h0'].value, result.params['h1'].value, result.params['h2'].value, result.params['h3'].value])
    residuals = result.residual  # Access the residuals from the minimize result
    return h, residuals


def compute_error(Z, sensor_temp, h, P):
    residuals = []
    N = len(sensor_temp) - 1
    for k in range(N - P):
        for i in range(1, P+1):
            y_pred = np.dot(Z[k + i - 1], h)
            residual = y_pred - sensor_temp[k + i-1]
            residuals.append(residual)
    return np.mean(np.array(residuals)**2)  # Return the Mean Squared Error (MSE)

def compute_rmse(errors):
    """Compute the root mean squared error from a list of errors."""
    mse = np.mean(np.array(errors)**2)
    return np.sqrt(mse)
"""
def compute_sensor_rmse(X, y, identified_params, P):
    residuals = []
    N = len(X) - P

    for k in range(N):
        Z = create_regressor(X[:, 1], X[:, 3], X[:, 0], X[:, 2],identified_params, 1, 1, 1, k, P)
        for i in range(1, P):
            y_pred = np.dot(Z[i - 1], identified_params)
            residual = y_pred - y[k + i]
            residuals.append(residual)

    return compute_rmse(residuals)
"""
def compute_sensor_rmse(X, y, identified_params, P):
    residuals = []
    N = len(X) - P

    for k in range(N):
        Z = create_regressor(X[:, 1], X[:, 3], X[:, 0], X[:, 2], identified_params, 1, 1, 1, k, P)
        
        # Only use the prediction at point P
        y_pred = np.dot(Z[P - 1], identified_params)
        residual = y_pred - y[k + P]
        residuals.append(residual)

    return compute_rmse(residuals)

#%%


def process_houses_genetic_algorithm(houses_dict, sensor_count, initial_parameters):
    results_lm = {}

    # Define the sensor columns
    sensor_columns = ['random_average','average_all','worst_case_average','motion_average','Indoor_AverageTemperature','Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, sensor_count + 1)]
    training_errors = {}
    testing_errors = {}

    for house_id, data in houses_dict.items():
        house_models = {}
        house_training_errors = {}
        house_testing_errors = {}

        print('house id: ', house_id)

        for sensor_col in sensor_columns:
            print(f"Processing sensor: {sensor_col}")

            # Target
            y = data[sensor_col].values[1:]

            # Check if y has only nan values
            print(f"y has {np.isnan(y).sum()} nan values out of {len(y)}")

            ## Replace nan values in 'CoolingEquipmentStage1_RunTime' with 0
            data['RunTime'].fillna(0, inplace=True)
            
            X = data[['RunTime', 'Outdoor_Temperature','GHI']].values[:-1]
            sensor_values = data[sensor_col].values[:-1].reshape(-1, 1)
            X = np.hstack([X, sensor_values])

            # Check shapes and sample values
            #print(f"X shape: {X.shape}, y shape: {y.shape}")
            #print(f"Sample X values: {X[:5]}, Sample y values: {y[:5]}")


            # Split the data into halves
            half_index = int(0.5 * len(X))
            first_half_X = X[:half_index]
            first_half_y = y[:half_index]

            # Within the first half, split into training and testing
            split_index = int(0.875 * len(first_half_X))
            X_train = first_half_X[:split_index]
            X_test = first_half_X[split_index:]
            y_train = first_half_y[:split_index]
            y_test = first_half_y[split_index:]
            

            # Check shapes and sample values after splitting
            #print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            #print(f"Sample X_train values: {X_train[:5]}, Sample y_train values: {y_train[:5]}")
            
            # Prediction horizon
            P = 24

            # Check if there are initial parameters for current sensor_col
            if sensor_col not in initial_parameters[house_id]:
                print(f"No initial parameters found for house {house_id} sensor {sensor_col}. Skipping...")
                continue

            init_guess = initial_parameters[house_id][sensor_col]
            print(f"Initial guess for {sensor_col}: {init_guess}")

            print('training started for', sensor_col)
            identified_params, residuals = identify_parameters(X_train[:, 1], X_train[:,3], X_train[:, 0], X_train[:,2], y_train, P, init_guess, len(X_train))
            print('Identified parameters:', identified_params)
            
            # Compute RMSE for training and testing data
            train_rmse = compute_sensor_rmse(X_train, y_train, identified_params, P)
            test_rmse = compute_sensor_rmse(X_test, y_test, identified_params, P)
            
            print(f"Training RMSE for {sensor_col}: {train_rmse}")
            print(f"Testing RMSE for {sensor_col}: {test_rmse}")
            
            house_training_errors[sensor_col] = train_rmse
            house_testing_errors[sensor_col] = test_rmse
            
            house_models[sensor_col] = identified_params
            
        training_errors[house_id] = house_training_errors
        testing_errors[house_id] = house_testing_errors

        # Store results
        results_lm[house_id] = {
            'models': house_models,
            'train_errors': house_training_errors,
            'test_errors': house_testing_errors
        }

    return results_lm

#%%
initial_params = {}
for idx, house in results_onestep.items():
    for house_id, result_dict in house.items():
        initial_params[house_id]=result_dict['models']
        
#%%
results_lm={}
for sensor_count, houses in all_houses_reduced.items():
    
    results_lm[sensor_count]=process_houses_genetic_algorithm(houses, sensor_count, initial_params)

#%%

def print_house_counts(results_dict, description):
    print(f"House counts for {description}:")
    for sensor_count, houses in results_dict.items():
        # Extracting unique house IDs from the dictionary
        house_ids = houses.keys()
        unique_house_count = len(set(house_ids))
        print(f"Sensor Count {sensor_count}: {unique_house_count} houses")


print_house_counts(results_lm, "Long-term Memory Results")
print_house_counts(results_onestep, "One-Step Results")
print_house_counts(all_houses_reduced,'all_houses')
#%%
house_counts = {}
for sensor_count in results_lm:
    house_counts[sensor_count] = len(results_lm[sensor_count])


# Sorting the dictionary by sensor count
sorted_house_counts = dict(sorted(house_counts.items()))

# Extracting the sensor counts and corresponding house numbers
sensor_counts = list(sorted_house_counts.keys())
number_of_houses = list(sorted_house_counts.values())

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(sensor_counts, number_of_houses, color='salmon')

# Adding the number of houses on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval-30, yval, ha='center', va='bottom', fontsize=20, fontweight='bold')

# Setting labels and title
plt.xlabel('Number of Additional Sensors', fontsize=20)
plt.ylabel('Number of Houses', fontsize=20)
#plt.title('Distribution of number of sensors among the houses trained', fontsize=16)
plt.xticks(fontsize=20)
# Remove y-axis line and ticks
#plt.gca().spines.set_visible(False)
plt.tick_params(axis='y', which='both', left=False, labelleft=False)
plt.savefig('trained_sensors.pdf', bbox_inches='tight')
# Show plot
plt.tight_layout()
plt.show()



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


