#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:26:01 2023

@author: ozanbaris
"""
import os
import pandas as pd
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy.core.fromnumeric')
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
                    
                    # Add the DataFrame to the nested dictionary
                    all_houses_reduced[int(house_group)][house_id] = df

    return all_houses_reduced


main_output_directory = "house_data_csvs1"
mode = "cooling"  # or "heating"
all_houses_reduced = read_csvs_to_dict(main_output_directory, mode)


#%%

def compute_motion_average(five_houses, num_sensors=5):
    # Define sensor names
    sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, num_sensors + 1)]
    sensor_temp_names = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, num_sensors + 1)]
    
    for house_id, dataset in five_houses.items():
        # Create a function to compute motion-based average for a row
        def motion_based_avg(row):
            occupied_temp_values = [row[temp] for temp, motion in zip(sensor_temp_names, sensor_motion_names) if row[motion] == 1]
            if occupied_temp_values:
                return sum(occupied_temp_values) / len(occupied_temp_values)
            else:
                return row['Thermostat_Temperature']
        
        # Apply the function to each row to compute the 'motion-average' column
        dataset['motion-average'] = dataset.apply(motion_based_avg, axis=1)
        five_houses[house_id] = dataset

    return five_houses

five_houses=all_houses_reduced[5]
updated_five_houses = compute_motion_average(five_houses)


#%%


def determine_scheduled_occupancy(time):
    # Define weekday and weekend occupancy schedules
    # Weekday schedule
    weekday_schedule = {
        'night': [0, 0, 1, 1, 1, 0],  # 10 pm - 7 am
        'morning': [1, 1, 0, 0, 0, 0],  # 7 am - 9 am
        'day': [1, 0, 0, 0, 0, 0],  # 9 am - 5 pm
        'evening': [0, 1, 0, 1, 0, 1],  # 5 pm - 8 pm
        'late_evening': [0, 0, 0, 0, 1, 1],  # 8 pm - 10 pm
    }

    # Weekend schedule
    weekend_schedule = {
        'night': [0, 0, 1, 0, 1, 0],  # Different for weekends
        'morning': [1, 1, 0, 1, 0, 1],  # Different for weekends
        'day': [0, 1, 0, 0, 1, 1],  # Different for weekends
        'evening': [1, 0, 1, 0, 0, 1],  # Different for weekends
        'late_evening': [0, 1, 0, 1, 0, 1],  # Different for weekends
    }

    hour = time.hour
    weekday = time.weekday()

    # Select the correct schedule based on whether it's a weekday (0-4) or weekend (5-6)
    schedule = weekday_schedule if weekday < 5 else weekend_schedule

    if 22 <= hour or hour < 7:  # 10 pm - 7 am
        return schedule['night']
    elif 7 <= hour < 9:  # 7 am - 9 am
        return schedule['morning']
    elif 9 <= hour < 17:  # 9 am - 5 pm
        return schedule['day']
    elif 17 <= hour < 20:  # 5 pm - 8 pm
        return schedule['evening']
    else:  # 8 pm - 10 pm
        return schedule['late_evening']

def add_scheduled_columns_and_average(five_houses, num_sensors=5):
    # Sensor names
    sensor_names = ['Thermostat'] + [f'RemoteSensor{i}' for i in range(1, num_sensors + 1)]
    sensor_temp_names = [f'{sensor}_Temperature' for sensor in sensor_names]

    for house_id, dataset in five_houses.items():
        # Ensure 'time' column is in datetime format
        dataset['time'] = pd.to_datetime(dataset['time'])
        
        # Add scheduled occupancy columns based on the datetime
        for sensor in sensor_names:
            dataset[f'{sensor}_Scheduled'] = dataset['time'].apply(
                lambda x: determine_scheduled_occupancy(x)[sensor_names.index(sensor)]
            )

        # Compute scheduled-average
        def scheduled_avg(row):
            occupied_temp_values = [
                row[temp] for temp, sched in zip(sensor_temp_names, [f'{sensor}_Scheduled' for sensor in sensor_names])
                if row[sched] == 1
            ]
            return sum(occupied_temp_values) / len(occupied_temp_values) if occupied_temp_values else float('nan')

        dataset['scheduled-average'] = dataset.apply(scheduled_avg, axis=1)

    return five_houses

updated_five_houses = add_scheduled_columns_and_average(five_houses)

#%%

def plot_occupancy_schedule():
    # Create a DataFrame to hold the occupancy data for both weekday and weekend
    hours = list(range(24))
    sensors = ['T', '1', '2', '3', '4', '5']
    occupancy_weekday = pd.DataFrame(index=sensors, columns=hours)
    occupancy_weekend = pd.DataFrame(index=sensors, columns=hours)

    # Populate the DataFrame using the determine_scheduled_occupancy function
    for hour in hours:
        time_weekday = pd.Timestamp('2024-01-03 ') + pd.Timedelta(hours=hour)  # A random weekday
        time_weekend = pd.Timestamp('2024-01-07 ') + pd.Timedelta(hours=hour)  # A random weekend day
        occupancy_weekday[hour] = determine_scheduled_occupancy(time_weekday)
        occupancy_weekend[hour] = determine_scheduled_occupancy(time_weekend)

    # Convert the DataFrames to a format suitable for a seaborn heatmap
    occupancy_weekday = occupancy_weekday # Transpose to have hours on the y-axis and sensors on the x-axis
    occupancy_weekend = occupancy_weekend

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot the weekday heatmap
    sns.heatmap(occupancy_weekday, cmap=['#444444', '#1c9545'], cbar=False, linewidths=.5, linecolor='grey', ax=ax1)
    ax1.set_title('Weekday', fontsize=16)
    ax1.set_ylabel('Sensor', fontsize=14)
    ax1.set_xlabel('Hour of Day', fontsize=14)
    ax1.set_xticks(np.arange(0.5, len(hours), 1))
    ax1.set_xticklabels(hours, rotation=0)
    ax1.set_yticks(np.arange(0.5, len(sensors), 1))
    ax1.set_yticklabels(sensors, rotation=0)

    # Plot the weekend heatmap
    sns.heatmap(occupancy_weekend, cmap=['#444444', '#1c9545'], cbar=False, linewidths=.5, linecolor='grey', ax=ax2)
    ax2.set_title('Weekend', fontsize=16)
    ax2.set_ylabel('Sensor', fontsize=14)
    ax2.set_xlabel('Hour of Day', fontsize=14)
    ax2.set_xticks(np.arange(0.5, len(hours), 1))
    ax2.set_xticklabels(hours, rotation=0)
    ax2.set_yticks(np.arange(0.5, len(sensors), 1))
    ax2.set_yticklabels(sensors, rotation=0)
    plt.savefig('occupancy.pdf', bbox_inches='tight')
    plt.show()
    
plot_occupancy_schedule()


#%% RUNNING RIDGE REGRESSION TO FIT ONE STEP AHEAD MODELS


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
    sensor_columns = ['motion-average','scheduled-average','Indoor_AverageTemperature','Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, sensor_count + 1)]

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
    
    #return models, train_errors, test_errors, singular_houses

        # Store results
        results[house_id] = {
            'models': house_models,
            'train_errors': house_train_errors,
            'test_errors': house_test_errors
        }

    return results

results_onestep = {}

results_onestep=process_houses(updated_five_houses, sensor_count=5)
    

#%%

def make_predictions(coefficients, data, feature_columns):
    X = data[feature_columns].values
    # Use matrix multiplication for prediction
    return np.dot(X, coefficients)

def process_house_data(num_sensors, houses_dict, sensor_motion_names, ground_truth_column):
    predicted_datasets = {}

    feature_columns = ['RunTime', 'Outdoor_Temperature', 'GHI', ground_truth_column]

    for house_id, house_data in houses_dict.items():
        print(f"Processing house: {house_id}")

        # Ensure all feature columns are present
        if not all(col in house_data.columns for col in feature_columns + sensor_motion_names):
            print(f"Missing feature columns in house {house_id}")
            continue


        split_index = int(0.5 * len(house_data))
        test_data = house_data.iloc[split_index:]
        print('num days in test data', len(test_data)/24)
        # Check if there are enough rows for prediction
        if test_data.shape[0] <= 1:
            print(f"Not enough data for predictions in house {house_id}")
            continue

        # Extract models for the current house
        models = results_onestep[house_id]['models']
        predictions_df = pd.DataFrame()

        # Iterate over each model and make predictions
        for i in range(num_sensors + 1):  # Including thermostat and remote sensors
            sensor_name = 'Thermostat_Temperature' if i == 0 else f'RemoteSensor{i}_Temperature'
            if sensor_name in models:
                coefficients = models[sensor_name]
                predictions = make_predictions(coefficients, test_data.iloc[:-1], feature_columns)
                predictions_df[f'{sensor_name}_predictions'] = predictions

        # Make predictions for the ground truth column itself
        if ground_truth_column in models:
            coefficients = models[ground_truth_column]
            ground_truth_predictions = make_predictions(coefficients, test_data.iloc[:-1], feature_columns)
            predictions_df[f'{ground_truth_column}_predictions'] = ground_truth_predictions

        # Add sensor motion columns from the next timestep
        for col in sensor_motion_names:
            predictions_df[f'{col}_next'] = test_data[col].values[1:]

        predictions_df[f'{ground_truth_column}_next'] = test_data[ground_truth_column].values[1:]
        predicted_datasets[house_id] = predictions_df

    return predicted_datasets



#%%
def calculate_sensor_based_predictions(predicted_datasets, sensor_motion_names, sensor_temp_pred_names):
    for house_id, predictions_df in predicted_datasets.items():
        sensor_based_predictions = []

        for index, row in predictions_df.iterrows():
            active_sensor_temps = []

            # Check each sensor's motion and collect corresponding temperature predictions if active
            for sensor_motion, sensor_temp_pred in zip(sensor_motion_names, sensor_temp_pred_names):
                if row[f'{sensor_motion}_next'] == 1:
                    active_sensor_temps.append(row[sensor_temp_pred])

            # If no sensors are active, use the Thermostat temperature prediction
            if not active_sensor_temps:
                print(' no active sensors')
                active_sensor_temps.append(row['Thermostat_Temperature_predictions'])

            # Calculate the average of the active sensor temperature predictions
            sensor_based_avg = np.mean(active_sensor_temps)
            sensor_based_predictions.append(sensor_based_avg)

        # Add sensor-based predictions to the DataFrame
        predictions_df['sensor_based_prediction'] = sensor_based_predictions

        # Update the dataset in the dictionary
        predicted_datasets[house_id] = predictions_df

    return predicted_datasets



#%%


def calculate_rmse(predicted_datasets, ground_truth_column):
    collective_rmse_data = pd.DataFrame()

    for house_id, predictions_df in predicted_datasets.items():
        # Calculate squared differences
        motion_column = f'{ground_truth_column}_next'
        predictions_column = f'{ground_truth_column}_predictions'
        sensor_column = 'sensor_based_prediction'

        predictions_df['motion_squared_diff'] = (predictions_df[motion_column] - predictions_df[predictions_column]) ** 2
        predictions_df['sensor_squared_diff'] = (predictions_df[motion_column] - predictions_df[sensor_column]) ** 2

        # Calculate RMSE
        motion_rmse = np.sqrt(predictions_df['motion_squared_diff'].mean())
        sensor_rmse = np.sqrt(predictions_df['sensor_squared_diff'].mean())

        # Add RMSE to collective RMSE data
        rmse_row = pd.DataFrame({
            'House': [house_id],
            f'{ground_truth_column}_RMSE': [motion_rmse],
            'Sensor_RMSE': [sensor_rmse]
        })
        collective_rmse_data = pd.concat([collective_rmse_data, rmse_row], ignore_index=True)

    # Plotting the box plot of errors
    plt.figure(figsize=(10, 6))
    collective_rmse_data.boxplot(column=[f'{ground_truth_column}_RMSE', 'Sensor_RMSE'])
    plt.title('Box Plot of Errors')
    plt.ylabel('Error')
    plt.grid(False)
    plt.show()

    # Calculate and print the average and standard deviation of each error type
    motion_error_mean = collective_rmse_data[f'{ground_truth_column}_RMSE'].mean()
    motion_error_std = collective_rmse_data[f'{ground_truth_column}_RMSE'].std()
    sensor_error_mean = collective_rmse_data['Sensor_RMSE'].mean()
    sensor_error_std = collective_rmse_data['Sensor_RMSE'].std()

    print(f"{ground_truth_column}_RMSE' Mean: {motion_error_mean}, Standard Deviation: {motion_error_std}")
    print(f"Sensor Error - Mean RMSE: {sensor_error_mean}, Standard Deviation: {sensor_error_std}")

    return collective_rmse_data

#%% MAKING PREDICTIONS FOR SCHEDULED OCCUPANCY
num_sensors = 5
sensor_motion_names = ['Thermostat_Scheduled'] + [f'RemoteSensor{i}_Scheduled' for i in range(1, num_sensors + 1)]
predictions= process_house_data(num_sensors, five_houses, sensor_motion_names, 'scheduled-average')

sensor_temp_pred_names = ['Thermostat_Temperature_predictions'] + [f'RemoteSensor{i}_Temperature_predictions' for i in range(1, num_sensors + 1)]

predicted_datasets = calculate_sensor_based_predictions(predictions, sensor_motion_names, sensor_temp_pred_names)

scheduled_rmse = calculate_rmse(predicted_datasets, 'scheduled-average')

#%% MAKING PREDICTIONS FOR MOTION OCCUPANCY
num_sensors = 5
sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, num_sensors + 1)]
predictions = process_house_data(num_sensors, five_houses, sensor_motion_names, 'motion-average')

sensor_temp_pred_names = ['Thermostat_Temperature_predictions'] + [f'RemoteSensor{i}_Temperature_predictions' for i in range(1, num_sensors + 1)]

predicted_datasets = calculate_sensor_based_predictions(predictions, sensor_motion_names, sensor_temp_pred_names)

motion_rmse = calculate_rmse(predicted_datasets, 'motion-average')

#%% DEFINING THE MODEL PREDICTIVE CONTROL RELEVANT IDENTIFICATION (MRI) to FIT ONE DAY AHEAD MODELS

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
    sensor_columns = ['motion-average','scheduled-average','Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, sensor_count + 1)]
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

#%% RUN THE MRI FOR EACH HOUSE
initial_params = {}

for house_id, data in results_onestep.items():
    models = data['models']
    
    # Initialize dictionaries for each house ID
    initial_params[house_id] = {}
    
    # Iterate through the models and classify them based on their type (motion or schedule)
    for sensor_name, parameters in models.items():
        initial_params[house_id][sensor_name] = parameters

results_lm=process_houses_genetic_algorithm(five_houses, num_sensors, initial_params)


#%%

def create_regressor( cooling_runtime, outdoor_temp, solar_data,sensor_temp, h, na, nb, nd, k, P):
    """
    Creates a regressor Z for the given inputs. 
    Uses recursively predicted sensor_temp values beyond the kth time step.
    """
    Z = []

    # Start with the actual sensor_temp at time k
    predicted_values = [sensor_temp[k]]
    #print('predcited vals', predicted_values)
    for i in range(P):
        
        z = [[cooling_runtime[k+i]]+[outdoor_temp[k+i]]+[solar_data[k+i]]+[predicted_values[-1]]]
        #print("Lengths of inputs at iteration {}:".format(i))
        #print("cooling_runtime:", len(cooling_runtime))
        #print("outdoor_temp:", len(outdoor_temp))
        #print("solar_data:", len(solar_data))
        #if k<5:
            #print('making predictions for', k+i)
            #print("k:", k)

        #print("i:", i)
        #print("k+i:", k+i)
        #print('z',z)
        #print(f"Iteration {i}, Z length: {len(Z)}, predicted_values length: {len(predicted_values)}")

        Z.append(z)
        z = np.array(z).flatten()
        # Compute the next predicted value using the current regressor
        y_pred = np.dot(z, h)
        #if i<20:
         #   print('in time k:',k,'i',i,'for Z:', z,'y_pred:',y_pred)
        predicted_values.append(y_pred)
    #print("Shape of Z:", np.array(Z).shape)
    return np.array(Z)
def compute_multistep_predictions(X, identified_params, k, P):
    """
    Compute the P-step ahead prediction at the k-th instance.

    Args:
    - X: Feature matrix.
    - identified_params: Identified model parameters.
    - P: Prediction step.
    - k: Current instance.

    Returns:
    - y_pred: P-step ahead prediction at the k-th instance.
    """
    Z = create_regressor(X[:, 0], X[:, 1], X[:, 2], X[:, 3], identified_params, 1, 1, 1, k, P)
    #print("Size of Z:", len(Z))
    #print("Value of P:", P)

    # Only use the prediction at point P
    y_pred = np.dot(Z[P - 1], identified_params)
    
    return y_pred


def compute_errors_and_predictions(house_data_full, initial_params, house_id, sensor_columns, sensor_motion_names, P, ground_truth_column):
    """
    Computes predictions for sensors, averages based on active sensors at prediction time,
    and calculates errors against ground truth data.

    Args:
    - house_data: DataFrame with sensor and motion data.
    - initial_params: Dictionary with parameters for each sensor's model.
    - house_id: Identifier for the house being processed.
    - sensor_columns: List of all sensor columns for prediction, in the same order as sensor_motion_names.
    - sensor_motion_names: List of columns indicating motion (sensor activity), corresponding to sensor_columns.
    - P: Prediction horizon.
    - ground_truth_column: Column name to use as ground truth for error calculation.

    Returns:
    - sensor_based_error: RMSE between sensor-based averaged predictions and ground truth.
    - ground_truth_error: RMSE between ground truth column predictions and actual ground truth.
    """
    # get the second half of the data for the analysis. 
    half_index = int(0.5 * len(house_data_full))
    house_data = house_data_full[:half_index]
    
    ground_truth_values = house_data[ground_truth_column].values[P:]
    active_sensor_matrix = house_data[sensor_motion_names].shift(-P).fillna(0).iloc[:-P].values
    house_models = initial_params[house_id]

    # Initialize lists to hold predictions
    sensor_based_predictions = []
    ground_truth_predictions = []

    for i in range(len(house_data) - P):
        # Get active sensor indicators for this prediction step
        active_sensors_indicator = active_sensor_matrix[i]

        # List to collect predictions from sensors that are active
        active_predictions = []

        for sensor_col, motion_col in zip(sensor_columns, sensor_motion_names):
            if sensor_col in house_models:
                # Use model parameters for the sensor
                theta = house_models[sensor_col]

                # Call compute_multistep_predictions with all data and current index i
                y_pred = compute_multistep_predictions(house_data[[ 'RunTime','Outdoor_Temperature', 'GHI', sensor_col] ].values, theta, i, P)
                
                # Find the corresponding motion column index
                motion_index = sensor_motion_names.index(motion_col)

                # If the sensor is active, add its prediction to active_predictions
                if active_sensors_indicator[motion_index]:
                    active_predictions.append(y_pred)

        # If no sensors are active, use the Thermostat, if it's one of the sensors
        if not active_predictions and 'Thermostat_Temperature' in sensor_columns:
            theta = house_models['Thermostat_Temperature']
            y_pred = compute_multistep_predictions(house_data[['RunTime','Outdoor_Temperature', 'GHI',  'Thermostat_Temperature']].values, theta, i, P)
            active_predictions = [y_pred]

        # Compute the average of active sensor predictions
        sensor_based_predictions.append(np.mean(active_predictions) if active_predictions else np.nan)

    # Calculate RMSE for sensor-based predictions
    sensor_based_error = np.sqrt(np.mean([(pred - true) ** 2 for pred, true in zip(sensor_based_predictions, ground_truth_values) if not np.isnan(pred)]))
    
    # Handle ground truth predictions
    if ground_truth_column in house_models:
        theta = house_models[ground_truth_column]
        for i in range(len(house_data) - P):
            y_pred = compute_multistep_predictions(house_data[['RunTime', 'Outdoor_Temperature', 'GHI', ground_truth_column]].values, theta, i, P)
            ground_truth_predictions.append(y_pred)

        # Calculate RMSE for ground truth predictions
        ground_truth_error = np.sqrt(np.mean([(pred - true) ** 2 for pred, true in zip(ground_truth_predictions, ground_truth_values)]))
    else:
        ground_truth_error = None

    print('sensor based error', sensor_based_error)
    print('ground_error', ground_truth_error)
    return sensor_based_error, ground_truth_error


#%% FOR MOTION

sensor_columns = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]  
sensor_motion_names = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, 6)]
P = 24 
ground_truth_column = 'motion-average'  

motion_errors_summary = {}
for house_id, house_data in five_houses.items():
    sensor_based_error, ground_truth_error = compute_errors_and_predictions(house_data, initial_params, house_id, sensor_columns, sensor_motion_names, P, ground_truth_column)
    motion_errors_summary[house_id] = {
        'sensor_based_error': sensor_based_error,
        'ground_truth_error': ground_truth_error
    }
#%% FOR SCHEDULE

sensor_columns = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]  
sensor_motion_names = ['Thermostat_Scheduled'] + [f'RemoteSensor{i}_Scheduled' for i in range(1, 6)]
P = 24  
ground_truth_column = 'scheduled-average' 

schedule_errors_summary = {}
for house_id, house_data in five_houses.items():
    sensor_based_error, ground_truth_error = compute_errors_and_predictions(house_data, initial_params, house_id, sensor_columns, sensor_motion_names, P, ground_truth_column)
    schedule_errors_summary[house_id] = {
        'sensor_based_error': sensor_based_error,
        'ground_truth_error': ground_truth_error
    }


#%%
def plot_combined_rmse_violins_enhanced(motion_rmse, scheduled_rmse, pred_type):
    sns.set_context("talk")  # Increase the font size globally

    # Preparing the data
    motion_data = motion_rmse.melt(id_vars='House', value_vars=['motion-average_RMSE', 'Sensor_RMSE'], var_name='ErrorType', value_name='RMSE')
    motion_data['Category'] = 'Motion'

    scheduled_data = scheduled_rmse.melt(id_vars='House', value_vars=['scheduled-average_RMSE', 'Sensor_RMSE'], var_name='ErrorType', value_name='RMSE')
    scheduled_data['Category'] = 'Schedule'

    combined_data = pd.concat([motion_data, scheduled_data])
    combined_data['ErrorType'] = combined_data['ErrorType'].apply(lambda x: 'Ground Truth' if 'average' in x else 'Sensor Based')

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.violinplot(x="Category", y="RMSE", hue="ErrorType", data=combined_data, split=True, inner="quart",
                   linewidth=2.5, palette={"Ground Truth": "#ff9999", "Sensor Based": "#add8e6"})
    
    # Adding mean values using pointplot
    sns.pointplot(x='Category', y='RMSE', hue='ErrorType', data=combined_data, dodge=0.07, join=False, 
                  markers='D', scale=1.5, ci=None, palette={"Ground Truth": "black", "Sensor Based": "black"})

    plt.title(f'{pred_type} Ahead Predictions', fontsize=20)
    plt.ylabel('RMSE(°F)', fontsize=20)
    plt.xlabel('')
    plt.tick_params(labelsize=20)

    # Calculate and annotate the mean values directly on the plot for clarity
    means = combined_data.groupby(['Category', 'ErrorType'])['RMSE'].mean().reset_index()
    for index, row in means.iterrows():
        x_location = index // 2 - 0.15 + (index % 2) * 0.3
        plt.text(x=x_location, y=row['RMSE'] , s=f"{row['RMSE']:.2f}", 
                 horizontalalignment='center', color='black', weight='bold', fontsize=16)

    # Custom legend creation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', edgecolor='black', label='Average-Based Modeling'),
        Patch(facecolor='#add8e6', edgecolor='black', label='Distributed Sensor Modeling')
    ]
    plt.legend(handles=legend_elements, title='Model Type', fontsize=16, title_fontsize='18')

    # Optionally, if you want to remove the automatic legend created by pointplot and violinplot, do it here
    # This gets the current axes and removes the existing legend
    plt.gca().legend_.remove()

    # Then create the custom legend as above
    plt.legend(handles=legend_elements, title='Model Type', fontsize=16, title_fontsize='18')
    plt.savefig(f'{pred_type}_model.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

plot_combined_rmse_violins_enhanced(motion_rmse, scheduled_rmse, 'One Step')
#%%

def prepare_rmse_dataframes(motion_errors_summary, schedule_errors_summary):
    # Initialize lists to store the data
    motion_data_list = []
    schedule_data_list = []
    
    # Process motion errors summary
    for house_id, errors in motion_errors_summary.items():
        motion_data_list.append({'House': house_id, 'motion-average_RMSE': errors['ground_truth_error'], 'Sensor_RMSE': errors['sensor_based_error']})
        
    # Process schedule errors summary
    for house_id, errors in schedule_errors_summary.items():
        schedule_data_list.append({'House': house_id, 'scheduled-average_RMSE': errors['ground_truth_error'], 'Sensor_RMSE': errors['sensor_based_error']})
    
    # Convert lists to DataFrames
    motion_rmse = pd.DataFrame(motion_data_list)
    scheduled_rmse = pd.DataFrame(schedule_data_list)
    
    return motion_rmse, scheduled_rmse

motion_rmse_oneday, scheduled_rmse_oneday = prepare_rmse_dataframes(motion_errors_summary, schedule_errors_summary)
plot_combined_rmse_violins_enhanced(motion_rmse_oneday, scheduled_rmse_oneday, 'One Day')




#%%

sensor_based_errors = [details['sensor_based_error'] for details in schedule_errors_summary.values() if details['sensor_based_error'] is not None]
ground_truth_errors = [details['ground_truth_error'] for details in schedule_errors_summary.values() if details['ground_truth_error'] is not None]

# Compute average and standard deviation for sensor-based errors
avg_sensor_based_error = statistics.mean(sensor_based_errors)
std_sensor_based_error = statistics.stdev(sensor_based_errors)

# Compute average and standard deviation for ground truth errors
avg_ground_truth_error = statistics.mean(ground_truth_errors)
std_ground_truth_error = statistics.stdev(ground_truth_errors)

# Print the statistics
print("Sensor-Based Errors:")
print(f"Average: {avg_sensor_based_error:.4f}, Standard Deviation: {std_sensor_based_error:.4f}")
print("Ground Truth Errors:")
print(f"Average: {avg_ground_truth_error:.4f}, Standard Deviation: {std_ground_truth_error:.4f}")

