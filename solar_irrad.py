#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderUnavailable
import time
# File names
file_names = ['Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc']

# Initialize an empty DataFrame
df = pd.DataFrame()

# Load each file and append it to df
for file_name in file_names:
    data = xr.open_dataset(file_name)
    temp_df = data.to_dataframe()
    temp_df = temp_df.reset_index()
    df = pd.concat([df, temp_df], ignore_index=True)

state_dict = {}
reduced_data = {}

remote_sensor_cols = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

unique_house_ids = df['id'].unique()

for house_id in unique_house_ids:
    house_df = df[df['id'] == house_id].copy()
    house_df.sort_values(by="time", inplace=True)
    
    # Extract state value for this house
    state_values = house_df['State'].dropna().unique()
    assigned_state = 'Unknown'
    for state in state_values:
        if state != '':
            assigned_state = state
            break
    state_dict[house_id] = assigned_state

    valid_sensors_for_house = [col for col in remote_sensor_cols if not house_df[col].isna().all()]

    for sensor_col in valid_sensors_for_house:
        relevant_cols = ['Indoor_AverageTemperature', 'Thermostat_Temperature', 'Outdoor_Temperature', sensor_col, 'time']
        valid_rows = house_df.dropna(subset=relevant_cols)

        # Determine continuous data points
        valid_rows['time_diff'] = valid_rows['time'].diff().dt.total_seconds()
        
        # Identify start of the 21-day period
        continuous_regions = (valid_rows['time_diff'] == 300).rolling(window=21*288).sum()
        valid = continuous_regions == 21*288

        if valid.any():
            end_idx = valid.idxmax()
            subset = valid_rows.loc[end_idx - 21*288 + 1:end_idx]
            if house_id not in reduced_data:
                reduced_data[house_id] = subset
            else:
                reduced_data[house_id] = pd.concat([reduced_data[house_id], subset])
#%%

all_data = pd.concat(reduced_data, names=['house_id', 'row_id'])
all_data.reset_index(level=1, drop=True, inplace=True)  # We don't need the second level of index

all_data.to_csv("reduced_data.csv")

reduced_data_from_csv = {}
for house_id in read_df['house_id'].unique():
    reduced_data_from_csv[house_id] = read_df[read_df['house_id'] == house_id].drop(columns=['house_id'])


#%%
remote_sensor_cols = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

# Dictionaries to classify houses by their sensor count
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}

for house_id, house_df in reduced_data.items():
    # Since the data in reduced_data is already sorted and has extracted the state value, 
    # you don't have to repeat those steps.

    valid_sensors_for_house = [col for col in remote_sensor_cols if not house_df[col].isna().all()]

    # Classify the house by the number of valid sensors
    sensor_counts = len(valid_sensors_for_house)
    if sensor_counts == 1:
        one_houses[house_id] = house_df
    elif sensor_counts == 2:
        two_houses[house_id] = house_df
    elif sensor_counts == 3:
        three_houses[house_id] = house_df
    elif sensor_counts == 4:
        four_houses[house_id] = house_df
    elif sensor_counts == 5:
        five_houses[house_id] = house_df

# Printing the number of houses in each category
print(f"Number of houses with 1 additional sensor: {len(one_houses)}")
print(f"Number of houses with 2 additional sensors: {len(two_houses)}")
print(f"Number of houses with 3 additional sensors: {len(three_houses)}")
print(f"Number of houses with 4 additional sensors: {len(four_houses)}")
print(f"Number of houses with 5 additional sensors: {len(five_houses)}")


#%%    

ca_data = pd.read_csv('CA.csv')
il_data = pd.read_csv('IL.csv')
ny_data = pd.read_csv('NY.csv')
tx_data = pd.read_csv('TX.csv')
#%% 
state_dict = {}

house_dictionaries = [one_houses, two_houses, three_houses, four_houses, five_houses]

for house_dict in house_dictionaries:
    for house_id, single_house_data in house_dict.items():
        state_values = single_house_data['State'].dropna().unique()
        assigned_state = 'Unknown'
        for state in state_values:
            if state != '':
                assigned_state = state
                break
        state_dict[house_id] = assigned_state

data_map = {
    'CA': ca_data,
    'TX': tx_data,
    'IL': il_data,
    'NY': ny_data
}

# Assign city to each house in house dictionaries based on the state and house_id
for house_id, state in state_dict.items():
    if state in data_map:
        house_row = data_map[state][data_map[state]['Identifier'] == house_id]
        if not house_row.empty:
            city = house_row['City'].iloc[0]

            # Find the dictionary where this house belongs and update the city
            for house_dict in house_dictionaries:
                if house_id in house_dict:
                    if 'City' not in house_dict[house_id]:
                        house_dict[house_id]['City'] = city
                    else:
                        house_dict[house_id]['City'] = city
                    print(city)
                    break
#%%
def get_lat_lon(city, state, retries=3):
    geolocator = Nominatim(user_agent="yourUniqueAppNameHere")
    while retries > 0:
        try:
            location = geolocator.geocode(f"{city}, {state}")
            return (location.latitude, location.longitude)
        except (GeocoderServiceError, GeocoderUnavailable):
            retries -= 1
            time.sleep(2)  # Wait for 2 seconds before retrying
    raise Exception("Failed to get coordinates after multiple retries")



# Create a set to store unique city names
unique_cities = set()

# Create dictionaries to store latitude and longitude for each city-state combination
city_latitudes = {}
city_longitudes = {}

# Iterate over each house dictionary and each house data within them
for house_dict in house_dictionaries:
    for house_id, house_data in house_dict.items():
        
        # Get city name and state from the data and state_dict respectively
        city_name = house_data['City'].mode().iloc[0]
        state_name = state_dict[house_id]
        
        # Form the address by combining city and state
        address = f"{city_name}, {state_name}"
        
        # If we haven't fetched the coordinates for this city-state combination before
        if address not in city_latitudes:
            latitude, longitude = get_lat_lon(city_name, state_name)
            city_latitudes[address] = latitude
            city_longitudes[address] = longitude
        
        # Update the house_data with latitude and longitude
        house_data['Latitude'] = city_latitudes[address]
        house_data['Longitude'] = city_longitudes[address]

        unique_cities.add(city_name)

print(len(unique_cities))
#%% 
lat_lon_couples = list(zip(city_latitudes.values(), city_longitudes.values()))
print(lat_lon_couples)

lat_lon_couples = list(zip(city_latitudes.values(), city_longitudes.values()))
wkt_string = "MULTIPOINT(" + ", ".join(["{} {}".format(lon, lat) for lat, lon in lat_lon_couples]) + ")"
#%% 

def batch(iterable, n=1):
    """Helper function to create batches from a list."""
    length = len(iterable)
    for index in range(0, length, n):
        yield iterable[index:min(index + n, length)]

# For testing purposes, get only the first batch of 10 lat_lon_couples
first_batch = next(batch(lat_lon_couples, 10))

# Define the endpoint and parameters
endpoint = "https://api.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.json"
params = {
    'api_key': '3hCXpYR2zw9uQcyzYvUWQMNmcJBekjyfZtDSldmt',
    'email': 'omulayim@andrew.cmu.edu',
    'wkt': 'MULTIPOINT(' + ', '.join([f"{lat} {lon}" for lat, lon in first_batch]) + ')',
    'names': '2017',
    'attributes': 'ghi,dni',
    'interval': '30',
    'utc': 'false',
    'leap_day': 'false',
    'full_name': 'Ozan Baris Mulayim',
    'affiliation': 'CMU',
    'reason': 'research',
    'mailing_list': 'false'
}

try:
    # Send the request
    response = requests.get(endpoint, params=params)
    
    # Check the HTTP status code
    if response.status_code == 200:
        json_response = response.json()
        
        # Check for errors in the JSON response
        if not json_response['errors']:
            print("Request was successful!")
            print(json_response['outputs']['message'])
        else:
            print("Request had errors:", json_response['errors'])
            
    else:
        print(f"HTTP Error {response.status_code}: {response.text}")

except requests.ConnectionError:
    print("Connection error. The service might be down or there's a network issue.")
except requests.Timeout:
    print("Request timed out. The service might be slow to respond or there's a network delay.")
except Exception as e:
    print(f"An error occurred: {e}")

#%%    
def divide_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def get_hourly_solar_data(lat_lon_list, year, api_key):
    base_url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download"
    
    # Construct MULTIPOINT WKT with the lat-lon list
    multipoint_wkt = 'MULTIPOINT(' + ', '.join([f"{lon} {lat}" for lat, lon in lat_lon_list]) + ')'
    
    params = {
        'api_key': api_key,
        'names': str(year),
        'leap_day': 'false',
        'interval': '30',
        'utc': 'false',
        'full_name': 'Ozan Baris Mulayim',
        'email': 'omulayim@andrew.cmu.edu',
        'affiliation': 'CMU',
        'reason': 'research',
        'mailing_list': 'false',
        'wkt': multipoint_wkt,
        'attributes': 'ghi,dni',
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        print('success!')
        with open(f"solar_data.csv", 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def main():
    # Assuming lat_lon_couples is defined somewhere# Replace with your list of lat-lon couples
    
    batch_size = 50  # Set the size of each batch
    lat_lon_batches = list(divide_into_batches(lat_lon_couples, batch_size))
    
    year = 2017  # Replace with desired year
    api_key = '3hCXpYR2zw9uQcyzYvUWQMNmcJBekjyfZtDSldmt'  # Replace with your NREL API Key
    
    # Skip the first batch (since it already worked), and download the rest
    for i, batch in enumerate(lat_lon_batches[1:]):
        get_hourly_solar_data(batch, year, api_key)
        
        # Wait for 5 minutes before making another API call, but not after the last batch
        if i < len(lat_lon_batches) - 2:
            print("Waiting for 5 minutes before the next API call...")
            time.sleep(300)  # Wait for 300 seconds or 5 minutes



