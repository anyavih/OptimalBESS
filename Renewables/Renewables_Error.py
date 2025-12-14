import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def fit_kde_for_hour(df: pd.DataFrame, zone: str, hour: int):
    #validate zone and create column names
    valid_zones = ['HOUSTON', 'NORTH', 'SOUTH', 'WEST']
    if zone not in valid_zones:
        print(f"Error: Zone '{zone}' is not valid. Must be one of {valid_zones}")
        return None, None, None
    rt_col = f'{zone}_RT_Actual'
    da_col = f'{zone}_DA_Forecast'
    #filter data for specific hour 
    try:
        df_hour = df[df['Hour Ending'].astype(int) == hour].copy()
    except Exception as e:
        print(f"Error filtering hour {hour}: {e}")
        return None, None, None
    if df_hour.empty:
        print(f"Warning: No data found for {zone}, Hour Ending = {hour}")
        return None, None, None
    #calculate RT - DA
    try:
        df_hour['error'] = df_hour[rt_col] - df_hour[da_col]
    except KeyError:
        print(f"Error: Columns '{rt_col}' or '{da_col}' not found.")
        return None, None, None
    #prepare data for KDE
    error_data = df_hour['error'].dropna()
    if error_data.empty:
        print(f"Warning: No error data found for {zone}, Hour {hour}")
        return None, None, None
    error_reshaped = error_data.values.reshape(-1, 1)
    #fit the KDE
    try:
        kde = KernelDensity(kernel='gaussian', bandwidth='scott')
        kde.fit(error_reshaped)
    except ValueError:
        print(f"Warning: 'scott' bandwidth estimation failed for Hour {hour}. Using bandwidth=1.0")
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
        kde.fit(error_reshaped)
    return kde, df_hour, error_data

def fit_kde_with_simulation(df: pd.DataFrame, zone: str, hour: int, H: int):
    #simulate H future hours 
    simulated_errors = []
    if H > 0:
        print(f"\n--- Simulating {H} Future Hours ---")
        for i in range(1, H + 1):
            #24 hour cycle 
            future_hour = (hour + i - 1) % 24 + 1
            #fit new KDE for future hour 
            kde_future, _, _ = fit_kde_for_hour(df, zone, future_hour)
            
            if kde_future:
                #draw one sample from that hour's error distribution
                sample = kde_future.sample(1)[0][0]
                simulated_errors.append(sample)
                print(f"Simulated error for Hour {future_hour} (H+{i}): {sample:.2f} MW")
            else:
                # Append NaN if we couldn't fit a KDE (e.g., no data)
                simulated_errors.append(np.nan)
                print(f"Could not simulate for Hour {future_hour} (H+{i}): No data")
    simulated_errors_array = np.array(simulated_errors)
    
    return simulated_errors_array



if __name__ == "__main__":
    try:
        df_final = pd.read_csv('/Users/anyavih/zone_renewable_data.csv')
        print("Successfully loaded 'zonal_data.csv'")
    except FileNotFoundError:
        print("Error: 'zonal_data.csv' not found.")
        print("Please place the data file in the same folder as this script.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the data: {e}")
        exit()
    
    #get zone
    valid_zones = ['HOUSTON', 'NORTH', 'SOUTH', 'WEST']
    target_zone = input(f"Enter the zone ({', '.join(valid_zones)}): ").upper()
    if target_zone not in valid_zones:
        print(f"Error: Invalid zone '{target_zone}'. Please run again.")
        exit()

    #get hour and horizon 
    try:
        target_hour_str = input("Enter the target hour (1-24): ")
        target_hour = int(target_hour_str)
        
        H_to_simulate_str = input("Enter the number of hours to simulate (H): ")
        H_to_simulate = int(H_to_simulate_str)
    except ValueError:
        print("Error: Hour and H must be valid integers.")
        exit()
    #validate hour range
    if not (1 <= target_hour <= 24):
        print(f"Error: Invalid hour '{target_hour}'. Must be between 1 and 24.")
        exit()
    if H_to_simulate < 0:
        print(f"Error: Invalid simulation hours '{H_to_simulate}'. Must be 0 or greater.")
        exit()

    #run the simulation 
    print(f"\nStarting simulation for {target_zone}, Hour {target_hour}, H={H_to_simulate}...")
    simulated_errors = fit_kde_with_simulation(
        df_final, 
        target_zone, 
        target_hour, 
        H_to_simulate)

