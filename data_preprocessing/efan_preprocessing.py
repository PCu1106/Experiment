import pandas as pd
import os

# Directory containing CSV files
csv_directory = 'D:\\ExperimentData\\0318_92589_wireless_train_raw_data\\0318_92589_train\\GalaxyA51'

# Output files
merged_csv_filename = 'BLE_data.csv'
timestamp_csv_filename = 'timestamp.csv'

# List to store dataframes
dfs = []

# List to store timestamp data
timestamp_data = []

# Iterate through CSV files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(csv_directory, filename)
        positionX, positionY = filename[:2], filename[2:4]

        # Read CSV file
        df = pd.read_csv(filepath, parse_dates=['Time'])

        # Store timestamp data
        timestamp_data.append({
            'positionX': int(positionX),
            'positionY': int(positionY),
            'start': df['Time'].min().strftime('%Y%m%d %H:%M:%S'),
            'end': df['Time'].max().strftime('%Y%m%d %H:%M:%S')
        })

        # Format the 'Time' column
        df['Time'] = df['Time'].dt.strftime('%Y%m%d %H:%M:%S')
        
        # Append dataframe to the list
        dfs.append(df[['Time', 'UUID', 'RSSI']])

# Concatenate dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Save merged dataframe to "BLE_data.csv"
merged_df.to_csv(merged_csv_filename, index=False)

# Create timestamp dataframe and save to "timestamp.csv"
timestamp_df = pd.DataFrame(timestamp_data)
timestamp_df.to_csv(timestamp_csv_filename, index=False)
