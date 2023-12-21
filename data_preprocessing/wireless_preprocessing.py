'''
Required Input files:
1. BLE_data.csv (from the app "Camera2Video")
2. timestamp.csv (manually record the start and end time of each RP)
3. routes.csv (testing routes' labels)
Output file
1. wireless_training.csv (dataset for training)
2. wireless_testing.csv (1 sample each label cut from training dataset)
3. routes/{route_name}.csv (evaluate performance)
3. max_beacon_rssi.pickle (record the max RSSI value of each beacon)
4. min_beacon_rssi.pickle (record the min RSSI value of each beacon)

python .\wireless_preprocessing.py --directory D:\Experiment\data\231218\GalaxyA51
'''

import numpy as np
import pandas as pd
import pickle
import argparse
import os

class WirelessPreprocessor:
    def __init__(self, directory):
        self.directory = directory
        self.table = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                                [ 0, 1, 0, 0, 0, 0,11, 0, 0, 0, 0,21],\
                                [ 0, 2, 0, 0, 0, 0,12, 0, 0, 0, 0,22],\
                                [ 0, 3, 0, 0, 0, 0,13, 0, 0, 0, 0,23],\
                                [ 0, 4, 0, 0, 0, 0,14, 0, 0, 0, 0,24],\
                                [ 0, 5, 0, 0, 0, 0,15, 0, 0, 0, 0,25],\
                                [ 0, 6, 0, 0, 0, 0,16, 0, 0, 0, 0,26],\
                                [ 0, 7, 0, 0, 0, 0,17, 0, 0, 0, 0,27],\
                                [ 0, 8, 0, 0, 0, 0,18, 0, 0, 0, 0,28],\
                                [ 0, 9, 0, 0, 0, 0,19, 0, 0, 0, 0,29],\
                                [ 0,10, 0, 0, 0, 0,20, 0, 0, 0, 0,30],\
                                [ 0,31,32,33,34,35,36,37,38,39,40,41] ], dtype = int)
        self.min_beacon = {}
        self.max_beacon = {}

    def change_directory(self):
        try:
            os.chdir(self.directory)
            print("Change Directory to:", self.directory)
        except FileNotFoundError:
            print("Can not find target directory:", self.directory)
        except Exception as e:
            print("Error:", e)

    def uuid_to_b_num(self, s):
        if s[0:30] == 'e20a39f4-73f5-4bc4-a12f-17d1ad':
            uuid_num = int(s[-1])  # last char of UUID
        else:
            uuid_num = 0
        return uuid_num

    def read_data(self):
        # Read BLE_data.csv
        ble_data = pd.read_csv('BLE_data.csv')

        # Read timestamp.csv
        timestamp_data = pd.read_csv('timestamp.csv')

        # Convert 'Time' column to Timestamp objects
        ble_data['Time'] = pd.to_datetime(ble_data['Time'], format="%Y%m%d %H:%M:%S")

        # Create a new DataFrame for processed data
        self.wireless_fingerprint = pd.DataFrame(
            columns=['label', 'Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7'])

        # Process data from timestamp_data
        for index, row in timestamp_data.iterrows():
            position = self.table[row['positionY']][row['positionX']]
            start_time = pd.to_datetime(row['start'])
            end_time = pd.to_datetime(row['end'])

            # Filter BLE_data within the time range
            subset_data = ble_data[(ble_data['Time'] >= start_time) & (ble_data['Time'] <= end_time)]

            if not subset_data.empty:
                subset_data['UUID'] = subset_data['UUID'].apply(self.uuid_to_b_num)
                rssi = [np.zeros((8))]
                rssi[0][:] = np.nan
                rssi[0][0] = position
                sample = 0

                for i in range(len(subset_data) - 1):
                    curr = subset_data.iloc[i]['Time']
                    next_time = subset_data.iloc[i + 1]['Time']
                    rssi_idx = int(subset_data.iloc[i]['UUID'])

                    if rssi_idx > 0:
                        rssi[sample][rssi_idx] = subset_data.iloc[i]['RSSI']

                    if curr != next_time:
                        sample += 1
                        rssi.append(np.zeros((8)))
                        rssi[sample][:] = np.nan
                        rssi[sample][0] = position

                rssi_idx = int(subset_data.iloc[len(subset_data) - 1]['UUID'])
                if rssi_idx > 0:
                    rssi[sample][rssi_idx] = subset_data.iloc[len(subset_data) - 1]['RSSI']

                new_row = pd.DataFrame(np.array(rssi),
                                       columns=['label', 'Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5',
                                                'Beacon_6', 'Beacon_7'])

                self.wireless_fingerprint = pd.concat([self.wireless_fingerprint, new_row], ignore_index=True)

    def normalize_data(self):
        self.wireless_fingerprint = self.wireless_fingerprint.astype(float)
        self.wireless_fingerprint = self.wireless_fingerprint.interpolate(limit_direction='both')

        beacons = ['Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7']

        for bc in beacons:
            beacon_value = self.wireless_fingerprint[bc]
            minimum = min(beacon_value)
            maximum = max(beacon_value)
            self.min_beacon[bc] = minimum
            self.max_beacon[bc] = maximum

        for i in range(len(self.wireless_fingerprint)):
            row = self.wireless_fingerprint.iloc[i]
            for bc in beacons:
                normalize_value = (row[bc] - self.min_beacon.get(bc)) / (
                            self.max_beacon.get(bc) - self.min_beacon.get(bc))
                row[bc] = normalize_value

    def save_min_max_rssi(self):
        with open(f'min_beacon_rssi.pickle', 'wb') as f:
            pickle.dump(self.min_beacon, f)
        with open(f'max_beacon_rssi.pickle', 'wb') as f:
            pickle.dump(self.max_beacon, f)

    def create_testing_set(self):
        testing_set = pd.DataFrame(columns=self.wireless_fingerprint.columns)

        for label in self.wireless_fingerprint['label'].unique():
            sample = self.wireless_fingerprint[self.wireless_fingerprint['label'] == label].sample(n=1)
            testing_set = pd.concat([testing_set, sample], ignore_index=True)
            self.wireless_fingerprint = self.wireless_fingerprint.drop(sample.index)
            print(f"  Label{label} samples: {len(self.wireless_fingerprint[self.wireless_fingerprint['label'] == label])}")

        testing_set.to_csv('wireless_testing.csv', index=False)

    def save_training_set(self):
        self.wireless_fingerprint.to_csv('wireless_training.csv', index=False)

    def create_route_csv(self, route_name, label_order):
        route_data = pd.DataFrame(columns=self.wireless_fingerprint.columns)
        testing_set = pd.read_csv('wireless_testing.csv')
        for label in label_order:
            sample = testing_set[testing_set['label'] == label].sample(n=1)
            route_data = pd.concat([route_data, sample], ignore_index=True)

        routes_dir = 'routes'
        os.makedirs(routes_dir, exist_ok=True)
        route_data.to_csv(os.path.join(routes_dir, f'{route_name}.csv'), index=False)

    def read_routes(self, filename='routes.csv'):
        routes = pd.read_csv(filename, header=None)
        for index, row in routes.iterrows():
            route_name = row[0]
            label_order = row[1:].dropna().astype(int).tolist()
            self.create_route_csv(route_name, label_order)


def main():
    parser = argparse.ArgumentParser(description='Wireless Preprocessing')
    parser.add_argument('--directory', type=str, default=os.getcwd(),
                        help='Change to this directory')

    args = parser.parse_args()

    # Instantiate the WirelessPreprocessor class
    wireless_preprocessor = WirelessPreprocessor(args.directory)

    # Perform wireless preprocessing steps
    wireless_preprocessor.change_directory()
    wireless_preprocessor.read_data()
    wireless_preprocessor.normalize_data()
    wireless_preprocessor.save_min_max_rssi()
    wireless_preprocessor.create_testing_set()
    wireless_preprocessor.save_training_set()
    wireless_preprocessor.read_routes(filename='D:\\Experiment\\data\\routes.csv')


if __name__ == "__main__":
    main()
