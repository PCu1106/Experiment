'''
python .\compare_rssi_histograms.py D:\Experiment\data\220318\GalaxyA51\wireless_training.csv D:\Experiment\data\231116\GalaxyA51\wireless_training.csv
python .\compare_rssi_histograms.py D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-06-11\wireless_training.csv D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data\2020-02-19\wireless_training.csv
'''
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_rssi_csv(filename):
    df = pd.read_csv(filename)
    labels = df['label'].values
    rssi_data = df.drop(columns=['label']).values.astype(np.float32)
    return labels, rssi_data

def calculate_histogram(filename):
    labels, rssi_data = load_rssi_csv(filename)
    # Randomly select 500 values from rssi_data
    selected_indices = np.random.choice(rssi_data.size, size=500, replace=False)
    selected_values = rssi_data.flatten()[selected_indices]
    # Calculate histogram of selected values with dynamic range
    hist_range = (np.min(selected_values), np.max(selected_values))
    hist = np.histogram(selected_values, bins=100, range=hist_range)[0].astype(np.float32)
    return hist

def calculate_similarity(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate RSSI histogram similarity between two CSV files.')
    parser.add_argument('file1', type=str, help='First CSV file containing RSSI data')
    parser.add_argument('file2', type=str, help='Second CSV file containing RSSI data')
    args = parser.parse_args()

    hist1 = calculate_histogram(args.file1)
    hist2 = calculate_histogram(args.file2)
    print(hist1)
    print(hist2)

    similarity = calculate_similarity(hist1, hist2)
    print(f'Similarity between {args.file1} and {args.file2}: {similarity}')
