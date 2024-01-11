import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_and_save_dataset(input_file, output_file_1, output_file_2, test_size=0.2, random_seed=42):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Split the dataset while maintaining a balanced distribution of labels
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, stratify=df['label'])

    # Save the split datasets to separate CSV files
    train_df.to_csv(output_file_1, index=False)
    test_df.to_csv(output_file_2, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test for corr')
    parser.add_argument('--data', type=str, required=True, help='Path to the source domain data file')
    parser.add_argument('--part1', type=str, required=True, help='Path to the target domain data file')
    parser.add_argument('--part2', type=str, required=True, help='Path to the target domain data file')

    args = parser.parse_args()

    # Specify the input and output file paths
    input_csv_path = args.data  # Update with your actual file path
    output_csv_path_1 = args.part1
    output_csv_path_2 = args.part2

    # Set the test_size parameter for train_test_split (adjust as needed)
    test_size_ratio = 0.5

    # Set a random seed for reproducibility (optional)
    random_seed_value = 42

    # Perform the split and save the datasets
    split_and_save_dataset(input_csv_path, output_csv_path_1, output_csv_path_2, test_size=test_size_ratio, random_seed=random_seed_value)
