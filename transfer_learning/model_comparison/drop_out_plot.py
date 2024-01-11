'''
python .\drop_out_plot.py \
    --model_name_list DANN DANN_AE \
    --list_data_list D:\Experiment\transfer_learning\DANN\231116_231117\12\my_list.pkl \
                    D:\Experiment\transfer_learning\DANN_AE\231116_231117\122\my_list.pkl \
    --title Model_Comparison_in_Space_Changing
'''

import matplotlib.pyplot as plt
import os
import pickle
import argparse

def plot_lines(data_drop_out_list, domain1, domain2, domain3, output_path, title):
    domain = domain3
    # plt.plot(data_drop_out_list, domain1, marker='o', label='Target Domain', color='blue')
    plt.plot(data_drop_out_list, domain, marker='o', label='Target Domain', color='orange')
    # plt.plot(data_drop_out_list, domain3, marker='o', label='Target Domain', color='green')

    plt.xlabel('data dropout ratio')
    plt.ylabel('MDE (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(data_drop_out_list, labels=[f'{x:.1f}' for x in data_drop_out_list])
    plt.ylim(0, 3)

    # 在每個點的上方顯示數字
    for x, y1, y2, y3 in zip(data_drop_out_list, domain1, domain2, domain3):
        # plt.text(x, y1, f'{y1:.3f}', ha='center', va='bottom')
        plt.text(x, y2, f'{y2:.3f}', ha='center', va='bottom')
        # plt.text(x, y3, f'{y3:.3f}', ha='center', va='bottom')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.chdir(output_path)
    plt.savefig('Dropout_Data.png')
    file_name = 'my_list.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(domain, file)
    os.chdir('..')

def model_comparison(model_name_list, list_data_list, title):
    dropout_ratios = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    plt.figure(figsize=(10, 6))

    # Define colors for the first three models
    colors = ['red', 'black', 'purple']

    for i, (model_name, data_path) in enumerate(zip(model_name_list, list_data_list)):
        with open(data_path, 'rb') as file:
            data_list = pickle.load(file)

        # Use specified color for the first three models, and then cycle for additional models
        color = colors[i % len(colors)]

        plt.plot(dropout_ratios, data_list, label=model_name, marker='o', color=color)

        # Annotate each point with its value
        for x, y in zip(dropout_ratios, data_list):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('data dropout ratio')
    plt.ylabel('MDE (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(dropout_ratios, labels=[f'{x:.1f}' for x in dropout_ratios])
    plt.ylim(0, 2)
    plt.savefig(f'Dropout_Data_{title}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--model_name_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--list_data_list', nargs='+', type=str, help='Path to the pickle file')
    parser.add_argument('--title', type=str, help='Path to the pickle file')
    args = parser.parse_args()
    model_comparison(args.model_name_list, args.list_data_list, args.title)