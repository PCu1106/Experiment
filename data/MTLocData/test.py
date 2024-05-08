import pickle
import os
folder_path = 'OfficeP1+P2'
# folder_path = 'Mall'
label_map_file = os.path.join(folder_path, 'label_map.pkl')

# 讀取label_map.pkl文件
with open(label_map_file, 'rb') as f:
    label_map = pickle.load(f)

# 使用label_map
for key, value in label_map.items():
    print(f'Coordinate: {key}, Label: {value}')

# 使用label_map並顯示最大最小值
x_values = [coord[0] for coord in label_map.keys()]
y_values = [coord[1] for coord in label_map.keys()]

print(f'Minimum x value: {min(x_values)}, Maximum x value: {max(x_values)}')
print(f'Minimum y value: {min(y_values)}, Maximum y value: {max(y_values)}')

flipped_dict = {value: key for key, value in label_map.items()}
print(flipped_dict)