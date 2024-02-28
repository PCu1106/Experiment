import parser
from pathlib import Path
import os
import pandas as pd

dataset_folder = Path('..')
ssv_folder = dataset_folder/'data'/'site_surveys'

site_surveys = [f.name for f in os.scandir(ssv_folder) if f.is_dir()]
site_surveys = parser.sorted_nicely(site_surveys)
# for site_survey in site_surveys:
#     dataset = parser.parse_dataset(ssv_folder/site_survey)
dataset0611 = parser.parse_dataset(ssv_folder/'2019-06-11')
dataset1211 = parser.parse_dataset(ssv_folder/'2019-12-11')
# print(dataset['rssis'])
# print(dataset['coordinates'])
count = 0
pre = 0
cor2la = {}
labels0611 = pd.DataFrame(columns=['label', 'x', 'y'])
labels1211 = pd.DataFrame(columns=['label', 'x', 'y'])
for i, line in enumerate(dataset0611['coordinates']):
    if pre != line:
        count+=1
    pre = line
    cor2la[line[0]] = count
print(cor2la)

for i, line in enumerate(dataset0611['coordinates']):
    datamap = {'label':cor2la[line[0]], 'x': line[0], 'y': line[1]}
    labels0611.loc[i] = datamap
for i, line in enumerate(dataset1211['coordinates']):
    datamap = {'label':cor2la[line[0]], 'x': line[0], 'y': line[1]}
    labels1211.loc[i] = datamap
labels0611.to_csv('labels0611.csv', index=False)
labels1211.to_csv('labels1211.csv', index=False)