import parser
from pathlib import Path
import os
import pandas as pd

dataset_folder = Path('..')
ssv_folder = dataset_folder/'data'/'site_surveys'

site_surveys = [f.name for f in os.scandir(ssv_folder) if f.is_dir()]
site_surveys = parser.sorted_nicely(site_surveys)
cor2la = {}

for site_survey in site_surveys:
    if(site_survey == '2019-02-19' or site_survey == '2019-03-25'): # number of RP different, don't use them
        continue
    dataset = parser.parse_dataset(ssv_folder / site_survey)
    
    count = 0
    pre = 0
    
    labels = pd.DataFrame(columns=['label', 'x', 'y'])
    if len(cor2la) == 0:
        for i, line in enumerate(dataset['coordinates']):
            if pre != line:
                count += 1
            pre = line
            cor2la[line[0]] = count

    for i, line in enumerate(dataset['coordinates']):
        datamap = {'label': cor2la[line[0]], 'x': line[0], 'y': line[1]}
        labels.loc[i] = datamap
    
    labels.to_csv(f'labels_{site_survey}.csv', index=False)