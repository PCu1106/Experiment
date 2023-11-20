python .\DNN.py \
--training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
--model_path 231116.h5

python .\dnn.py \
--testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                    D:\Experiment\data\220318\GalaxyA51\routes \
                    D:\Experiment\data\231117\GalaxyA51\routes \
--model_path 231116.h5

python .\evaluator.py \
--model_name 231116 \
--directory DNN