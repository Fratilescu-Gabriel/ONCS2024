import numpy as np 
import pandas as pd 
import csv
import os

PATH_TO_DATA = r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0\package\DATA"
SIMULATION_DATASET_DEFAULT_NAME = r"sim-dataset-"

def my_to_csv( content:pd.DataFrame, column_names = [], file_name = None, new_file = True, file_append = True):
    if not new_file:
        if file_append:
            content.to_csv( PATH_TO_DATA + r"\\" + get_last_filename(), mode='a')
        else:
            content.to_csv( PATH_TO_DATA + r"\\" + get_last_filename(), mode= 'w')
    else:
        if file_name == None:
            index = int(get_last_filename().split('-')[-1].split('.')[0]) + 1
            new_filename = f"sim-dataset-{index}.csv"
            content.to_csv( PATH_TO_DATA + r"\\" + new_filename, ' ', np.nan, header=column_names, mode='w')
        else:
            content.to_csv( PATH_TO_DATA + r"\\" + file_name, ' ', np.nan, header=column_names, mode='w')


def get_last_filename():
    files = [filename for filename in os.listdir(PATH_TO_DATA) if filename.startswith(SIMULATION_DATASET_DEFAULT_NAME)]
    if not files:
        return 'sim-dataset-1'  # No files matching the format found
    files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    return files[-1]

print(get_last_filename())
