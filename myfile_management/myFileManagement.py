import numpy as np
import pandas as pd 
import csv
import os
from decimal import *

PATH_TO_DATA = r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0-new\ONCS2024\DATA"
if not os.path.exists(PATH_TO_DATA):
    PATH_TO_DATA = r"C:\Eu\ONCS\v1.0\ONCS2024\DATA"
SIMULATION_DATASET_DEFAULT_NAME = r"sim-dataset-"
DEFAULT_CHUNKING_VOLUME = 1000  # Default number of rows per chunk
DEFAULT_DELIMITER = ','  # Default delimiter for CSV files


def my_to_csv(data, column_names=[], file_name=None, new_file=True, file_append=True):
    """
    Save data to a CSV file.

    Args:
        data: The data to be saved. It can be either a pandas DataFrame or a list of numpy arrays.
        chunk_size (int): Number of rows per chunk.
        column_names (list, optional): List of column names. Defaults to [].
        file_name (str, optional): Name of the file to save. Defaults to None.
        new_file (bool, optional): Whether to create a new file (True) or append to an existing one (False). Defaults to True.
        file_append (bool, optional): If new_file is False, whether to append to the existing file (True) or overwrite it (False). Defaults to True.
    """
    # if not column_names:
    #     raise ValueError("Column names must be provided.")

    if new_file:
        if file_name is None:
            index = int(get_last_filename().split('-')[-1].split('.')[0]) + 1
            file_name = f"sim-dataset-{index}.csv"
    else:
        file_name = get_last_filename()

    file_path = os.path.join(PATH_TO_DATA, file_name)
    mode = 'a' if not new_file and file_append else 'w'

    with open(file_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=DEFAULT_DELIMITER)
        if new_file:
            writer.writerow(column_names)

        # Transpose data arrays if necessary and write to the file
        if isinstance(data, pd.DataFrame):
            # Transpose DataFrame and convert to a list of lists
            writer.writerows(data.transpose().values.tolist())
        else:
            # Convert each array to a list of lists and cast values to appropriate numeric type
            for array in data:
                if isinstance(array, np.ndarray):
                    array = array.T.tolist()  # Transpose and convert to list of lists if it's a NumPy array
                writer.writerows(map(list, map(np.float64, row)) for row in array)

    # print(f"Data saved to '{file_path}'.")


def get_last_filename():
    files = [filename for filename in os.listdir(PATH_TO_DATA) if filename.startswith(SIMULATION_DATASET_DEFAULT_NAME)]
    if not files:
        return 'sim-dataset-0.csv'  # No files matching the format found
    files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    return files[-1]



def my_chunking(filename, numpy_array=True, chunk_size=DEFAULT_CHUNKING_VOLUME, yield_header=False):
    file_path = os.path.join(PATH_TO_DATA, filename)
    with open(file_path, 'r') as file:
        # Read the header (assuming the first row is a header)
        header = file.readline().strip().split(',')
        if yield_header:
            if numpy_array:
                header = np.array(header)
            yield header  # Yield the header as the first chunk

        # Read the remaining rows in chunks
        while True:
            chunk = []
            for _ in range(chunk_size):
                line = file.readline().strip()
                if not line:
                    break  # End of file
                values = [float(x) for x in line.split(',')]
                chunk.append(values)
            if not chunk:
                return  # End of file
            if numpy_array:
                chunk = np.array(chunk)
                chunk = np.transpose(chunk)
                print(type(chunk))
            yield chunk

def check_file_exists(file_name):
    """
    Check if a file exists in the directory described by PATH_TO_DATA.

    Args:
        file_name (str): Name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = os.path.join(PATH_TO_DATA, file_name)
    return os.path.isfile(file_path)
        
