import numpy as np 
import pandas as pd 
import csv
import os

PATH_TO_DATA = r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0\package\DATA"
if not os.path.exists(PATH_TO_DATA):
    PATH_TO_DATA = r"C:\Eu\ONCS\v1.0\ONCS2024\DATA"
SIMULATION_DATASET_DEFAULT_NAME = r"sim-dataset-"
DEFAULT_CHUNKING_VOLUME = 100000  # Default number of rows per chunk
DEFAULT_DELIMITER = ','  # Default delimiter for CSV files

def my_to_csv(data, column_names=[], file_name=None, new_file=True, file_append=True):
    """
    Save data to a CSV file.

    Args:
        data: The data to be saved. It can be either a pandas DataFrame or a list of numpy arrays.
        column_names (list, optional): List of column names. Defaults to [].
        file_name (str, optional): Name of the file to save. Defaults to None.
        new_file (bool, optional): Whether to create a new file (True) or append to an existing one (False). Defaults to True.
        file_append (bool, optional): If new_file is False, whether to append to the existing file (True) or overwrite it (False). Defaults to True.
    """
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

        # Write data to the file in chunks
        if isinstance(data, pd.DataFrame):
            for i in range(0, len(data), DEFAULT_CHUNKING_VOLUME):
                chunk = data.iloc[i:i + DEFAULT_CHUNKING_VOLUME]
                writer.writerows(chunk.values)
        else:
            for i in range(0, len(data[0]), DEFAULT_CHUNKING_VOLUME):
                chunk = [column[i:i + DEFAULT_CHUNKING_VOLUME] for column in data]
                writer.writerows(zip(*chunk))

    print(f"Data saved to '{file_path}'.")


def get_last_filename():
    files = [filename for filename in os.listdir(PATH_TO_DATA) if filename.startswith(SIMULATION_DATASET_DEFAULT_NAME)]
    if not files:
        return 'sim-dataset-0'  # No files matching the format found
    files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    return files[-1]


def my_chunking(filename, numpy_array=True, chunk_size=DEFAULT_CHUNKING_VOLUME, yield_header=False):
    file_path = os.path.join(PATH_TO_DATA, filename)
    with open(file_path, 'r') as file:
        # Read the header (assuming the first row is a header)
        header = file.readline().strip().split()
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
                chunk.append(line.split())
            if not chunk:
                break  # End of file
            if numpy_array:
                chunk = np.array(chunk)
            yield chunk
