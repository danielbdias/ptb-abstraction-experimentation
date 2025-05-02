import csv
import pickle
import os

def save_pickle_data(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_raw_data(data, file_path):
    with open(file_path, 'w') as file:
        file.write(data)

def load_pickle_data(file_path):
    if not file_exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def get_ground_fluents_to_ablate_from_csv(file_path: str):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            return set(row)

    raise ValueError(f'No fluents to ablate found in file {file_path}')

def file_exists(file_path : str) -> bool:
    return os.path.exists(file_path)
        
def read_file(file_path : str) -> str:
    with open(file_path, 'r') as file:
        return file.read()
