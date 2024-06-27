import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

import pickle

class CustomException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

def save_obj(file_path, obj):
    try:
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f'Object saved successfully at {file_path}')
    except Exception as e:
        raise CustomException(f"Error saving object: {e}")

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print(f'Object loaded successfully from {file_path}')
        return obj
    except Exception as e:
        raise CustomException(f"Error loading object: {e}")
