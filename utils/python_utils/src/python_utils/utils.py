# -*- coding: UTF-8 -*-
"""Python Utils.
"""


import pickle


def save_object(obj, file_path):
    """
    Save a Python object to a file.
    
    :param obj: The Python object to save
    :param file_path: The path of the file where the object will be stored
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the object: {e}")
        


def load_object(file_path):
    """
    Load a Python object from a file.
    
    :param file_path: The path of the file where the object is stored
    :return: The loaded Python object
    """
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        return None