import yaml
from concrete.exception import ConcreteException
import os,sys
import pandas as pd
from concrete.constants import *
import numpy as np
import dill


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise ConcreteException(e,sys)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ConcreteException(e,sys) from e

def load_data(file_path:str, schema_file_path:str)-> pd.DataFrame:
    try:
        schema = read_yaml_file(schema_file_path)
        columns = schema[SCHEMA_COLUMNS_KEY]
        df = pd.read_csv(file_path)
        error_message = ""
        for column in df.columns:
            if column in list(columns.keys()):
                df[column].astype(columns[column])
            else:
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        if len(error_message) > 0:
            raise Exception(error_message)
        return df
    except Exception as e:
        raise ConcreteException(e,sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ConcreteException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ConcreteException(e, sys) from e


def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ConcreteException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ConcreteException(e,sys) from e


def get_previous_timestamp_dir(dir:str):
    try:
        folder_name = os.listdir(dir)
        previous_timestamp_dir = os.path.join(dir, f"{folder_name[-2]}")
        return previous_timestamp_dir
    except Exception as e:
        raise ConcreteException(e, sys) from e
