import os
import sys
from concrete.exception import ConcreteException
from concrete.util.util import load_object
import pandas as pd, numpy as np


class ConcreteData:

    def __init__(self,
                cement: np.float64,
                blast_furnace_slag: np.float64,
                fly_ash: np.float64,
                water: np.float64,
                superplasticizer: np.float64,
                coarse_aggregate: np.float64,
                fine_aggregate: np.float64,
                age: np.int64,
                concrete_compressive_strength: np.float64 = None
                ):
        try:
            self.cement = cement
            self.blast_furnace_slag = blast_furnace_slag
            self.fly_ash = fly_ash
            self.water = water
            self.superplasticizer = superplasticizer
            self.coarse_aggregate = coarse_aggregate
            self.fine_aggregate = fine_aggregate
            self.age = age
            self.concrete_compressive_strength = concrete_compressive_strength
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_concrete_input_data_frame(self):

        try:
            concrete_input_dict = self.get_concrete_data_as_dict()
            return pd.DataFrame(concrete_input_dict)
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_concrete_data_as_dict(self):
        try:
            input_data = {
                "cement": [self.cement],
                "blast_furnace_slag": [self.blast_furnace_slag],
                "fly_ash": [self.fly_ash],
                "water": [self.water],
                "superplasticizer": [self.superplasticizer],
                "coarse_aggregate": [self.coarse_aggregate],
                "fine_aggregate": [self.fine_aggregate],
                "age": [self.age]
                }
            return input_data
        except Exception as e:
            raise ConcreteException(e, sys)


class ConcretePredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            concrete_compressive_strength = model.predict(X)
            return concrete_compressive_strength
        except Exception as e:
            raise ConcreteException(e, sys) from e