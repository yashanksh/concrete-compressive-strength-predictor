from concrete.entity.config_entity import DataInjestionConfig
from concrete.exception import ConcreteException
import sys, os
from concrete.logger import logging
from concrete.entity.artifact_entity import DataIngestionArtifact
import tarfile
from six.moves import urllib
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    def __init__(self, data_ingestion_config: DataInjestionConfig) -> None:
        try:
            logging.info(f"{'='*20} Data Ingestion Log Started {'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def download_concrete_data(self)-> str:
        try:
            #Extracting remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url
            #downloaded file's directory
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(raw_data_dir, exist_ok=True)
            raw_file_name = os.path.basename(download_url)
            raw_file_path = os.path.join(raw_data_dir,
                                raw_file_name)
            logging.info(f"Downloading [{raw_file_name}] from [{download_url}] to [{raw_data_dir}]")
            urllib.request.urlretrieve(download_url, raw_file_path)
            logging.info(f"Downloaded [{raw_file_name}] successfully.")
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def get_previous_train_file_path(self, train_file_name):
        try:
            ingested_data_dir, train_folder = os.path.split(self.data_ingestion_config.ingested_train_dir)
            timestamp_dir, ingested_data_folder = os.path.split(ingested_data_dir)
            data_ingestion_dir =  os.path.dirname(timestamp_dir)
            previous_timestamp_folder = os.listdir(data_ingestion_dir)[-1]
            previous_train_file_path = os.path.join(data_ingestion_dir,
                                                    previous_timestamp_folder,
                                                    ingested_data_folder,
                                                    train_folder,
                                                    train_file_name)
            return previous_train_file_path
        except Exception as e:
            raise ConcreteException(e,sys) from e


    def split_data_as_train_test(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            previous_train_file_path = self.get_previous_train_file_path(file_name)
            concrete_file_path = os.path.join(raw_data_dir,
                                file_name)
            logging.info(f'Reading xls file: [{concrete_file_path}]')
            concrete_df = pd.read_csv(concrete_file_path)
            concrete_df['strength_cat'] = pd.cut(concrete_df['concrete_compressive_strength'],
                                       bins= [0,20,40,60,80,np.inf],
                                       labels= [1,2,3,4,5])
            logging.info(f"Splitting the dataset into train and test")
            strat_train_set = None
            strat_test_set = None
            split = StratifiedShuffleSplit(n_splits=1,
                    test_size=0.2, 
                    random_state=13)
            for train_index, test_index in split.split(concrete_df,concrete_df['strength_cat']):
                strat_train_set = concrete_df.loc[train_index].drop(['strength_cat'], axis=1)
                strat_test_set = concrete_df.loc[test_index].drop(['strength_cat'], axis=1)
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                              file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                             file_name)
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,
                            exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,
                                       index=False)
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,
                            exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,
                                      index=False)
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data Ingestion Completed sucessfully",
                                                            previous_train_file_path=previous_train_file_path)
            logging.info(f'Data Ingestion Artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.download_concrete_data()
            return self.split_data_as_train_test()
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log Ended{'='*20} \n\n")