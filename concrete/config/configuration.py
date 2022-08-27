import sys,os
from concrete.entity.config_entity import DataInjestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig, TrainingPipelineConfig
from concrete.util.util import read_yaml_file
from concrete.constants import *
from concrete.exception import ConcreteException
from concrete.logger import logging

class Configuration:
    def __init__(self,
                config_file_path:str = CONFIG_FILE_PATH,
                current_time_stamp:str = get_current_time_stamp()
                ) -> None:
        try:
            self.config_info = read_yaml_file(config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_data_ingestion_config(self) -> DataInjestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(artifact_dir,
                                            DATA_INGESTION_ARTIFACT_DIR,
                                            self.time_stamp)

            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                            data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])

            ingested_dir = os.path.join(data_ingestion_artifact_dir,
                            data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            ingested_train_dir = os.path.join(ingested_dir,
                                data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])
            ingested_test_dir = os.path.join(ingested_dir,
                                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])

            data_ingestion_config = DataInjestionConfig(
                                    dataset_download_url=dataset_download_url, 
                                    raw_data_dir=raw_data_dir, 
                                    ingested_train_dir=ingested_train_dir, 
                                    ingested_test_dir=ingested_test_dir
            )
            logging.info(f'DataInjestionConfig: {data_ingestion_config}')
            return data_ingestion_config
        except Exception as e:
            raise ConcreteException(e,sys) from e


    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_validation_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            data_validation_artifact_dir = os.path.join(artifact_dir,
                                                        DATA_VALIDATION_ARTIFACT_DIR,
                                                        self.time_stamp)
            report_file_path = os.path.join(data_validation_artifact_dir,
                                            data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            report_page_file_path = os.path.join(data_validation_artifact_dir,
                                                data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])
            schema_file_path = os.path.join(ROOT_DIR,
                                            data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                            data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            data_validation_config = DataValidationConfig(schema_file_path=schema_file_path,
                                                            report_file_path=report_file_path,
                                                            report_page_file_path=report_page_file_path)
            logging.info(f"DataValidationConfig: {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise ConcreteException(e,sys) from e


    def get_data_transformation_config(self)-> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_artifact_dir = os.path.join(artifact_dir,
                                            DATA_TRANSFORMATION_ARTIFACT_DIR,
                                            self.time_stamp)
            transformed_dir = os.path.join(
                            data_transformation_artifact_dir,
                            data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY])
            transformed_train_dir = os.path.join(
                                    transformed_dir,
                                    data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY])
            transformed_test_dir = os.path.join(
                                transformed_dir,
                                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY])
            preprocessed_object_file_path = os.path.join(
                                            data_transformation_artifact_dir,
                                            data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                            data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME_KEY])
            data_transformation_config = DataTransformationConfig(
                                        transformed_train_dir= transformed_train_dir,
                                        transformed_test_dir= transformed_test_dir,
                                        preprocessed_object_file_path= preprocessed_object_file_path)
            logging.info(f"DataTransformationConfig: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def get_model_trainer_config(self)-> ModelTrainerConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_trainer_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            model_trainer_artifact_dir = os.path.join(artifact_dir,
                                        MODEL_TRAINER_ARTIFACT_DIR,
                                        self.time_stamp)
            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
                                model_trainer_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                                model_trainer_info[MODEL_TRAINER_MODEL_FILE_NAME_KEY])
            base_accuracy = model_trainer_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            model_config_file_path = os.path.join(model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_DIR_NAME_KEY],
                                                  model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY])
            model_trainer_config = ModelTrainerConfig(
                                    trained_model_file_path= trained_model_file_path,
                                    base_accuracy= base_accuracy,
                                    model_config_file_path=model_config_file_path)
            logging.info(f"Model Trainer Config: {model_trainer_config}")
            return model_trainer_config                 
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_evaluation_info = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            model_evaluation_file_path = os.path.join(artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR,
                                        model_evaluation_info[MODEL_EVALUATION_FILE_NAME_KEY])
            model_evaluation_config = ModelEvaluationConfig(
                                    model_evaluation_file_path=model_evaluation_file_path,
                                    time_stamp=self.time_stamp)
            logging.info(f"Model Evaluation Config: {model_evaluation_config}")
            return model_evaluation_config
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def get_model_pusher_config(self)-> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(ROOT_DIR,
                                           model_pusher_info[MODEL_PUSHER_EXPORT_DIR_KEY],
                                           time_stamp)
            model_pusher_config = ModelPusherConfig(export_dir_path= export_dir_path)
            logging.info(f"Model Pusher Config : {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_info = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
                                        training_pipeline_info[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_info[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipeling config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise ConcreteException(e,sys) from e