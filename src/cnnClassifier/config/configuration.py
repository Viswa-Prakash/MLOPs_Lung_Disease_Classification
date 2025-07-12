import os
from pathlib import Path
from typing import Dict, Any
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    FineTuneConfig,
    EvaluationConfig,
    ModelPusherConfig
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            source_URL=config["source_URL"],
            local_data_file=Path(config["local_data_file"]),
            unzip_dir=Path(config["unzip_dir"])
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config["prepare_base_model"]
        params = self.params
        create_directories([config["root_dir"]])
        return PrepareBaseModelConfig(
            root_dir=Path(config["root_dir"]),
            base_model_path=Path(config["base_model_path"]),
            updated_base_model_path=Path(config["updated_base_model_path"]),
            base_model_name=params["BASE_MODEL"]["NAME"],
            include_top=params["BASE_MODEL"]["INCLUDE_TOP"],
            weights=params["BASE_MODEL"]["WEIGHTS"],
            input_shape=params["BASE_MODEL"]["INPUT_SHAPE"],
            trainable=params["BASE_MODEL"]["TRAINABLE"],
            dropout=params["UPDATED_BASE_MODEL"]["DROPOUT"],
            classes=params["CLASSES"],
            activation=params["UPDATED_BASE_MODEL"]["ACTIVATION"],
            kernel_regularizer_l2=params["UPDATED_BASE_MODEL"]["KERNEL_REGULARIZER_L2"],
            learning_rate=params["TRAINING"]["LEARNING_RATE"],
            optimizer=params["TRAINING"]["OPTIMIZER"],
            loss=params["TRAINING"]["LOSS"],
            metrics=[params["TRAINING"]["METRICS"]]
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config["training"]
        prepare_base_model = self.config["prepare_base_model"]
        params = self.params
        create_directories([config["root_dir"]])
        return TrainingConfig(
            root_dir=Path(config["root_dir"]),
            trained_model_path=Path(config["trained_model_path"]),
            updated_base_model_path=Path(prepare_base_model["updated_base_model_path"]),
            training_data=Path(self.config["data_ingestion"]["unzip_dir"]),
            epochs=params["TRAINING"]["EPOCHS"],
            learning_rate=params["TRAINING"]["LEARNING_RATE"],
            batch_size=params["BATCH_SIZE"],
            is_augmentation=params["AUGMENTATION"],
            image_height=params["IMAGE_HEIGHT"],
            image_width=params["IMAGE_WIDTH"],
            loss=params["TRAINING"]["LOSS"],
            optimizer=params["TRAINING"]["OPTIMIZER"],
            metrics=[params["TRAINING"]["METRICS"]],
            base_model_name=params["BASE_MODEL"]["NAME"],
            include_top=params["BASE_MODEL"]["INCLUDE_TOP"],
            weights=params["BASE_MODEL"]["WEIGHTS"],
            input_shape=params["BASE_MODEL"]["INPUT_SHAPE"],
            trainable=params["BASE_MODEL"]["TRAINABLE"],
            dropout=params["UPDATED_BASE_MODEL"]["DROPOUT"],
            classes=params["CLASSES"],
            activation=params["UPDATED_BASE_MODEL"]["ACTIVATION"],
            kernel_regularizer_l2=params["UPDATED_BASE_MODEL"]["KERNEL_REGULARIZER_L2"]
        )

    def get_fine_tune_config(self) -> FineTuneConfig:
        ft = self.params["FINE_TUNE"]
        return FineTuneConfig(
            enabled=ft["ENABLED"],
            epochs=ft["EPOCHS"],
            learning_rate=ft["LEARNING_RATE"],
            unfreeze_from_layer=ft["UNFREEZE_FROM_LAYER"]
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_cfg = self.config["evaluation"]
        return EvaluationConfig(
            path_of_model=Path(eval_cfg["path_of_model"]),
            training_data=Path(self.config["data_ingestion"]["unzip_dir"]),
            all_params=self.params,
            mlflow_uri=eval_cfg["mlflow_uri"],
            experiment_name=eval_cfg["experiment_name"],
            registered_model_name=eval_cfg["registered_model_name"],
            dagshub_repo_owner=eval_cfg["dagshub_repo_owner"],
            dagshub_repo_name=eval_cfg["dagshub_repo_name"],
            params_image_size=[self.params["IMAGE_HEIGHT"], self.params["IMAGE_WIDTH"], 3],
            params_batch_size=self.params["BATCH_SIZE"]
        )

    def get_model_pusher_config(self) -> ModelPusherConfig:
        push_cfg = self.config["model_pusher"]
        return ModelPusherConfig(
            bucket_name=push_cfg["bucket_name"],
            s3_model_path=push_cfg["s3_model_path"],
            local_model_path=Path(self.config["training"]["trained_model_path"])
        )
