import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger
import dagshub


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.val_ds = None
        self.score = None

    def load_model(self):
        logger.info(f"Loading model from: {self.config.path_of_model}")
        self.model = tf.keras.models.load_model(self.config.path_of_model)

    def prepare_validation_dataset(self):
        logger.info("Preparing validation dataset using image_dataset_from_directory...")

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            label_mode="int",
            shuffle=False
        )

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        AUTOTUNE = tf.data.AUTOTUNE

        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=AUTOTUNE
        ).cache().prefetch(buffer_size=AUTOTUNE)

        self.val_ds = val_ds
        logger.info("Validation dataset ready.")

    def evaluate(self):
        logger.info("Evaluating model...")
        self.score = self.model.evaluate(self.val_ds)
        logger.info(f"Evaluation complete. Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self._save_scores()

    def _save_scores(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }
        save_json(path=Path("scores.json"), data=scores)
        logger.info("Saved evaluation scores to scores.json.")

    def log_to_mlflow(self):
        logger.info("Logging to MLflow and DagsHub...")

        # Initialize DagsHub MLflow setup
        dagshub.init(
            repo_owner=self.config.dagshub_repo_owner,
            repo_name=self.config.dagshub_repo_name,
            mlflow=True
        )

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics({
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            })

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name=self.config.registered_model_name
                )
                logger.info(f"Model registered to DagsHub MLflow: {self.config.registered_model_name}")
            else:
                mlflow.keras.log_model(self.model, "model")
                logger.info("Model logged locally")
