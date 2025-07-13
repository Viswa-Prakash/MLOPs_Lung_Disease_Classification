import os
import boto3
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger


class PredictionPipeline:
    _model = None  # Class-level cache for the loaded model
    _model_path = None  # Local path to downloaded model

    def __init__(self, filename):
        self.filename = filename
        config = ConfigurationManager()
        pusher_config = config.get_model_pusher_config()

        self.bucket_name = pusher_config.bucket_name
        self.s3_model_path = pusher_config.s3_model_path

        self.image_size = (224, 224)
        self.class_names = [
            "Bacterial Pneumonia",
            "Corona Virus Disease",
            "Normal",
            "Tuberculosis",
            "Viral Pneumonia"
        ]

    def download_model_from_s3(self):
        if not PredictionPipeline._model_path or not os.path.exists(PredictionPipeline._model_path):
            logger.info("Attempting to download model from S3 bucket: %s", self.bucket_name)
            local_model_path = os.path.join(tempfile.gettempdir(), "model.h5")
            boto3.client("s3").download_file(self.bucket_name, self.s3_model_path, local_model_path)
            PredictionPipeline._model_path = local_model_path
            logger.info("Model downloaded to: %s", local_model_path)
        return PredictionPipeline._model_path

    def predict(self):
        if PredictionPipeline._model is None:
            logger.info("Loading model from S3...")
            model_path = self.download_model_from_s3()
            PredictionPipeline._model = load_model(model_path)

        logger.info("Preparing image for prediction...")
        img = image.load_img(self.filename, target_size=self.image_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        logger.info("Making prediction...")
        preds = PredictionPipeline._model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_class = self.class_names[pred_idx]

        logger.info(f"Prediction completed: {pred_class}")
        return [{"predicted_class": pred_class}]
