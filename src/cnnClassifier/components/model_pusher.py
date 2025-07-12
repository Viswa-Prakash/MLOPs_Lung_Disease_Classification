import os
import boto3
from cnnClassifier.entity.config_entity import ModelPusherConfig
from cnnClassifier import logger
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure environment variables are correctly loaded into boto3's environment
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

class ModelPusher:
    def __init__(self, config: ModelPusherConfig):
        self.config = config

    def upload_model_to_s3(self):
        try:
            logger.info(f"Uploading model from {self.config.local_model_path} to s3://{self.config.bucket_name}/{self.config.s3_model_path}")
            
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            s3.upload_file(
                Filename=str(self.config.local_model_path),
                Bucket=self.config.bucket_name,
                Key=self.config.s3_model_path
            )

            logger.info("✅ Model uploaded successfully to S3.")

        except Exception as e:
            logger.error("❌ Failed to upload model to S3.")
            raise e
