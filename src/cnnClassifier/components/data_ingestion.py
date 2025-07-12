import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Downloads the dataset zip file from Google Drive.
        Returns the path to the downloaded file.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_path = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_path), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_path}")

            file_id = dataset_url.split("/")[-2]
            gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_download_path), quiet=False)

            logger.info(f"Successfully downloaded data to: {zip_download_path} | Size: {get_size(Path(zip_download_path))}")

            return zip_download_path

        except Exception as e:
            logger.error(f"Error while downloading file: {e}")
            raise e

    def extract_zip_file(self) -> None:
        """
        Extracts the downloaded zip file to the specified directory.
        """
        try:
            unzip_dir = self.config.unzip_dir
            zip_file_path = self.config.local_data_file
            os.makedirs(unzip_dir, exist_ok=True)

            logger.info(f"Extracting zip file: {zip_file_path} to directory: {unzip_dir}")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)

            logger.info(f"Successfully extracted files to: {unzip_dir}")

        except Exception as e:
            logger.error(f"Error while extracting zip file: {e}")
            raise e
