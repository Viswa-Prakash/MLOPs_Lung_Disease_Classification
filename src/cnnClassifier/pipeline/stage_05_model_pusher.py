from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_pusher import ModelPusher
from cnnClassifier import logger

STAGE_NAME = "Model Pusher to S3"

class ModelPusherPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        pusher_config = config.get_model_pusher_config()
        pusher = ModelPusher(config=pusher_config)
        pusher.upload_model_to_s3()

if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelPusherPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
