from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import ModelTrainer
from cnnClassifier import logger

STAGE_NAME = "Model Trainer"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load training configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Initialize and build model
        trainer = ModelTrainer(config=training_config)
        trainer.build_model()

        # Train model with callbacks
        trainer.train()


if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'*' * 25}")
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
        logger.info(f"{'x' * 25}\n\n")
    except Exception as e:
        logger.exception(f"Exception occurred in stage {STAGE_NAME}: {e}")
        raise e
