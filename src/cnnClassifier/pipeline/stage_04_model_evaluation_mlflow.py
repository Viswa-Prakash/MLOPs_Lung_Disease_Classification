from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Model Evaluation with MLflow"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load evaluation config
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        # Initialize evaluation
        evaluation = Evaluation(config=eval_config)

        # Load model and data
        evaluation.load_model()
        evaluation.prepare_validation_dataset()

        # Evaluate and log to MLflow
        evaluation.evaluate()
        evaluation.log_to_mlflow()


if __name__ == "__main__":
    try:
        logger.info("\n\n" + "*" * 25)
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
        logger.info("x" * 25 + "\n")
    except Exception as e:
        logger.exception(f"Exception occurred in stage {STAGE_NAME}: {e}")
        raise e
