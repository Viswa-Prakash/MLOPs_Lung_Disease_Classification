from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation_mlflow import ModelEvaluationPipeline
from cnnClassifier.pipeline.stage_05_model_pusher import ModelPusherPipeline


def run_pipeline_stage(stage_name, stage_instance):
    try:
        logger.info("\n" + "*" * 50)
        logger.info(f">>>>>> Stage: {stage_name} started <<<<<<")
        stage_instance.main()
        logger.info(f">>>>>> Stage: {stage_name} completed <<<<<<")
        logger.info("x" * 50 + "\n")
    except Exception as e:
        logger.exception(f"Exception occurred in stage: {stage_name}: {e}")
        raise e


if __name__ == "__main__":
    run_pipeline_stage("Data Ingestion", DataIngestionTrainingPipeline())
    run_pipeline_stage("Prepare Base Model", PrepareBaseModelTrainingPipeline())
    run_pipeline_stage("Training", ModelTrainingPipeline())
    run_pipeline_stage("Evaluation", ModelEvaluationPipeline())
    run_pipeline_stage("Model Pusher", ModelPusherPipeline())
