from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


# Shared model config (used in multiple stages)
@dataclass(frozen=True)
class BaseModelParams:
    base_model_name: str
    include_top: bool
    weights: str
    input_shape: List[int]
    trainable: bool
    dropout: float
    classes: int
    activation: str
    kernel_regularizer_l2: float


# Training hyperparameters (used by PrepareBaseModel and Training)
@dataclass(frozen=True)
class TrainingParams:
    learning_rate: float
    optimizer: str
    loss: str
    metrics: List[str]


# Stage 01: Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


# Stage 02: Prepare Base Model
@dataclass(frozen=True)
class PrepareBaseModelConfig(BaseModelParams, TrainingParams):
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path


# Stage 03: Training
@dataclass(frozen=True)
class TrainingConfig(BaseModelParams, TrainingParams):
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    epochs: int
    batch_size: int
    is_augmentation: bool
    image_height: int
    image_width: int


# Stage 03-1: Fine-tuning
@dataclass(frozen=True)
class FineTuneConfig:
    enabled: bool
    epochs: int
    learning_rate: float
    unfreeze_from_layer: int


# Stage 04: Evaluation
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: Dict
    mlflow_uri: str
    experiment_name: str
    registered_model_name: str
    dagshub_repo_owner: str
    dagshub_repo_name: str
    params_image_size: List[int]
    params_batch_size: int


# Stage 05: Model Pusher
@dataclass(frozen=True)
class ModelPusherConfig:
    bucket_name: str
    s3_model_path: str
    local_model_path: Path
