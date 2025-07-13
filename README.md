# End-to-End Lung Disease Image Classification

## Project Overview
This project focuses on building an End-to-End Image Classification Pipeline to detect lung diseases using chest X-ray images. The pipeline includes:

- Data ingestion, augmentation, and preprocessing
- Transfer learning using EfficientNetB0
- Training and evaluation pipeline
- MLflow tracking with DagsHub integration
- Dockerization
- CI/CD deployment to AWS EC2 via GitHub Actions

## Problem Statement
Lung diseases such as Pneumonia, Tuberculosis, and COVID-19 are critical health issues requiring early and accurate detection. Manual diagnosis through X-rays is time-consuming and may suffer from inter-reader variability.

### Objective:
Build an automated classification model that classifies X-ray images into 5 classes:
- Bacterial Pneumonia
- Corona Virus Disease (COVID-19)
- Normal
- Tuberculosis
- Viral Pneumonia

## Dataset
The dataset consists of labeled chest X-ray images for the above 5 categories. It is preprocessed and stored in the artifacts/data_ingestion directory during pipeline execution.

## Workflow & Pipeline Architecture
constants/
├── All file paths, model names, etc.

entity/
├── Dataclasses for config structures

components/
├── data_ingestion.py
├── prepare_base_model.py
├── training.py
├── evaluation.py
├── model_pusher.py

pipeline/
├── stage_01_data_ingestion.py
├── stage_02_prepare_base_model.py
├── stage_03_training.py
├── stage_04_evaluation.py
├── stage_05_model_pusher.py

main.py
├── Orchestrates all pipeline stages

## How to Run
### Step 1: Setup Environment
'''bash
conda create -n lung-tf291 python=3.8 -y
conda activate lung-tf291
pip install -r requirements.txt

### Step 2: Set Environment Variables
'''bash
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>

### Step 3: Run Pipeline
'''bash
python main.py

## Deployment using AWS & GitHub Actions (CI/CD)
### Steps:
- Login to AWS Console

- Create IAM user with the following permissions:
 - AmazonEC2FullAccess
 - AmazonEC2ContainerRegistryFullAccess

- Create ECR Repository
 - Example URI: 123456789012.dkr.ecr.us-east-1.amazonaws.com/lung-disease-classifier

- Create EC2 instance (Ubuntu) and install Docker:
'''bash
sudo apt-get update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

- Configure EC2 as GitHub self-hosted runner
 - Go to Repo > Settings > Actions > Runners > New self-hosted runner

- Create GitHub Secrets:

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
AWS_ECR_LOGIN_URI


