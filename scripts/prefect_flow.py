from prefect import flow, task
import requests
import pandas as pd
import subprocess
from ml_pipeline.config import Config
from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.model_training import train_model
from ml_pipeline.model_evaluation import evaluate_model
from ml_pipeline.eda import perform_eda

# Define tasks
@task
def load_data_task():
    return load_data()

@task
def preprocess_data_task(df):
    return preprocess_data(df)

@task
def train_model_task(processed_data):
    return train_model(processed_data)

@task
def evaluate_model_task(model, X_val, y_val, scaler):
    return evaluate_model(model, X_val, y_val, scaler)

@task
def perform_eda_task(df):
    perform_eda(df)

@task
def build_and_push_docker_image_task():
    """
    Task to build and push the Docker image.
    """
    try:
        # Log in to Docker Hub
        subprocess.run(
            ["docker", "login", "-u", Config.DOCKER_HUB_USERNAME, "-p", Config.DOCKER_HUB_TOKEN],
            check=True,
        )

        # Build the Docker image
        subprocess.run(
            ["docker", "build", "-t", f"{Config.DOCKER_HUB_USERNAME}/flask-app:latest", "."],
            check=True,
        )

        # Push the Docker image
        subprocess.run(
            ["docker", "push", f"{Config.DOCKER_HUB_USERNAME}/flask-app:latest"],
            check=True,
        )

        print("Docker image built and pushed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error building or pushing Docker image: {e}")
        raise


# Define the flow
@flow(name="ML Pipeline Flow")
def ml_pipeline_flow():
    # Load data
    df = load_data_task()
    
    # Perform EDA
    perform_eda_task(df)
    
    # Preprocess data
    processed_data = preprocess_data_task(df)
    
    # Train model
    model, scaler, X_train, X_val, y_train, y_val = train_model_task(processed_data)
    
    # Evaluate model
    evaluation_metrics = evaluate_model_task(model, X_val, y_val, scaler)
    
    # Log evaluation metrics
    print("Evaluation Metrics:", evaluation_metrics)

    # Build and push Docker image
    build_and_push_docker_image_task()

# Run the flow
if __name__ == "__main__":
    ml_pipeline_flow()