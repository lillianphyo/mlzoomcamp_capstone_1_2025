# from prefect import flow, task
# from ml_pipeline.data_processing import load_data, preprocess_data
# from ml_pipeline.eda import perform_eda
# from ml_pipeline.model_training import train_model
# from ml_pipeline.model_evaluation import evaluate_model
# from ml_pipeline.config import Config

# @task
# def load_and_preprocess_data():
#     data = load_data()
#     return preprocess_data(data)

# @task
# def perform_eda_task(data):
#     perform_eda(data)

# @task
# def train_and_evaluate_model(data):
#     model, scaler = train_model(data)
#     evaluate_model(model, (data, scaler))
#     return model, scaler

# @flow(name="ML Pipeline")
# def ml_pipeline_flow():
#     data = load_and_preprocess_data()
#     perform_eda_task(data)
#     model, scaler = train_and_evaluate_model(data)

# if __name__ == "__main__":
#     ml_pipeline_flow()

# from prefect import flow, task
# from ml_pipeline.data_processing import load_data, preprocess_data
# from ml_pipeline.eda import perform_eda
# from ml_pipeline.model_training import train_model
# from ml_pipeline.model_evaluation import evaluate_model
# from ml_pipeline.config import Config

# @task
# def load_and_preprocess_data():
#     """Load and preprocess the data."""
#     data = load_data()
#     processed_data = preprocess_data(data)
#     return processed_data

# @task
# def perform_eda_task(data):
#     """Perform exploratory data analysis."""
#     perform_eda(data)

# @task
# def train_and_evaluate_model(processed_data):
#     """Train the model and evaluate it."""
#     # Train the model
#     model, scaler, X_train, X_val, y_train, y_val = train_model(processed_data)
    
#     # Evaluate the model
#     metrics = evaluate_model(model, X_val, y_val, scaler)
    
#     # Log evaluation metrics
#     print("Evaluation Metrics:")
#     print(f"MAE: {metrics['mae']}")
#     print(f"MSE: {metrics['mse']}")
#     print(f"RMSE: {metrics['rmse']}")
    
#     return model, scaler

# @flow(name="ML Pipeline")
# def ml_pipeline_flow():
#     """Main Prefect flow for the ML pipeline."""
#     # Load and preprocess data
#     processed_data = load_and_preprocess_data()
    
#     # Perform EDA
#     perform_eda_task(processed_data)
    
#     # Train and evaluate the model
#     model, scaler = train_and_evaluate_model(processed_data)

# if __name__ == "__main__":
#     ml_pipeline_flow()

# from prefect import flow, task
# from ml_pipeline.data_processing import load_data, preprocess_data
# from ml_pipeline.eda import perform_eda
# from ml_pipeline.model_training import train_model
# from ml_pipeline.model_evaluation import evaluate_model
# from ml_pipeline.config import Config

# @task
# def load_and_preprocess_data():
#     """Load and preprocess the data."""
#     data = load_data()
#     processed_data = preprocess_data(data)
#     return processed_data

# @task
# def perform_eda_task(data):
#     """Perform exploratory data analysis."""
#     perform_eda(data)

# @task
# def train_model_task(processed_data):
#     """Train the model."""
#     model, scaler, X_train, X_val, y_train, y_val = train_model(processed_data)
#     return model, scaler, X_val, y_val

# @task
# def evaluate_model_task(model, X_val, y_val, scaler):
#     """Evaluate the model."""
#     metrics = evaluate_model(model, X_val, y_val, scaler)
#     return metrics

# @flow(name="ML Pipeline")
# def ml_pipeline_flow():
#     """Main Prefect flow for the ML pipeline."""
#     # Load and preprocess data
#     processed_data = load_and_preprocess_data()
    
#     # Perform EDA
#     perform_eda_task(processed_data)
    
#     # Train the model
#     model, scaler, X_val, y_val = train_model_task(processed_data)
    
#     # Evaluate the model
#     metrics = evaluate_model_task(model, X_val, y_val, scaler)
    
#     # Log evaluation metrics
#     print("Evaluation Metrics:")
#     print(f"MAE: {metrics['mae']}")
#     print(f"MSE: {metrics['mse']}")
#     print(f"RMSE: {metrics['rmse']}")

# if __name__ == "__main__":
#     ml_pipeline_flow()

from prefect import flow, task
from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.eda import perform_eda
from ml_pipeline.model_training import train_model
from ml_pipeline.model_evaluation import evaluate_model
from ml_pipeline.config import Config

@task
def load_and_preprocess_data():
    """Load and preprocess the data."""
    df = load_data()
    processed_data = preprocess_data(df)
    return processed_data

@task
def perform_eda_task(df):
    """Perform exploratory data analysis."""
    perform_eda(df)

@task
def train_model_task(processed_data):
    """Train the model."""
    model, scaler, X_train, X_val, y_train, y_val = train_model(processed_data)
    return model, scaler, X_val, y_val

@task
def evaluate_model_task(model, X_val, y_val, scaler):
    """Evaluate the model."""
    metrics = evaluate_model(model, X_val, y_val, scaler)
    return metrics

@flow(name="ML Pipeline")
def ml_pipeline_flow():
    """Main Prefect flow for the ML pipeline."""
    # Load and preprocess data
    df, processed_data = load_and_preprocess_data()
    
    # Perform EDA on the raw DataFrame
    perform_eda_task(df)
    
    # Train the model using the processed data
    model, scaler, X_val, y_val = train_model_task(processed_data)
    
    # Evaluate the model
    metrics = evaluate_model_task(model, X_val, y_val, scaler)
    
    # Log evaluation metrics
    print("Evaluation Metrics:")
    print(f"MAE: {metrics['mae']}")
    print(f"MSE: {metrics['mse']}")
    print(f"RMSE: {metrics['rmse']}")

if __name__ == "__main__":
    ml_pipeline_flow()