# from ml_pipeline.data_processing import load_data, preprocess_data
# from ml_pipeline.eda import perform_eda
# from ml_pipeline.model_training import train_model
# from ml_pipeline.model_evaluation import evaluate_model
# from ml_pipeline.config import Config

# def main():
#     # Load and preprocess data
#     df = load_data()
#     processed_data = preprocess_data(df)
    
#     # Perform EDA
#     perform_eda(processed_data)
    
#     # Train and evaluate the model
#     model, scaler = train_model(processed_data)
#     # evaluate_model(model, (processed_data, scaler))

#     # Assuming processed_data is a DataFrame with features and target
#     X = processed_data.drop(columns=['Cost'])  # Features (drop the target column)
#     y = processed_data['Cost']  # Target

#     # Drop non-numeric columns from X (if not already done)
#     X = X.select_dtypes(include='number')

#     # Evaluate the model
#     metrics = evaluate_model(model, (X, y), scaler)

# if __name__ == "__main__":
#     main()

from ml_pipeline.data_processing import load_data, preprocess_data
from ml_pipeline.eda import perform_eda
from ml_pipeline.model_training import train_model
from ml_pipeline.model_evaluation import evaluate_model
from ml_pipeline.config import Config

def main():
    # Load and preprocess data
    df = load_data()
    processed_data = preprocess_data(df)
    
    # Perform EDA (optional)
    perform_eda(df)
    
    # Train the model
    model, scaler, X_train, X_val, y_train, y_val = train_model(processed_data)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_val, y_val, scaler)
    
    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"MAE: {metrics['mae']}")
    print(f"MSE: {metrics['mse']}")
    print(f"RMSE: {metrics['rmse']}")

if __name__ == "__main__":
    main()