# **AWS EC2 Spot Price Forecasting**

## **Overview**

This project focuses on forecasting **AWS EC2 Spot Instance prices** using machine learning techniques. Spot Instances allow you to use spare AWS EC2 capacity at a significantly reduced cost, but their prices fluctuate based on supply and demand. Accurately predicting these prices can help optimize costs and improve resource allocation.

The project includes:

- **Data Collection**: Fetching historical EC2 Spot Price data.
- **Data Preprocessing**: Cleaning and preparing the data for modeling.
- **Exploratory Data Analysis (EDA)**: Visualizing trends and patterns in the data.
- **Model Training**: Building and training machine learning models to forecast spot prices.
- **Deployment**: Deploying the trained model as a Flask app using Docker and Koyeb.

---

## **Features**

- **Data Pipeline**: Automated data collection and preprocessing.
- **Machine Learning Models**: Includes models like LSTM, ARIMA, and XGBoost for forecasting.
- **API Endpoint**: A Flask app to serve predictions via an API.
- **CI/CD Pipeline**: Automated deployment using GitHub Actions and Koyeb.

---

## **Project Structure**

```
.
├── app/
│   ├── app.py                # Flask app for serving predictions
│   └── requirements.txt      # Python dependencies for the Flask app
├── data/
│   ├── processed/            # Processed data files
│   └── raw/                  # Raw data files
├── ml_pipeline/
│   ├── config.py             # Configuration settings
│   ├── data_processing.py    # Data preprocessing scripts
│   ├── eda.py                # Exploratory data analysis scripts
│   ├── model_training.py     # Model training scripts
│   ├── model_evaluation.py   # Model evaluation scripts
│   └── utils.py              # Utility functions
├── models/
│   └── lstm_model/           # Trained LSTM model files
├── notebooks/
│   ├── eda.ipynb             # Jupyter notebook for EDA
│   └── model_training.ipynb  # Jupyter notebook for model training
├── scripts/
│   ├── prefect_flow.py       # Prefect pipeline for automation
│   └── run_pipeline.py       # Script to run the entire pipeline
├── Dockerfile                # Dockerfile for containerizing the Flask app
├── requirements.txt          # Python dependencies for the project
├── README.md                 # Project documentation
└── .github/workflows/        # GitHub Actions workflows for CI/CD
```

---

## **Setup Instructions**

### **1. Prerequisites**

- **Python 3.10**: Install Python 3.10 or higher.
- **Docker**: Install Docker to containerize the Flask app.
- **AWS CLI**: Install the AWS CLI to fetch EC2 Spot Price data.
- **Koyeb Account**: Create a Koyeb account for deployment.

### **2. Clone the Repository**

```bash
git clone https://github.com/lillianphyo/mlzoomcamp_capstone_1_2025.git
cd mlzoomcamp_capstone_1_2025
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. EC2 Spot Price Data**

The script to fetch historical EC2 Spot Price data was provided in notebook as this part not exported.:

```bash
/data/raw/p2-east-1b.csv
```

### **5. Run the Data Pipeline**

Execute the Prefect pipeline to preprocess data, train models, and evaluate performance:

```bash
python scripts/run_pipeline.py
```

---

# **Model Serving**

### 1 Flask

Start the Flask app to serve predictions:

```bash
python app/app.py
```

The API will be available at `http://localhost:5000`.

#### **API Endpoints**

- **GET `/predict`**: Get spot price forecasts for a specific instance type.
  - **Parameters**:
    - `instance_type`: The EC2 instance type (e.g., `m5.large`).
    - `region`: The AWS region (e.g., `us-east-1`).
  - **Example**:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
    ```

---

## **Deployment**

### **1. Build Docker Image**

Build the Docker image for the Flask app:

```bash
docker build -t your-dockerhub-username/flask-app:latest .
```

### **2. Push Docker Image to Docker Hub**

Push the Docker image to Docker Hub:

```bash
docker push your-dockerhub-username/flask-app:latest
```

### **3. Deploy to Koyeb**

Deploy the Docker image to Koyeb using the Koyeb CLI:

```bash
koyeb service create \
  --name flask-app \
  --image your-dockerhub-username/flask-app:latest \
  --ports 5000:http \
  --env FLASK_APP=app.py \
  --env FLASK_ENV=production \
  --regions fra
```

---

## **CI/CD Pipeline**

The project includes a GitHub Actions workflow to automate the deployment process. The workflow:

1. Builds and pushes the Docker image to Docker Hub.
2. Deploys the Docker image to Koyeb.

To trigger the workflow, push changes to the `master` branch.

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **AWS**: For providing EC2 Spot Price data.
- **Koyeb**: For simplifying deployment with their platform.
- **Prefect**: For enabling workflow automation.

---

## **Contact**

For questions or feedback, please contact:

- **Your Name**: khinpyaephyosan@gmail.com
- **GitHub**: lillianphyo

---

This README provides a comprehensive guide to using and deploying your AWS EC2 Spot Price Forecasting project. 🚀
