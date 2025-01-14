# EC2 Spot Instance Price Forecasting

This project provides a machine learning system for forecasting Amazon EC2 Spot Instance prices. The system uses an LSTM (Long Short-Term Memory) model to predict future spot prices based on historical data. The project is modularized, containerized with Docker, and includes a Flask API for serving predictions. It also supports deployment on Saturn Cloud and CI/CD integration with GitHub Actions.

---

## Table of Contents

1. [Business Domain Knowledge](#business-domain-knowledge)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Deployment](#deployment)
   - [Docker](#docker)
   - [BentoML](#bentoml)
   - [Saturn Cloud](#saturn-cloud)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [License](#license)

---

## Business Domain Knowledge

### What are EC2 Spot Instances?

Amazon EC2 Spot Instances allow you to use spare EC2 capacity at a significantly reduced cost compared to On-Demand Instances. However, the prices for Spot Instances fluctuate based on supply and demand, and AWS can terminate instances with a two-minute warning when capacity is no longer available.

### Why Forecast Spot Prices?

Forecasting EC2 Spot Instance prices helps businesses:

- **Optimize Costs**: Predict low-price periods to launch workloads cost-effectively.
- **Plan Workloads**: Schedule non-critical workloads during low-price periods.
- **Reduce Risk**: Avoid unexpected terminations by anticipating price spikes.

### Key Challenges

- **Price Volatility**: Spot prices can change rapidly.
- **Data Quality**: Historical price data may contain gaps or anomalies.
- **Model Accuracy**: Accurate forecasting requires handling time-series data effectively.

---

## Project Structure

```
ec2-spot-forecast/
├── api/                  # Flask API for serving predictions
├── artifacts/            # Data and trained models
│   ├── data/             # Historical EC2 spot price data
│   └── models/           # Trained LSTM models
├── config/               # Configuration files and environment variables
├── services/             # BentoML service for model serving
├── tests/                # Unit tests for the project
├── utils/                # Utility functions (data preprocessing, model training)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # Project documentation
└── .github/workflows/    # GitHub Actions CI/CD workflows
```

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/ec2-spot-forecast.git
   cd ec2-spot-forecast
   ```
2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```
3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Data Preprocessing**

Run the data preprocessing script to prepare the data for training:

```bash
python -m utils.data_preprocessing
```

### 2. **Model Training**

Train the LSTM model using the preprocessed data:

```bash
python -m utils.model_training
```

### 3. **Run the Flask API**

Start the Flask API to serve predictions:

```bash
python api/app.py
```

### 4. **Make Predictions**

Send a POST request to the `/predict` endpoint with historical price data:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
```

---

## Deployment

### Docker

1. **Build the Docker Image**:

   ```bash
   docker build -t ec2-spot-forecast .
   ```
2. **Run the Docker Container**:

   ```bash
   docker run -p 5000:5000 ec2-spot-forecast
   ```

### BentoML

1. **Package the Model**:

   ```bash
   bentoml build
   ```
2. **Serve the Model**:

   ```bash
   bentoml serve EC2SpotForecastService:latest
   ```

### Saturn Cloud

1. **Push the Docker Image to Docker Hub**:

   ```bash
   docker tag ec2-spot-forecast your-dockerhub-username/ec2-spot-forecast:latest
   docker push your-dockerhub-username/ec2-spot-forecast:latest
   ```
2. **Deploy on Saturn Cloud**:

   - Create a new deployment on Saturn Cloud.
   - Use the Docker image from Docker Hub.
   - Set environment variables (e.g., `MODEL_PATH`, `SCALER_PATH`).

---

## Testing

Run unit tests to ensure the project works as expected:

```bash
pytest tests/ --cov=utils --cov=api --cov=services --cov-report=html
```

View the coverage report in the `htmlcov` folder.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Amazon Web Services (AWS) for providing EC2 Spot Instance data.
- TensorFlow and Keras for the LSTM model implementation.
- Flask and BentoML for model serving.

---

This `README.md` provides a comprehensive guide for using, deploying, and understanding the EC2 Spot Instance price forecasting project. Let me know if you need further assistance!
