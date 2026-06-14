# 📉 End-to-End Telco Customer Churn Prediction

## 📌 Project Overview

This project is an end-to-end production machine learning pipeline designed to predict customer churn in the telecommunications industry.

By identifying customers highly likely to cancel their subscriptions, businesses can proactively target them with retention campaigns. The model is optimized for **Recall** to ensure the lowest possible rate of false negatives (missed churners).

The pipeline goes beyond a Jupyter Notebook—it encompasses data validation, modular ML scripting, experiment tracking, API serving, containerization, and automated cloud deployment.

## 🏗️ Architecture & Pipeline

1. **Data Engineering:** Ingestion and automated cleaning of the Blastchar IBM Telco dataset.
2. **Data Validation:** Boundary and schema checks using `great_expectations`.
3. **Model Training:** XGBoost Classifier optimized via `Optuna` for hyperparameter tuning.
4. **Experiment Tracking:** Logging parameters, metrics, and serialized models locally via `MLflow`.
5. **Model Serving:** Exposing the inference endpoint via a `FastAPI` REST app and a `Gradio` web UI.
6. **Containerization:** Packaging the environment into a lightweight `Docker` image.
7. **CI/CD & Cloud:** Automated image builds via `GitHub Actions` and serverless deployment to `AWS ECS (Fargate)` accessed via an Application Load Balancer.

## 🗂️ Project Structure

```text
telco-churn-project/
├── .github/workflows/      # CI/CD pipelines (GitHub Actions)
├── app/                    # FastAPI & Gradio UI application
├── configs/                # Configuration files (YAML/JSON)
├── data/
│   ├── external/           # Third-party data sources
│   ├── processed/          # Cleaned, encoded data ready for modeling
│   └── raw/                # Original immutable Kaggle dataset
├── docker/                 # Dockerfiles and container configs
├── great_expectations/     # Data validation suites
├── mlruns/                 # Local MLflow tracking logs (git-ignored)
├── notebooks/              # Jupyter notebooks for initial EDA
├── scripts/                # Utility execution scripts
├── src/                    # Core modular Python package
│   ├── data/               # Ingestion and cleaning scripts
│   ├── features/           # Encoding and transformation pipelines
│   ├── models/             # XGBoost training and tuning scripts
│   └── utils/              # Helper functions
├── tests/                  # Pytest unit and integration tests
├── .gitignore              # Files to hide from Git tracking
├── README.md               # Project documentation
└── requirements.txt        # Pinned Python dependencies
```
